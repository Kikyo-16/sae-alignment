from __future__ import annotations

import typing as tp

import torch


@torch.no_grad()
def _decoder_columns(module: torch.nn.Module, eps: float = 1e-8) -> torch.Tensor:
    """Return unit-normalized decoder feature vectors with shape [D, H].

    The SAE decoder is expected to be `fc2: Linear(H, D)`, whose weight has
    shape [D, H]. Each column is one decoder feature.
    """
    if not hasattr(module, "fc2") or not hasattr(module.fc2, "weight"):
        raise TypeError("Each SAE module must expose fc2.weight with shape [D, H].")
    w = module.fc2.weight.detach().float()
    return w / w.norm(dim=0, keepdim=True).clamp_min(eps)


@torch.no_grad()
def compute_mid_sae_alignment_scores(
    sae_modules: tp.Sequence[torch.nn.Module],
    target_idx: int = 1,
    eps: float = 1e-8,
    mid_neighbor_topk: int = 8,
) -> dict[str, torch.Tensor]:
    """Compute decoder-orbit alignment scores for one middle sub-SAE.

    For feature i in target SAE s, with low neighbor s-1 and high neighbor s+1:

    low_score(i):
        J_i = top-k nearest neighbors in SAE_s for decoder feature (s-1, i)
              using cosine similarity, equivalently smallest cosine distance.
        candidate_score(i, j) =
            0.5 * [cosdist((s-1, i), (s, j)) + cosdist((s+1, j), (s, i))]
        low_score(i) = min_j candidate_score(i, j), for j in J_i

    high_score(i):
        K_i = top-k nearest neighbors in SAE_s for decoder feature (s+1, i)
              using cosine similarity, equivalently smallest cosine distance.
        candidate_score(i, k) =
            0.5 * [cosdist((s+1, i), (s, k)) + cosdist((s-1, k), (s, i))]
        high_score(i) = min_k candidate_score(i, k), for k in K_i

    alignment_score(i) = 0.5 * [low_score(i) + high_score(i)]

    Lower scores mean better alignment. The top-k step first finds the k nearest
    mid-SAE candidates by cosine similarity, then keeps the candidate path with
    the smallest cycle-consistency distance. Set mid_neighbor_topk=1 to exactly
    recover the old top-1-NN behavior. Returned tensors are detached CPU tensors.
    """
    if target_idx <= 0 or target_idx >= len(sae_modules) - 1:
        raise ValueError(
            f"target_idx={target_idx} needs both low and high neighbors; "
            f"got {len(sae_modules)} SAE modules."
        )

    dec_low = _decoder_columns(sae_modules[target_idx - 1], eps=eps)
    dec_mid = _decoder_columns(sae_modules[target_idx], eps=eps)
    dec_high = _decoder_columns(sae_modules[target_idx + 1], eps=eps)

    if dec_low.shape != dec_mid.shape or dec_mid.shape != dec_high.shape:
        raise ValueError(
            "Neighbor SAE decoders must have the same [D, H] shape; "
            f"got low={tuple(dec_low.shape)}, mid={tuple(dec_mid.shape)}, "
            f"high={tuple(dec_high.shape)}."
        )

    hidden = dec_mid.shape[1]
    feat_idx = torch.arange(hidden, device=dec_mid.device)
    nn_topk = max(1, min(int(mid_neighbor_topk), int(hidden)))

    # sim_low_mid[a, b] = cos(dec_low[:, a], dec_mid[:, b])
    sim_low_mid = dec_low.t().matmul(dec_mid).clamp(-1.0, 1.0)
    sim_high_mid = dec_high.t().matmul(dec_mid).clamp(-1.0, 1.0)

    feat_idx_2d = feat_idx[:, None].expand(-1, nn_topk)

    # Low-side path: low_i -> top-k mid_j candidates -> high_j compared with mid_i.
    low_topk_sim, low_topk_idx = torch.topk(
        sim_low_mid, k=nn_topk, dim=1, largest=True, sorted=True
    )
    low_bridge_dist_k = 1.0 - low_topk_sim
    low_cross_sim_k = sim_high_mid[low_topk_idx, feat_idx_2d]
    low_cross_dist_k = 1.0 - low_cross_sim_k
    low_candidate_score = 0.5 * (low_bridge_dist_k + low_cross_dist_k)
    low_score, low_pick_pos = low_candidate_score.min(dim=1)
    low_nn_idx = low_topk_idx.gather(1, low_pick_pos[:, None]).squeeze(1)

    # High-side path: high_i -> top-k mid_k candidates -> low_k compared with mid_i.
    high_topk_sim, high_topk_idx = torch.topk(
        sim_high_mid, k=nn_topk, dim=1, largest=True, sorted=True
    )
    high_bridge_dist_k = 1.0 - high_topk_sim
    high_cross_sim_k = sim_low_mid[high_topk_idx, feat_idx_2d]
    high_cross_dist_k = 1.0 - high_cross_sim_k
    high_candidate_score = 0.5 * (high_bridge_dist_k + high_cross_dist_k)
    high_score, high_pick_pos = high_candidate_score.min(dim=1)
    high_nn_idx = high_topk_idx.gather(1, high_pick_pos[:, None]).squeeze(1)

    alignment_score = 0.5 * (low_score + high_score)
    sorted_score, sorted_feature_idx = torch.sort(alignment_score, dim=0, descending=False)

    return {
        "alignment_score": alignment_score.detach().cpu(),
        "low_score": low_score.detach().cpu(),
        "high_score": high_score.detach().cpu(),
        "low_nn_idx": low_nn_idx.detach().cpu(),
        "high_nn_idx": high_nn_idx.detach().cpu(),
        "low_topk_idx": low_topk_idx.detach().cpu(),
        "high_topk_idx": high_topk_idx.detach().cpu(),
        "low_candidate_score": low_candidate_score.detach().cpu(),
        "high_candidate_score": high_candidate_score.detach().cpu(),
        "mid_neighbor_topk": torch.tensor(nn_topk, dtype=torch.long),
        "sorted_score": sorted_score.detach().cpu(),
        "sorted_feature_idx": sorted_feature_idx.detach().cpu(),
    }


@torch.no_grad()
def min_alignment_score_for_feature_ids(
    scores: dict[str, torch.Tensor],
    feature_ids: tp.Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    """Return the minimum alignment score among the given SAE feature ids.

    Invalid / out-of-range ids are ignored. If no valid ids remain, returns NaN.
    Alignment scores are cosine-distance based, so lower is better.
    """
    score = scores["alignment_score"].detach().float().cpu().flatten()
    if feature_ids is None:
        return torch.tensor(float("nan"), dtype=torch.float32)
    ids = torch.as_tensor(feature_ids, dtype=torch.long).flatten().cpu()
    if ids.numel() == 0:
        return torch.tensor(float("nan"), dtype=torch.float32)
    valid = (ids >= 0) & (ids < score.numel())
    if not bool(valid.any()):
        return torch.tensor(float("nan"), dtype=torch.float32)
    return score.index_select(0, ids[valid]).min()


def make_alignment_figure(
    scores: dict[str, torch.Tensor],
    *,
    title: str = "SAE feature alignment scores",
):
    """Create a matplotlib figure showing sorted per-feature alignment scores."""
    import matplotlib.pyplot as plt

    sorted_score = scores["sorted_score"].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sorted_score)
    ax.set_title(title)
    ax.set_xlabel("Feature rank, sorted by alignment score")
    ax.set_ylabel("Alignment score (cosine distance, lower is better)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def log_alignment_scores_to_tensorboard(
    logger: tp.Any,
    scores: dict[str, torch.Tensor],
    *,
    global_step: int,
    tag: str = "val/sae_1_alignment_sorted",
    title: str = "SAE-1 feature alignment scores",
) -> bool:
    """Log the sorted alignment figure to a TensorBoard-like logger.

    Returns True if a figure was logged. Works with PyTorch Lightning's
    TensorBoardLogger, or any logger whose `.experiment` has `add_figure`.
    """
    experiment = getattr(logger, "experiment", None)
    add_figure = getattr(experiment, "add_figure", None)
    if add_figure is None:
        return False

    fig = make_alignment_figure(scores, title=title)
    try:
        add_figure(tag, fig, global_step=global_step)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return True


def _feature_usage_to_frequency(
    feature_usage_counts: torch.Tensor,
    *,
    topk: int | None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert per-feature shared-mask counts to per-frame selection frequency.

    `feature_usage_counts[i]` counts how often feature i appears in the shared
    top-k mask over the validation epoch. Since exactly topk features are chosen
    per frame, the estimated number of evaluated frames is sum(counts) / topk.
    The returned value is therefore approximately P(feature i is selected in a
    validation frame). Its range is [0, 1] when top-k indices are unique.
    """
    counts = feature_usage_counts.detach().float().cpu().flatten()
    total_selected = counts.sum().clamp_min(eps)
    if topk is None or int(topk) <= 0:
        return counts / total_selected
    n_frames = (total_selected / float(int(topk))).clamp_min(eps)
    return counts / n_frames


def _topk_agreement_to_normalized(
    agreement_scores: torch.Tensor,
    agreement_counts: torch.Tensor,
) -> torch.Tensor:
    """Return SAE-1-conditioned mean activation disagreement per feature.

    The legacy argument names are kept for API compatibility. In the current
    metric, `agreement_scores[i]` is the validation sum of relative activation
    disagreement values over frames where SAE-1 selected feature i, and
    `agreement_counts[i]` is the number of such SAE-1 selections. Lower values
    are better. Features never selected by SAE-1 are set to NaN so plots do not
    silently treat inactive features as perfectly agreeing.
    """
    numer = agreement_scores.detach().float().cpu().flatten()
    denom = agreement_counts.detach().float().cpu().flatten()
    if numer.numel() != denom.numel():
        raise ValueError(
            "agreement_scores and agreement_counts must have the same length; "
            f"got {numer.numel()} and {denom.numel()}."
        )
    mean_disagreement = torch.full_like(numer, float("nan"))
    selected = denom > 0
    mean_disagreement[selected] = numer[selected] / denom[selected].clamp_min(1.0)
    return mean_disagreement


def build_alignment_validation_arrays(
    scores: dict[str, torch.Tensor],
    feature_usage_counts: torch.Tensor,
    agreement_scores: torch.Tensor,
    agreement_counts: torch.Tensor,
    *,
    topk: int | None,
) -> dict[str, torch.Tensor]:
    """Build all per-feature arrays shared by TensorBoard plots and npz export.

    Returned tensors are CPU tensors. The `ordered_*` arrays use the same feature
    order as `scores["sorted_feature_idx"]`, i.e. increasing alignment score.
    The `*_by_feature_id` arrays use the original SAE feature id as index.
    """
    sorted_feature_idx = scores["sorted_feature_idx"].detach().cpu().long()
    alignment_score = scores["alignment_score"].detach().float().cpu().flatten()
    sorted_score = scores["sorted_score"].detach().float().cpu().flatten()
    usage_freq = _feature_usage_to_frequency(feature_usage_counts, topk=topk)
    agreement = _topk_agreement_to_normalized(agreement_scores, agreement_counts)

    n = alignment_score.numel()
    if sorted_feature_idx.numel() != n or usage_freq.numel() != n or agreement.numel() != n:
        raise ValueError(
            "alignment, usage, and agreement tensors must have matching feature counts; "
            f"alignment={n}, sorted={sorted_feature_idx.numel()}, "
            f"usage={usage_freq.numel()}, agreement={agreement.numel()}."
        )

    ordered_usage = usage_freq.index_select(0, sorted_feature_idx)
    ordered_disagreement = agreement.index_select(0, sorted_feature_idx)
    return {
        "feature_id_sorted_by_alignment": sorted_feature_idx,
        "sae_1_alignment_score": sorted_score,
        "sae_1_alignment_ordered_feature_usage": ordered_usage,
        # Keep the legacy key requested by earlier analysis scripts, but its
        # value is now activation disagreement rather than +/- top-k support
        # agreement. Lower is better; inactive SAE-1 features are NaN.
        "sae_1_alignment_ordered_topk_agreement": ordered_disagreement,
        "sae_1_alignment_ordered_activation_disagreement": ordered_disagreement,
        "sae_1_alignment_score_by_feature_id": alignment_score,
        "sae_1_feature_usage_by_feature_id": usage_freq,
        "sae_1_topk_agreement_by_feature_id": agreement,
        "sae_1_activation_disagreement_by_feature_id": agreement,
    }


def save_alignment_validation_numpy(
    path: str | "tp.Any",
    scores: dict[str, torch.Tensor],
    feature_usage_counts: torch.Tensor,
    agreement_scores: torch.Tensor,
    agreement_counts: torch.Tensor,
    *,
    topk: int | None,
    epoch: int | None = None,
    global_step: int | None = None,
) -> None:
    """Save per-feature alignment/usage/agreement arrays to a NumPy .npz file."""
    import numpy as np

    arrays = build_alignment_validation_arrays(
        scores,
        feature_usage_counts,
        agreement_scores,
        agreement_counts,
        topk=topk,
    )
    payload = {k: v.detach().cpu().numpy() for k, v in arrays.items()}
    if epoch is not None:
        payload["epoch"] = np.asarray(int(epoch), dtype=np.int64)
    if global_step is not None:
        payload["global_step"] = np.asarray(int(global_step), dtype=np.int64)
    payload["topk"] = np.asarray(-1 if topk is None else int(topk), dtype=np.int64)
    np.savez_compressed(str(path), **payload)


def make_alignment_ordered_usage_figure(
    scores: dict[str, torch.Tensor],
    feature_usage_counts: torch.Tensor,
    *,
    topk: int | None,
    title: str = "Validation feature usage vs alignment score",
):
    """Scatter validation feature usage against alignment score.

    x-axis is the sorted alignment score itself, not feature rank. Each feature is
    one point; y uses the same ordered feature-usage values as the old second
    plot.
    """
    import matplotlib.pyplot as plt

    sorted_feature_idx = scores["sorted_feature_idx"].detach().cpu().long()
    sorted_score = scores["sorted_score"].detach().cpu().numpy()
    freq = _feature_usage_to_frequency(feature_usage_counts, topk=topk)
    if sorted_feature_idx.numel() != freq.numel():
        raise ValueError(
            "feature_usage_counts length must match the number of alignment scores; "
            f"got counts={freq.numel()} and scores={sorted_feature_idx.numel()}."
        )
    ordered_freq = freq.index_select(0, sorted_feature_idx).numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(sorted_score, ordered_freq, s=6, alpha=0.55, linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel("Alignment score (cosine distance, lower is better)")
    ax.set_ylabel("Validation activation frequency per frame")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def log_alignment_ordered_usage_to_tensorboard(
    logger: tp.Any,
    scores: dict[str, torch.Tensor],
    feature_usage_counts: torch.Tensor,
    *,
    topk: int | None,
    global_step: int,
    tag: str = "val/sae_1_alignment_ordered_feature_usage",
    title: str = "SAE-1 validation feature usage vs alignment score",
) -> bool:
    """Log validation feature usage as a scatter against alignment score."""
    experiment = getattr(logger, "experiment", None)
    add_figure = getattr(experiment, "add_figure", None)
    if add_figure is None:
        return False

    fig = make_alignment_ordered_usage_figure(
        scores,
        feature_usage_counts,
        topk=topk,
        title=title,
    )
    try:
        add_figure(tag, fig, global_step=global_step)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return True


def make_alignment_ordered_topk_agreement_figure(
    scores: dict[str, torch.Tensor],
    agreement_scores: torch.Tensor,
    agreement_counts: torch.Tensor,
    *,
    title: str = "SAE-1 activation disagreement vs alignment score",
):
    """Scatter SAE-1-conditioned activation disagreement against alignment score."""
    import matplotlib.pyplot as plt

    sorted_feature_idx = scores["sorted_feature_idx"].detach().cpu().long()
    sorted_score = scores["sorted_score"].detach().cpu().numpy()
    agreement = _topk_agreement_to_normalized(agreement_scores, agreement_counts)
    if sorted_feature_idx.numel() != agreement.numel():
        raise ValueError(
            "agreement tensors must match the number of alignment scores; "
            f"got sorted={sorted_feature_idx.numel()} and agreement={agreement.numel()}."
        )
    ordered = agreement.index_select(0, sorted_feature_idx).numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(sorted_score, ordered, s=6, alpha=0.55, linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel("Alignment score (cosine distance, lower is better)")
    ax.set_ylabel("SAE-1-conditioned activation disagreement (lower is better)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def log_alignment_ordered_topk_agreement_to_tensorboard(
    logger: tp.Any,
    scores: dict[str, torch.Tensor],
    agreement_scores: torch.Tensor,
    agreement_counts: torch.Tensor,
    *,
    global_step: int,
    tag: str = "val/sae_1_alignment_ordered_topk_agreement",
    title: str = "SAE-1 activation disagreement vs alignment score",
) -> bool:
    """Log SAE-1-conditioned activation disagreement against alignment score."""
    experiment = getattr(logger, "experiment", None)
    add_figure = getattr(experiment, "add_figure", None)
    if add_figure is None:
        return False

    fig = make_alignment_ordered_topk_agreement_figure(
        scores,
        agreement_scores,
        agreement_counts,
        title=title,
    )
    try:
        add_figure(tag, fig, global_step=global_step)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return True


def make_alignment_usage_vs_topk_agreement_figure(
    scores: dict[str, torch.Tensor],
    feature_usage_counts: torch.Tensor,
    agreement_scores: torch.Tensor,
    agreement_counts: torch.Tensor,
    *,
    topk: int | None,
    title: str = "SAE-1 activation disagreement vs validation feature usage",
):
    """Scatter activation disagreement against usage, one point per feature.

    x-axis is the old second figure's y-value; y-axis is the current third
    figure's y-value. The alignment score is not an axis in this plot.
    """
    import matplotlib.pyplot as plt

    # Use scores only for consistency validation and future extensibility.
    sorted_feature_idx = scores["sorted_feature_idx"].detach().cpu().long()
    usage = _feature_usage_to_frequency(feature_usage_counts, topk=topk)
    agreement = _topk_agreement_to_normalized(agreement_scores, agreement_counts)
    if sorted_feature_idx.numel() != usage.numel() or usage.numel() != agreement.numel():
        raise ValueError(
            "usage/agreement tensors must match the number of alignment scores; "
            f"sorted={sorted_feature_idx.numel()}, usage={usage.numel()}, "
            f"agreement={agreement.numel()}."
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(usage.numpy(), agreement.numpy(), s=6, alpha=0.55, linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel("Validation activation frequency per frame")
    ax.set_ylabel("SAE-1-conditioned activation disagreement (lower is better)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def log_alignment_usage_vs_topk_agreement_to_tensorboard(
    logger: tp.Any,
    scores: dict[str, torch.Tensor],
    feature_usage_counts: torch.Tensor,
    agreement_scores: torch.Tensor,
    agreement_counts: torch.Tensor,
    *,
    topk: int | None,
    global_step: int,
    tag: str = "val/sae_1_alignment_usage_vs_topk_agreement",
    title: str = "SAE-1 activation disagreement vs validation feature usage",
) -> bool:
    """Log usage-vs-activation-disagreement scatter to TensorBoard."""
    experiment = getattr(logger, "experiment", None)
    add_figure = getattr(experiment, "add_figure", None)
    if add_figure is None:
        return False

    fig = make_alignment_usage_vs_topk_agreement_figure(
        scores,
        feature_usage_counts,
        agreement_scores,
        agreement_counts,
        topk=topk,
        title=title,
    )
    try:
        add_figure(tag, fig, global_step=global_step)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return True
