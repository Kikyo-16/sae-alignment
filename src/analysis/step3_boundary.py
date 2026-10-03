#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Detect chord-boundary features and silence features, then append them
into an existing feature_ids.txt created by step2_rings.py.

Boundary detection
------------------
Each feature's activation is passed through the same boundary-candidate pipeline
used by chord-recognition inference:

    moving_average_1d -> detect_peaks_with_drop(tol/drop) -> thin_times
    -> remove endpoint peaks -> boundary_min_gap -> boundary_score_floor

The PEAK MAGNITUDE IS IGNORED after thresholding/filtering: each retained peak
counts as 1. Retained peaks are scored against GT chord boundaries:

Accepted stars are REWARDED; missing boundaries and extra peaks are PENALISED:

  * For each GT boundary, find the nearest peak of each feature.
  * That peak becomes a "star" only when its own nearest GT boundary is the same
    GT boundary (mutual nearest match) AND its distance to the boundary is not
    larger than star_max_dist. A star receives a POSITIVE narrow Gaussian reward
    w: ~1 right on the boundary, decaying toward 0 as it drifts away.
  * If the nearest peak is closer to another GT boundary, or is farther than
    star_max_dist, this GT boundary is treated as MISSING and charged the larger
    missing_boundary_penalty.
  * EVERY non-star peak is charged the smaller flat boundary_penalty.

    star_reward[t] = w[t]   (w = narrow Gaussian, boundary_sigma)
    score[d] = ( sum_{star peaks} star_reward
                 - missing_boundary_penalty * #missing_boundaries
                 - boundary_penalty * #other_peaks )
               / n_segments

The highest-scoring feature is fB. A good boundary feature has exactly one mutual
star sitting on each boundary (near-one star reward), no missing boundaries, and
no other peaks. Missing boundaries are penalised more heavily than false-alarm
peaks by setting missing_boundary_penalty >= boundary_penalty.

Silence detection
-----------------
For each selected song, use the first segment only. The intro frames
[0, intro_sec) are treated as positive silence frames, while the fixed later
window [silence_neg_start_sec, silence_neg_end_sec) (default: 20s-30s) is treated
as negative/background frames. A feature is scored by TOTAL positive intro activation minus TOTAL negative-window
activation:

    score[d] = sum(z[intro, d]) - silence_neg_weight * sum(z[neg_window, d])

Top-K features by this sum-minus-sum contrast score are treated as silence features.

Appends two lines to feature_ids.txt:
    [Chord Boundary]: 12 top1 -> top2 -> ... -> top12
    [Silence]: id0 id1 ...

Usage:
    python -m src.analysis.step3_boundary_gaussian \
        --ckpt-path         store/.../ckpts/last.ckpt \
        --h5-path           sae-data/pop909/muq/pop909_muq_30s_layer_2.h5 \
        --split             train \
        --feature-ds        muq_layer \
        --feature-ids-file  store/.../analysis/feature_ids.txt \
        [--sae-idx 1] [--device cuda] [--batch-size 64] \
        [--max-songs 2] [--boundary-sigma 2.0] [--boundary-penalty 1.0] \
        [--missing-boundary-penalty 3.0] [--star-max-dist 4.0] \
        [--smooth-win 9] [--peak-tol 0.2] \
        [--top-k-silence 2] [--silence-songs 0] [--intro-sec 2.0] \
        [--silence-neg-start-sec 20.0] [--silence-neg-end-sec 30.0] \
        [--silence-neg-weight 1.0] [--fps 25] \
        [--debug-dir store/.../analysis/debug] [--debug-segments 4]

Debug mode (--debug-dir): writes one activation heatmap per segment for the
found boundary top-12 and silence features — rows = features, x = time,
red vertical lines = GT chord boundaries, colour = activation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from .find_chord_rings import load_sae_and_factory, standardize_to_BTD_batch
from .infer_key import (
    decode_str_array,
    decode_str_scalar,
    smooth_cols,
)
from .boundary_utils import (
    detect_boundary_peaks_1d as _boundary_peaks_1d,
    boundary_peak_mask as _peak_mask,
    mutual_boundary_star_mask as _mutual_boundary_star_mask,
)


# ---------------------------------------------------------------------------
# Helpers shared by both detectors
# ---------------------------------------------------------------------------

def _run_sae(
    module,
    model_factory,
    use_tag: str,
    use_idx: int,
    x_np: np.ndarray,
    t_feat: int,
    device: str,
) -> Optional[np.ndarray]:
    """Return [B, T, D] float32 sparse activations, or None on failure."""
    x = torch.from_numpy(x_np).float().to(device)
    with torch.no_grad():
        feat_batch: Dict[str, torch.Tensor] = model_factory({use_tag: x}, t=0)
    if use_tag not in feat_batch:
        return None

    feat = feat_batch[use_tag]
    x_sae = standardize_to_BTD_batch(feat, t_candidates=[t_feat, int(x_np.shape[-1])])
    with torch.no_grad():
        z = module.inference(x_sae, idx=use_idx, topk_wide=0)

    z_np = z.detach().float().cpu().numpy()
    return z_np if z_np.ndim == 3 else None


def _resample(frame: np.ndarray, target_t: int) -> np.ndarray:
    """Nearest-neighbour resampling of a frame-level label array."""
    t0 = int(frame.shape[0])
    if target_t <= 0:
        return frame[:0]
    if t0 == target_t:
        return frame
    if t0 <= 0:
        return np.array(["N"] * target_t, dtype=object)

    idx = np.clip(
        np.rint(np.linspace(0, t0 - 1, num=target_t)).astype(np.int64),
        0,
        t0 - 1,
    )
    return frame[idx]


def _change_points(chord_frame: np.ndarray) -> List[int]:
    """Return frame indices t where chord_frame[t] differs from t - 1."""
    if len(chord_frame) <= 1:
        return []

    cps: List[int] = []
    prev = str(chord_frame[0])
    for t in range(1, len(chord_frame)):
        cur = str(chord_frame[t])
        if cur != prev:
            cps.append(t)
            prev = cur
    return cps



def _boundary_gaussian_weight(
    t_len: int,
    centers: List[int],
    sigma: float,
) -> np.ndarray:
    """
    Return [T] float32 weights in [0, 1].

    Each GT chord boundary contributes a Gaussian bump. Multiple boundaries are
    combined by max, not sum, so dense boundary regions do not explode.
    """
    if t_len <= 0 or not centers:
        return np.zeros(max(0, t_len), dtype=np.float32)
    if sigma <= 0:
        raise ValueError(f"boundary sigma must be > 0, got {sigma}")

    t = np.arange(t_len, dtype=np.float32)
    w = np.zeros(t_len, dtype=np.float32)
    for c in centers:
        c = int(np.clip(c, 0, t_len - 1))
        wc = np.exp(-0.5 * ((t - float(c)) / float(sigma)) ** 2)
        w = np.maximum(w, wc.astype(np.float32))
    return w


def _collect_song_segments(
    group: h5py.Group,
    n_total: int,
    max_songs: int,
) -> Dict[str, List[int]]:
    """Collect segment indices grouped by trackId, preserving HDF5 order."""
    has_track = "trackId" in group
    song_segs: Dict[str, List[int]] = {}

    for si in range(n_total):
        tid = decode_str_scalar(group["trackId"][si]) if has_track else f"seg{si:06d}"
        if tid not in song_segs:
            if max_songs > 0 and len(song_segs) >= max_songs:
                break
            song_segs[tid] = []
        song_segs[tid].append(si)

    return song_segs


# ---------------------------------------------------------------------------
# Boundary feature detection
# ---------------------------------------------------------------------------

def find_boundary_feature(
    module,
    model_factory,
    use_tag: str,
    use_idx: int,
    h5_path: Path,
    split: str,
    feature_ds: str,
    device: str,
    batch_size: int = 64,
    max_songs: int = 2,
    boundary_sigma: float = 2.0,
    boundary_penalty: float = 1.0,
    missing_boundary_penalty: float = 3.0,
    star_max_dist: float = 4.0,
    boundary_peak_drop: float = 0.0,
    boundary_min_gap: int = 2,
    boundary_score_floor: float = 0.0,
    boundary_max_labels: int = 120,
    smooth_win: int = 9,
    peak_tol: float = 0.2,
) -> Tuple[int, List[int]]:
    """
    Return (fB, top12_ids).

    Activation is smoothed (smooth_win) and reduced to peaks
    (detect_peaks_with_drop, tol=peak_tol). For each GT boundary, the nearest
    peak is accepted as a "star" only if the peak's own nearest GT boundary is
    the same one (mutual nearest match) and its distance is <= star_max_dist.
    Accepted stars receive a positive Gaussian reward (narrow Gaussian, ~1 right
    on a boundary). Boundaries without an accepted star are missing and charged
    the larger
    missing_boundary_penalty; all non-star peaks are charged boundary_penalty.
    score = star_reward - missing_penalty - false_alarm_penalty, and the
    highest-scoring feature is fB.
    """
    if boundary_sigma <= 0:
        raise ValueError(f"boundary_sigma must be > 0, got {boundary_sigma}")
    if boundary_penalty < 0.0:
        raise ValueError(f"boundary_penalty must be >= 0.0, got {boundary_penalty}")
    if missing_boundary_penalty < boundary_penalty:
        raise ValueError(
            f"missing_boundary_penalty must be >= boundary_penalty so missing "
            f"boundaries are not cheaper than false-alarm peaks; got "
            f"missing_boundary_penalty={missing_boundary_penalty}, "
            f"boundary_penalty={boundary_penalty}"
        )
    if star_max_dist < 0:
        raise ValueError(f"star_max_dist must be >= 0, got {star_max_dist}")
    if boundary_peak_drop < 0:
        raise ValueError(f"boundary_peak_drop must be >= 0, got {boundary_peak_drop}")
    if boundary_min_gap < 0:
        raise ValueError(f"boundary_min_gap must be >= 0, got {boundary_min_gap}; use 0 to disable min-gap thinning")
    if boundary_score_floor < 0:
        raise ValueError(f"boundary_score_floor must be >= 0, got {boundary_score_floor}")
    if boundary_max_labels <= 0:
        raise ValueError(f"boundary_max_labels must be > 0, got {boundary_max_labels}")
    if smooth_win < 1:
        raise ValueError(f"smooth_win must be >= 1, got {smooth_win}")
    if peak_tol < 0:
        raise ValueError(f"peak_tol must be >= 0, got {peak_tol}")

    score_sum: Optional[np.ndarray] = None
    n_used_segments = 0
    n_skipped_no_boundary = 0

    with h5py.File(h5_path, "r") as f:
        g = f[split]
        ds_feat = g[feature_ds]
        n_total = int(ds_feat.shape[0])
        t_feat = int(ds_feat.shape[-1])
        has_chord = "labels" in g and "chord_frame" in g["labels"]
        if not has_chord:
            raise RuntimeError(
                f"GT chord_frame labels not found in {h5_path} split={split}. "
                "Gaussian boundary scoring needs GT chord boundaries."
            )

        song_segs = _collect_song_segments(g, n_total, max_songs)
        sel: List[int] = [si for segs in song_segs.values() for si in segs]
        print(
            f"[INFO] Boundary: {len(song_segs)} song(s), {len(sel)} segment(s), "
            f"star_sigma={boundary_sigma}, star_max_dist={star_max_dist}, "
            f"nonstar_penalty={boundary_penalty}, "
            f"missing_penalty={missing_boundary_penalty}, "
            f"peak_drop={boundary_peak_drop}, min_gap={boundary_min_gap}, "
            f"score_floor={boundary_score_floor}, max_labels={boundary_max_labels}, "
            f"smooth_win={smooth_win}, peak_tol={peak_tol}"
        )

        bs = max(1, batch_size)
        for b0 in tqdm(range(0, len(sel), bs), desc="Boundary", unit="batch"):
            idxs = sel[b0: b0 + bs]
            x_np = np.stack([np.array(ds_feat[i], copy=True) for i in idxs])
            z_batch = _run_sae(
                module,
                model_factory,
                use_tag,
                use_idx,
                x_np,
                t_feat,
                device,
            )
            if z_batch is None:
                continue

            for j, si in enumerate(idxs):
                z = z_batch[j]  # [T, D]
                t_z = int(z.shape[0])
                if t_z <= 0:
                    continue

                chord = decode_str_array(g["labels"]["chord_frame"][si])
                if len(chord) != t_z:
                    chord = _resample(chord, t_z)

                cps = _change_points(chord)
                w = _boundary_gaussian_weight(t_z, cps, boundary_sigma)
                w_sum = float(w.sum())
                if w_sum <= 0.0:
                    n_skipped_no_boundary += 1
                    continue

                # Detect the exact boundary candidates that step4 inference would use
                # for each feature, then score those candidates against GT boundaries.
                if score_sum is None:
                    score_sum = np.zeros(z.shape[1], dtype=np.float64)

                pm = _peak_mask(
                    z, smooth_win, peak_tol, boundary_peak_drop,
                    boundary_min_gap, boundary_score_floor, boundary_max_labels,
                )                                                        # [T, D] peaks
                star, miss = _mutual_boundary_star_mask(pm, cps, t_z, star_max_dist)

                # Score = positive star reward - missing/false-alarm penalties:
                #   accepted stars   -> Gaussian reward w (near 1 on boundary)
                #   missing boundary -> larger missing_boundary_penalty
                #   other peaks      -> smaller boundary_penalty (false alarms)
                star_reward = w.astype(np.float32)                      # [T] positive Gaussian
                other = pm & ~star
                score_sum += (star_reward[:, None] * star).sum(axis=0)
                score_sum -= missing_boundary_penalty * miss
                score_sum -= boundary_penalty * other.sum(axis=0).astype(np.float64)
                n_used_segments += 1

    if score_sum is None or n_used_segments == 0:
        raise RuntimeError("No valid chord-boundary segments processed.")

    # Highest score wins: good stars add reward; missing/extra peaks subtract.
    score = score_sum / float(n_used_segments)
    ranked = np.argsort(-score)
    top12_ids = [int(x) for x in ranked[:12]]
    fB = int(top12_ids[0])

    print(
        f"[INFO] Boundary used {n_used_segments} segment(s); "
        f"skipped {n_skipped_no_boundary} segment(s) with no chord changes."
    )
    print(f"[INFO] Boundary feature fB={fB}  score={float(score[fB]):.6f}")
    print("[INFO] Boundary top-12:")
    for rank, fid in enumerate(top12_ids, start=1):
        print(f"       #{rank:02d}  feature={fid}  score={float(score[fid]):.6f}")

    return fB, top12_ids


# ---------------------------------------------------------------------------
# Silence feature detection
# ---------------------------------------------------------------------------

def find_silence_features(
    module,
    model_factory,
    use_tag: str,
    use_idx: int,
    h5_path: Path,
    split: str,
    feature_ds: str,
    device: str,
    top_k: int = 2,
    intro_sec: float = 2.0,
    fps: float = 25.0,
    song_indices: Optional[List[int]] = None,
    silence_neg_start_sec: float = 20.0,
    silence_neg_end_sec: float = 30.0,
    silence_neg_weight: float = 1.0,
) -> List[int]:
    """
    Return top-k silence feature IDs by sum-minus-sum intro-vs-background score.

    song_indices: list of 0-based song positions to use. Default: [0].
    Only the first segment of each selected song is used.

    Positive frames: [0, intro_sec) within the first segment.
    Negative frames: [silence_neg_start_sec, silence_neg_end_sec) within the
    same first segment; by default this is 20s-30s.

    Score for each feature and each selected song:
        sum(z[positive_frames, d])
        - silence_neg_weight * sum(z[negative_frames, d])

    Both terms are sums, not means, so window length intentionally affects the
    score. With the default 2s positive window and 20s-30s negative window, a
    feature that stays active later is heavily penalised.
    """
    if song_indices is None:
        song_indices = [0]
    if not song_indices:
        raise ValueError("song_indices must not be empty")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if intro_sec <= 0:
        raise ValueError(f"intro_sec must be > 0, got {intro_sec}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if silence_neg_start_sec < 0:
        raise ValueError(
            f"silence_neg_start_sec must be >= 0, got {silence_neg_start_sec}"
        )
    if silence_neg_end_sec <= silence_neg_start_sec:
        raise ValueError(
            "silence_neg_end_sec must be greater than silence_neg_start_sec, got "
            f"{silence_neg_end_sec} <= {silence_neg_start_sec}"
        )
    if silence_neg_weight < 0:
        raise ValueError(f"silence_neg_weight must be >= 0, got {silence_neg_weight}")

    target_songs = set(song_indices)
    intro_frames = max(1, int(round(intro_sec * fps)))
    neg_start_frame = max(0, int(round(silence_neg_start_sec * fps)))
    neg_end_frame = max(neg_start_frame + 1, int(round(silence_neg_end_sec * fps)))

    with h5py.File(h5_path, "r") as f:
        g = f[split]
        ds_feat = g[feature_ds]
        n_total = int(ds_feat.shape[0])
        t_feat = int(ds_feat.shape[-1])
        has_track = "trackId" in g

        # Collect the first segment index of each target song.
        current_song = -1
        current_tid: Optional[str] = None
        intro_segs: List[Tuple[int, str]] = []
        for si in range(n_total):
            tid = decode_str_scalar(g["trackId"][si]) if has_track else "song0"
            if tid != current_tid:
                current_song += 1
                current_tid = tid
                if current_song in target_songs:
                    intro_segs.append((si, tid))
            if current_song > max(target_songs):
                break

        if not intro_segs:
            raise RuntimeError(
                f"None of song_indices={song_indices} found in {h5_path} "
                f"split={split} (only {current_song + 1} song(s) available)."
            )

        print(
            f"[INFO] Silence: song_indices={song_indices}  "
            f"found {len(intro_segs)}/{len(target_songs)} song(s), "
            f"positive=0.0s-{intro_sec:.1f}s sum, "
            f"negative={silence_neg_start_sec:.1f}s-{silence_neg_end_sec:.1f}s sum, "
            f"neg_weight={silence_neg_weight}"
        )

        score_sum: Optional[np.ndarray] = None
        pos_sum: Optional[np.ndarray] = None
        neg_sum_total: Optional[np.ndarray] = None
        count = 0
        skipped_short_neg = 0
        for si, tid in intro_segs:
            x_np = np.array(ds_feat[si], copy=True)[np.newaxis]
            z_batch = _run_sae(
                module,
                model_factory,
                use_tag,
                use_idx,
                x_np,
                t_feat,
                device,
            )
            if z_batch is None:
                continue

            z = z_batch[0]  # [T, D]
            t_pos_end = min(intro_frames, z.shape[0])
            t_neg_start = min(neg_start_frame, z.shape[0])
            t_neg_end = min(neg_end_frame, z.shape[0])
            if t_pos_end <= 0:
                continue
            if t_neg_end <= t_neg_start:
                skipped_short_neg += 1
                print(
                    f"       seg {si}  tid={tid!r}: skipped; segment length "
                    f"{z.shape[0] / fps:.1f}s does not cover negative window "
                    f"{silence_neg_start_sec:.1f}s-{silence_neg_end_sec:.1f}s"
                )
                continue

            if score_sum is None:
                score_sum = np.zeros(z.shape[1], dtype=np.float64)
                pos_sum = np.zeros(z.shape[1], dtype=np.float64)
                neg_sum_total = np.zeros(z.shape[1], dtype=np.float64)

            pos_sum_seg = z[:t_pos_end].sum(axis=0).astype(np.float64)
            neg_sum = z[t_neg_start:t_neg_end].sum(axis=0).astype(np.float64)
            score = pos_sum_seg - float(silence_neg_weight) * neg_sum

            pos_end_sec = t_pos_end / fps
            neg_start_sec_used = t_neg_start / fps
            neg_end_sec_used = t_neg_end / fps
            print(
                f"       seg {si}  tid={tid!r}  pos frames 0-{t_pos_end} "
                f"(0.0s-{pos_end_sec:.1f}s sum), neg frames {t_neg_start}-{t_neg_end} "
                f"({neg_start_sec_used:.1f}s-{neg_end_sec_used:.1f}s sum)"
            )

            score_sum += score
            pos_sum += pos_sum_seg
            neg_sum_total += neg_sum
            count += 1

    if score_sum is None or count == 0:
        raise RuntimeError(
            "No valid segments for silence detection. Check --silence-songs and "
            "that the selected first segments cover --silence-neg-start-sec to "
            "--silence-neg-end-sec."
        )

    score_mean = (score_sum / count).astype(np.float32)
    pos_sum_mean = (pos_sum / count).astype(np.float32) if pos_sum is not None else None
    neg_sum_mean = (
        (neg_sum_total / count).astype(np.float32)
        if neg_sum_total is not None else None
    )

    k = min(top_k, int(score_mean.shape[0]))
    top_idx = np.argpartition(score_mean, -k)[-k:]
    top_idx = top_idx[np.argsort(score_mean[top_idx])[::-1]]
    ids = [int(x) for x in top_idx]

    if skipped_short_neg:
        print(f"[INFO] Silence skipped {skipped_short_neg} short segment(s).")
    print(f"[INFO] Silence features (top-{k}, contrast score): {ids}")
    if pos_sum_mean is not None and neg_sum_mean is not None:
        for rank, fid in enumerate(ids, start=1):
            print(
                f"       #{rank:02d}  feature={fid}  score={float(score_mean[fid]):.6f}  "
                f"pos_sum={float(pos_sum_mean[fid]):.6f}  "
                f"neg_sum={float(neg_sum_mean[fid]):.6f}"
            )
    return ids

# ---------------------------------------------------------------------------
# Debug visualisation: per-segment feature-activation heatmaps
# ---------------------------------------------------------------------------

def _chord_spans_labels(chord_frame: np.ndarray) -> List[Tuple[int, int, str]]:
    """Constant-chord spans as (start, end, label) over the frame axis."""
    spans: List[Tuple[int, int, str]] = []
    if len(chord_frame) == 0:
        return spans
    s, prev = 0, str(chord_frame[0])
    for t in range(1, len(chord_frame)):
        cur = str(chord_frame[t])
        if cur != prev:
            spans.append((s, t, prev))
            s, prev = t, cur
    spans.append((s, len(chord_frame), prev))
    return spans


def _first_segments_of_songs(g: h5py.Group, n_total: int, target_songs: List[int]) -> List[int]:
    """First segment index of each 0-based song position in target_songs."""
    has_track = "trackId" in g
    target = set(target_songs)
    out: List[int] = []
    current_song, current_tid = -1, None
    for si in range(n_total):
        tid = decode_str_scalar(g["trackId"][si]) if has_track else "song0"
        if tid != current_tid:
            current_song += 1
            current_tid = tid
            if current_song in target:
                out.append(si)
        if target and current_song > max(target):
            break
    return out


def _plot_segment_heatmap(
    out_path: Path,
    z: np.ndarray,
    feature_ids: List[int],
    feat_labels: List[str],
    chord_frame: Optional[np.ndarray],
    title: str,
    fps: float,
    intro_frames: Optional[int] = None,
    neg_frames: Optional[Tuple[int, int]] = None,
    smooth_win: int = 1,
    mark_peaks: bool = False,
    peak_tol: float = 0.2,
    boundary_peak_drop: float = 0.0,
    boundary_min_gap: int = 2,
    boundary_score_floor: float = 0.0,
    boundary_max_labels: int = 120,
    star_max_dist: float = 4.0,
) -> None:
    """Heatmap: rows = feature_ids, cols = time, value = activation, with GT
    chord-boundary vertical lines + per-span chord labels and a seconds axis.
    Optionally smooths the activation and marks per-feature peaks (matching how
    the boundary feature is consumed at inference)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, D = z.shape
    fids = [fid for fid in feature_ids if 0 <= fid < D]
    labs = [feat_labels[i] for i, fid in enumerate(feature_ids) if 0 <= fid < D]
    if not fids:
        return
    mat_raw = np.stack([z[:, fid] for fid in fids], axis=0)               # [n, T]
    if smooth_win and smooth_win > 1:
        mat_raw = smooth_cols(mat_raw.T, smooth_win).T
    mat = np.maximum(mat_raw, 0.0)                                       # display only
    n = mat.shape[0]

    fig_w = max(10.0, 0.03 * T)
    fig_h = max(2.5, 0.5 * n + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title(title, fontsize=10)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_ylabel("feature")
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", axis="y", linewidth=0.25)
    ax.tick_params(which="minor", left=False)

    # Bottom x-axis: GT chord label per span + red boundary lines.
    if chord_frame is not None and len(chord_frame) == T:
        spans = _chord_spans_labels(chord_frame)
        ax.set_xticks([(s + e - 1) // 2 for s, e, _ in spans])
        ax.set_xticklabels([lab for _, _, lab in spans], rotation=90, fontsize=6)
        ax.set_xlabel("GT chord per span (red lines = boundaries)")
        for s, _e, _lab in spans[1:]:
            ax.vlines(s - 0.5, -0.5, n - 0.5, color="r", linewidth=1.0)
    else:
        ax.set_xlabel("frame")

    # Top x-axis: time in seconds.
    if fps and fps > 0:
        dur = T / float(fps)
        sec_marks = np.arange(0, int(np.floor(dur)) + 1, 1, dtype=np.int64)
        frame_pos = np.clip(np.rint(sec_marks * float(fps)).astype(np.int64), 0, max(0, T - 1))
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xlabel("Time (s)")
        ax_top.set_xticks(frame_pos.tolist())
        ax_top.set_xticklabels([f"{int(s)}s" for s in sec_marks], fontsize=7)

    # Optional intro cutoff and negative/background window (silence debug).
    if intro_frames is not None and 0 < intro_frames < T:
        ax.vlines(intro_frames - 0.5, -0.5, n - 0.5, color="cyan", linewidth=1.5, linestyle="--")
    if neg_frames is not None:
        n0, n1 = neg_frames
        n0 = int(np.clip(n0, 0, T))
        n1 = int(np.clip(n1, 0, T))
        if n1 > n0:
            ax.axvspan(n0 - 0.5, n1 - 0.5, color="orange", alpha=0.12)
            ax.vlines([n0 - 0.5, n1 - 0.5], -0.5, n - 0.5,
                      color="orange", linewidth=1.0, linestyle="--")

    # Per-feature peaks (local maxima of the smoothed row). Use the same mutual
    # nearest-boundary + max-distance star rule as scoring: accepted stars are
    # lime; non-star peaks are white. Missing boundaries have no marker but are
    # charged in scoring.
    if mark_peaks:
        cps_plot = (
            _change_points(chord_frame)
            if chord_frame is not None and len(chord_frame) == T else []
        )
        pm_plot = np.zeros((T, n), dtype=bool)
        for i in range(n):
            fid = fids[i]
            pk_list, _sm = _boundary_peaks_1d(
                z[:, fid], smooth_win, peak_tol, boundary_peak_drop,
                boundary_min_gap, boundary_score_floor, boundary_max_labels,
            )
            pk = np.array(pk_list, dtype=int)
            if pk.size == 0:
                continue
            pm_plot[pk, i] = True

        star_plot, _miss_plot = _mutual_boundary_star_mask(pm_plot, cps_plot, T, star_max_dist)
        for i in range(n):
            pk = np.where(pm_plot[:, i])[0]
            if pk.size == 0:
                continue
            rew = np.where(star_plot[:, i])[0]
            oth = np.array([p for p in pk if not star_plot[p, i]], dtype=int)
            if oth.size:
                ax.scatter(oth, np.full(oth.shape[0], i), s=10, c="white",
                           edgecolors="black", linewidths=0.3, zorder=6)
            if rew.size:
                ax.scatter(rew, np.full(rew.shape[0], i), s=42, marker="*", c="lime",
                           edgecolors="black", linewidths=0.4, zorder=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Activation", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def debug_plot_segments(
    module,
    model_factory,
    use_tag: str,
    use_idx: int,
    h5_path: Path,
    split: str,
    feature_ds: str,
    device: str,
    feature_ids: List[int],
    feat_labels: List[str],
    seg_indices: List[int],
    out_dir: Path,
    prefix: str,
    fps: float,
    intro_frames: Optional[int] = None,
    neg_frames: Optional[Tuple[int, int]] = None,
    smooth_win: int = 1,
    mark_peaks: bool = False,
    peak_tol: float = 0.2,
    boundary_peak_drop: float = 0.0,
    boundary_min_gap: int = 2,
    boundary_score_floor: float = 0.0,
    boundary_max_labels: int = 120,
    star_max_dist: float = 4.0,
) -> None:
    """Render one activation heatmap per segment for the given feature_ids."""
    out_dir = Path(out_dir)
    with h5py.File(h5_path, "r") as f:
        g = f[split]
        ds_feat = g[feature_ds]
        t_feat = int(ds_feat.shape[-1])
        has_chord = "labels" in g and "chord_frame" in g["labels"]
        has_track = "trackId" in g
        for si in seg_indices:
            x_np = np.array(ds_feat[si], copy=True)[np.newaxis]
            z_batch = _run_sae(module, model_factory, use_tag, use_idx, x_np, t_feat, device)
            if z_batch is None:
                print(f"[DEBUG] {prefix} seg {si}: SAE inference failed, skipped")
                continue
            z = z_batch[0]  # [T, D]
            chord = None
            if has_chord:
                chord = decode_str_array(g["labels"]["chord_frame"][si])
                if len(chord) != z.shape[0]:
                    chord = _resample(chord, z.shape[0])
            tid = decode_str_scalar(g["trackId"][si]) if has_track else f"seg{si}"
            out_path = out_dir / f"{prefix}_seg{si:06d}.png"
            _plot_segment_heatmap(
                out_path, z, feature_ids, feat_labels, chord,
                title=f"[{prefix}] seg={si}  tid={tid}", fps=fps,
                intro_frames=intro_frames, neg_frames=neg_frames,
                smooth_win=smooth_win, mark_peaks=mark_peaks,
                peak_tol=peak_tol, boundary_peak_drop=boundary_peak_drop,
                boundary_min_gap=boundary_min_gap,
                boundary_score_floor=boundary_score_floor,
                boundary_max_labels=boundary_max_labels,
                star_max_dist=star_max_dist,
            )
            print(f"[DEBUG] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Step 3: Gaussian-reward chord boundary + silence -> append to feature_ids.txt"
    )
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--h5-path", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--feature-ds", required=True)
    ap.add_argument(
        "--feature-ids-file",
        required=True,
        help="feature_ids.txt from step2_rings.py; this script appends two lines",
    )
    ap.add_argument("--sae-idx", type=int, default=0)  # 0 = anchor sub-SAE
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--max-songs",
        type=int,
        default=2,
        help="Songs for boundary detection; 0 = all songs",
    )
    ap.add_argument(
        "--boundary-sigma",
        type=float,
        default=2.0,
        help="Sigma of the narrow Gaussian that gives a positive reward to a "
             "mutual-star peak by its distance to the GT boundary, in SAE frames",
    )
    ap.add_argument(
        "--boundary-penalty",
        type=float,
        default=1.0,
        help="Flat penalty weight for non-star (false-alarm) peaks",
    )
    ap.add_argument(
        "--missing-boundary-penalty",
        type=float,
        default=5.0,
        help="Larger flat penalty for each GT boundary that has no accepted star peak; "
             "must be >= --boundary-penalty.",
    )
    ap.add_argument(
        "--star-max-dist",
        type=float,
        default=4.0,
        help="Maximum frame distance allowed between a GT boundary and its mutual "
             "nearest peak. If exceeded, the peak is not a star and the boundary "
             "is counted as missing.",
    )
    ap.add_argument(
        "--smooth-win",
        type=int,
        default=9,
        help="Moving-average window on the activation before boundary scoring / "
             "plotting, matching chord-recognition inference (1 = no smoothing)",
    )
    ap.add_argument(
        "--peak-tol",
        "--boundary-peak-tol",
        dest="peak_tol",
        type=float,
        default=0.2,
        help="Plateau tolerance for boundary peak detection; keep identical to "
             "step4 --boundary-peak-tol.",
    )
    ap.add_argument(
        "--boundary-peak-drop",
        type=float,
        default=0.0,
        help="Minimum rise/drop used by detect_peaks_with_drop for boundary peaks; "
             "keep identical to step4 --boundary-peak-drop.",
    )
    ap.add_argument(
        "--boundary-min-gap",
        type=int,
        default=0,
        help="Minimum frame gap after boundary peak detection; 0 disables "
             "min-gap thinning. Keep identical to step4 --boundary-min-gap.",
    )
    ap.add_argument(
        "--boundary-score-floor",
        type=float,
        default=0.0,
        help="Drop boundary peaks whose smoothed activation is below this floor; "
             "keep identical to step4 --boundary-score-floor.",
    )
    ap.add_argument(
        "--boundary-max-labels",
        type=int,
        default=120,
        help="Maximum number of boundary peaks kept by thin_times; keep identical "
             "to step4 --boundary-max-labels.",
    )
    ap.add_argument("--top-k-silence", type=int, default=2)
    ap.add_argument(
        "--silence-songs",
        type=int,
        nargs="+",
        default=[0],
        help="0-based song indices for silence detection, e.g. 0 1 4",
    )
    ap.add_argument("--intro-sec", type=float, default=2.0)
    ap.add_argument(
        "--silence-neg-start-sec",
        type=float,
        default=20.0,
        help="Start time (seconds) of the negative/background window in each "
             "selected song's first segment for silence scoring.",
    )
    ap.add_argument(
        "--silence-neg-end-sec",
        type=float,
        default=30.0,
        help="End time (seconds) of the negative/background window in each "
             "selected song's first segment for silence scoring.",
    )
    ap.add_argument(
        "--silence-neg-weight",
        type=float,
        default=1.0,
        help="Weight applied to the summed activation in the negative/background "
             "window before subtracting it from the silence intro mean.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="SAE frames per second; used for silence intro duration and the "
             "debug seconds axis",
    )
    ap.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="If set, write per-segment activation heatmaps (boundary + silence "
             "features) into this directory.",
    )
    ap.add_argument(
        "--debug-segments",
        type=int,
        default=4,
        help="Number of boundary segments to plot in debug mode.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ids_path = Path(args.feature_ids_file)
    if not ids_path.exists():
        raise FileNotFoundError(f"feature_ids.txt not found: {ids_path}")

    lit, sae, model_factory = load_sae_and_factory(Path(args.ckpt_path), args.device)
    module = sae.sae
    use_tag = str(lit.data_tag)
    n_sub = int(getattr(module, "n_saes", 0))
    use_idx = max(0, min(args.sae_idx, n_sub - 1))
    print(f"[INFO] use_tag={use_tag}  sae_idx={use_idx}  split={args.split}")

    h5_path = Path(args.h5_path)

    fB, top12_ids = find_boundary_feature(
        module,
        model_factory,
        use_tag,
        use_idx,
        h5_path,
        args.split,
        args.feature_ds,
        args.device,
        args.batch_size,
        args.max_songs,
        args.boundary_sigma,
        args.boundary_penalty,
        args.missing_boundary_penalty,
        args.star_max_dist,
        args.boundary_peak_drop,
        args.boundary_min_gap,
        args.boundary_score_floor,
        args.boundary_max_labels,
        args.smooth_win,
        args.peak_tol,
    )

    silence_ids = find_silence_features(
        module,
        model_factory,
        use_tag,
        use_idx,
        h5_path,
        args.split,
        args.feature_ds,
        args.device,
        args.top_k_silence,
        args.intro_sec,
        args.fps,
        args.silence_songs,
        args.silence_neg_start_sec,
        args.silence_neg_end_sec,
        args.silence_neg_weight,
    )

    # Build boundary line: ranked top-12 boundary feature IDs.
    # Inference should use boundary_ids[0] (top-1 / fB).
    boundary_ids = top12_ids
    boundary_ids_str = " -> ".join(str(x) for x in boundary_ids)
    boundary_line = f"[Chord Boundary]: {len(boundary_ids)} {boundary_ids_str}"

    silence_line = f"[Silence]: {' '.join(str(x) for x in silence_ids)}"

    with ids_path.open("a", encoding="utf-8") as f:
        f.write(boundary_line + "\n")
        f.write(silence_line + "\n")

    print(f"\n[DONE] Appended to {ids_path}:")
    print(f"  {boundary_line}")
    print(f"  {silence_line}")

    # ── Debug heatmaps ───────────────────────────────────────────────────────
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        print(f"\n[DEBUG] writing activation heatmaps → {debug_dir}")

        # Boundary: the ranked top-12 features over the first few segments.
        with h5py.File(h5_path, "r") as f:
            g = f[args.split]
            n_total = int(g[args.feature_ds].shape[0])
            song_segs = _collect_song_segments(g, n_total, args.max_songs)
            bnd_segs = [si for segs in song_segs.values() for si in segs][: args.debug_segments]
            sil_segs = _first_segments_of_songs(g, n_total, args.silence_songs)
        bnd_labels = [f"{fid} (top1/fB)" if fid == fB else str(fid) for fid in top12_ids]
        debug_plot_segments(
            module, model_factory, use_tag, use_idx, h5_path, args.split,
            args.feature_ds, args.device, top12_ids, bnd_labels, bnd_segs,
            debug_dir, "boundary", args.fps,
            smooth_win=args.smooth_win, mark_peaks=True, peak_tol=args.peak_tol,
            boundary_peak_drop=args.boundary_peak_drop,
            boundary_min_gap=args.boundary_min_gap,
            boundary_score_floor=args.boundary_score_floor,
            boundary_max_labels=args.boundary_max_labels,
            star_max_dist=args.star_max_dist,
        )

        # Silence: the found silence features over each silence song's first
        # segment, with the intro cutoff and negative/background window marked.
        intro_frames = max(1, int(round(args.intro_sec * args.fps)))
        neg_frames = (
            max(0, int(round(args.silence_neg_start_sec * args.fps))),
            max(0, int(round(args.silence_neg_end_sec * args.fps))),
        )
        sil_labels = [str(fid) for fid in silence_ids]
        debug_plot_segments(
            module, model_factory, use_tag, use_idx, h5_path, args.split,
            args.feature_ds, args.device, silence_ids, sil_labels, sil_segs,
            debug_dir, "silence", args.fps, intro_frames=intro_frames,
            neg_frames=neg_frames,
        )


if __name__ == "__main__":
    main()
