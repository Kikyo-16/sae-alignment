from __future__ import annotations

import typing as tp
import torch
import numpy as np
import pytorch_lightning as pl

from .group_sae import GroupSAE
from .model_factory import ModelFactory
from .score import (
    compute_mid_sae_alignment_scores,
    log_alignment_ordered_topk_agreement_to_tensorboard,
    log_alignment_ordered_usage_to_tensorboard,
    log_alignment_scores_to_tensorboard,
    log_alignment_usage_vs_topk_agreement_to_tensorboard,
    save_alignment_validation_numpy,
)


class LitSAE(pl.LightningModule):
    def __init__(
        self,
        exp_config,
        topk: int = 8,
        lr: float = 1e-4,
        lr_warmup_frac: float = 0.05,
        lr_decay_frac: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        model_config = exp_config["model_config"]
        self.data_tag = model_config["data_tag"]
        d_model = int(model_config["feature_dims"][self.data_tag])

        self.model_factory = ModelFactory(model_config=model_config)
        self.sae = GroupSAE(
            d_model=d_model,
            topk=topk,
            mlp_ratio=exp_config.get("mlp_ratio", 4),
            n_saes=exp_config["n_saes"]
        )

        self.share_mask = exp_config["share_mask"]
        self.n_saes = exp_config.get("n_saes", 1)
        self.norm_every = exp_config.get("norm_every", 10)
        self.synchronize = exp_config.get("synchronize", False)
        self.dead_threshold = exp_config.get("dead_threshold", 1e-4)
        self.topk = topk
        self.use_feature_zscore = exp_config.get("use_feature_zscore", False)
        self.norm_stats_path: str | None = exp_config.get("norm_stats_path", None)
        self.pre_bias_init = str(exp_config.get("pre_bias_init", "geomedian"))
        self.pre_bias_init_batches = int(exp_config.get("pre_bias_init_batches", 50))
        self.pre_bias_max_vectors = int(exp_config.get("pre_bias_max_vectors", 50_000))
        self.pre_bias_geomedian_iters = int(exp_config.get("pre_bias_geomedian_iters", 20))
        self.pre_bias_geomedian_eps = float(exp_config.get("pre_bias_geomedian_eps", 1e-6))
        self.orbit_init = exp_config.get("orbit_init", True)
        self.orbit_init_batches = int(exp_config.get("orbit_init_batches", 50))
        self.orbit_init_min_count = float(exp_config.get("orbit_init_min_count", 5.0))
        self.orbit_init_noise_std = float(exp_config.get("orbit_init_noise_std", 1e-4))
        self.orbit_init_chunk_frames = int(exp_config.get("orbit_init_chunk_frames", 2048))
        # Validation feature liveness logging is intentionally kept to one scalar:
        # val/feature_alive_ratio = fraction of hidden features selected at least
        # once by the shared mask over the full validation epoch.
        self.log_val_feature_alive_ratio = bool(exp_config.get("log_val_feature_alive_ratio", True))
        self._val_shared_feature_usage_counts: torch.Tensor | None = None
        # Decoder-orbit alignment diagnostic.  For n_saes >= 3, this evaluates
        # the middle SAE by default (sae_idx=1): lower scores mean better
        # pitch-transposition alignment.
        self.log_val_alignment_scores = bool(exp_config.get("log_val_alignment_scores", True))
        self.log_val_alignment_usage = bool(exp_config.get("log_val_alignment_usage", True))
        # Legacy config/log names keep "topk_agreement", but this metric is now
        # SAE-1-conditioned normalized activation disagreement: lower is better.
        self.log_val_alignment_topk_agreement = bool(
            exp_config.get("log_val_alignment_topk_agreement", True)
        )
        self.log_val_alignment_usage_vs_topk_agreement = bool(
            exp_config.get("log_val_alignment_usage_vs_topk_agreement", True)
        )
        self.save_val_alignment_numpy = bool(exp_config.get("save_val_alignment_numpy", True))
        self.alignment_target_idx = int(exp_config.get("alignment_target_idx", 1))
        self.alignment_mid_neighbor_topk = int(exp_config.get("alignment_mid_neighbor_topk", 8))
        self._val_sae1_topk_agreement_scores: torch.Tensor | None = None
        self._val_sae1_topk_agreement_counts: torch.Tensor | None = None

        # if self.use_feature_zscore:
        #     self.register_buffer(f"x_mu__{self.data_tag}", torch.zeros(d_model), persistent=True)
        #     self.register_buffer(f"x_std__{self.data_tag}", torch.ones(d_model), persistent=True)

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if "models" in k]:
            sd.pop(k, None)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr)

        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = int(self.hparams.lr_warmup_frac * total)
        decay_start = total - int(self.hparams.lr_decay_frac * total)

        def lr_lambda(step: int) -> float:
            # Linear warmup 0 -> 1 over the first `warmup` steps.
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            # Constant plateau, then linear decay 1 -> 0 over the final stretch.
            if step >= decay_start and total > decay_start:
                return max(0.0, (total - step) / (total - decay_start))
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: tp.Callable | None = None,
        *args,
        **kwargs,
    ) -> None:
        if optimizer_closure is not None:
            optimizer.step(optimizer_closure)
        else:
            optimizer.step()

        if self.global_step % self.norm_every == 0:
            for sae in self.sae.sae.sae_modules:
                sae.decoder_unit_norm_()

        optimizer.zero_grad()

    def _normalize_feature_batch(self, feature_batch):
        if not self.use_feature_zscore:
            return feature_batch
        out = {}
        for tag, x in feature_batch.items():
            if not torch.is_tensor(x):
                continue
            mu = getattr(self, f"x_mu__{tag}", None)
            std = getattr(self, f"x_std__{tag}", None)
            if mu is None or std is None:
                out[tag] = x
                continue
            if x.dim() == 4:
                mu_view, std_view = mu[None, None, None, :], std[None, None, None, :]
            elif x.dim() == 3:
                mu_view, std_view = mu[None, None, :], std[None, None, :]
            else:
                out[tag] = x
                continue
            out[tag] = (x - mu_view) / std_view.clamp_min(1e-8)
        return out

    def on_fit_start(self) -> None:
        self.model_factory.to(self.device)
        if self.use_feature_zscore:
            if self.norm_stats_path is None:
                raise ValueError(
                    "use_feature_zscore=True requires norm_stats_path to be set "
                    "(online feature-stat computation has been removed)."
                )
            self._load_norm_stats_from_file_(self.norm_stats_path)
        # Init the shared pre_bias on the post-normalization input center only
        # when training from scratch, so a resumed pre_bias is not overwritten.
        if self.trainer.ckpt_path is None:
            self._init_pre_bias_from_data(max_batches=self.pre_bias_init_batches)
            if self.orbit_init and self.n_saes > 1:
                self._init_orbit_from_data()

    @staticmethod
    @torch.no_grad()
    def _weiszfeld_geometric_median(
        samples: torch.Tensor,
        max_iters: int = 20,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Approximate geometric median by Weiszfeld iterations.

        samples: [M, D], usually kept on CPU to avoid GPU memory spikes.
        """
        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError("samples must have shape [M, D] with M > 0")
        y = samples.mean(dim=0)
        eps = float(eps)
        for _ in range(max(0, int(max_iters))):
            diff = samples - y[None, :]
            dist = diff.norm(dim=1).clamp_min(eps)
            weights = dist.reciprocal()
            y_next = (samples * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(eps)
            if torch.norm(y_next - y) <= eps * max(1.0, float(torch.norm(y))):
                y = y_next
                break
            y = y_next
        return y

    @torch.no_grad()
    def _init_pre_bias_from_data(self, max_batches: int = 50) -> None:
        """Set MultiSAE.pre_bias to a robust center of the input the SAE sees.

        Default is an approximate geometric median over a per-batch frame sample.
        Set exp_config["pre_bias_init"] = "mean" to use the streaming mean.
        """
        pre_bias = self.sae.sae.pre_bias
        d = pre_bias.shape[0]
        total = torch.zeros(d, device=self.device)
        count = 0
        samples_cpu = []
        max_vectors = max(0, int(self.pre_bias_max_vectors))
        per_batch_cap = max(1, (max_vectors + max(1, max_batches) - 1) // max(1, max_batches))
        kept = 0

        dl = self.trainer.datamodule.train_dataloader()
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            feat = self._normalize_feature_batch(self.model_factory(batch))
            x = feat[self.data_tag]            # [B, N, T, d]
            xf = x.reshape(-1, d)              # pool over B, N, T
            total += xf.sum(0)
            count += xf.shape[0]

            if max_vectors > 0 and kept < max_vectors:
                n_take = min(per_batch_cap, xf.shape[0], max_vectors - kept)
                if n_take > 0:
                    if xf.shape[0] > n_take:
                        idx = torch.randperm(xf.shape[0], device=xf.device)[:n_take]
                        sample = xf.index_select(0, idx)
                    else:
                        sample = xf
                    samples_cpu.append(sample.detach().float().cpu())
                    kept += int(sample.shape[0])

        if count == 0:
            print("[LitSAE] pre_bias init skipped (no batches)")
            return

        mean = total / count
        method = self.pre_bias_init.lower().replace("-", "_")
        if method in {"geomedian", "geometric_median", "median"} and samples_cpu:
            samples = torch.cat(samples_cpu, dim=0)
            center = self._weiszfeld_geometric_median(
                samples,
                max_iters=self.pre_bias_geomedian_iters,
                eps=self.pre_bias_geomedian_eps,
            ).to(device=pre_bias.device, dtype=pre_bias.dtype)
            pre_bias.copy_(center)
            print(
                f"[LitSAE] shared pre_bias init by geometric median "
                f"from {kept}/{count} sampled vectors, |b|={pre_bias.norm():.4f}"
            )
        elif method == "mean" or not samples_cpu:
            pre_bias.copy_(mean)
            print(
                f"[LitSAE] shared pre_bias init by mean from {count} vectors, "
                f"|b|={pre_bias.norm():.4f}"
            )
        else:
            raise ValueError(
                f"Unknown pre_bias_init='{self.pre_bias_init}'. "
                "Use 'geomedian' or 'mean'."
            )

    @torch.no_grad()
    def _init_orbit_from_data(self) -> None:
        """Initialize sub-SAE decoders from a random anchor SAE0 orbit map.

        SAE0 keeps its random decoder. We run SAE0 over paired augmentation
        frames [slot0, slot1, ...]. If slot0 top-1 is i and slot s top-1 is j,
        then sub-SAE s feature i accumulates SAE0 decoder feature j. This gives
        every feature index an initial transposition-orbit interpretation.
        """
        multi_sae = self.sae.sae
        modules = multi_sae.sae_modules
        if len(modules) <= 1:
            return

        sae0 = modules[0]
        sae0.decoder_unit_norm_()
        sae0.fc1.weight.copy_(sae0.fc2.weight.t())

        D0 = sae0.fc2.weight.detach()  # [D, H], columns are decoder features
        device = D0.device
        dtype = D0.dtype
        d_model, hidden = D0.shape
        pre_bias = multi_sae.pre_bias.detach().to(device=device, dtype=dtype)
        eps = float(getattr(multi_sae, "eps", 1e-6))
        chunk_frames = max(1, int(self.orbit_init_chunk_frames))

        acc = [torch.zeros(d_model, hidden, device=device, dtype=dtype)
               for _ in range(1, len(modules))]
        counts = [torch.zeros(hidden, device=device, dtype=dtype)
                  for _ in range(1, len(modules))]

        dl = self.trainer.datamodule.train_dataloader()
        total_pairs = 0
        used_pairs = [0 for _ in range(1, len(modules))]

        for batch_idx, batch in enumerate(dl):
            if batch_idx >= self.orbit_init_batches:
                break
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            feat = self._normalize_feature_batch(self.model_factory(batch))
            x = feat[self.data_tag]  # [B, N, T, D]
            if x.dim() != 4:
                raise ValueError(
                    f"orbit_init expects feature tensor [B,N,T,D], got shape {tuple(x.shape)}"
                )
            if x.shape[1] < len(modules):
                raise ValueError(
                    f"orbit_init expects at least {len(modules)} slots, got {x.shape[1]}"
                )

            x = x[:, :len(modules)].reshape(-1, len(modules), d_model).to(dtype=dtype)
            total_pairs += x.shape[0]

            for start in range(0, x.shape[0], chunk_frames):
                xc = x[start:start + chunk_frames]
                centered0 = xc[:, 0] - pre_bias
                z0 = torch.relu(centered0 @ D0)
                v0, i0 = z0.max(dim=-1)

                for slot in range(1, len(modules)):
                    centered_s = xc[:, slot] - pre_bias
                    zs = torch.relu(centered_s @ D0)
                    vs, js = zs.max(dim=-1)

                    valid = (v0 > eps) & (vs > eps)
                    if not torch.any(valid):
                        continue

                    i_valid = i0[valid]
                    j_valid = js[valid]
                    weights = torch.sqrt((v0[valid] * vs[valid]).clamp_min(eps))
                    contrib = D0[:, j_valid] * weights.unsqueeze(0)
                    acc[slot - 1].index_add_(dim=1, index=i_valid, source=contrib)
                    counts[slot - 1].index_add_(dim=0, index=i_valid, source=weights)
                    used_pairs[slot - 1] += int(valid.sum().item())

        if total_pairs == 0:
            print("[LitSAE] orbit init skipped (no paired frames)")
            return

        for slot in range(1, len(modules)):
            cnt = counts[slot - 1]
            enough = cnt >= self.orbit_init_min_count
            low = ~enough

            new_dec = torch.empty_like(D0)
            if torch.any(enough):
                new_dec[:, enough] = acc[slot - 1][:, enough] / cnt[enough].clamp_min(eps).unsqueeze(0)
            if torch.any(low):
                fallback = D0[:, low]
                if self.orbit_init_noise_std > 0:
                    fallback = fallback + self.orbit_init_noise_std * torch.randn_like(fallback)
                new_dec[:, low] = fallback

            new_dec = new_dec / new_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
            modules[slot].fc2.weight.copy_(new_dec)
            modules[slot].decoder_unit_norm_()
            modules[slot].fc1.weight.copy_(modules[slot].fc2.weight.t())

            enough_count = int(enough.sum().item())
            print(
                f"[LitSAE] orbit init slot={slot}: paired_frames={total_pairs} "
                f"used={used_pairs[slot - 1]} initialized={enough_count}/{hidden} "
                f"min_count={self.orbit_init_min_count:g}"
            )

    @torch.no_grad()
    def _load_norm_stats_from_file_(self, path: str) -> None:
        data = np.load(path)
        mean = torch.from_numpy(data["mean"]).float().to(self.device)
        std = torch.from_numpy(data["std"]).float().to(self.device)
        tag = self.data_tag
        if not hasattr(self, f"x_mu__{tag}"):
            self.register_buffer(f"x_mu__{tag}", mean, persistent=True)
            self.register_buffer(f"x_std__{tag}", std, persistent=True)
        else:
            getattr(self, f"x_mu__{tag}").copy_(mean)
            getattr(self, f"x_std__{tag}").copy_(std)
        print(f"[LitSAE] loaded norm stats from {path}  mean={mean[:4]}  std={std[:4]}")

    def _log_diagnostic_metrics(
        self,
        prefix: str,
        res: dict[str, tp.Any],
        *,
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        mean_keys = (
            "shared_winner_ratio_mean",
            "shared_energy_ratio_mean",
        )
        for k in mean_keys:
            if k in res:
                self.log(
                    f"{prefix}/{k}",
                    res[k],
                    prog_bar=False,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )

        per_sae_prefixes = (
            "shared_winner_ratio_sae_",
            "shared_energy_ratio_sae_",
        )
        for i in range(self.n_saes):
            for p in per_sae_prefixes:
                k = f"{p}{i}"
                if k in res:
                    self.log(
                        f"{prefix}/{k}",
                        res[k],
                        prog_bar=False,
                        on_step=on_step,
                        on_epoch=on_epoch,
                        batch_size=batch_size,
                    )

    def _init_val_feature_alive_buffer(self) -> None:
        """Allocate validation feature-usage and SAE-1 agreement counters."""
        hidden = self.sae.sae.sae_modules[0].hidden
        self._val_shared_feature_usage_counts = torch.zeros(
            hidden, device=self.device, dtype=torch.float32
        )
        self._val_sae1_topk_agreement_scores = torch.zeros(
            hidden, device=self.device, dtype=torch.float32
        )
        self._val_sae1_topk_agreement_counts = torch.zeros(
            hidden, device=self.device, dtype=torch.float32
        )

    def on_validation_epoch_start(self) -> None:
        if (
            self.log_val_feature_alive_ratio
            or self.log_val_alignment_usage
            or self.log_val_alignment_topk_agreement
            or self.log_val_alignment_usage_vs_topk_agreement
        ):
            self._init_val_feature_alive_buffer()

    def _accumulate_val_feature_alive(self, res: dict[str, tp.Any]) -> None:
        if not (
            self.log_val_feature_alive_ratio
            or self.log_val_alignment_usage
            or self.log_val_alignment_topk_agreement
            or self.log_val_alignment_usage_vs_topk_agreement
        ):
            return
        if self._val_shared_feature_usage_counts is None:
            self._init_val_feature_alive_buffer()

        shared_counts = res.get("shared_feature_usage_counts", None)
        if shared_counts is not None:
            self._val_shared_feature_usage_counts.add_(
                shared_counts.detach().to(device=self.device, dtype=torch.float32)
            )

        agreement_scores = res.get("sae1_topk_agreement_scores", None)
        if agreement_scores is not None and self._val_sae1_topk_agreement_scores is not None:
            self._val_sae1_topk_agreement_scores.add_(
                agreement_scores.detach().to(device=self.device, dtype=torch.float32)
            )

        agreement_counts = res.get("sae1_topk_agreement_counts", None)
        if agreement_counts is not None and self._val_sae1_topk_agreement_counts is not None:
            self._val_sae1_topk_agreement_counts.add_(
                agreement_counts.detach().to(device=self.device, dtype=torch.float32)
            )

    def _sum_across_processes_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.trainer, "world_size", 1) > 1:
            gathered = self.all_gather(x)
            return gathered.sum(dim=0)
        return x

    def _log_val_alignment_scores(
        self,
        val_shared_feature_usage_counts: torch.Tensor | None = None,
        val_sae1_topk_agreement_scores: torch.Tensor | None = None,
        val_sae1_topk_agreement_counts: torch.Tensor | None = None,
    ) -> None:
        """Log sorted decoder alignment and matching validation figures."""
        if not self.log_val_alignment_scores:
            return
        if self.n_saes < 3:
            return
        if self.alignment_target_idx <= 0 or self.alignment_target_idx >= self.n_saes - 1:
            return

        scores = compute_mid_sae_alignment_scores(
            self.sae.sae.sae_modules,
            target_idx=self.alignment_target_idx,
            mid_neighbor_topk=self.alignment_mid_neighbor_topk,
        )
        score = scores["alignment_score"].to(device=self.device, dtype=torch.float32)
        self.log(
            f"val/sae_{self.alignment_target_idx}_alignment_mean",
            score.mean(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            f"val/sae_{self.alignment_target_idx}_alignment_median",
            score.median(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        trainer = getattr(self, "trainer", None)
        if trainer is not None and not getattr(trainer, "is_global_zero", True):
            return

        if (
            self.save_val_alignment_numpy
            and val_shared_feature_usage_counts is not None
            and val_sae1_topk_agreement_scores is not None
            and val_sae1_topk_agreement_counts is not None
        ):
            from pathlib import Path
            data_dir = Path(getattr(trainer, "default_root_dir", ".") if trainer is not None else ".") / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            fname = (
                f"sae_{self.alignment_target_idx}_alignment_"
                f"epoch{int(self.current_epoch):04d}_step{int(self.global_step):08d}.npz"
            )
            for out_path in (data_dir / fname, data_dir / f"sae_{self.alignment_target_idx}_alignment_latest.npz"):
                save_alignment_validation_numpy(
                    out_path,
                    scores,
                    val_shared_feature_usage_counts.detach().cpu(),
                    val_sae1_topk_agreement_scores.detach().cpu(),
                    val_sae1_topk_agreement_counts.detach().cpu(),
                    topk=self.topk,
                    epoch=int(self.current_epoch),
                    global_step=int(self.global_step),
                )

        loggers = []
        if trainer is not None and getattr(trainer, "loggers", None):
            loggers = list(trainer.loggers)
        elif getattr(self, "logger", None) is not None:
            loggers = [self.logger]

        for logger in loggers:
            log_alignment_scores_to_tensorboard(
                logger,
                scores,
                global_step=int(self.global_step),
                tag=f"val/sae_{self.alignment_target_idx}_alignment_sorted",
                title=f"SAE-{self.alignment_target_idx} feature alignment scores",
            )
            if self.log_val_alignment_usage and val_shared_feature_usage_counts is not None:
                log_alignment_ordered_usage_to_tensorboard(
                    logger,
                    scores,
                    val_shared_feature_usage_counts.detach().cpu(),
                    topk=self.topk,
                    global_step=int(self.global_step),
                    tag=f"val/sae_{self.alignment_target_idx}_alignment_ordered_feature_usage",
                    title=(
                        f"SAE-{self.alignment_target_idx} validation feature usage "
                        "ordered by alignment"
                    ),
                )
            if (
                self.log_val_alignment_topk_agreement
                and val_sae1_topk_agreement_scores is not None
                and val_sae1_topk_agreement_counts is not None
            ):
                log_alignment_ordered_topk_agreement_to_tensorboard(
                    logger,
                    scores,
                    val_sae1_topk_agreement_scores.detach().cpu(),
                    val_sae1_topk_agreement_counts.detach().cpu(),
                    global_step=int(self.global_step),
                    tag=f"val/sae_{self.alignment_target_idx}_alignment_ordered_topk_agreement",
                    title=(
                        f"SAE-{self.alignment_target_idx} activation disagreement "
                        "vs alignment score"
                    ),
                )
            if (
                self.log_val_alignment_usage_vs_topk_agreement
                and val_shared_feature_usage_counts is not None
                and val_sae1_topk_agreement_scores is not None
                and val_sae1_topk_agreement_counts is not None
            ):
                log_alignment_usage_vs_topk_agreement_to_tensorboard(
                    logger,
                    scores,
                    val_shared_feature_usage_counts.detach().cpu(),
                    val_sae1_topk_agreement_scores.detach().cpu(),
                    val_sae1_topk_agreement_counts.detach().cpu(),
                    topk=self.topk,
                    global_step=int(self.global_step),
                    tag=f"val/sae_{self.alignment_target_idx}_alignment_usage_vs_topk_agreement",
                    title=(
                        f"SAE-{self.alignment_target_idx} activation disagreement "
                        "vs validation feature usage"
                    ),
                )

    def on_validation_epoch_end(self) -> None:
        counts_all = None
        agreement_scores_all = None
        agreement_counts_all = None
        if self._val_shared_feature_usage_counts is not None:
            counts_all = self._sum_across_processes_if_needed(self._val_shared_feature_usage_counts)
        if self._val_sae1_topk_agreement_scores is not None:
            agreement_scores_all = self._sum_across_processes_if_needed(
                self._val_sae1_topk_agreement_scores
            )
        if self._val_sae1_topk_agreement_counts is not None:
            agreement_counts_all = self._sum_across_processes_if_needed(
                self._val_sae1_topk_agreement_counts
            )

        if self.log_val_feature_alive_ratio and counts_all is not None:
            feature_alive_ratio = (counts_all > 0).float().mean()
            self.log(
                "val/feature_alive_ratio",
                feature_alive_ratio,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        self._log_val_alignment_scores(
            counts_all,
            agreement_scores_all,
            agreement_counts_all,
        )

    def training_step(self, batch: dict[str, tp.Any], _: int) -> torch.Tensor:
        with torch.no_grad():
            feature_batch = self.model_factory(batch)
            feature_batch = self._normalize_feature_batch(feature_batch)

        res = self.sae(
            feature_batch,
            dead_threshold=self.dead_threshold,
            share_mask=self.share_mask,
            synchronize=self.synchronize,
        )

        B = next((len(v) for v in batch.values() if hasattr(v, "__len__")), 1)
        loss = res["mse_loss"]
        self.log("train/mse_loss", res["mse_loss"], prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        self.log("train/anchor_loss", res["anchor_loss"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self.log("train/alive_ratio_mean", res["alive_ratio_mean"], prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        if "shared_eq" in res and self.n_saes > 1:
            for j in range(1, self.n_saes + 1):
                k = f"shared_eq_{j}"
                if k in res["shared_eq"]:
                    self.log(f"train/{k}", res["shared_eq"][k], prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        for i in range(self.n_saes):
            k = f"alive_ratio_sae_{i}"
            if k in res:
                self.log(f"train/{k}", res[k], prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        self._log_diagnostic_metrics("train", res, batch_size=B, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def validation_step(self, batch: dict[str, tp.Any], batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            feature_batch = self.model_factory(batch)
            feature_batch = self._normalize_feature_batch(feature_batch)
            res = self.sae(
                feature_batch,
                dead_threshold=self.dead_threshold,
                share_mask=self.share_mask,
                synchronize=self.synchronize,
            )

        B = next((len(v) for v in batch.values() if hasattr(v, "__len__")), 1)
        val_loss = res["mse_loss"]
        self._accumulate_val_feature_alive(res)
        self.log("val/mse_loss", res["mse_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
        self.log("val/anchor_loss", res["anchor_loss"], prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
        if "shared_eq" in res and self.n_saes > 1:
            for j in range(1, self.n_saes + 1):
                k = f"shared_eq_{j}"
                if k in res["shared_eq"]:
                    self.log(f"val/{k}", res["shared_eq"][k], prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
        # Do not log validation feature-activation diagnostics here; the single
        # validation liveness scalar is emitted in on_validation_epoch_end.
        self.log("val_mse_loss", res["mse_loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
        return val_loss
