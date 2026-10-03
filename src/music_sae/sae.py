from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_tools.sae_utils import ActivationStats

from .modules import (
    apply_topk as apply_topk_utils,
    per_frame_shared_counts,
)


class SAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        topk: int = 32,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        self.l1_coeff = l1_coeff
        self.topk = topk
        hidden = int(d_model * mlp_ratio)
        self.hidden = hidden
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        # pre_bias is a single parameter shared across all sub-SAEs; it is owned
        # by the parent MultiSAE and injected via attach_pre_bias(). Stored as a
        # plain (non-registered) reference so it is not duplicated in state_dict.
        self._pre_bias: nn.Parameter | None = None
        self.normalize_decoder = True
        self.stats = ActivationStats(hidden, ema_beta=0.99, eps=1e-6, device=None)

        print(f"[SAE] {d_model} -> {hidden} -> {d_model} (topk={topk})")
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.0, nonlinearity="relu")
        self.decoder_unit_norm_()
        self.fc1.weight.data.copy_(self.fc2.weight.data.t())

    def attach_pre_bias(self, pre_bias: nn.Parameter) -> None:
        """Bind the MultiSAE-owned shared pre_bias without nn.Module registration."""
        object.__setattr__(self, "_pre_bias", pre_bias)

    @property
    def pre_bias(self) -> nn.Parameter:
        return self._pre_bias

    @torch.no_grad()
    def decoder_unit_norm_(self, eps: float = 1e-8):
        """Normalize decoder columns to unit norm."""
        if not self.normalize_decoder:
            return
        w = self.fc2.weight
        norms = torch.norm(w, dim=0, keepdim=True).clamp_min(eps)
        self.fc2.weight.div_(norms)

    @torch.no_grad()
    def inference(self, x: torch.Tensor, topk_wide: float = 0.0) -> torch.Tensor:
        """Return post-ReLU activations [B,T,H]. topk_wide>0 masks to int(topk*topk_wide) active units."""
        z = F.relu(self.fc1(x - self.pre_bias))
        if topk_wide > 0:
            mask = apply_topk_utils(z, k=int(self.topk * topk_wide), return_indices=False)
            return z * mask
        return z

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Return SAE reconstruction x_hat. Same shape as x."""
        z = F.relu(self.fc1(x - self.pre_bias))
        return self.fc2(z) + self.pre_bias

    def encode(self, x: torch.Tensor, topk: int):
        z = F.relu(self.fc1(x - self.pre_bias))
        topk_idx = apply_topk_utils(z, k=topk, return_indices=True)
        if self.training:
            self.stats.update_from_topk(topk_idx)
        return z, topk_idx

    def decode(
        self,
        z,
        topk_idx,
        x: torch.Tensor = None,
        return_mse: bool = False,
    ):
        """z: [B,T,E], topk_idx: [B,T,K], x: [B,T,D]"""
        mask = torch.zeros_like(z.flatten(0, 1), dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_idx.flatten(0, 1), value=True)
        mask = mask.reshape_as(z)
        x_hat = self.fc2(z * mask) + self.pre_bias

        if return_mse:
            return F.mse_loss(x_hat, x)

        return x_hat


class MultiSAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        topk: int = 32,
        l1_coeff: float = 1e-4,
        eps: float = 1e-6,
        n_sp: int = 4,
        matching: int = 64,
    ):
        super().__init__()
        self.sae_modules = nn.ModuleList(
            SAE(d_model=d_model, mlp_ratio=mlp_ratio, l1_coeff=l1_coeff, topk=topk)
            for _ in range(n_sp)
        )
        # single pre_bias shared by every sub-SAE (centering is common to the group)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        for m in self.sae_modules:
            m.attach_pre_bias(self.pre_bias)
        self.n_saes = n_sp
        self.topk = topk
        self.matching = matching
        self.eps = eps

    def encode(self, x: torch.Tensor, topk: int):
        """x: [B, N, T, d]"""
        zs, topk_indices = [], []
        for idx in range(len(self.sae_modules)):
            z, topk_idx = self.sae_modules[idx].encode(x=x[:, idx], topk=topk)
            zs.append(z)
            topk_indices.append(topk_idx)
        return zs, topk_indices

    def decode_mse(self, zs, x: torch.Tensor, topk_idx_shared):
        """zs: list of [B,T,E], x: [B,N,T,D], topk_idx_shared: [B,T,K]"""
        x_hats, losses = [], []
        for i in range(self.n_saes):
            x_hat_i = self.sae_modules[i].decode(
                z=zs[i],
                topk_idx=topk_idx_shared,
                x=x[:, i],
            )
            x_hats.append(x_hat_i)
            losses.append(F.mse_loss(x_hat_i, x[:, i]))

        losses = torch.stack(losses)
        mse_loss = losses.mean()
        anchor_loss = losses[0]
        return mse_loss, anchor_loss, x_hats

    def forward(
        self,
        x: torch.Tensor,
        share_mask: bool,
        dead_threshold: float = 1e-4,
        topk: int | None = None,
    ):
        """x: [B, N, T, d]"""
        if topk is None:
            topk = self.topk

        zs, topk_indices = self.encode(x, topk=topk)

        if share_mask:
            z_normed = []
            for z in zs:
                scale = z.detach().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
                z_normed.append(z / scale)
            z_normed_stack = torch.stack(z_normed, dim=1)  # [B, N, T, H]
            z_shared_score, _ = torch.max(z_normed_stack, dim=1)
            topk_idx_shared = apply_topk_utils(z_shared_score, k=topk, return_indices=True)
        else:
            raise NotImplementedError("share_mask=False is not supported")

        mse_loss, anchor_loss, x_hats = self.decode_mse(
            zs, x=x, topk_idx_shared=topk_idx_shared
        )

        res = {
            "mse_loss": mse_loss,
            "anchor_loss": anchor_loss,
        }

        if self.n_saes > 1:
            topk_idx_all = torch.stack(topk_indices, dim=2)[..., : self.topk]
            shared_eq = per_frame_shared_counts(topk_idx_all=topk_idx_all, max_n=self.n_saes)
            res["shared_eq"] = {k: (v * 1.0 / self.topk).mean() for k, v in shared_eq.items()}

            # Which slot proposed each selected shared-mask feature?  This uses
            # the same normalized activations used to build the shared mask.
            winner = z_normed_stack.argmax(dim=1)  # [B, T, H]
            winner_on_shared = torch.gather(winner, dim=-1, index=topk_idx_shared)  # [B, T, K]
            winner_vals = []
            for i in range(self.n_saes):
                v = (winner_on_shared == i).float().mean()
                res[f"shared_winner_ratio_sae_{i}"] = v
                winner_vals.append(v)
            res["shared_winner_ratio_mean"] = torch.stack(winner_vals).mean()

            # How much of each SAE's raw activation energy is covered by the
            # final shared mask? Low values indicate that the shared mask is not
            # selecting what that SAE itself strongly activates.
            shared_mask = torch.zeros_like(zs[0].flatten(0, 1), dtype=torch.bool)
            shared_mask.scatter_(dim=-1, index=topk_idx_shared.flatten(0, 1), value=True)
            shared_mask = shared_mask.reshape_as(zs[0])
            energy_vals = []
            for i, z in enumerate(zs):
                used_energy = (z * shared_mask).abs().sum(dim=-1)
                total_energy = z.abs().sum(dim=-1).clamp_min(self.eps)
                v = (used_energy / total_energy).mean()
                res[f"shared_energy_ratio_sae_{i}"] = v
                energy_vals.append(v)
            res["shared_energy_ratio_mean"] = torch.stack(energy_vals).mean()

            # Validation diagnostic for SAE-1-conditioned activation disagreement.
            # For every feature id selected by SAE-1's own top-k, compare the
            # normalized activations of the same feature id across SAE-0/1/2.
            # The per-frame contribution is a relative pairwise L1 disagreement:
            #   (|a0-a1| + |a1-a2| + |a0-a2|) / (2*(a0+a1+a2)+eps)
            # Values are near 0 when all three normalized activations agree and
            # near 1 when only one side fires. Counts are returned separately so
            # the epoch-level plot can show mean disagreement over SAE-1-selected
            # frames, in alignment-score order. This is only defined for the
            # 3-SAE low/mid/high setup.
            if self.n_saes >= 3:
                mid_idx = topk_indices[1]   # [B, T, K]
                a0 = torch.gather(z_normed_stack[:, 0], dim=-1, index=mid_idx)
                a1 = torch.gather(z_normed_stack[:, 1], dim=-1, index=mid_idx)
                a2 = torch.gather(z_normed_stack[:, 2], dim=-1, index=mid_idx)
                disagreement = (
                    (a0 - a1).abs() + (a1 - a2).abs() + (a0 - a2).abs()
                ) / (2.0 * (a0 + a1 + a2).clamp_min(self.eps))
                mid_flat = mid_idx.reshape(-1)
                hidden = self.sae_modules[1].hidden
                disagreement_scores = torch.zeros(hidden, device=x.device, dtype=torch.float32)
                disagreement_scores.scatter_add_(0, mid_flat, disagreement.float().reshape(-1))
                disagreement_counts = torch.bincount(
                    mid_flat, minlength=hidden
                ).to(device=x.device, dtype=torch.float32)
                # Keep the legacy result keys so the Lightning/TensorBoard path
                # and downstream npz names remain backward compatible. The value
                # is now activation disagreement: lower is better.
                res["sae1_topk_agreement_scores"] = disagreement_scores.detach()
                res["sae1_topk_agreement_counts"] = disagreement_counts.detach()
        else:
            res["shared_eq"] = {}

        # Validation-set alive metric support.  We only expose shared-mask usage
        # counts because those are the actual feature indices used for decode in
        # the shared-mask model.  Expensive per-SAE usage histograms, entropy,
        # max/median summaries, and TensorBoard figures are intentionally removed.
        shared_flat = topk_idx_shared.reshape(-1)
        shared_hidden = self.sae_modules[0].hidden
        res["shared_feature_usage_counts"] = torch.bincount(
            shared_flat, minlength=shared_hidden
        ).to(device=x.device, dtype=torch.float32).detach()

        alive = [self.sae_modules[i].stats.alive_count(threshold=dead_threshold) for i in range(self.n_saes)]
        for i, a in enumerate(alive):
            res[f"alive_ratio_sae_{i}"] = torch.tensor(
                a / float(self.sae_modules[i].hidden), device=x.device, dtype=x.dtype
            )
        res["alive_ratio_mean"] = torch.stack([res[f"alive_ratio_sae_{i}"] for i in range(self.n_saes)]).mean()

        return res

    @torch.no_grad()
    def inference(self, x: torch.Tensor, idx: int, topk_wide: float = 0.0) -> torch.Tensor:
        """Delegate to sae_modules[idx].inference. Returns [B,T,H]."""
        return self.sae_modules[idx].inference(x, topk_wide=topk_wide)

   