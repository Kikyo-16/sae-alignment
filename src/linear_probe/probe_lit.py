#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linear probe LightningModule supporting three tasks:

    key_relative : 12-class scale-degree probe (relative-major key detection).
                   Per-frame BCE with N-rejection; tonic/forth/fifth selected by
                   the data loader's label_tag.
    chord        : 25-class chord probe (12 maj + 12 min + N).  Per-frame
                   cross-entropy with ignore_index for non-maj/min-triad frames.
    key_strict   : 24-class strict key probe (12 maj + 12 min).  Per-frame logits
                   are mean-pooled to one segment vector, cross-entropy on the
                   segment key class.  (Placeholder global head; structure TBD.)

Checkpoint-selection metrics are computed by ValMetric callbacks (see
linear_probe/metrics.py); the validation_step here only logs a light loss/acc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .data_loader import (
    TASK_KEY_RELATIVE, TASK_CHORD, TASK_KEY_STRICT, ALL_TASKS,
    DEGREE_N, CHORD_IGNORE,
)

NUM_CLASSES_BY_TASK = {
    TASK_KEY_RELATIVE: 12,
    TASK_CHORD: 25,
    TASK_KEY_STRICT: 24,
}


class LitLinearProbe(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        feature_tag: str,
        task: str = TASK_KEY_RELATIVE,
        lr: float = 1e-3,
        threshold: float = 0.7,
        sae: nn.Module | None = None,
        sae_ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        if task not in ALL_TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {ALL_TASKS}")
        self.save_hyperparameters(ignore=["sae"])

        # When loading from a probe checkpoint, sae is None (excluded from hparams)
        # but sae_ckpt_path is stored — use it to reconstruct the SAE automatically.
        if sae is None and sae_ckpt_path is not None:
            from ..music_sae.sae_lit import LitSAE
            lit_sae = LitSAE.load_from_checkpoint(sae_ckpt_path, map_location="cpu")
            lit_sae.eval()
            sae = lit_sae.sae.sae.sae_modules[0]
            del lit_sae
        if sae is not None:
            self.sae_encoder = sae
            for p in self.sae_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.sae_encoder = None

        self.linear = nn.Linear(d_model, NUM_CLASSES_BY_TASK[task])

    # ── checkpoint: do not persist the frozen SAE encoder ────────────────────
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint["state_dict"]
        for k in [k for k in sd if k.startswith("sae_encoder.")]:
            del sd[k]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.sae_encoder is not None:
            for k, v in self.sae_encoder.state_dict().items():
                checkpoint["state_dict"][f"sae_encoder.{k}"] = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sae_encoder is not None:
            with torch.no_grad():
                x = self.sae_encoder.fc2(F.relu(self.sae_encoder.fc1(x)))
        return self.linear(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    # ── per-task steps ───────────────────────────────────────────────────────
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        task = self.hparams.task
        if task == TASK_CHORD:
            return self._step_chord(batch, stage)
        if task == TASK_KEY_STRICT:
            return self._step_key_strict(batch, stage)
        return self._step_key_relative(batch, stage)

    def _step_key_relative(self, batch: dict, stage: str) -> torch.Tensor:
        x = batch[self.hparams.feature_tag]   # (B, T, D)
        y = batch["label"]                    # (B, T)  int64, 0-11 pc, 12=N
        B, T, _ = x.shape
        logits = self(x).reshape(B * T, -1)   # (B*T, 12)
        targets = y.reshape(B * T)

        is_pitch = targets < DEGREE_N
        onehot = torch.zeros_like(logits)
        if is_pitch.any():
            onehot[is_pitch] = F.one_hot(targets[is_pitch], num_classes=DEGREE_N).float()
        loss = F.binary_cross_entropy_with_logits(logits, onehot)

        with torch.no_grad():
            best_logit, best_cls = logits.max(-1)
            preds = torch.where(
                best_logit.sigmoid() >= self.hparams.threshold,
                best_cls, torch.full_like(best_cls, DEGREE_N),
            )
            acc = (preds == targets).float().mean()
        self._log_common(stage, loss, acc, B)
        return loss

    def _step_chord(self, batch: dict, stage: str) -> torch.Tensor:
        x = batch[self.hparams.feature_tag]   # (B, T, D)
        y = batch["label"]                    # (B, T)  0-23 chord, 24=N, -100=ignore
        B, T, _ = x.shape
        logits = self(x).reshape(B * T, -1)   # (B*T, 25)
        targets = y.reshape(B * T)
        loss = F.cross_entropy(logits, targets, ignore_index=CHORD_IGNORE)

        with torch.no_grad():
            keep = targets != CHORD_IGNORE
            preds = logits.argmax(-1)
            acc = (preds[keep] == targets[keep]).float().mean() if keep.any() else torch.tensor(0.0)
        self._log_common(stage, loss, acc, B)
        return loss

    def _step_key_strict(self, batch: dict, stage: str) -> torch.Tensor:
        x = batch[self.hparams.feature_tag]   # (B, T, D)
        y = batch["label"]                    # (B,)  segment key class 0-23
        logits = self(x).mean(dim=1)          # (B, 24)  mean-pool over time
        loss = F.cross_entropy(logits, y)

        with torch.no_grad():
            acc = (logits.argmax(-1) == y).float().mean()
        self._log_common(stage, loss, acc, x.shape[0])
        return loss

    def _log_common(self, stage: str, loss: torch.Tensor, acc: torch.Tensor, bs: int) -> None:
        on_step = (stage == "train")
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/frame_acc", acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, _: int) -> torch.Tensor:
        return self._step(batch, "val")
