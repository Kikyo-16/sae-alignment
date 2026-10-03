from __future__ import annotations

import copy
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm.auto import tqdm

from ..linear_probe.probe_lit import LitLinearProbe
from ..analysis.infer_key import (
    NOTE_NAMES_12,
    decode_str_array,
    parse_key_signature_major_pc,
    ref_segment_signature_from_key_frame,
)

LABEL_TAG_TO_DEGREE: dict[str, int] = {
    "tonic_frame": 0,
    "forth_frame": 5,
    "fifth_frame": 7,
}

ALL_LABEL_TAGS: tuple[str, ...] = ("tonic_frame", "forth_frame", "fifth_frame")


def _mean_probs_to_est_key(mean_probs: np.ndarray, degree_semitones: int) -> str:
    best_degree_pc = int(np.argmax(mean_probs))
    tonic_pc = (best_degree_pc - degree_semitones + 12) % 12
    return f"{NOTE_NAMES_12[tonic_pc]}:maj"


class ProbeKeyMetric:
    """
    Segment-level key accuracy via SAE encoder NN weight replacement.

    For each of the 12 probe output weight vectors, the nearest SAE encoder
    vector (cosine similarity) from sae_modules[sae_idx] is substituted.
    The modified probe is then run over the H5 split to compute accuracy.

    The base probe is loaded once and cached; a deepcopy is made each call so
    the base weights are never mutated.
    """

    def __init__(
        self,
        probe_ckpt: str,
        h5_path: str,
        feature_ds: str,
        label_tag: str,
        split: str = "test",
        sae_idx: int = 0,
        batch_size: int = 64,
        max_segs: int = -1,
    ) -> None:
        if label_tag not in LABEL_TAG_TO_DEGREE:
            raise ValueError(f"Unknown label_tag '{label_tag}'. Choose from {list(LABEL_TAG_TO_DEGREE)}")
        self.probe_ckpt = probe_ckpt
        self.h5_path = h5_path
        self.feature_ds = feature_ds
        self.label_tag = label_tag
        self.split = split
        self.sae_idx = sae_idx
        self.batch_size = batch_size
        self.max_segs = max_segs
        self._probe_base: Optional[LitLinearProbe] = None

    @property
    def name(self) -> str:
        return f"val/probe_key_{self.label_tag}_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _base_probe(self) -> LitLinearProbe:
        if self._probe_base is None:
            print(f"[ProbeKeyMetric] Loading probe from {self.probe_ckpt}")
            self._probe_base = LitLinearProbe.load_from_checkpoint(
                self.probe_ckpt, map_location="cpu", strict=False
            )
            self._probe_base.eval()
        return self._probe_base

    @torch.no_grad()
    def _build_replaced_probe(self, pl_module: pl.LightningModule, device: str) -> LitLinearProbe:
        probe = copy.deepcopy(self._base_probe()).to(device)

        sae_module = pl_module.sae.sae.sae_modules[self.sae_idx]
        enc_w = sae_module.fc1.weight.data.to(device)  # (hidden, d_model)
        probe_w = probe.linear.weight.data              # (12, d_model)

        cos_sim = F.normalize(probe_w, dim=1) @ F.normalize(enc_w, dim=1).T  # (12, hidden)
        best_idx = cos_sim.argmax(dim=1)
        best_scores = cos_sim[torch.arange(12), best_idx]

        print(f"[ProbeKeyMetric] SAE NN indices:  {best_idx.tolist()}")
        print(f"[ProbeKeyMetric] cosine sims:     {[round(float(v), 4) for v in best_scores.tolist()]}")

        probe.linear.weight.data = enc_w[best_idx].clone()
        return probe

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        device = str(pl_module.device)
        probe = self._build_replaced_probe(pl_module, device)
        degree_semitones = LABEL_TAG_TO_DEGREE[self.label_tag]

        eval_total = 0
        eval_correct = 0

        with h5py.File(self.h5_path, "r") as f:
            if self.split not in f:
                raise KeyError(f"Split '{self.split}' not found. Available: {list(f.keys())}")
            g = f[self.split]

            if self.feature_ds not in g:
                raise KeyError(
                    f"Feature '{self.feature_ds}' not in split '{self.split}'. "
                    f"Available: {list(g.keys())}"
                )

            has_labels = ("labels" in g) and ("key_frame" in g["labels"])
            ds_feat = g[self.feature_ds]
            n_total = int(ds_feat.shape[0])
            end = min(n_total, int(self.max_segs)) if self.max_segs > 0 else n_total
            bs = max(1, self.batch_size)

            pbar = tqdm(total=end, desc=f"ProbeKey({self.label_tag})", unit="seg", leave=False)

            for s0 in range(0, end, bs):
                s1 = min(end, s0 + bs)

                feat_np = np.array(ds_feat[s0:s1], copy=True)  # (B, D, T)
                feat_t = torch.from_numpy(feat_np).float().to(device)
                feat_t = feat_t.permute(0, 2, 1)              # (B, T, D)

                with torch.no_grad():
                    logits = torch.relu(probe(feat_t))           # (B, T, 12)
                    mean_probs_batch = logits.mean(dim=1).cpu().numpy()  # (B, 12)

                for j in range(s1 - s0):
                    mean_probs = mean_probs_batch[j].astype(np.float32)
                    key_frame = decode_str_array(g["labels"]["key_frame"][s0 + j]) if has_labels else None

                    ref_sig, has_change, has_ref = ref_segment_signature_from_key_frame(key_frame)
                    if has_change or not has_ref:
                        continue

                    eval_total += 1
                    est_key = _mean_probs_to_est_key(mean_probs, degree_semitones)
                    est_sig = parse_key_signature_major_pc(est_key)
                    if est_sig is not None and ref_sig is not None and int(est_sig) == int(ref_sig):
                        eval_correct += 1

                pbar.update(s1 - s0)
            pbar.close()

        acc = (100.0 * eval_correct / eval_total) if eval_total > 0 else 0.0
        print(f"[ProbeKeyMetric] {self.label_tag}: {eval_correct}/{eval_total} = {acc:.2f}%")
        return acc


class MultiProbeKeyMetric:
    """
    Computes segment-level key accuracy for multiple degree labels.

    Each label_tag has its own probe checkpoint (trained on that specific
    degree).  Per epoch, each probe independently undergoes SAE encoder weight
    replacement and a full inference pass over the H5 split.  Individual
    accuracies are logged to the trainer; the maximum is returned as the
    aggregate score consumed by TopKEvalCallback for checkpoint selection.

    Args:
        probe_ckpts: Mapping from label_tag to checkpoint path.  Must contain
            at least one entry; keys must be a subset of ALL_LABEL_TAGS.
    """

    def __init__(
        self,
        probe_ckpts: dict[str, str],
        h5_path: str,
        feature_ds: str,
        split: str = "test",
        sae_idx: int = 0,
        batch_size: int = 64,
        max_segs: int = -1,
    ) -> None:
        unknown = set(probe_ckpts) - set(LABEL_TAG_TO_DEGREE)
        if unknown:
            raise ValueError(f"Unknown label_tag(s): {unknown}. Choose from {list(LABEL_TAG_TO_DEGREE)}")
        if not probe_ckpts:
            raise ValueError("probe_ckpts must contain at least one entry.")
        self.probe_ckpts = probe_ckpts
        self.h5_path = h5_path
        self.feature_ds = feature_ds
        self.split = split
        self.sae_idx = sae_idx
        self.batch_size = batch_size
        self.max_segs = max_segs
        self._probe_bases: dict[str, Optional[LitLinearProbe]] = {tag: None for tag in probe_ckpts}

    @property
    def name(self) -> str:
        return "val/probe_key_best_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _base_probe(self, tag: str) -> LitLinearProbe:
        if self._probe_bases[tag] is None:
            ckpt_path = self.probe_ckpts[tag]
            print(f"[MultiProbeKeyMetric] Loading probe ({tag}) from {ckpt_path}")
            self._probe_bases[tag] = LitLinearProbe.load_from_checkpoint(
                ckpt_path, map_location="cpu", strict=False
            )
            self._probe_bases[tag].eval()
        return self._probe_bases[tag]  # type: ignore[return-value]

    @torch.no_grad()
    def _build_replaced_probe(
        self, pl_module: pl.LightningModule, device: str, tag: str
    ) -> LitLinearProbe:
        probe = copy.deepcopy(self._base_probe(tag)).to(device)

        sae_module = pl_module.sae.sae.sae_modules[self.sae_idx]
        enc_w = sae_module.fc1.weight.data.to(device)  # (hidden, d_model)
        probe_w = probe.linear.weight.data              # (12, d_model)

        cos_sim = F.normalize(probe_w, dim=1) @ F.normalize(enc_w, dim=1).T  # (12, hidden)
        best_idx = cos_sim.argmax(dim=1)
        best_scores = cos_sim[torch.arange(12), best_idx]

        print(f"[MultiProbeKeyMetric] ({tag}) SAE NN indices: {best_idx.tolist()}")
        print(f"[MultiProbeKeyMetric] ({tag}) cosine sims:    {[round(float(v), 4) for v in best_scores.tolist()]}")

        probe.linear.weight.data = enc_w[best_idx].clone()
        return probe

    def _run_inference(
        self, probe: LitLinearProbe, degree_semitones: int, device: str, tag: str
    ) -> tuple[int, int]:
        """Full H5 pass for one probe. Returns (correct, total)."""
        correct = 0
        total = 0

        with h5py.File(self.h5_path, "r") as f:
            if self.split not in f:
                raise KeyError(f"Split '{self.split}' not found. Available: {list(f.keys())}")
            g = f[self.split]

            if self.feature_ds not in g:
                raise KeyError(
                    f"Feature '{self.feature_ds}' not in split '{self.split}'. "
                    f"Available: {list(g.keys())}"
                )

            has_labels = ("labels" in g) and ("key_frame" in g["labels"])
            ds_feat = g[self.feature_ds]
            n_total = int(ds_feat.shape[0])
            end = min(n_total, int(self.max_segs)) if self.max_segs > 0 else n_total
            bs = max(1, self.batch_size)

            pbar = tqdm(total=end, desc=f"ProbeKey({tag})", unit="seg", leave=False)

            for s0 in range(0, end, bs):
                s1 = min(end, s0 + bs)

                feat_np = np.array(ds_feat[s0:s1], copy=True)  # (B, D, T)
                feat_t = torch.from_numpy(feat_np).float().to(device).permute(0, 2, 1)  # (B, T, D)

                with torch.no_grad():
                    logits = torch.relu(probe(feat_t))               # (B, T, 12)
                    mean_probs_batch = logits.mean(dim=1).cpu().numpy()  # (B, 12)

                for j in range(s1 - s0):
                    mean_probs = mean_probs_batch[j].astype(np.float32)
                    key_frame = decode_str_array(g["labels"]["key_frame"][s0 + j]) if has_labels else None

                    ref_sig, has_change, has_ref = ref_segment_signature_from_key_frame(key_frame)
                    if has_change or not has_ref:
                        continue

                    total += 1
                    est_key = _mean_probs_to_est_key(mean_probs, degree_semitones)
                    est_sig = parse_key_signature_major_pc(est_key)
                    if est_sig is not None and ref_sig is not None and int(est_sig) == int(ref_sig):
                        correct += 1

                pbar.update(s1 - s0)
            pbar.close()

        return correct, total

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        device = str(pl_module.device)
        accs: dict[str, float] = {}

        for tag in ALL_LABEL_TAGS:
            if tag not in self.probe_ckpts:
                continue
            probe = self._build_replaced_probe(pl_module, device, tag)
            degree_semitones = LABEL_TAG_TO_DEGREE[tag]
            correct, total = self._run_inference(probe, degree_semitones, device, tag)
            acc = (100.0 * correct / total) if total > 0 else 0.0
            accs[tag] = acc
            metric_key = f"val/probe_key_{tag}_acc"
            print(f"[MultiProbeKeyMetric] {tag}: {correct}/{total} = {acc:.2f}%")
            if trainer.logger is not None:
                trainer.logger.log_metrics({metric_key: acc}, step=trainer.global_step)
            trainer.callback_metrics[metric_key] = torch.tensor(acc)

        best_tag = max(accs, key=lambda t: accs[t])
        best_acc = accs[best_tag]
        print(f"[MultiProbeKeyMetric] best={best_tag} @ {best_acc:.2f}%")
        return best_acc
