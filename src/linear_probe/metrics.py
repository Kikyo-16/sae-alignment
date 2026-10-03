#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation metrics for the linear probe, mirroring the music_sae ValMetric /
TopKEvalCallback design.  Each metric reads the eval H5 song-by-song
(``orig_only=True`` skips pitch-shift aug copies) and scores the live probe:

    KeyDegreeMetric  (task=key_relative) – segment relative-major key accuracy
    ChordProbeMetric (task=chord)        – GT-boundary chord WCSR (root + majmin)
    KeyStrictMetric  (task=key_strict)   – strict 24-class key acc + MIREX score

Validation songs are defined by a track-id file so the probe and the SAE share
the same validation set.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch
import pytorch_lightning as pl
from tqdm.auto import tqdm

from ..analysis.find_chord_rings import iter_songs
from ..analysis.infer_key import ref_segment_signature_from_key_frame
from ..analysis.eval_metrics import _mirex_weighted_score
from ..music_sae.chord_ring_metric import _chord_rows_for_segment
from ..evaluation.evaluate_chord import ChordRow, evaluate
from .data_loader import _key_str_to_int24_global, KEY_INVALID

# Maj-side ring degree per scale-degree label (chord-root → key-tonic offset).
_LABEL_TAG_TO_DEGREE = {"tonic_frame": 0, "forth_frame": 5, "fifth_frame": 7}


def _read_track_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            tid = line.strip()
            if tid:
                ids.add(tid)
    return ids


@torch.no_grad()
def _segment_logits(pl_module: pl.LightningModule, feat_np: np.ndarray,
                    device: str, batch_size: int) -> torch.Tensor:
    """Run the probe over one song's segments.  feat_np: (n_segs, D, T).

    Returns logits (n_segs, T, num_classes) on CPU.
    """
    outs: List[torch.Tensor] = []
    n = int(feat_np.shape[0])
    for s0 in range(0, n, batch_size):
        x = torch.from_numpy(feat_np[s0:s0 + batch_size]).float().to(device)  # (B, D, T)
        x = x.permute(0, 2, 1)                                                # (B, T, D)
        outs.append(pl_module(x).detach().float().cpu())
    return torch.cat(outs, dim=0)


class KeyDegreeMetric:
    """Segment relative-major key accuracy for the scale-degree probe."""

    def __init__(self, h5_path: str, feature_ds: str, label_tag: str,
                 val_track_id_file: str, split: str = "train",
                 batch_size: int = 64, max_songs: Optional[int] = None) -> None:
        if label_tag not in _LABEL_TAG_TO_DEGREE:
            raise ValueError(f"Unknown label_tag '{label_tag}'.")
        self.h5_path = h5_path
        self.feature_ds = feature_ds
        self.degree = _LABEL_TAG_TO_DEGREE[label_tag]
        self.track_ids = _read_track_ids(val_track_id_file)
        self.split = split
        self.batch_size = batch_size
        self.max_songs = max_songs

    @property
    def name(self) -> str:
        return "val/key_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        device = str(pl_module.device)
        correct = total = 0
        songs = iter_songs(Path(self.h5_path), self.split, self.feature_ds,
                           max_songs=self.max_songs, track_ids=self.track_ids, orig_only=True)
        for _tid, feat_np, _chord_2d, key_2d in tqdm(
            songs, total=len(self.track_ids), desc="KeyDegree", unit="song", leave=False
        ):
            if key_2d is None:
                continue
            logits = _segment_logits(pl_module, feat_np, device, self.batch_size)  # (S, T, 12)
            mean_probs = torch.relu(logits).mean(dim=1).numpy()                    # (S, 12)
            for si in range(mean_probs.shape[0]):
                ref_sig, has_change, has_ref = ref_segment_signature_from_key_frame(key_2d[si])
                if has_change or not has_ref:
                    continue
                pred_pc = int(np.argmax(mean_probs[si]))
                pred_sig = (pred_pc - self.degree) % 12
                total += 1
                if pred_sig == int(ref_sig):
                    correct += 1
        acc = (100.0 * correct / total) if total > 0 else 0.0
        print(f"[KeyDegreeMetric] {correct}/{total} = {acc:.2f}%")
        return acc


class ChordProbeMetric:
    """GT-boundary chord WCSR (root + majmin) from the 25-class chord probe.

    The probe's 24 chord logits are relu'd and treated as ring activations with
    major cols 0-11 and minor cols 12-23, then scored exactly like the music_sae
    ChordRingMetric (GT boundaries / GT silence, evaluate() WCSR).
    """

    MAJOR_COLS = list(range(12))
    MINOR_COLS = list(range(12, 24))

    def __init__(self, h5_path: str, feature_ds: str, val_track_id_file: str,
                 split: str = "train", batch_size: int = 64,
                 smooth_win: int = 9, qual_mode: str = "sum_peak",
                 max_songs: Optional[int] = None) -> None:
        self.h5_path = h5_path
        self.feature_ds = feature_ds
        self.track_ids = _read_track_ids(val_track_id_file)
        self.split = split
        self.batch_size = batch_size
        self.smooth_win = smooth_win
        self.qual_mode = qual_mode
        self.max_songs = max_songs

    @property
    def name(self) -> str:
        return "val/chord_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        device = str(pl_module.device)
        correct = {"root": 0.0, "majmin": 0.0}
        total = {"root": 0.0, "majmin": 0.0}
        songs = iter_songs(Path(self.h5_path), self.split, self.feature_ds,
                           max_songs=self.max_songs, track_ids=self.track_ids, orig_only=True)
        for tid, feat_np, chord_2d, _key_2d in tqdm(
            songs, total=len(self.track_ids), desc="ChordProbe", unit="song", leave=False
        ):
            logits = _segment_logits(pl_module, feat_np, device, self.batch_size)  # (S, T, 25)
            z_all = torch.relu(logits[..., :24]).numpy()                          # (S, T, 24)
            rows: List[ChordRow] = []
            for si in range(z_all.shape[0]):
                rows.extend(_chord_rows_for_segment(
                    tid, z_all[si], chord_2d[si], self.MAJOR_COLS, self.MINOR_COLS,
                    self.smooth_win, self.qual_mode,
                ))
            if not rows:
                continue
            c, t, _s, _tc, _tt = evaluate(rows)
            for v in ("root", "majmin"):
                correct[v] += c[v]
                total[v] += t[v]

        root = (100.0 * correct["root"] / total["root"]) if total["root"] > 0 else 0.0
        majmin = (100.0 * correct["majmin"] / total["majmin"]) if total["majmin"] > 0 else 0.0
        score = 0.5 * (root + majmin)
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"val/chord_root": root, "val/chord_majmin": majmin}, step=trainer.global_step
            )
        trainer.callback_metrics["val/chord_root"] = torch.tensor(root)
        trainer.callback_metrics["val/chord_majmin"] = torch.tensor(majmin)
        print(f"[ChordProbeMetric] root={root:.2f}%  majmin={majmin:.2f}%  avg={score:.2f}%")
        return score


class KeyStrictMetric:
    """Strict 24-class key accuracy + MIREX weighted score (drives checkpointing).

    The probe's per-frame 24-class logits are mean-pooled over time to a single
    per-segment vector; argmax gives (root, mode).  References come from the
    segment's key_frame (within-segment key changes are skipped).
    """

    def __init__(self, h5_path: str, feature_ds: str, val_track_id_file: str,
                 split: str = "train", batch_size: int = 64,
                 max_songs: Optional[int] = None) -> None:
        self.h5_path = h5_path
        self.feature_ds = feature_ds
        self.track_ids = _read_track_ids(val_track_id_file)
        self.split = split
        self.batch_size = batch_size
        self.max_songs = max_songs

    @property
    def name(self) -> str:
        return "val/key_mirex_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        device = str(pl_module.device)
        strict_correct = 0
        mirex_sum = 0.0
        total = 0
        songs = iter_songs(Path(self.h5_path), self.split, self.feature_ds,
                           max_songs=self.max_songs, track_ids=self.track_ids, orig_only=True)
        for _tid, feat_np, _chord_2d, key_2d in tqdm(
            songs, total=len(self.track_ids), desc="KeyStrict", unit="song", leave=False
        ):
            if key_2d is None:
                continue
            logits = _segment_logits(pl_module, feat_np, device, self.batch_size)  # (S, T, 24)
            seg_vec = logits.mean(dim=1).numpy()                                   # (S, 24)
            for si in range(seg_vec.shape[0]):
                ref = _key_str_to_int24_global(np.array(key_2d[si]))
                if ref == KEY_INVALID:
                    continue
                pred = int(np.argmax(seg_vec[si]))
                ref_t = (ref % 12, 1 if ref >= 12 else 0)
                pred_t = (pred % 12, 1 if pred >= 12 else 0)
                total += 1
                if pred == ref:
                    strict_correct += 1
                mirex_sum += _mirex_weighted_score(ref_t, pred_t)

        strict = (100.0 * strict_correct / total) if total > 0 else 0.0
        mirex = (100.0 * mirex_sum / total) if total > 0 else 0.0
        if trainer.logger is not None:
            trainer.logger.log_metrics({"val/key_strict_acc": strict}, step=trainer.global_step)
        trainer.callback_metrics["val/key_strict_acc"] = torch.tensor(strict)
        print(f"[KeyStrictMetric] strict={strict:.2f}%  mirex={mirex:.2f}%  ({total} segs)")
        return mirex
