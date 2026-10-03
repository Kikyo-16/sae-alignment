#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAE / linear-probe runners for audio inference.

SAE activations are computed exactly as in ``analysis.step4_infer``:
float16 features -> ``model_factory`` -> ``standardize_to_BTD_batch`` ->
``MultiSAE.inference(x, idx, topk_wide=0)``. Single-ring tasks and chord
recognition use sub-SAE 0; key-chord uses sub-SAEs 0/1/2 of an o-SAE.

Probe outputs are computed exactly as in ``linear_probe.infer_key*`` /
``infer_chord_gt_boundary``: float16 features -> (B, T, D) -> ``model(x)``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..analysis.boundary_utils import boundary_segments_from_feature
from ..analysis.find_chord_rings import load_sae_and_factory, standardize_to_BTD_batch
from ..analysis.infer_key import _segments_from_boundaries, predict_chord_segments, smooth_cols
from ..analysis.step4_infer import _parse_feature_groups
from ..analysis.step4_infer_gt_boundary import _gt_chord_boundaries, _gt_silence_score
from ..linear_probe.data_loader import TASK_CHORD as PROBE_TASK_CHORD
from ..linear_probe.data_loader import TASK_KEY_RELATIVE as PROBE_TASK_KEY_RELATIVE
from ..linear_probe.probe_lit import LitLinearProbe
from .audio import AudioSegment, stack_features
from .decode import (
    CHORD_DEFAULTS,
    DEGREE_BY_NAME,
    TASK_CHORD,
    TASK_KEY_CHORD,
    TASK_KEY_SIGNATURE,
    ChordCollector,
    KeyChordDecoder,
    KeySignatureDecoder,
    lab_frames,
)

VOTE_SAE_IDXS = (0, 1, 2)


def parse_data_tag(data_tag: str) -> Tuple[str, int]:
    """'muq_layer_2' -> ('muq', 2)."""
    m = re.match(r"^(.+)_layer_(\d+)$", str(data_tag))
    if not m:
        raise ValueError(f"Cannot parse foundation model / layer from data_tag={data_tag!r}")
    return m.group(1), int(m.group(2))


def _batches(n: int, bs: int):
    for s0 in range(0, n, max(1, bs)):
        yield s0, min(n, s0 + max(1, bs))


class SAERunner:
    def __init__(self, ckpt_path: Path, feature_ids_path: Path, device: str, batch_size: int = 32) -> None:
        lit, sae, model_factory = load_sae_and_factory(Path(ckpt_path), device)
        self.module = sae.sae
        self.model_factory = model_factory
        self.use_tag = str(lit.data_tag)
        self.n_saes = int(getattr(self.module, "n_saes", 0))
        self.device = device
        self.batch_size = int(batch_size)
        self.foundation_model, self.layer = parse_data_tag(self.use_tag)
        self.groups = _parse_feature_groups(Path(feature_ids_path))
        print(f"[INFO] SAE data_tag={self.use_tag}  n_saes={self.n_saes}  "
              f"feature groups={sorted(self.groups)}")

    def ring(self, name: str) -> List[int]:
        ids = self.groups.get(name.lower(), [])
        if len(ids) < 12:
            raise RuntimeError(f"[{name}] with 12 ids not found in feature_ids.txt "
                               f"(found groups: {sorted(self.groups)}).")
        return [int(i) for i in ids[:12]]

    @torch.no_grad()
    def encode(self, x_np: np.ndarray, sae_idxs: Sequence[int]) -> Dict[int, np.ndarray]:
        """x_np: [B, D, T] float16 -> {sub-SAE idx: [B, T, M] sparse codes}."""
        t_feat = int(x_np.shape[-1])
        x = torch.from_numpy(np.array(x_np, copy=True)).float().to(self.device)
        feat_batch = self.model_factory({self.use_tag: x}, t=0)
        x_sae = standardize_to_BTD_batch(feat_batch[self.use_tag], t_candidates=[t_feat, int(x_np.shape[-1])])
        return {
            si: self.module.inference(x_sae, idx=si, topk_wide=0).detach().float().cpu().numpy()
            for si in sae_idxs
        }

    def run(
        self,
        segments: List[AudioSegment],
        tasks: Sequence[str],
        key_ring: str = "forth",
        chord_lab: Optional[List[Tuple[float, float, str]]] = None,
        boundary_rank: int = 1,
        chord_opts: Optional[dict] = None,
    ) -> Dict[str, object]:
        opts = dict(CHORD_DEFAULTS)
        opts.update(chord_opts or {})
        out: Dict[str, object] = {}

        sig_cols: List[int] = []
        if TASK_KEY_SIGNATURE in tasks:
            sig_cols = self.ring({"forth": "Forth", "fifth": "Fifth", "tonic": "Tonic"}[key_ring])
            out[TASK_KEY_SIGNATURE] = KeySignatureDecoder(DEGREE_BY_NAME[key_ring])
        maj_cols: List[int] = []
        min_cols: List[int] = []
        if TASK_KEY_CHORD in tasks or TASK_CHORD in tasks:
            maj_cols, min_cols = self.ring("Major Chord"), self.ring("Minor Chord")
        if TASK_KEY_CHORD in tasks:
            if not (self.n_saes == 1 or self.n_saes >= 3):
                raise RuntimeError(f"key-chord needs an o-SAE with >= 3 sub-SAEs or an s-SAE; n_saes={self.n_saes}")
            out[TASK_KEY_CHORD] = KeyChordDecoder()
        silence_cols: List[int] = []
        boundary_id: Optional[int] = None
        if TASK_CHORD in tasks:
            silence_cols = [int(i) for i in self.groups.get("silence", [])]
            if chord_lab is None:
                bnd = self.groups.get("chord boundary", [])
                if len(bnd) < boundary_rank:
                    raise RuntimeError("[Chord Boundary] missing in feature_ids.txt; estimated-boundary "
                                       "chord recognition needs it (or pass --chord-lab).")
                boundary_id = int(bnd[boundary_rank - 1])
            out[TASK_CHORD] = ChordCollector()

        sae_idxs = [0]
        if TASK_KEY_CHORD in tasks and self.n_saes >= 3:
            sae_idxs = list(VOTE_SAE_IDXS)

        feats = stack_features(segments)
        qual_mode = str(opts["majmin_mode"]).replace("-", "_")
        for s0, s1 in _batches(len(segments), self.batch_size):
            z_by_idx = self.encode(feats[s0:s1], sae_idxs)
            for j, seg in enumerate(segments[s0:s1]):
                z = z_by_idx[0][j]                      # [T, M], sub-SAE 0
                t_z, d_z = int(z.shape[0]), int(z.shape[1])
                keep = seg.keep_from_frame()

                if TASK_KEY_SIGNATURE in out:
                    cols = [i for i in sig_cols if 0 <= i < d_z]
                    out[TASK_KEY_SIGNATURE].add(seg, z[keep:][:, cols])

                if TASK_KEY_CHORD in out:
                    if self.n_saes >= 3:
                        maj_blocks = [z_by_idx[si][j][keep:, maj_cols].sum(axis=0) for si in VOTE_SAE_IDXS]
                        min_blocks = [z_by_idx[si][j][keep:, min_cols].sum(axis=0) for si in VOTE_SAE_IDXS]
                    else:  # single sub-SAE: blocks 1/2 zeroed (step4 joint1 layout)
                        maj_blocks = [z[keep:, maj_cols].sum(axis=0), np.zeros(12), np.zeros(12)]
                        min_blocks = [z[keep:, min_cols].sum(axis=0), np.zeros(12), np.zeros(12)]
                    out[TASK_KEY_CHORD].add(seg, maj_blocks, min_blocks)

                if TASK_CHORD in out:
                    major = [i for i in maj_cols if 0 <= i < d_z]
                    minor = [i for i in min_cols if 0 <= i < d_z]
                    if chord_lab is not None:      # Oracle Bd: reference boundaries + N/X silence
                        frames = lab_frames(chord_lab, seg)[:t_z]
                        chord_segs = _segments_from_boundaries(_gt_chord_boundaries(frames, t_z), t_z) or [(0, t_z)]
                        silence_sc_t = _gt_silence_score(frames, t_z)
                    else:                          # Est. Bd: SAE boundary + silence features
                        frames = None
                        sil = [i for i in silence_cols if 0 <= i < d_z]
                        silence_sc_t = smooth_cols(z[:, sil], opts["smooth_win"]).min(axis=1) if sil else None
                        chord_segs = boundary_segments_from_feature(
                            z=z, feature_id=boundary_id, t_z=t_z,
                            smooth_win=opts["smooth_win"],
                            peak_tol=float(opts["boundary_peak_tol"]),
                            peak_drop=float(opts["boundary_peak_drop"]),
                            min_gap=int(opts["boundary_min_gap"]),
                            score_floor=float(opts["boundary_score_floor"]),
                            max_labels=int(opts["boundary_max_labels"]),
                        )
                    rows = predict_chord_segments(
                        z, major, minor, chord_segs,
                        silence_sc_t=silence_sc_t, chord_frame_seg=frames, t_z=t_z,
                        smooth_win=opts["smooth_win"], qual_mode=qual_mode,
                        conf_thr=opts["majmin_conf_thr"], alpha=opts["majmin_alpha"],
                    )
                    out[TASK_CHORD].add(seg, rows)
        return out


class ProbeRunner:
    MAJOR_COLS = list(range(12))
    MINOR_COLS = list(range(12, 24))

    def __init__(self, ckpt_path: Path, device: str, batch_size: int = 64) -> None:
        self.model = LitLinearProbe.load_from_checkpoint(str(ckpt_path), map_location=device)
        self.model.eval()
        self.model.to(device)
        self.task = str(self.model.hparams.task)
        self.device = device
        self.batch_size = int(batch_size)
        print(f"[INFO] probe task={self.task}  feature_tag={self.model.hparams.feature_tag}")

    @torch.no_grad()
    def logits(self, x_np: np.ndarray) -> np.ndarray:
        """x_np: [B, D, T] float16 -> [B, T, C] logits."""
        x = torch.from_numpy(np.array(x_np, copy=True)).float().to(self.device).permute(0, 2, 1)
        return self.model(x).detach().float().cpu().numpy()

    def run(
        self,
        segments: List[AudioSegment],
        task: str,
        degree: str = "forth",
        chord_lab: Optional[List[Tuple[float, float, str]]] = None,
        chord_opts: Optional[dict] = None,
    ) -> Dict[str, object]:
        opts = dict(CHORD_DEFAULTS)
        opts.update(chord_opts or {})
        if task == TASK_KEY_SIGNATURE and self.task != PROBE_TASK_KEY_RELATIVE:
            raise ValueError(f"key-signature needs a '{PROBE_TASK_KEY_RELATIVE}' probe, got '{self.task}'.")
        if task in (TASK_KEY_CHORD, TASK_CHORD) and self.task != PROBE_TASK_CHORD:
            raise ValueError(f"{task} needs a '{PROBE_TASK_CHORD}' probe, got '{self.task}'.")
        if task == TASK_CHORD and chord_lab is None:
            raise ValueError("Probe chord recognition is defined with reference boundaries "
                             "only (Oracle Bd); pass --chord-lab.")

        if task == TASK_KEY_SIGNATURE:
            dec = KeySignatureDecoder(DEGREE_BY_NAME[degree])
        elif task == TASK_KEY_CHORD:
            dec = KeyChordDecoder()
        else:
            dec = ChordCollector()

        feats = stack_features(segments)
        qual_mode = str(opts["majmin_mode"]).replace("-", "_")
        zero12 = np.zeros(12, dtype=np.float64)
        for s0, s1 in _batches(len(segments), self.batch_size):
            logits = self.logits(feats[s0:s1])
            if task == TASK_CHORD:
                probs_all = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            for j, seg in enumerate(segments[s0:s1]):
                lg = logits[j]
                keep = seg.keep_from_frame()
                if task == TASK_KEY_SIGNATURE:
                    # relu'd logits, time-averaged (linear_probe.infer_key_relative)
                    ring_mean = np.maximum(lg[keep:], 0.0).mean(axis=0).reshape(1, 12)
                    dec.add(seg, ring_mean)
                elif task == TASK_KEY_CHORD:
                    # relu'd major/minor logits, time-summed (linear_probe.infer_key)
                    zr = np.maximum(lg[keep:, :24], 0.0)
                    dec.add(seg, [zr[:, :12].sum(axis=0), zero12, zero12],
                            [zr[:, 12:24].sum(axis=0), zero12, zero12])
                else:
                    # softmax probabilities + reference boundaries (linear_probe.infer_chord_gt_boundary)
                    z = probs_all[j, :, :24]
                    t_z = int(z.shape[0])
                    frames = lab_frames(chord_lab, seg)[:t_z]
                    chord_segs = _segments_from_boundaries(_gt_chord_boundaries(frames, t_z), t_z) or [(0, t_z)]
                    silence_sc_t = smooth_cols(probs_all[j, :t_z, 24][:, None], opts["smooth_win"])[:, 0]
                    rows = predict_chord_segments(
                        z, self.MAJOR_COLS, self.MINOR_COLS, chord_segs,
                        silence_sc_t=silence_sc_t, chord_frame_seg=frames, t_z=t_z,
                        smooth_win=opts["smooth_win"], qual_mode=qual_mode,
                        conf_thr=opts["majmin_conf_thr"], alpha=opts["majmin_alpha"],
                    )
                    dec.add(seg, rows)
        return {task: dec}
