#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear-probe key inference from the 25-class chord probe.

Counterpart of the SAE driver ``src.analysis.step4_infer`` (key half) for the
linear probe — the chord probe already emits a [Major Chord] ring (cols 0-11)
and a [Minor Chord] ring (cols 12-23), so no rings / feature-id file are needed.
Each segment's pre-softmax logits are relu'd and time-summed into a major and a
minor 12-vector in chord-root space, then the *same* key-decision functions used
by the SAE drivers turn those two rings into a key.

The probe is a single decoder (no sub-SAE pitch-rotation voting), so only the
major/minor variants are produced — the forth-ring variants (forth /
forth_minor_tonic / *_gt) have no probe equivalent and are omitted.  To stay
byte-compatible with ``eval_metrics key``, each ring is written in the 3-sub-SAE
"vote2" layout with the probe's single decode in sub-SAE slot 0 and slots 1/2
zeroed: the eval's align-average + L1 normalisation makes the zeroed slots cancel
exactly, so ``--vote2-decision {avg,joint,joint_gt}`` reproduces the SAE rule.

Variants (output file per variant):
    major_tonic           major ring only, tonic (I)
    minor_tonic           minor ring only, tonic (i)
    major_minor_tonic     major+minor, align-avg argmax  (--vote2-decision avg)
    major_minor_joint     major+minor, joint template     (--vote2-decision joint)
    major_minor_joint_gt  joint template, gt mode         (--vote2-decision joint_gt)
    major_minor_tonic_gt  gt mode picks the ring          (--vote2-decision avg)

Usage:
    python -m src.linear_probe.infer_key \\
        --h5-path     sae-data/rwc/muq/rwc_muq_30s_layer_2.h5 \\
        --split       test \\
        --ckpt-path   store/linear_chord_new/chord_muq-2/ckpts/val_chord_acc-...ckpt \\
        --major-minor-joint-out-file    store/.../rwc_test_key_major_minor_joint.txt \\
        --major-minor-joint-gt-out-file store/.../rwc_test_key_major_minor_joint_gt.txt
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..analysis.infer_key import (
    NOTE_NAMES_12,
    decide_joint_major_minor,
    decide_maj_min_multisae,
    decode_str_array,
    decode_str_scalar,
    ref_key_label_for_segment,
    segment_time_range_str,
)
from .data_loader import TASK_CHORD
from .probe_lit import LitLinearProbe

# The probe's single decode sits in sub-SAE slot 0; slots 1/2 are zero so the
# eval's offset roll + L1 normalisation cancels them. offsets must match the
# eval's hard-coded (0, 1, 2).
VOTE_OFFSETS: Tuple[int, ...] = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Linear-probe key inference from the chord probe (vote2 layout)"
    )
    ap.add_argument("--h5-path",     required=True)
    ap.add_argument("--split",       default="test")
    ap.add_argument("--ckpt-path",   required=True)
    ap.add_argument("--feature-ds",  default=None,
                    help="Feature dataset key in the H5 (default: probe's feature_tag).")
    ap.add_argument("--device",      type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--allow-missing-labels", action="store_true")

    # One output file per variant (all optional; emit only the ones requested).
    ap.add_argument("--major-out-file",                default=None)  # major_tonic
    ap.add_argument("--minor-out-file",                default=None)  # minor_tonic
    ap.add_argument("--major-minor-tonic-out-file",    default=None)  # avg
    ap.add_argument("--major-minor-joint-out-file",    default=None)  # joint
    ap.add_argument("--major-minor-joint-gt-out-file", default=None)  # joint_gt
    ap.add_argument("--major-minor-tonic-gt-out-file", default=None)  # gt ring select
    return ap.parse_args()


def _est_for_variant(
    variant: str, maj_blocks: List[np.ndarray], min_blocks: List[np.ndarray],
    zeros: List[np.ndarray], ref_is_minor: bool,
) -> Tuple[int, int, float]:
    """(tonic_pc, is_minor, score) for one variant, matching the eval decisions."""
    if variant == "major_tonic":
        return decide_maj_min_multisae(maj_blocks, zeros, VOTE_OFFSETS, maj_degree=0)
    if variant == "minor_tonic":
        return decide_maj_min_multisae(zeros, min_blocks, VOTE_OFFSETS, min_degree=0)
    if variant == "major_minor_tonic":
        return decide_maj_min_multisae(maj_blocks, min_blocks, VOTE_OFFSETS,
                                       maj_degree=0, min_degree=0)
    if variant == "major_minor_joint":
        return decide_joint_major_minor(maj_blocks, min_blocks, VOTE_OFFSETS, gt_is_min=None)
    if variant == "major_minor_joint_gt":
        return decide_joint_major_minor(maj_blocks, min_blocks, VOTE_OFFSETS,
                                        gt_is_min=ref_is_minor)
    if variant == "major_minor_tonic_gt":
        if ref_is_minor:
            return decide_maj_min_multisae(zeros, min_blocks, VOTE_OFFSETS, min_degree=0)
        return decide_maj_min_multisae(maj_blocks, zeros, VOTE_OFFSETS, maj_degree=0)
    raise ValueError(f"Unknown variant '{variant}'")


def _store_blocks(variant: str, maj_blocks: List[np.ndarray], min_blocks: List[np.ndarray],
                  zeros: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Ring blocks to persist (maj_sae, min_sae); single-ring variants zero the other."""
    if variant == "major_tonic":
        return maj_blocks, zeros
    if variant == "minor_tonic":
        return zeros, min_blocks
    return maj_blocks, min_blocks


def main() -> None:
    args = parse_args()

    h5_path   = Path(args.h5_path).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()

    model = LitLinearProbe.load_from_checkpoint(str(ckpt_path), map_location=args.device)
    model.eval()
    model.to(args.device)
    if model.hparams.task != TASK_CHORD:
        raise ValueError(
            f"Checkpoint task is '{model.hparams.task}', expected '{TASK_CHORD}'. "
            "Key inference here reuses the chord probe's major/minor rings."
        )
    feature_ds = args.feature_ds or model.hparams.feature_tag

    # (variant, out_file_arg) for every requested output.
    VARIANTS: List[Tuple[str, Optional[str]]] = [
        ("major_tonic",          args.major_out_file),
        ("minor_tonic",          args.minor_out_file),
        ("major_minor_tonic",    args.major_minor_tonic_out_file),
        ("major_minor_joint",    args.major_minor_joint_out_file),
        ("major_minor_joint_gt", args.major_minor_joint_gt_out_file),
        ("major_minor_tonic_gt", args.major_minor_tonic_gt_out_file),
    ]
    active = [(v, Path(p).resolve()) for v, p in VARIANTS if p is not None]
    if not active:
        raise ValueError("No output file requested; pass at least one --*-out-file.")
    for _v, p in active:
        p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] feature_ds={feature_ds}  split={args.split}")
    for v, p in active:
        print(f"[INFO] {v} -> {p}")

    # vote2 header: maj ring (s0/s1/s2) then min ring (s0/s1/s2), 12 each.
    def _ring_header(ring: str) -> str:
        return "\t".join(f"{ring}_s{i}_{n}" for i in range(3) for n in NOTE_NAMES_12)
    header = (f"# track_id\tseg_range\tref_key\test_key\tscore"
              f"\t{_ring_header('majring')}\t{_ring_header('minring')}")
    zeros_str = "\t".join(["0.000000"] * 72)

    @contextlib.contextmanager
    def _open(path: Path):
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            yield fh

    with h5py.File(h5_path, "r") as f, contextlib.ExitStack() as stack:
        fhs = {v: stack.enter_context(_open(p)) for v, p in active}

        if args.split not in f:
            raise KeyError(f"Split '{args.split}' not found in {h5_path}.")
        g = f[args.split]
        if feature_ds not in g:
            raise KeyError(f"Feature dataset '{feature_ds}' not in split '{args.split}'.")
        has_key_labels = ("labels" in g) and ("key_frame" in g["labels"])
        if not has_key_labels and not args.allow_missing_labels:
            raise RuntimeError("Missing key_frame labels; pass --allow-missing-labels.")

        ds_feat = g[feature_ds]
        n_total = int(ds_feat.shape[0])
        t_feat  = int(ds_feat.shape[-1])
        bs      = max(1, args.batch_size)

        pbar = tqdm(total=n_total, desc="Infer", unit="seg")
        for s0 in range(0, n_total, bs):
            s1   = min(n_total, s0 + bs)
            idxs = list(range(s0, s1))

            x_np = np.array(ds_feat[s0:s1], copy=True)                 # (B, D, T)
            x = torch.from_numpy(x_np).float().to(args.device).permute(0, 2, 1)  # (B, T, D)
            with torch.no_grad():
                logits = model(x).detach().float().cpu().numpy()      # (B, T, 25)

            tracks: List[str] = (
                [decode_str_scalar(v) for v in g["trackId"][s0:s1]]
                if "trackId" in g else [f"seg{i:06d}" for i in idxs]
            )
            key_frames: List[Optional[np.ndarray]] = (
                [decode_str_array(g["labels"]["key_frame"][i]) for i in idxs]
                if has_key_labels else [None] * len(idxs)
            )

            for j, seg_idx in enumerate(idxs):
                lg = logits[j]
                seg_range = segment_time_range_str(g, seg_idx, t_feat)
                key_frame = key_frames[j]
                ref_key = ref_key_label_for_segment(key_frame) if key_frame is not None else "N/A"

                if lg.ndim != 2 or lg.shape[0] <= 1:
                    for v, fh in fhs.items():
                        fh.write(f"{tracks[j]}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zeros_str}\n")
                    pbar.update(1)
                    continue

                z = np.maximum(lg[:, :24], 0.0)              # (T, 24) relu'd
                maj_sum = z[:, :12].sum(axis=0)              # chord-root space
                min_sum = z[:, 12:24].sum(axis=0)
                zero12 = np.zeros(12, dtype=np.float64)
                maj_blocks = [maj_sum, zero12, zero12]
                min_blocks = [min_sum, zero12, zero12]
                zeros = [zero12, zero12, zero12]
                ref_is_minor = "min" in ref_key.lower()

                for v, fh in fhs.items():
                    tonic, is_min, score = _est_for_variant(
                        v, maj_blocks, min_blocks, zeros, ref_is_minor
                    )
                    est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                    s_maj, s_min = _store_blocks(v, maj_blocks, min_blocks, zeros)
                    act = "\t".join(
                        [f"{float(val):.6f}" for blk in s_maj for val in blk.tolist()]
                        + [f"{float(val):.6f}" for blk in s_min for val in blk.tolist()]
                    )
                    fh.write(f"{tracks[j]}\t{seg_range}\t{ref_key}\t{est_key}\t{score:.6f}\t{act}\n")
                pbar.update(1)
        pbar.close()
    print("[DONE] key inference complete.")


if __name__ == "__main__":
    main()
