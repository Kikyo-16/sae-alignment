#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear-probe key inference from the 12-class relative-key (scale-degree) probe.

Counterpart of ``src.linear_probe.infer_key`` (which reads the chord probe) for
the ``key_relative`` probe.  That probe emits 12 per-frame scale-degree logits;
the degree it was trained on (tonic / forth / fifth — i.e. tonic / subdominant /
dominant) is a property of the run, not of the checkpoint, so it is supplied via
``--label-tag``.

Key estimation is the single-ring signature predictor shared with the SAE forth /
fifth / tonic rings (``src.analysis.infer_key.estimate_key_from_ring``): each
segment's logits are relu'd and time-averaged into one 12-vector, whose argmax
position ``p`` maps to the key signature ``(p - degree) % 12``.  The probe gives
no major/minor cue, so every estimate is reported in relative-major (key-
signature) space as ``<root>:maj`` — the merged-signature accuracy is the
meaningful metric.  The file layout (track_id, seg_range, ref_key, est_key,
score, 12 signature-space activations) is the 17-column single-ring format
``eval_metrics key`` reads with the default decision (no maj/min block).

Degrees (mirrors ``linear_probe.metrics._LABEL_TAG_TO_DEGREE``):
    tonic_frame -> 0   (tonic / I)
    forth_frame -> 5   (subdominant / IV)
    fifth_frame -> 7   (dominant / V)

Usage:
    python -m src.linear_probe.infer_key_relative \\
        --h5-path    sae-data/rwc/muq/rwc_muq_30s_layer_2.h5 \\
        --split      test \\
        --ckpt-path  store/.../forth_muq-2/ckpts/val_key_acc-...ckpt \\
        --label-tag  forth_frame \\
        --key-out-file store/.../rwc_test_muq-2_key_forth.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..analysis.infer_key import (
    NOTE_NAMES_12,
    decode_str_array,
    decode_str_scalar,
    estimate_key_from_ring,
    ref_key_label_for_segment,
    segment_time_range_str,
)
from .data_loader import TASK_KEY_RELATIVE
from .probe_lit import LitLinearProbe

# Scale-degree label -> semitones above the tonic (chord-root -> key-signature).
LABEL_TAG_TO_DEGREE = {"tonic_frame": 0, "forth_frame": 5, "fifth_frame": 7}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Linear-probe key inference from the relative-key (scale-degree) probe."
    )
    ap.add_argument("--h5-path",      required=True)
    ap.add_argument("--split",        default="test")
    ap.add_argument("--ckpt-path",    required=True)
    ap.add_argument("--key-out-file", required=True,
                    help="Output TSV (17-column single-ring layout).")
    ap.add_argument("--label-tag",    required=True, choices=list(LABEL_TAG_TO_DEGREE),
                    help="Scale degree the probe was trained on (sets the signature offset).")
    ap.add_argument("--feature-ds",   default=None,
                    help="Feature dataset key in the H5 (default: probe's feature_tag).")
    ap.add_argument("--device",       type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",   type=int, default=64)
    ap.add_argument("--allow-missing-labels", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    h5_path   = Path(args.h5_path).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    out_path  = Path(args.key_out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    degree = LABEL_TAG_TO_DEGREE[args.label_tag]

    model = LitLinearProbe.load_from_checkpoint(str(ckpt_path), map_location=args.device)
    model.eval()
    model.to(args.device)
    if model.hparams.task != TASK_KEY_RELATIVE:
        raise ValueError(
            f"Checkpoint task is '{model.hparams.task}', expected '{TASK_KEY_RELATIVE}'. "
            "Relative-key inference requires a scale-degree probe."
        )
    feature_ds = args.feature_ds or model.hparams.feature_tag
    print(f"[INFO] feature_ds={feature_ds}  split={args.split}  "
          f"label_tag={args.label_tag}  degree={degree}")
    print(f"[INFO] key -> {out_path}")

    # 17-column single-ring header: signature-space activations are tagged ':maj'.
    key_act_header = "\t".join([f"act_{n}:maj" for n in NOTE_NAMES_12])
    header = f"# track_id\tseg_range\tref_key\test_key\tscore\t{key_act_header}"
    zeros_str = "\t".join(["0.000000"] * 12)

    with h5py.File(h5_path, "r") as f, out_path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")

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

            x_np = np.array(ds_feat[s0:s1], copy=True)                            # (B, D, T)
            x = torch.from_numpy(x_np).float().to(args.device).permute(0, 2, 1)   # (B, T, D)
            with torch.no_grad():
                logits = model(x).detach().float().cpu().numpy()                  # (B, T, 12)

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
                ref_key = (
                    ref_key_label_for_segment(key_frame) if key_frame is not None else "N/A"
                )

                if lg.ndim != 2 or lg.shape[0] <= 1:
                    fh.write(f"{tracks[j]}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zeros_str}\n")
                    pbar.update(1)
                    continue

                # relu'd per-frame logits, time-averaged into one 12-vector, then
                # mapped to a key signature exactly like the SAE single-ring path.
                ring_mean = np.maximum(lg, 0.0).mean(axis=0).reshape(1, 12)        # (1, 12)
                est_key, est_score, key_scores_12 = estimate_key_from_ring(
                    ring_mean, list(range(12)), degree,
                )
                act = "\t".join(f"{float(v):.6f}" for v in key_scores_12.tolist())
                fh.write(f"{tracks[j]}\t{seg_range}\t{ref_key}\t{est_key}\t{est_score:.6f}\t{act}\n")
                pbar.update(1)
        pbar.close()
    print("[DONE] relative-key inference complete.")


if __name__ == "__main__":
    main()
