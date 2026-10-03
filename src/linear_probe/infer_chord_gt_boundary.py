#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear-probe chord inference with reference chord boundaries.

Offline counterpart of the SAE chord driver, producing a chord output file with
the identical schema (``# track_id\tseg_range\tref_chord\test_chord``) so it
can be scored by ``src.analysis.eval_metrics chord`` exactly like the SAE
results.

The 25-class probe outputs 12 major chords, 12 minor chords, and no chord.
Reference annotations provide chord boundaries only. Chord root, major/minor
quality, and no-chord are all inferred from the probe outputs.

Usage:
    python -m src.linear_probe.infer_chord_gt_boundary \\
        --h5-path        sae-data/pop909/muq/pop909_muq_30s_layer_2.h5 \\
        --split          train \\
        --ckpt-path      store/linear_chord_new/chord_muq-2/ckpts/val_chord_acc-...ckpt \\
        --chord-out-file store/.../pop909_train_chord_gtbnd.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

from ..analysis.find_chord_rings import count_songs, iter_songs
from ..analysis.infer_key import (
    _segments_from_boundaries,
    predict_chord_segments,
    smooth_cols,
)
from ..analysis.step4_infer_gt_boundary import _gt_chord_boundaries
from .data_loader import TASK_CHORD
from .probe_lit import LitLinearProbe

# The 25-class chord probe lays out its logits as 12 major + 12 minor + N.
MAJOR_COLS = list(range(12))
MINOR_COLS = list(range(12, 24))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Linear-probe chord inference with ground-truth boundaries"
    )
    ap.add_argument("--h5-path",        required=True)
    ap.add_argument("--split",          default="test")
    ap.add_argument("--ckpt-path",      required=True)
    ap.add_argument("--chord-out-file", required=True)
    ap.add_argument("--feature-ds",     default=None,
                    help="Feature dataset key in the H5 (default: probe's feature_tag).")
    ap.add_argument("--device",         type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",     type=int, default=64)
    ap.add_argument("--max-songs",      type=int, default=None)
    ap.add_argument("--smooth-win",     type=int, default=9,
                    help="Moving-average window on chord activations (<=1 disables).")
    ap.add_argument("--majmin-mode",    default="seg-mean-max",
                    choices=["sum-peak", "seg-mean-max", "seg-mean-mean", "raw-tmpl"],
                    help="Per-segment chord decision (matches the SAE drivers).")
    ap.add_argument("--majmin-conf-thr", type=float, default=0.18,
                    help="raw-tmpl: top-2 raw-margin for the template fallback.")
    ap.add_argument("--majmin-alpha",    type=float, default=0.4,
                    help="raw-tmpl: weight of the normalized triad-template score.")
    return ap.parse_args()


@torch.no_grad()
def _segment_logits(model: LitLinearProbe, feat_np: np.ndarray,
                    device: str, batch_size: int) -> np.ndarray:
    """Run the probe over one song's segments. feat_np: (n_segs, D, T).

    Returns logits (n_segs, T, num_classes) as a numpy array. Mirrors
    ``linear_probe.metrics._segment_logits`` so the offline path matches training.
    """
    outs: List[torch.Tensor] = []
    n = int(feat_np.shape[0])
    for s0 in range(0, n, batch_size):
        x = torch.from_numpy(feat_np[s0:s0 + batch_size]).float().to(device)  # (B, D, T)
        x = x.permute(0, 2, 1)                                                # (B, T, D)
        outs.append(model(x).detach().float().cpu())
    return torch.cat(outs, dim=0).numpy()


def main() -> None:
    args = parse_args()

    h5_path   = Path(args.h5_path).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    out_file  = Path(args.chord_out_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model = LitLinearProbe.load_from_checkpoint(str(ckpt_path), map_location=args.device)
    model.eval()
    model.to(args.device)
    if model.hparams.task != TASK_CHORD:
        raise ValueError(
            f"Checkpoint task is '{model.hparams.task}', expected '{TASK_CHORD}'. "
            "This driver only handles the chord probe."
        )
    feature_ds = args.feature_ds or model.hparams.feature_tag
    qual_mode = args.majmin_mode.replace("-", "_")
    print(f"[INFO] feature_ds={feature_ds}  split={args.split}  smooth_win={args.smooth_win}  "
          f"majmin_mode={qual_mode}")
    print(
        "[INFO] Reference boundaries only; root, quality, and N are "
        "predicted by the 25-class probe"
    )

    n_rows = 0
    with out_file.open("w", encoding="utf-8") as f_chord:
        f_chord.write("# track_id\tseg_range\tref_chord\test_chord\n")
        n_songs_total = count_songs(h5_path, args.split, feature_ds,
                                    max_songs=args.max_songs, track_ids=None, orig_only=True)
        songs = iter_songs(h5_path, args.split, feature_ds,
                           max_songs=args.max_songs, track_ids=None, orig_only=True)
        for tid, feat_np, chord_2d, _key_2d in tqdm(
            songs, total=n_songs_total, desc="Infer", unit="song"
        ):
            if chord_2d is None:
                continue
            logits = _segment_logits(model, feat_np, args.device, args.batch_size)
            if logits.ndim != 3 or logits.shape[-1] != 25:
                raise RuntimeError(
                    f"Expected probe logits with shape (S, T, 25), got {logits.shape}."
                )

            # The probe is trained with 25-way cross-entropy. Use the joint class
            # probabilities as comparable evidence for major, minor, and N.
            probs_all = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            z_all = probs_all[..., :24]       # 12 major + 12 minor
            silence_all = probs_all[..., 24]  # predicted no-chord probability

            for si in range(z_all.shape[0]):
                z = z_all[si]
                if z.ndim != 2 or z.shape[0] <= 1:
                    continue
                t_z = int(z.shape[0])
                chord_frame_seg = chord_2d[si]
                if chord_frame_seg is None:
                    raise RuntimeError(
                        f"Missing chord_frame for track={tid!r}, segment={si}."
                    )
                chord_frame_seg = chord_frame_seg[:t_z]

                # Reference annotations provide boundaries only.
                boundaries = _gt_chord_boundaries(chord_frame_seg, t_z)
                chord_segs = _segments_from_boundaries(boundaries, t_z) or [(0, t_z)]

                # Predict no-chord from the 25th probe output. The shared chord
                # decoder smooths major/minor evidence internally, so smooth N
                # with the same window before the segment-level decision.
                silence_sc_t = smooth_cols(
                    silence_all[si, :t_z, None],
                    args.smooth_win,
                )[:, 0]

                rows = predict_chord_segments(
                    z, MAJOR_COLS, MINOR_COLS, chord_segs,
                    silence_sc_t=silence_sc_t, chord_frame_seg=chord_frame_seg,
                    track=tid, t_z=t_z, f_chord=f_chord,
                    smooth_win=args.smooth_win, qual_mode=qual_mode,
                    conf_thr=args.majmin_conf_thr, alpha=args.majmin_alpha,
                )
                n_rows += len(rows)

    print(f"[DONE] wrote {n_rows} segment rows -> {out_file}")


if __name__ == "__main__":
    main()
