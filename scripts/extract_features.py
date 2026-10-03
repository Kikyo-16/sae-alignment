#!/usr/bin/env python3
"""
Extract frozen foundation-model features (MuQ / MusicFM) into an H5 file.

Wraps src/feature_extraction/process_datasets.py with the settings used in the
paper (mono mix, 16 kHz input, 30 s segments, float16 storage):

  --purpose train   SAE / probe training data (Slakh2100, App. B.1):
                    10 s hop, pitch-shifted views at -2..+2 semitones,
                    silent segments removed, optional track-id allowlists.
  --purpose eval    evaluation / anchor data (App. B.2-B.3): non-overlapping
                    30 s segments, no augmentation; with --label-root the
                    frame-level labels are stored and required.

Supported --dataset values: slakh, pop909, rwc, gtzan, giantsteps_key, fmakv2,
sae_diagnostic (see src/feature_extraction/datasets/).

Examples:
    # training H5 (both track-id lists go into one file; train_sae.py splits them)
    python scripts/extract_features.py --purpose train --model muq --layer 2 \\
        --dataset slakh --root dataset/slakh2100_flac_redux --label-root dataset/slakh2100_flac_redux \\
        --split train --track-id-files splits/slakh_train.txt splits/slakh_val.txt \\
        --out-h5 sae-data/slakh2100_train_16/muq/slakh2100_train_16_muq_30s_layer_2.h5

    # anchor / evaluation H5
    python scripts/extract_features.py --purpose eval --model muq --layer 2 \\
        --dataset pop909 --root dataset/pop909/pop909_audio --label-root dataset/pop909/POP909 \\
        --split test --out-h5 sae-data/pop909-test/muq/pop909-test_muq_30s_layer_2.h5
"""
from __future__ import annotations

import argparse

from _common import run_module

MODEL_IDS = {"muq": "OpenMuQ/MuQ-large-msd-iter", "musicfm": "tky823/MusicFM"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--purpose", required=True, choices=["train", "eval"])
    ap.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    ap.add_argument("--layer", required=True, type=int, help="Hidden-state layer index.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True, help="Dataset audio root.")
    ap.add_argument("--label-root", default=None, help="Annotation root (needed for training and anchors).")
    ap.add_argument("--split", required=True, help="Dataset split folder, e.g. train / test.")
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--track-id-files", nargs="+", default=None,
                    help="Optional allowlist(s), one track id per line.")
    ap.add_argument("--model-id", default=None, help="Override the Hugging Face model id.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "--dataset", args.dataset,
        "--root", args.root,
        "--split", args.split,
        "--out-h5", args.out_h5,
        "--rep", f"{args.model}_layer",
        f"--{args.model}-layer", args.layer,
        f"--{args.model}-model-id", args.model_id or MODEL_IDS[args.model],
        "--mode", "mix",
        "--duration", "30",
        "--dtype", "float16",
        "--compression", "lzf",
    ]
    if args.label_root:
        cmd += ["--label-root", args.label_root]
    if args.purpose == "train":
        cmd += ["--hop", "10"]
    else:
        cmd += ["--inference"]
        if args.label_root:
            cmd += ["--require-labels"]
    if args.track_id_files:
        cmd += ["--track-id-files", *args.track_id_files]
    run_module("src.feature_extraction.process_datasets", cmd)


if __name__ == "__main__":
    main()
