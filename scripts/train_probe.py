#!/usr/bin/env python3
"""
Train the supervised linear-probe baselines (App. C) on pre-extracted features.

  --task chord : 25-class chord probe (12 major + 12 minor + N), cross-entropy.
                 Also used as the semantic reference for SAE checkpoint
                 selection (scripts/train_sae.py --chord-probe-ckpt) and for
                 retrieving s-SAE features (scripts/find_orbits.py).
  --task key   : 12-output scale-degree probe (BCE) for the tonic, forth
                 (subdominant) or fifth (dominant) function, set by --degree.

Settings follow the paper: lr 1e-3, batch 32, 80 epochs, validation every
2 epochs on --val-track-ids, best 3 checkpoints kept.

Example:
    python scripts/train_probe.py --task chord --model muq --layer 2 \\
        --h5 sae-data/slakh2100_train_16/muq/slakh2100_train_16_muq_30s_layer_2.h5 \\
        --train-track-ids splits/slakh_train.txt --val-track-ids splits/slakh_val.txt
"""
from __future__ import annotations

import argparse

from _common import run_module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["chord", "key"])
    ap.add_argument("--degree", default="forth", choices=["forth", "fifth", "tonic"],
                    help="--task key: harmonic function to probe. Default: forth.")
    ap.add_argument("--model", required=True, choices=["muq", "musicfm"])
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--h5", required=True, nargs="+", help="Training feature H5 file(s).")
    ap.add_argument("--train-track-ids", required=True)
    ap.add_argument("--val-track-ids", required=True)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--log-dir", default=None,
                    help="Default: runs/probes/chord or runs/probes/key_<degree>.")
    ap.add_argument("--run-name", default=None,
                    help="Default: chord_<model>-<layer> or key_<degree>_<model>-<layer>.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    mode = f"{args.model}-{args.layer}"
    common = [
        "--h5", *args.h5,
        "--mode", mode,
        "--train-track-ids", args.train_track_ids,
        "--val-track-ids", args.val_track_ids,
        "--batchSize", args.batch_size,
        "--lr", args.lr,
        "--maxEpochs", args.max_epochs,
        "--top-k", 3,
    ]
    if args.task == "chord":
        cmd = [
            "--task", "chord",
            "--logDir", args.log_dir or "runs/probes/chord",
            "--runName", args.run_name or f"chord_{mode}",
            "--val-smooth-win", 9,
        ] + common
    else:
        cmd = [
            "--task", "key_relative",
            "--labelTag", f"{args.degree}_frame",
            "--logDir", args.log_dir or f"runs/probes/key_{args.degree}",
            "--runName", args.run_name or f"key_{args.degree}_{mode}",
        ] + common
    run_module("src.linear_probe.train", cmd)


if __name__ == "__main__":
    main()
