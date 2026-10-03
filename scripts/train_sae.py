#!/usr/bin/env python3
"""
Train an o-SAE (aligned multi-SAE, the proposed method) or an s-SAE (standard
Top-K SAE, ablation) on pre-extracted features.

The script writes <log-dir>/<run-name>/config.json, holding
  exp_config : model settings (foundation model / layer, number of sub-SAEs,
               shared Top-K support, pitch-shift views, decoder normalisation)
  train      : trainer settings (data, Top-K, lr, batch size, epochs, seed, ...)
and launches  python -m src.music_sae.train --config <config.json>.

Paper settings (Sec. 4.1, App. B.1): 3 sub-SAEs over adjacent semitone views,
expansion 4 (4096 latents for d = 1024), batch 32 x 30 s, lr 1e-3,
K in {32, 48, 64} (main tables: K = 48), 400 epochs, seed 42.

Checkpoint selection (App. F): validation runs every 10 epochs on the
--val-track-ids songs. With --chord-probe-ckpt the SAE features closest to the
chord-probe weights are scored (the setting used in the paper); without it,
chord rings are enumerated from the SAE itself.

Example:
    python scripts/train_sae.py --model muq --layer 2 --topk 48 \\
        --h5 sae-data/slakh2100_train_16/muq/slakh2100_train_16_muq_30s_layer_2.h5 \\
        --train-track-ids splits/slakh_train.txt --val-track-ids splits/slakh_val.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import run_module

from src.music_sae.config import build_exp_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["muq", "musicfm"])
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--sae-type", default="o-sae", choices=["o-sae", "s-sae"],
                    help="o-sae: 3 aligned sub-SAEs (default); s-sae: one standard SAE.")
    ap.add_argument("--topk", type=int, default=48, help="Top-K sparsity K. Default: 48.")
    ap.add_argument("--h5", required=True, nargs="+", help="Training feature H5 file(s).")
    ap.add_argument("--train-track-ids", required=True, help="Training track ids, one per line.")
    ap.add_argument("--val-track-ids", required=True, help="Validation track ids, one per line.")
    ap.add_argument("--chord-probe-ckpt", default=None,
                    help="Optional chord-probe checkpoint for checkpoint selection (App. F).")
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-dir", default="runs/sae")
    ap.add_argument("--run-name", default=None,
                    help="Default: slakh2100_train_16_<model>-<layer>_<n>-1_<lr>_<topk>_max "
                         "(the naming parsed by scripts/analyze/*.sh).")
    ap.add_argument("--resume", default=None, help="Resume from this checkpoint.")
    ap.add_argument("--config-only", action="store_true", help="Write config.json and exit.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    n_saes = 3 if args.sae_type == "o-sae" else 1
    run_name = args.run_name or f"slakh2100_train_16_{args.model}-{args.layer}_{n_saes}-1_{args.lr}_{args.topk}_max"

    cfg = {
        "exp_config": build_exp_config(args.model, args.layer, n_saes),
        "train": {
            "h5": [str(Path(p)) for p in args.h5],
            "split": "train",
            "featureTag": f"{args.model}_layer",
            "train_track_ids": args.train_track_ids,
            "val_track_ids": args.val_track_ids,
            "chord_probe_ckpt": args.chord_probe_ckpt,
            "probe_sae_idx": 0,
            "topk": args.topk,
            "lr": float(args.lr),
            "lr_warmup_frac": 0.0,
            "lr_decay_frac": 0.0,
            "batchSize": args.batch_size,
            "maxEpochs": args.max_epochs,
            "numWorkers": args.num_workers,
            "precision": "32",
            "val_smooth_win": 9,
            "val_majmin_mode": "seg-mean-max",
            "top_k": 6,
            "seed": args.seed,
            "logDir": args.log_dir,
            "runName": run_name,
        },
    }

    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] config -> {cfg_path}")
    if args.config_only:
        return

    extra = ["--ckpt", args.resume] if args.resume else []
    run_module("src.music_sae.train", ["--config", cfg_path, *extra])


if __name__ == "__main__":
    main()
