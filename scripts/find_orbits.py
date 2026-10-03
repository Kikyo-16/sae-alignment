#!/usr/bin/env python3
"""
Recover orbits (rings / sequences / fixed points) from a trained o-SAE checkpoint.

Only the decoder dictionaries are used; no data or labels are needed.

  default   successor-map recovery of Sec. 3.3
            (src/orbits_recovery/obits_recovery.py, threshold --threshold)
  --refine  sparse directed-graph recovery of App. G.2
            (src/analysis/step1_graph.py), which also keeps near-cyclic orbits

Outputs in --out-dir:
  orbits_raw.txt        recovered rings and sequences
  orbits_raw_score.txt  edge scores of each structure (--refine only)
  timbre.txt            fixed points (candidate pitch-invariant features)
  invalid.txt           features without an accepted successor (default only)

To interpret the orbits, plot their activations on a few songs with
scripts/plot_activations.py, then copy the orbits you need into feature_ids.txt.

Example:
    python scripts/find_orbits.py --ckpt o-sae.ckpt --out-dir results/muq-2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import default_device, read_sae_hparams, run_module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="o-SAE checkpoint (.ckpt).")
    ap.add_argument("--out-dir", required=True, help="Directory for the output text files.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Successor-confidence threshold of the default recovery (Sec. 3.3). Default: 0.5.")
    ap.add_argument("--refine", action="store_true",
                    help="Use the sparse directed-graph recovery (App. G.2) instead.")
    ap.add_argument("--device", default=default_device())
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.ckpt).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saes = int(read_sae_hparams(ckpt)["n_saes"])
    if n_saes < 2:
        raise SystemExit(f"{ckpt} has {n_saes} sub-SAE; orbit recovery needs an aligned multi-SAE (o-SAE).")

    if args.refine:
        run_module("src.analysis.step1_graph",
                   ["--ckpt-path", ckpt, "--out-dir", out_dir, "--device", args.device])
    else:
        run_module("src.orbits_recovery.obits_recovery",
                   ["--ckpt-path", ckpt, "--out-dir", out_dir,
                    "--prob", args.threshold, "--device", args.device])
    print(f"[DONE] orbits written to {out_dir}")


if __name__ == "__main__":
    main()
