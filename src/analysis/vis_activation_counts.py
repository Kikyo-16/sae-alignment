#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize activation counts of an o-SAE / s-SAE pair in one mirrored figure.

Left half  (negative x): o-SAE (mode 1-1), counts sorted ASCENDING
                         (least active far left, most active next to centre).
Right half (positive x): s-SAE (mode 3-1), counts sorted DESCENDING
                         (most active next to centre, least active far right).
X-axis: feature rank (tick labels show the absolute rank).
Y-axis: raw activation count (segments, from step_activation_counts).

Counts are read from <counts-root>/<group>/<run>/activation_counts.npy where
the run directory is located by globbing *_{feature}-{lid}_{mode}_*_{exp}_max,
so both the old short and the current long run-name styles match.

Usage:
    python -m src.analysis.vis_activation_counts \\
        --counts-root store/results_latest/analysis/activation_counts \\
        --feature muq --lid 2 --expansion 64 \\
        [--o-mode 1-1] [--s-mode 3-1] [--out figs/act_counts_muq-2_64.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_O = "#1f77b4"   # blue — o-SAE (1-1)
C_S = "#d62728"   # red  — s-SAE (3-1)


def find_counts(
    counts_root: Path, feature: str, lid: int, mode: str, expansion: int,
) -> Optional[np.ndarray]:
    """Locate <run>/activation_counts.npy for one (feature, lid, mode, exp)."""
    run_glob = f"*_{feature}-{lid}_{mode}_*_{expansion}_max"
    candidates = sorted(counts_root.glob(f"*/{run_glob}/activation_counts.npy")) \
        + sorted(counts_root.glob(f"{run_glob}/activation_counts.npy"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"[WARN] {len(candidates)} matches for {run_glob}; using {candidates[0]}")
    print(f"[INFO] {mode}: {candidates[0]}")
    return np.load(candidates[0]).astype(np.int64)


def plot_pair(
    o_counts: np.ndarray,
    s_counts: np.ndarray,
    o_label: str,
    s_label: str,
    out_path: Path,
) -> None:
    from matplotlib.ticker import FuncFormatter

    d_o, d_s = len(o_counts), len(s_counts)

    # Left: ascending toward the centre; right: descending away from it.
    y_o = np.sort(o_counts)                # ascending
    x_o = np.arange(-d_o, 0)               # -D_o … -1
    y_s = np.sort(s_counts)[::-1]          # descending
    x_s = np.arange(1, d_s + 1)            # 1 … D_s

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x_o, y_o, color=C_O, lw=1.4, label=o_label)
    ax.plot(x_s, y_s, color=C_S, lw=1.4, label=s_label)
    ax.axvline(0, color="gray", lw=0.9, ls="--", zorder=0)

    ax.set_xlim(-d_o * 1.02, d_s * 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.0f}"))

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", lw=0.5, alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("Feature", fontsize=20)
    ax.set_ylabel("Feature count", fontsize=20)
    ax.legend(fontsize=16, loc="upper left", framealpha=0.88)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Mirrored o-SAE / s-SAE activation-count figure."
    )
    ap.add_argument("--counts-root",
                    default="store/results_latest/analysis/activation_counts",
                    help="Root written by run_activation_counts.sh")
    ap.add_argument("--feature",   required=True, help="muq / musicfm")
    ap.add_argument("--lid",       type=int, required=True)
    ap.add_argument("--expansion", type=int, required=True)
    ap.add_argument("--o-mode",    default="1-1", help="mode of the o-SAE (left half)")
    ap.add_argument("--s-mode",    default="3-1", help="mode of the s-SAE (right half)")
    ap.add_argument("--out",       default=None,
                    help="Output PNG (default: <counts-root>/figs/act_counts_"
                         "<feature>-<lid>_<expansion>.png)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    counts_root = Path(args.counts_root)

    o_counts = find_counts(counts_root, args.feature, args.lid, args.o_mode, args.expansion)
    s_counts = find_counts(counts_root, args.feature, args.lid, args.s_mode, args.expansion)
    if o_counts is None or s_counts is None:
        missing = [m for m, c in ((args.o_mode, o_counts), (args.s_mode, s_counts)) if c is None]
        raise FileNotFoundError(
            f"activation_counts.npy not found under {counts_root} for mode(s) "
            f"{missing} ({args.feature}-{args.lid}, exp={args.expansion}). "
            "Run run_activation_counts.sh first."
        )

    out = Path(args.out) if args.out else (
        counts_root / "figs" /
        f"act_counts_{args.feature}-{args.lid}_{args.expansion}.png"
    )
    plot_pair(o_counts, s_counts, "O-SAE", "S-SAE", out)


if __name__ == "__main__":
    main()
