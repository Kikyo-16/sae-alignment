#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Orbit Recovery Summary across varying thresholds (--prob).

Layout:
  Generates grid subplots corresponding to combinations of Representation (muq/musicfm)
  and TopK values.
  
Axes & Colors:
  - Y-axis : Threshold (0.00 to 0.95)
  - X-axis : Layer ID
  - Colors : Bright Green (Yes / Recovered), Bright Red (No / N/A)
  - Shapes : Major (Circle 'o'), Minor (Square 's'), Forth (Triangle '^')
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== Simplified Two-Color Palette =====
COLOR_YES = "#00C853"  # Bright / Vibrant Green (Recovered)
COLOR_NO = "#FF1744"   # Bright Red (Not Recovered / NA)

# Distinct marker shapes for ring identification
MARKER_MAP = {
    "major": "o",  # Circle
    "minor": "s",  # Square
    "forth": "^",  # Triangle
}

# Horizontal offset to prevent marker overlap on the same layer & threshold
X_OFFSET = {
    "major": -0.28,
    "minor": 0.0,
    "forth": 0.28,
}

MODEL_NAME_RE = re.compile(
    r"^[^_]+_([^-]+)_([^-]+)_([^-]+)-([0-9]+)_([0-9]+-[0-9]+)_([^_]+)_([0-9]+)_max$"
)


def parse_model_info(model_name: str, group_name: str) -> Tuple[str, int, int]:
    """Parse (rep, layer, topk) from model or group directory name."""
    parts = model_name.split("_")
    if len(parts) >= 7 and "-" in parts[3]:
        rep, layer_str = parts[3].split("-", 1)
        topk_str = parts[6]
        if layer_str.isdigit() and topk_str.isdigit():
            return rep.lower(), int(layer_str), int(topk_str)

    g_parts = group_name.split("-")
    if len(g_parts) >= 3:
        rep = g_parts[1].lower()
        topk_str = g_parts[2]
        layer_match = re.search(r"-(\d+)_", model_name)
        layer = int(layer_match.group(1)) if layer_match else 0
        if topk_str.isdigit():
            return rep, layer, int(topk_str)

    raise ValueError(f"Failed to parse model information: group='{group_name}', model='{model_name}'")


def load_all_summaries(base_dir: Path) -> pd.DataFrame:
    """Traverse all threshold directories to load orbit_summary.tsv files."""
    records = []

    summary_files = list(base_dir.rglob("orbit_summary.tsv"))
    if not summary_files:
        raise FileNotFoundError(f"No orbit_summary.tsv files found under {base_dir}!")

    for file_path in summary_files:
        try:
            threshold = float(file_path.parent.name)
        except ValueError:
            print(f"[WARN] Skipping directory with unparseable threshold: {file_path.parent}")
            continue

        df = pd.read_csv(file_path, sep="\t")
        for _, row in df.iterrows():
            group = str(row["group"])
            model = str(row["model"])
            try:
                rep, layer, topk = parse_model_info(model, group)
            except ValueError as e:
                print(f"[WARN] {e}")
                continue

            for ring in ["major", "minor", "forth"]:
                status = str(row[ring]).strip()
                is_yes = (status.lower() == "yes")
                records.append({
                    "threshold": threshold,
                    "rep": rep,
                    "topk": topk,
                    "layer": layer,
                    "ring": ring,
                    "status": status,
                    "is_yes": is_yes,
                })

    return pd.DataFrame(records)


def plot_grid(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot grid of subplots for each Representation x TopK pair and save the figure."""
    reps = sorted(df["rep"].unique())
    topks = sorted(df["topk"].unique())

    fig, axes = plt.subplots(
        nrows=len(reps),
        ncols=len(topks),
        figsize=(8.0 * len(topks), 4 * len(reps)),
        sharex=False,
        sharey=True,
        dpi=200,
    )

    if len(reps) == 1 and len(topks) == 1:
        axes = np.array([[axes]])
    elif len(reps) == 1:
        axes = np.array([axes])
    elif len(topks) == 1:
        axes = axes[:, np.newaxis]

    layers_all = sorted(df["layer"].unique())
    thresholds_all = sorted(df["threshold"].unique())

    for r_idx, rep in enumerate(reps):
        for t_idx, topk in enumerate(topks):
            ax = axes[r_idx, t_idx]
            sub_df = df[(df["rep"] == rep) & (df["topk"] == topk)]

            if sub_df.empty:
                ax.set_title(f"{rep.upper()} (TopK={topk}) - No Data")
                continue

            # Plot scatter points using binary color scheme
            for ring in ["major", "minor", "forth"]:
                ring_df = sub_df[sub_df["ring"] == ring]
                marker = MARKER_MAP[ring]
                x_shift = X_OFFSET[ring]

                for is_yes in [True, False]:
                    pts = ring_df[ring_df["is_yes"] == is_yes]
                    if pts.empty:
                        continue

                    color = COLOR_YES if is_yes else COLOR_NO

                    ax.scatter(
                        pts["layer"] + x_shift,
                        pts["threshold"],
                        c=color,
                        marker=marker,
                        s=30,
                        alpha=0.85,
                        edgecolors="none",
                    )

            # Subplot layout adjustments
            ax.set_title(f"{rep.upper()} | TopK = {topk}", fontsize=16, fontweight="bold")
            ax.set_xlabel("Layer", fontsize=20)
            if t_idx == 0:
                ax.set_ylabel(r'$\tau$', fontsize=20)

            ax.set_xticks(layers_all)
            ax.set_yticks(thresholds_all)
            ax.set_yticklabels([f"{t:.2f}" for t in thresholds_all], fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)

    # Simplified legend with green/red symbols per ring shape
    legend_elements = [
        # Yes / Recovered (Green)
        plt.Line2D([0], [0], marker=MARKER_MAP["major"], color="w", label="Major (Y)", markerfacecolor=COLOR_YES, markersize=8),
        plt.Line2D([0], [0], marker=MARKER_MAP["minor"], color="w", label="Minor (Y)", markerfacecolor=COLOR_YES, markersize=8),
        plt.Line2D([0], [0], marker=MARKER_MAP["forth"], color="w", label="Subdominant (Y)", markerfacecolor=COLOR_YES, markersize=8),
        # No / N/A (Red)
        plt.Line2D([0], [0], marker=MARKER_MAP["major"], color="w", label="Major (N)", markerfacecolor=COLOR_NO, markersize=8),
        plt.Line2D([0], [0], marker=MARKER_MAP["minor"], color="w", label="Minor (N)", markerfacecolor=COLOR_NO, markersize=8),
        plt.Line2D([0], [0], marker=MARKER_MAP["forth"], color="w", label="Subdominant (N)", markerfacecolor=COLOR_NO, markersize=8),
    ]

    plt.tight_layout()

    # Single-row legend placed entirely above the figure, clear of the subplot titles
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=6,
        frameon=True,
        fontsize=18,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "orbits_summary_threshold_grid.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Figure successfully saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot orbit recovery summary across thresholds.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("store/results_latest/analysis/orbit_recovery_successor"),
        help="Base directory containing subdirectories for each prob value.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("store/results_latest/analysis/plots"),
        help="Directory to save output plots.",
    )
    args = parser.parse_args()

    df = load_all_summaries(args.base_dir)
    print(f"[INFO] Successfully loaded {len(df)} records. Reps: {df['rep'].unique()}, TopKs: {df['topk'].unique()}")
    plot_grid(df, args.out_dir)


if __name__ == "__main__":
    main()