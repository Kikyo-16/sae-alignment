#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw five layer-wise GTZAN genre bar charts.

Figures
-------
1. MuQ 3-1: strict accuracy
2. MuQ 3-1: MIREX score
3. MuQ 3-1: forth accuracy (layers 2--8 only)
4. MusicFM 3-1: strict accuracy
5. MusicFM 3-1: MIREX score

Expected input layout
---------------------

    <results_root>/
      blues/
        slakh-muq-32-3-1.txt
        slakh-muq-48-3-1.txt
        slakh-muq-64-3-1.txt
        slakh-musicfm-32-3-1.txt
        ...
      country/
        ...

Each TXT contains one row per layer and at least these columns:

    layer  strict_acc  mirex  forth_acc

Because the requested figures identify only the representation and mode, not a
single SAE width, the default behavior averages the available widths (32/48/64)
within each genre and layer. Use ``--width 32`` (or another width) to plot one
fixed width instead.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_VERSION = "2026-07-27-layerwise-blue-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "store/results_latest/analysis/gtzan_genre_results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "store/results_latest/analysis/gtzan_genre_layer_figures"

GENRE_ORDER = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
)

METRIC_ALIASES = {
    "strict_acc": ("strict_acc", "joint_acc_strict"),
    "mirex": ("mirex", "joint_mirex"),
    "forth_acc": ("forth_acc",),
}

FIGURE_SPECS = (
    {
        "feature": "muq",
        "mode": "3-1",
        "metric": "strict_acc",
        "ylabel": "Strict accuracy (%)",
        "filename": "gtzan_muq_3-1_strict_acc_by_genre",
        "layer_range": None,
    },
    {
        "feature": "muq",
        "mode": "3-1",
        "metric": "mirex",
        "ylabel": "MIREX (%)",
        "filename": "gtzan_muq_3-1_mirex_by_genre",
        "layer_range": None,
    },
    {
        "feature": "muq",
        "mode": "3-1",
        "metric": "forth_acc",
        "ylabel": "Accuracy (%)",
        "filename": "gtzan_muq_3-1_forth_acc_by_genre",
        "layer_range": (2, 8),
    },
    {
        "feature": "musicfm",
        "mode": "3-1",
        "metric": "strict_acc",
        "ylabel": "Strict accuracy (%)",
        "filename": "gtzan_musicfm_3-1_strict_acc_by_genre",
        "layer_range": None,
    },
    {
        "feature": "musicfm",
        "mode": "3-1",
        "metric": "mirex",
        "ylabel": "MIREX (%)",
        "filename": "gtzan_musicfm_3-1_mirex_by_genre",
        "layer_range": None,
    },
)

MODEL_FILE_RE = re.compile(
    r"^(?:slakh-)?(?P<feature>muq|musicfm)-(?P<width>\d+)-(?P<mode>1-1|3-1)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class Config:
    feature: str
    mode: str
    width: int
    layer: int


# metric -> genre -> config -> value
MetricData = Dict[str, Dict[str, Dict[Config, float]]]


def _as_float(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NAN", "NONE", "-"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _first_metric_value(row: Mapping[str, str], aliases: Sequence[str]) -> Optional[float]:
    for alias in aliases:
        if alias in row:
            value = _as_float(row[alias])
            if value is not None:
                return value
    return None


def _parse_model_file(path: Path) -> Optional[Tuple[str, int, str]]:
    match = MODEL_FILE_RE.match(path.stem)
    if match is None:
        return None
    return (
        match.group("feature").lower(),
        int(match.group("width")),
        match.group("mode"),
    )


def load_results(results_root: Path) -> MetricData:
    data: MetricData = {metric: defaultdict(dict) for metric in METRIC_ALIASES}
    genre_dirs = [path for path in results_root.iterdir() if path.is_dir()]
    if not genre_dirs:
        raise RuntimeError(f"No genre directories found under {results_root}")

    loaded_rows = 0
    for genre_dir in sorted(genre_dirs):
        genre = genre_dir.name.lower()
        for model_file in sorted(genre_dir.glob("*.txt")):
            parsed = _parse_model_file(model_file)
            if parsed is None:
                continue
            feature, width, mode = parsed
            if mode != "3-1":
                continue

            with model_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames is None:
                    continue
                for row_no, row in enumerate(reader, start=2):
                    layer_value = _as_float(row.get("layer", ""))
                    if layer_value is None:
                        print(f"[SKIP] missing layer: {model_file}:{row_no}")
                        continue
                    layer = int(layer_value)
                    config = Config(feature, mode, width, layer)
                    found_any = False
                    for metric, aliases in METRIC_ALIASES.items():
                        value = _first_metric_value(row, aliases)
                        if value is None:
                            continue
                        data[metric][genre][config] = value
                        found_any = True
                    loaded_rows += int(found_any)

    if loaded_rows == 0:
        raise RuntimeError(
            "No metric rows were loaded. Expected tab-separated files with "
            "layer and strict_acc/mirex/forth_acc columns."
        )

    # Support either percentages [0, 100] or fractions [0, 1].
    for metric in METRIC_ALIASES:
        values = [
            value
            for genre_values in data[metric].values()
            for value in genre_values.values()
        ]
        if values and max(values) <= 1.5:
            for genre_values in data[metric].values():
                for config in list(genre_values):
                    genre_values[config] *= 100.0

    return data


def _genres_present(metric_data: Mapping[str, Mapping[Config, float]]) -> List[str]:
    present = set(metric_data)
    ordered = [genre for genre in GENRE_ORDER if genre in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _layers_present(
    metric_data: Mapping[str, Mapping[Config, float]],
    feature: str,
    mode: str,
    layer_range: Optional[Tuple[int, int]],
    fixed_width: Optional[int],
) -> List[int]:
    layers = {
        config.layer
        for genre_values in metric_data.values()
        for config in genre_values
        if config.feature == feature
        and config.mode == mode
        and (fixed_width is None or config.width == fixed_width)
    }
    if layer_range is not None:
        lo, hi = layer_range
        layers = {layer for layer in layers if lo <= layer <= hi}
    return sorted(layers)


def aggregate_layer_values(
    metric_data: Mapping[str, Mapping[Config, float]],
    genres: Sequence[str],
    feature: str,
    mode: str,
    layers: Sequence[int],
    fixed_width: Optional[int],
) -> Tuple[Dict[int, List[float]], Dict[int, List[int]]]:
    values: Dict[int, List[float]] = {layer: [] for layer in layers}
    widths_used: Dict[int, set[int]] = {layer: set() for layer in layers}

    for layer in layers:
        for genre in genres:
            candidates: List[Tuple[int, float]] = []
            for config, value in metric_data.get(genre, {}).items():
                if config.feature != feature or config.mode != mode or config.layer != layer:
                    continue
                if fixed_width is not None and config.width != fixed_width:
                    continue
                candidates.append((config.width, value))

            if not candidates:
                values[layer].append(math.nan)
                continue

            widths_used[layer].update(width for width, _ in candidates)
            values[layer].append(float(np.mean([value for _, value in candidates])))

    return values, {layer: sorted(widths) for layer, widths in widths_used.items()}


def blue_palette(n: int) -> List[Tuple[float, float, float, float]]:
    """Return n blue shades ordered from dark to light."""
    if n <= 0:
        return []
    positions = np.linspace(0.90, 0.36, n)
    return [plt.cm.Blues(float(position)) for position in positions]


def draw_figure(
    spec: Mapping[str, object],
    metric_data: Mapping[str, Mapping[Config, float]],
    output_dir: Path,
    fixed_width: Optional[int],
    formats: Sequence[str],
    dpi: int,
    show_values: bool,
) -> Tuple[List[int], Dict[int, List[int]]]:
    feature = str(spec["feature"])
    mode = str(spec["mode"])
    metric = str(spec["metric"])
    layer_range = spec["layer_range"]
    assert layer_range is None or isinstance(layer_range, tuple)

    genres = _genres_present(metric_data)
    layers = _layers_present(
        metric_data,
        feature,
        mode,
        layer_range,
        fixed_width,
    )
    if not genres or not layers:
        print(f"[SKIP] no data for {feature=} {mode=} {metric=}")
        return [], {}

    layer_values, widths_used = aggregate_layer_values(
        metric_data,
        genres,
        feature,
        mode,
        layers,
        fixed_width,
    )

    # Preserve the user's wide, low-height layout while allowing more bars.
    genre_spacing = 1.28
    x = np.arange(len(genres), dtype=float) * genre_spacing
    n_layers = len(layers)
    group_width = 0.92
    bar_width = group_width / n_layers
    offsets = (np.arange(n_layers) - (n_layers - 1) / 2.0) * bar_width
    colors = blue_palette(n_layers)

    fig_width = max(24.0, 2.45 * len(genres) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))

    for index, layer in enumerate(layers):
        heights = np.asarray(layer_values[layer], dtype=float)
        bars = ax.bar(
            x + offsets[index],
            heights,
            width=bar_width * 0.90,
            label=f"Layer {layer}",
            color=colors[index],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.84,
            zorder=3,
        )
        if show_values:
            labels = [f"{height:.1f}" if np.isfinite(height) else "" for height in heights]
            ax.bar_label(
                bars,
                labels=labels,
                padding=2,
                fontsize=8.2 if n_layers >= 8 else 9.0,
                rotation=0,
                color="#2F2F2F",
            )

    ax.set_ylabel(str(spec["ylabel"]), fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [genre.capitalize() for genre in genres],
        fontsize=22,
        fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 4), minor=True)
    ax.grid(
        axis="y",
        which="major",
        linestyle="-",
        linewidth=0.95,
        color="#737373",
        alpha=0.62,
        zorder=0,
    )
    ax.grid(
        axis="y",
        which="minor",
        linestyle="-",
        linewidth=0.52,
        color="#9A9A9A",
        alpha=0.46,
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="y", labelsize=10)

    # One legend entry per layer, ordered from dark to light.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=n_layers,
        frameon=True,
        fontsize=22 ,
        columnspacing=1.15,
        handlelength=1.5,
        handletextpad=0.45,
    )

    fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.93))
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = output_dir / f"{spec['filename']}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[WRITE] {out_path}")
    plt.close(fig)
    return layers, widths_used


def write_plot_manifest(
    output_dir: Path,
    records: Sequence[Tuple[Mapping[str, object], Sequence[int], Mapping[int, Sequence[int]]]],
    fixed_width: Optional[int],
) -> None:
    path = output_dir / "plot_manifest.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["figure", "feature", "mode", "metric", "layer", "widths_averaged"])
        for spec, layers, widths_used in records:
            for layer in layers:
                widths = widths_used.get(layer, ())
                writer.writerow(
                    [
                        spec["filename"],
                        spec["feature"],
                        spec["mode"],
                        spec["metric"],
                        layer,
                        fixed_width if fixed_width is not None else ",".join(map(str, widths)),
                    ]
                )
    print(f"[WRITE] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw five layer-wise GTZAN genre bar charts in blue shades."
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Default: store/results_latest/analysis/gtzan_genre_results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Default: store/results_latest/analysis/gtzan_genre_layer_figures",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Plot one fixed SAE width. By default, average all available widths per layer.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--no-values",
        action="store_true",
        help="Do not draw numeric labels above bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[INFO] version={SCRIPT_VERSION}")
    print(f"[INFO] script={Path(__file__).resolve()}")
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"results root not found: {results_root}")

    data = load_results(results_root)
    records = []
    for spec in FIGURE_SPECS:
        metric = str(spec["metric"])
        layers, widths_used = draw_figure(
            spec,
            data[metric],
            output_dir,
            args.width,
            args.formats,
            args.dpi,
            show_values=not args.no_values,
        )
        records.append((spec, layers, widths_used))

    write_plot_manifest(output_dir, records, args.width)
    print(f"[DONE] figures written to {output_dir}")


if __name__ == "__main__":
    main()
