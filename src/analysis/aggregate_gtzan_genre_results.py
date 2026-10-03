#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate GTZAN key metrics by genre, model type, and model layer.

The script scans the two result roots produced by:

* ``run_minor_key.sh`` / ``run_minor_key_1sae.sh``
  - reads only ``*_key_major_minor_joint.txt``
  - deliberately ignores ``*_key_major_minor_joint_gt.txt``
* ``run_relative_key.sh``
  - reads ``*_key_forth.txt``, ``*_key_fifth.txt``, and ``*_key_tonic.txt``

Output layout
-------------

    <output_root>/
      blues/
        slakh-muq-32-3-1.txt
        slakh-musicfm-64-1-1.txt
        ...
      country/
        ...

Each model-type file has one row per model layer.  The joint and the three
relative-key variants are pivoted into columns, so a layer appears exactly once.
All metrics are song-level metrics.  For the current GTZAN cache format, every
song has one 30-second segment, but evaluation still goes through
``compute_key_metrics`` to preserve the project's exact Strict/MIREX logic.

Example
-------

python -m src.analysis.aggregate_gtzan_genre_results \\
    --joint-root store/results_latest/analysis/chord_key_results \\
    --relative-root store/results_latest/analysis/key_results \\
    --output-root store/results_latest/analysis/gtzan_genre_results \\
    --clean
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from .eval_metrics import compute_key_metrics


DEGREES = ("forth", "fifth", "tonic")
GTZAN_GENRE_ORDER = (
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

# Example run directory:
# slakh2100_train_16_muq-2_3-1_1e-3_32_max
RUN_RE = re.compile(r"_(?P<feature>muq|musicfm)-(?P<layer>\d+)_")
EPOCH_RE = re.compile(r"_ep(?P<epoch>\d+)_key_")
MODEL_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<feature>muq|musicfm)-"
    r"(?P<width>\d+)-(?P<mode>\d+-\d+)$"
)

OUTPUT_COLUMNS = [
    "model_type",
    "dataset",
    "feature",
    "width",
    "mode",
    "layer",
    "song_total",
    "song_maj",
    "song_min",
    "joint_epoch",
    "joint_acc",
    "joint_acc_strict",
    "joint_mirex",
    "joint_majmin_acc",
]
for _degree in DEGREES:
    OUTPUT_COLUMNS.extend(
        [
            f"{_degree}_epoch",
            f"{_degree}_acc",
            f"{_degree}_acc_strict",
            f"{_degree}_mirex",
        ]
    )


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def _genre_sort_key(genre: str) -> Tuple[int, str]:
    try:
        return GTZAN_GENRE_ORDER.index(genre), genre
    except ValueError:
        return len(GTZAN_GENRE_ORDER), genre


def _parse_model_type(model_type: str) -> Dict[str, str]:
    match = MODEL_RE.match(model_type)
    if match is None:
        return {
            "dataset": "",
            "feature": "",
            "width": "",
            "mode": "",
        }
    return match.groupdict()


def _parse_layer(run_dir: Path) -> Optional[int]:
    match = RUN_RE.search(run_dir.name)
    if match is None:
        _warn(f"cannot parse layer from run directory; skipped: {run_dir}")
        return None
    return int(match.group("layer"))


def _epoch(path: Path) -> int:
    match = EPOCH_RE.search(path.name)
    return int(match.group("epoch")) if match else -1


def _pick_latest(paths: Iterable[Path]) -> Optional[Path]:
    candidates = [p for p in paths if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (_epoch(p), p.stat().st_mtime_ns, p.name))


def _find_joint_cache(run_dir: Path) -> Optional[Path]:
    # The exact suffix prevents *_joint_gt.txt from being selected.
    return _pick_latest(run_dir.glob("gtzan*_key_major_minor_joint.txt"))


def _find_relative_cache(run_dir: Path, degree: str) -> Optional[Path]:
    return _pick_latest(run_dir.glob(f"gtzan*_key_{degree}.txt"))


def _read_cache_by_genre(path: Path) -> Tuple[str, Dict[str, List[str]]]:
    """Read a raw per-song cache and group its rows by ``track_id`` prefix."""
    header = ""
    grouped: Dict[str, List[str]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if line.startswith("#"):
                if not header:
                    header = line if line.endswith("\n") else line + "\n"
                continue

            track_id = line.split("\t", 1)[0].strip()
            if "/" not in track_id:
                raise ValueError(
                    f"{path}:{line_no}: expected raw genre/track rows, got "
                    f"track_id={track_id!r}. Do not pass key_song_*.tsv summaries."
                )
            genre = track_id.split("/", 1)[0].strip().lower()
            if not genre:
                raise ValueError(f"{path}:{line_no}: empty genre in track_id={track_id!r}")
            grouped[genre].append(line if line.endswith("\n") else line + "\n")

    if not grouped:
        raise ValueError(f"no per-song rows found in {path}")
    return header, dict(grouped)


def _metrics_by_genre(path: Path, *, vote2_decision: str) -> Dict[str, Dict]:
    """Run the project's exact key evaluator independently for each genre."""
    header, grouped = _read_cache_by_genre(path)
    output: Dict[str, Dict] = {}

    with tempfile.TemporaryDirectory(prefix="gtzan_genre_eval_") as tmp:
        tmp_root = Path(tmp)
        for genre, lines in grouped.items():
            genre_cache = tmp_root / f"{genre}.txt"
            with genre_cache.open("w", encoding="utf-8") as handle:
                if header:
                    handle.write(header)
                handle.writelines(lines)

            output[genre] = compute_key_metrics(
                genre_cache,
                maj_degree=0,
                vote2_decision=vote2_decision,
            )
    return output


def _base_row(model_type: str, layer: int) -> Dict[str, object]:
    metadata = _parse_model_type(model_type)
    row: Dict[str, object] = {column: "NA" for column in OUTPUT_COLUMNS}
    row.update(
        {
            "model_type": model_type,
            "dataset": metadata["dataset"],
            "feature": metadata["feature"],
            "width": metadata["width"],
            "mode": metadata["mode"],
            "layer": layer,
        }
    )
    return row


def _get_row(
    store: MutableMapping[str, MutableMapping[str, MutableMapping[int, Dict[str, object]]]],
    genre: str,
    model_type: str,
    layer: int,
) -> Dict[str, object]:
    model_rows = store.setdefault(genre, {}).setdefault(model_type, {})
    return model_rows.setdefault(layer, _base_row(model_type, layer))


def _fill_counts(row: Dict[str, object], metrics: Mapping[str, object]) -> None:
    """Fill common GT counts once; warn if another cache disagrees."""
    for out_key, metric_key in (
        ("song_total", "song_total"),
        ("song_maj", "song_maj"),
        ("song_min", "song_min"),
    ):
        value = metrics.get(metric_key, "NA")
        old = row.get(out_key, "NA")
        if old not in ("NA", "", value):
            _warn(
                f"count mismatch for {row['model_type']} layer={row['layer']} "
                f"{out_key}: existing={old}, new={value}"
            )
        if old in ("NA", ""):
            row[out_key] = value


def _scan_joint_root(
    root: Path,
    store: MutableMapping[str, MutableMapping[str, MutableMapping[int, Dict[str, object]]]],
) -> int:
    n_files = 0
    if not root.is_dir():
        _warn(f"joint root does not exist: {root}")
        return n_files

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model_type = model_dir.name
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            layer = _parse_layer(run_dir)
            if layer is None:
                continue
            cache = _find_joint_cache(run_dir)
            if cache is None:
                continue

            genre_metrics = _metrics_by_genre(cache, vote2_decision="joint")
            epoch = _epoch(cache)
            for genre, metrics in genre_metrics.items():
                row = _get_row(store, genre, model_type, layer)
                _fill_counts(row, metrics)
                row.update(
                    {
                        "joint_epoch": epoch if epoch >= 0 else "NA",
                        "joint_acc": metrics.get("song_acc", "NA"),
                        "joint_acc_strict": metrics.get("song_acc_strict", "NA"),
                        "joint_mirex": metrics.get("song_mirex", "NA"),
                        "joint_majmin_acc": metrics.get("song_majmin_acc", "NA"),
                    }
                )
            n_files += 1
            print(f"[JOINT]    {model_type} layer={layer}: {cache}")
    return n_files


def _scan_relative_root(
    root: Path,
    store: MutableMapping[str, MutableMapping[str, MutableMapping[int, Dict[str, object]]]],
) -> int:
    n_files = 0
    if not root.is_dir():
        _warn(f"relative root does not exist: {root}")
        return n_files

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model_type = model_dir.name
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            layer = _parse_layer(run_dir)
            if layer is None:
                continue

            for degree in DEGREES:
                cache = _find_relative_cache(run_dir, degree)
                if cache is None:
                    continue

                # Plain 17-column relative-key cache. vote2_decision is unused,
                # but "tally" is the evaluator's neutral default.
                genre_metrics = _metrics_by_genre(cache, vote2_decision="tally")
                epoch = _epoch(cache)
                for genre, metrics in genre_metrics.items():
                    row = _get_row(store, genre, model_type, layer)
                    _fill_counts(row, metrics)
                    row.update(
                        {
                            f"{degree}_epoch": epoch if epoch >= 0 else "NA",
                            f"{degree}_acc": metrics.get("song_acc", "NA"),
                            f"{degree}_acc_strict": metrics.get("song_acc_strict", "NA"),
                            f"{degree}_mirex": metrics.get("song_mirex", "NA"),
                        }
                    )
                n_files += 1
                print(f"[RELATIVE] {model_type} layer={layer} degree={degree}: {cache}")
    return n_files


def _write_outputs(
    output_root: Path,
    store: Mapping[str, Mapping[str, Mapping[int, Dict[str, object]]]],
) -> Tuple[int, int]:
    n_genres = 0
    n_files = 0
    output_root.mkdir(parents=True, exist_ok=True)

    for genre in sorted(store, key=_genre_sort_key):
        genre_dir = output_root / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        n_genres += 1

        for model_type in sorted(store[genre]):
            output_path = genre_dir / f"{model_type}.txt"
            layer_rows = store[genre][model_type]
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=OUTPUT_COLUMNS,
                    delimiter="\t",
                    lineterminator="\n",
                    extrasaction="ignore",
                )
                writer.writeheader()
                for layer in sorted(layer_rows):
                    writer.writerow(layer_rows[layer])
            n_files += 1
            print(f"[WRITE]    {output_path} ({len(layer_rows)} layers)")

    return n_genres, n_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate raw GTZAN joint and relative-key caches into one folder "
            "per genre and one file per model type, with one row per layer."
        )
    )
    parser.add_argument(
        "--joint-root",
        type=Path,
        required=True,
        help="Root produced by run_minor_key, e.g. .../analysis/chord_key_results",
    )
    parser.add_argument(
        "--relative-root",
        type=Path,
        required=True,
        help="Root produced by run_relative_key, e.g. .../analysis/key_results",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root containing one directory per genre.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output-root before writing, preventing stale model files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joint_root = args.joint_root.resolve()
    relative_root = args.relative_root.resolve()
    output_root = args.output_root.resolve()

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    # genre -> model_type -> layer -> wide result row
    store: Dict[str, Dict[str, Dict[int, Dict[str, object]]]] = {}

    joint_files = _scan_joint_root(joint_root, store)
    relative_files = _scan_relative_root(relative_root, store)
    if not store:
        raise RuntimeError(
            "No raw GTZAN caches were found. Check the two roots and make sure "
            "the per-run directories contain gtzan*_key_*.txt files."
        )

    n_genres, n_outputs = _write_outputs(output_root, store)
    print(
        f"[DONE] joint caches={joint_files}, relative caches={relative_files}, "
        f"genres={n_genres}, output files={n_outputs}\n"
        f"       output_root={output_root}"
    )


if __name__ == "__main__":
    main()
