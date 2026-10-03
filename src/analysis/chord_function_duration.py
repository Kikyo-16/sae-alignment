#!/usr/bin/env python3
"""
Compute total duration (seconds) of tonic / subdominant / dominant chords
across the dataset, plus overall dataset duration.

Usage:
    python src/analysis/chord_function_duration.py \
        --dataset dataset/slakh2100_flac_redux \
        [--splits train validation test]
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NOTE2PC: Dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "E#": 5, "F": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}


def note_to_pc(name: str) -> Optional[int]:
    name = name.strip()
    if not name:
        return None
    letter = name[0].upper()
    acc = name[1:].replace("b", "B") if len(name) > 1 else ""
    return NOTE2PC.get((letter + acc).upper())


def parse_lab(path: Path) -> List[Tuple[float, float, str]]:
    segments = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            segments.append((start, end, parts[2]))
    return segments


def parse_key_root_pc(label: str) -> Optional[int]:
    """Return tonic pitch class for I/IV/V analysis.
    Minor keys are converted to their relative major (+3 semitones) first.
    E.g. A:minor → C major (tonic=C, subdom=F, dom=G).
    """
    m = re.match(r"^([A-Ga-g][#b]?)(?::(.*))?$", label.strip())
    if not m:
        return None
    root_pc = note_to_pc(m.group(1))
    if root_pc is None:
        return None
    mode = (m.group(2) or "").lower().strip()
    if mode.startswith("min"):
        root_pc = (root_pc + 3) % 12  # relative major
    return root_pc


def chord_root_pc(label: str) -> Optional[int]:
    """Extract root pitch class from chord labels like 'F:maj', 'G#:min7'. N/X → None."""
    if label.upper() in ("N", "X", ""):
        return None
    return note_to_pc(label.split(":")[0])


FUNCTIONS = ("tonic", "subdominant", "dominant")
# semitone offsets from tonic: I=0, IV=5, V=7
OFFSETS = {"tonic": 0, "subdominant": 5, "dominant": 7}


def classify(chord_pc: int, key_pc: int) -> Optional[str]:
    interval = (chord_pc - key_pc) % 12
    for func, offset in OFFSETS.items():
        if interval == offset:
            return func
    return None


def analyze_track(chord_path: Path, key_path: Path) -> Tuple[Dict[str, float], float]:
    chord_segs = parse_lab(chord_path)
    key_segs = parse_lab(key_path)
    if not chord_segs:
        return {}, 0.0

    total = chord_segs[-1][1] - chord_segs[0][0]
    durations: Dict[str, float] = defaultdict(float)

    for c_start, c_end, chord_label in chord_segs:
        c_pc = chord_root_pc(chord_label)
        if c_pc is None:
            continue
        for k_start, k_end, key_label in key_segs:
            ov_start = max(c_start, k_start)
            ov_end = min(c_end, k_end)
            if ov_end <= ov_start:
                continue
            k_pc = parse_key_root_pc(key_label)
            if k_pc is None:
                continue
            func = classify(c_pc, k_pc)
            if func:
                durations[func] += ov_end - ov_start

    return durations, total


def fmt_dur(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{seconds:>12.2f}s  ({h:02d}h{m:02d}m{s:05.2f}s)"


def main():
    ap = argparse.ArgumentParser(description="Dataset chord-function duration statistics.")
    ap.add_argument("--dataset", default="dataset/slakh2100_flac_redux",
                    help="Root of dataset directory.")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="Subset of splits to process (default: all subdirectories).")
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset not found: {dataset_root}")

    if args.splits:
        split_dirs = [dataset_root / s for s in args.splits]
    else:
        split_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]

    total_durations: Dict[str, float] = defaultdict(float)
    dataset_total = 0.0
    track_count = 0
    skipped = 0

    for split_dir in split_dirs:
        split_total = 0.0
        split_durations: Dict[str, float] = defaultdict(float)
        split_tracks = 0

        for chord_path in sorted(split_dir.rglob("all_src.chord.lab")):
            key_path = chord_path.parent / "all_src.key.lab"
            if not key_path.exists():
                skipped += 1
                continue
            durs, track_dur = analyze_track(chord_path, key_path)
            for func, d in durs.items():
                split_durations[func] += d
                total_durations[func] += d
            split_total += track_dur
            dataset_total += track_dur
            split_tracks += 1
            track_count += 1

        if split_tracks == 0:
            continue
        print(f"=== {split_dir.name}  ({split_tracks} tracks) ===")
        print(f"  total duration : {fmt_dur(split_total)}")
        for func in FUNCTIONS:
            d = split_durations.get(func, 0.0)
            pct = 100 * d / split_total if split_total > 0 else 0.0
            print(f"  {func:<12s}  : {fmt_dur(d)}  {pct:6.2f}%")
        print()

    print(f"=== TOTAL  ({track_count} tracks) ===")
    print(f"  total duration : {fmt_dur(dataset_total)}")
    for func in FUNCTIONS:
        d = total_durations.get(func, 0.0)
        pct = 100 * d / dataset_total if dataset_total > 0 else 0.0
        print(f"  {func:<12s}  : {fmt_dur(d)}  {pct:6.2f}%")
    if skipped:
        print(f"\n  (skipped {skipped} tracks missing key.lab)")


if __name__ == "__main__":
    main()
