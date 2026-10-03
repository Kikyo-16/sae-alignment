#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate chord predictions (WCSR under multiple vocabularies).

Input: chord output TSV from infer_key --chord-out-file
  # track_id  seg_range  ref_chord  est_chord  block_consistency
  where seg_range = "cs-ce" (frame indices, fps=10 by default)

Vocabularies:
  root        – root pitch class only (N==N correct, N vs chord wrong)
  majmin      – root + {maj|min|N} (mir_eval rule); ref frames whose triad core
                is not a plain maj/min triad (dim, aug, sus, 9/11/13, …) skipped
  majmin_inv  – majmin + bass note
  mirex       – MIREX pitch-class set overlap: |ref_pcs ∩ est_pcs| >= 3; N==N correct
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Note / chord constants
# ---------------------------------------------------------------------------

NOTE2PC: Dict[str, int] = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

# Intervals (semitones above root) for each quality.
QUALITY_INTERVALS: Dict[str, List[int]] = {
    "maj":     [0, 4, 7],
    "min":     [0, 3, 7],
    "7":       [0, 4, 7, 10],
    "maj7":    [0, 4, 7, 11],
    "min7":    [0, 3, 7, 10],
    "dim":     [0, 3, 6],
    "dim7":    [0, 3, 6, 9],
    "hdim7":   [0, 3, 6, 10],
    "aug":     [0, 4, 8],
    "sus2":    [0, 2, 7],
    "sus4":    [0, 5, 7],
    "maj6":    [0, 4, 7, 9],
    "min6":    [0, 3, 7, 9],
    "9":       [0, 4, 7, 10, 2],
    "maj9":    [0, 4, 7, 11, 2],
    "min9":    [0, 3, 7, 10, 2],
    "minmaj7": [0, 3, 7, 11],
    "1":       [0],
    "5":       [0, 7],
    "6":       [0, 4, 7, 9],
    "add9":    [0, 4, 7, 2],
    "add2":    [0, 4, 7, 2],
    "sus":     [0, 5, 7],
    "m":       [0, 3, 7],
    "m7":      [0, 3, 7, 10],
    "m6":      [0, 3, 7, 9],
    "m9":      [0, 3, 7, 10, 2],
    "hdim":    [0, 3, 6, 10],
}

# Major / minor triad cores in pitch-class-interval space (semitones 0-7).
# mir_eval.chord.majmin scores a reference chord only when the pitch classes
# within the lower 8 semitones are exactly a major or minor triad.  Extensions
# that sit at semitones 8-11 (6th, b7, maj7) don't change the core, so 7 / maj7
# / 6 fold to 'maj' and min7 / min6 / minmaj7 fold to 'min'; but dim, aug, sus,
# power chords and added-lower-tone extensions (9 / 11 / 13 / add2 / add9) have
# a different core and are EXCLUDED from majmin scoring (not folded).
_MAJ_TRIAD_CORE = frozenset({0, 4, 7})
_MIN_TRIAD_CORE = frozenset({0, 3, 7})

# Harte shorthand interval→semitone for bass parsing (e.g. C:maj/3).
_INTERVAL_SEMITONES: Dict[str, int] = {
    "1": 0, "2": 2, "b2": 1, "#1": 1,
    "3": 4, "b3": 3, "#2": 3,
    "4": 5, "#3": 5, "b4": 4,
    "5": 7, "b5": 6, "#4": 6, "#5": 8,
    "6": 9, "b6": 8,
    "b7": 10, "7": 11,
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

NO_CHORD_STRS = frozenset({"N", "X", "NOCHORD", "N/A", ""})

# 'X' = unknown / unannotated reference.  mir_eval excludes such frames from
# every metric (unlike 'N' no-chord, which is scored).  Detected from the raw
# reference string before parsing, since parse_chord folds both N and X to None.
UNKNOWN_CHORD_STRS = frozenset({"X"})


def _is_unknown_ref(label: str) -> bool:
    """True for an 'X' (unknown) reference label → excluded from scoring."""
    return label.strip().upper() in UNKNOWN_CHORD_STRS


@dataclass(frozen=True)
class ParsedChord:
    root_pc: int            # 0–11
    quality: str            # lower-case quality string
    bass_pc: Optional[int]  # explicit bass pitch class, or None


@dataclass(frozen=True)
class ChordRow:
    track_id: str
    cs: int                 # start frame (inclusive)
    ce: int                 # end frame (exclusive)
    ref_chord: str
    est_chord: str
    block_consistency: float


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_note(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    # Normalise: upper first letter, keep accidental
    if len(s) >= 2 and s[1] in ("#", "b"):
        key = s[0].upper() + s[1]
    else:
        key = s[0].upper()
    return NOTE2PC.get(key.upper(), None)


def parse_chord(chord: str) -> Optional[ParsedChord]:
    """
    Parse a Harte-style chord string into root_pc, quality, bass_pc.
    Returns None for no-chord (N, X, …).
    """
    s = chord.strip()
    if s.upper() in NO_CHORD_STRS:
        return None

    colon = s.split(":", 1)
    root_pc = _parse_note(colon[0])
    if root_pc is None:
        return None

    rest = colon[1] if len(colon) > 1 else "maj"
    qual_bass = rest.split("/", 1)
    quality = qual_bass[0].strip().lower()

    bass_pc: Optional[int] = None
    if len(qual_bass) > 1:
        bass_str = qual_bass[1].strip()
        bp = _parse_note(bass_str)
        if bp is not None:
            bass_pc = bp
        else:
            interval = _INTERVAL_SEMITONES.get(bass_str)
            if interval is not None:
                bass_pc = (root_pc + interval) % 12

    return ParsedChord(root_pc=root_pc, quality=quality, bass_pc=bass_pc)


def _chord_pc_set(p: ParsedChord) -> frozenset:
    intervals = QUALITY_INTERVALS.get(p.quality, [0, 4, 7])
    return frozenset((p.root_pc + i) % 12 for i in intervals)


def _majmin_quality(quality: str) -> Optional[str]:
    """Return 'maj', 'min', or None (ref outside the major/minor vocab → skip).

    Mirrors mir_eval.chord.majmin: reduce the chord to the pitch classes within
    semitones 0-7 and keep it only if that triad core is exactly a major
    ({0,4,7}) or minor ({0,3,7}) triad.  dim / aug / sus / power / extended
    chords with altered lower voices have a different core and are skipped.
    """
    intervals = QUALITY_INTERVALS.get(quality.lower())
    if intervals is None:
        return None
    core = frozenset(i % 12 for i in intervals if i % 12 < 8)
    if core == _MAJ_TRIAD_CORE:
        return "maj"
    if core == _MIN_TRIAD_CORE:
        return "min"
    return None


# ---------------------------------------------------------------------------
# Vocabulary match functions
# Return True (correct), False (wrong), or None (skip / ref outside vocab)
# ---------------------------------------------------------------------------

MatchFn = Callable[[Optional[ParsedChord], Optional[ParsedChord]], Optional[bool]]


def match_root(ref: Optional[ParsedChord], est: Optional[ParsedChord]) -> Optional[bool]:
    """Root pitch class only; N==N is correct."""
    if ref is None and est is None:
        return True
    if ref is None or est is None:
        return False
    return ref.root_pc == est.root_pc


def match_majmin(ref: Optional[ParsedChord], est: Optional[ParsedChord]) -> Optional[bool]:
    """Root + {maj|min|N}. Frames where ref quality ∉ vocab are skipped."""
    if ref is None and est is None:
        return True         # N == N
    if ref is None:
        return False        # ref=N, est=chord
    ref_q = _majmin_quality(ref.quality)
    if ref_q is None:
        return None         # ref quality outside majmin vocab → skip
    if est is None:
        return False        # est=N, ref=maj/min
    est_q = _majmin_quality(est.quality)
    if est_q is None:
        return False
    return ref.root_pc == est.root_pc and ref_q == est_q


def match_majmin_inv(ref: Optional[ParsedChord], est: Optional[ParsedChord]) -> Optional[bool]:
    """Majmin + bass note (inversion)."""
    base = match_majmin(ref, est)
    if base is None or not base:
        return base
    # Both are None (N==N) or both matched as maj/min with same root+quality.
    if ref is None:         # N == N, no bass to compare
        return True
    return ref.bass_pc == est.bass_pc  # type: ignore[union-attr]


def match_mirex(ref: Optional[ParsedChord], est: Optional[ParsedChord]) -> Optional[bool]:
    """MIREX: pitch-class set overlap ≥ 3; N==N correct, N vs chord wrong."""
    if ref is None and est is None:
        return True
    if ref is None or est is None:
        return False
    return len(_chord_pc_set(ref) & _chord_pc_set(est)) >= 3


VOCABS: List[str] = ["root", "majmin", "majmin_inv", "mirex"]
MATCH_FNS: Dict[str, MatchFn] = {
    "root":        match_root,
    "majmin":      match_majmin,
    "majmin_inv":  match_majmin_inv,
    "mirex":       match_mirex,
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> List[ChordRow]:
    rows: List[ChordRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt or txt.startswith("#"):
                continue
            parts = txt.split("\t")
            if len(parts) < 4:
                continue
            track_id = parts[0].strip()
            seg_range = parts[1].strip()
            ref_chord = parts[2].strip()
            est_chord = parts[3].strip()
            block_consistency = float(parts[4].strip()) if len(parts) >= 5 else -1.0

            m = re.match(r"^(\d+)-(\d+)$", seg_range)
            if not m:
                continue
            cs, ce = int(m.group(1)), int(m.group(2))
            if ce <= cs:
                continue

            rows.append(ChordRow(
                track_id=track_id,
                cs=cs,
                ce=ce,
                ref_chord=ref_chord,
                est_chord=est_chord,
                block_consistency=block_consistency,
            ))
    return rows

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    rows: List[ChordRow],
) -> Tuple[
    Dict[str, float],   # correct frames per vocab
    Dict[str, float],   # total (evaluated) frames per vocab
    Dict[str, float],   # skipped frames per vocab
    Dict[str, Dict[str, float]],  # per-track correct
    Dict[str, Dict[str, float]],  # per-track total
]:
    correct: Dict[str, float] = defaultdict(float)
    total:   Dict[str, float] = defaultdict(float)
    skip:    Dict[str, float] = defaultdict(float)
    t_correct: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    t_total:   Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for r in rows:
        n = float(r.ce - r.cs)
        # mir_eval excludes 'X' (unknown) reference frames from all metrics.
        if _is_unknown_ref(r.ref_chord):
            for vocab in VOCABS:
                skip[vocab] += n
            continue
        ref = parse_chord(r.ref_chord)
        est = parse_chord(r.est_chord)
        for vocab in VOCABS:
            result = MATCH_FNS[vocab](ref, est)
            if result is None:
                skip[vocab] += n
            else:
                total[vocab] += n
                t_total[r.track_id][vocab] += n
                if result:
                    correct[vocab] += n
                    t_correct[r.track_id][vocab] += n

    return correct, total, skip, t_correct, t_total

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate chord predictions: WCSR under root / majmin / majmin_inv / mirex."
    )
    ap.add_argument("--result-file", type=str, required=True,
                    help="Chord output TSV from infer_key --chord-out-file.")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Frames per second used to convert frame counts to seconds (default: 10).")
    ap.add_argument("--per-track", action="store_true",
                    help="Print per-track WCSR breakdown.")
    return ap.parse_args()


def _wcsr(correct: float, total: float) -> float:
    return (100.0 * correct / total) if total > 0 else 0.0


def main():
    args = parse_args()
    fps = float(args.fps)
    path = Path(args.result_file).resolve()
    rows = load_rows(path)

    if not rows:
        print("[WARN] No rows loaded.")
        return

    rows_no_n = [r for r in rows if not (r.ref_chord.strip().upper() in NO_CHORD_STRS)]

    correct,    total,    skip,    t_correct,    t_total    = evaluate(rows)
    correct_nn, total_nn, skip_nn, t_correct_nn, t_total_nn = evaluate(rows_no_n)

    total_frames = sum(r.ce - r.cs for r in rows)
    total_sec = total_frames / fps
    n_ref_n_frames = sum(r.ce - r.cs for r in rows if r.ref_chord.strip().upper() in NO_CHORD_STRS)
    n_tracks = len(set(r.track_id for r in rows))

    print(f"[INFO] File   : {path.name}")
    print(f"[INFO] Blocks : {len(rows)}  ({len(rows_no_n)} non-N ref)  |  "
          f"Tracks: {n_tracks}  |  Duration: {total_sec:.1f}s  |  FPS: {fps}")
    print(f"[INFO] N-ref duration excluded: {n_ref_n_frames / fps:.1f}s")

    def _print_table(label: str, c_map, t_map, s_map):
        print()
        print(f"[SUMMARY] {label}")
        print(f"  {'vocab':<14} {'correct (s)':>12} {'total (s)':>12} {'skip (s)':>10} {'WCSR':>8}")
        print("  " + "-" * 60)
        for vocab in VOCABS:
            c, t, s = c_map[vocab], t_map[vocab], s_map[vocab]
            print(
                f"  {vocab:<14} {c/fps:>12.1f} {t/fps:>12.1f} {s/fps:>10.1f} "
                f"{_wcsr(c, t):>7.2f}%"
            )

    _print_table("WCSR — all blocks (including N-ref)", correct, total, skip)
    _print_table("WCSR — non-N ref only", correct_nn, total_nn, skip_nn)

    if args.per_track:
        def _print_per_track(label: str, tc, tt):
            if not tt:
                return
            print()
            print(f"[SUMMARY] Per-track WCSR (%) — {label}")
            col_w = max(30, max(len(tid) for tid in tt) + 2)
            header = f"  {'track_id':<{col_w}}" + "".join(f"{v:>14}" for v in VOCABS)
            print(header)
            print("  " + "-" * (col_w + 14 * len(VOCABS)))
            for tid in sorted(tt.keys()):
                row_str = f"  {tid:<{col_w}}" + "".join(
                    f"{_wcsr(tc[tid][v], tt[tid][v]):>13.2f}%"
                    for v in VOCABS
                )
                print(row_str)

        _print_per_track("all blocks", t_correct, t_total)
        _print_per_track("non-N ref only", t_correct_nn, t_total_nn)

    # Song-level: macro-average of per-track WCSR (non-N ref)
    if t_total_nn:
        macro: Dict[str, float] = {}
        for vocab in VOCABS:
            vals = [
                _wcsr(t_correct_nn[tid].get(vocab, 0.0), t_total_nn[tid].get(vocab, 0.0))
                for tid in t_total_nn
                if t_total_nn[tid].get(vocab, 0.0) > 0
            ]
            macro[vocab] = sum(vals) / len(vals) if vals else 0.0
        n_tracks_nn = len(t_total_nn)
        print(f"\n[SONG-LEVEL WCSR] Macro-average per-track (non-N ref, {n_tracks_nn} tracks)")
        for vocab in VOCABS:
            print(f"  {vocab:<14} {macro[vocab]:>7.2f}%")


if __name__ == "__main__":
    main()
