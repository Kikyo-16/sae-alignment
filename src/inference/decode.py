#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task decoders shared by SAE and linear-probe audio inference.

Every decision below calls the code path that produced the paper numbers:

key-signature  (Table 2, App. D.2; relative major/minor merged)
    segment: ``analysis.infer_key.estimate_key_from_ring`` on the time-averaged
             12-d ring -> argmax -> minus the scale-degree offset.
    song:    sum of the per-segment 12-d key-signature vectors -> argmax
             (``analysis.eval_metrics.compute_key_metrics``, single-ring files).

key-chord      (Table 3, App. D.3; 24 major/minor keys)
    segment: per-sub-SAE frame sums of the major/minor chord rings ->
             ``eval_metrics._joint_norms`` (+1/+2 roll, sqrt, L1) ->
             ``_joint_scores`` (template scores) -> ``_joint_decide``.
    song:    the 24 template scores averaged over segments -> ``_joint_decide``.

chord          (Table 1, App. D.1)
    ``analysis.infer_key.predict_chord_segments`` on the chord rings with either
    estimated boundaries (SAE boundary feature, ``boundary_utils``) or
    reference boundaries from a ``.lab`` file (Oracle Bd).

The evaluation scripts read intermediate activations back from text files
written with 6 decimals; ``_r6`` reproduces that rounding so the song-level
aggregation is numerically identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..analysis.eval_metrics import _joint_decide, _joint_norms, _joint_scores
from ..analysis.infer_key import NOTE_NAMES_12, estimate_key_from_ring
from ..feature_extraction.label_providers import _parse_time_seg_file, _segs_to_frames
from .audio import SEGMENT_SEC, AudioSegment

TASK_KEY_SIGNATURE = "key-signature"
TASK_KEY_CHORD = "key-chord"
TASK_CHORD = "chord"
TASKS = (TASK_KEY_SIGNATURE, TASK_KEY_CHORD, TASK_CHORD)
LEVELS = ("segment", "song")

# Scale degree of a 12-d ring, in semitones above the tonic.
DEGREE_BY_NAME = {"tonic": 0, "forth": 5, "fifth": 7}

# Defaults of the paper's batch evaluation (scripts/analyze/run_chord_gt_boundary.sh).
CHORD_DEFAULTS = dict(
    smooth_win=9,
    majmin_mode="seg-mean-max",
    majmin_conf_thr=0.18,
    majmin_alpha=0.4,
    boundary_peak_tol=0.2,
    boundary_peak_drop=0.0,
    boundary_min_gap=0,
    boundary_score_floor=0.0,
    boundary_max_labels=120,
)


def _r6(x) -> float:
    return float(f"{float(x):.6f}")


def signature_label(pc: int) -> str:
    """Key signature as '<major>:maj/<relative minor>:min', e.g. 'G:maj/E:min'."""
    return f"{NOTE_NAMES_12[pc % 12]}:maj/{NOTE_NAMES_12[(pc + 9) % 12]}:min"


def key_label(root: int, is_min: int) -> str:
    return f"{NOTE_NAMES_12[root % 12]}:{'min' if is_min else 'maj'}"


@dataclass
class Row:
    start_sec: float
    end_sec: float
    label: str


def kept_span(seg: AudioSegment) -> Tuple[float, float]:
    """Time span of the frames whose results are kept for this segment."""
    return seg.frame_time(seg.keep_from_frame()), seg.end_sec


# ---------------------------------------------------------------------------
# key-signature (single 12-d ring)
# ---------------------------------------------------------------------------

class KeySignatureDecoder:
    def __init__(self, degree: int) -> None:
        self.degree = int(degree)
        self.rows: List[Row] = []
        self.act_sum = [0.0] * 12
        self.n_seg = 0

    def add(self, seg: AudioSegment, ring_act: np.ndarray) -> None:
        """ring_act: [T_kept, 12] ring activations in ring (pitch-class) order."""
        est_key, _score, key_scores_12 = estimate_key_from_ring(ring_act, list(range(12)), self.degree)
        sig = NOTE_NAMES_12.index(est_key.split(":", 1)[0])
        self.rows.append(Row(*kept_span(seg), signature_label(sig)))
        for i, v in enumerate(key_scores_12.tolist()):
            self.act_sum[i] += _r6(v)
        self.n_seg += 1

    def song(self) -> Optional[str]:
        if self.n_seg == 0:
            return None
        pred = int(max(range(12), key=lambda i: self.act_sum[i]))
        return signature_label(pred)


# ---------------------------------------------------------------------------
# key-chord (major + minor chord rings, joint templates)
# ---------------------------------------------------------------------------

class KeyChordDecoder:
    def __init__(self) -> None:
        self.rows: List[Row] = []
        self.smaj_sum = [0.0] * 12
        self.smin_sum = [0.0] * 12
        self.n_seg = 0

    def add(self, seg: AudioSegment, maj_blocks: Sequence[np.ndarray], min_blocks: Sequence[np.ndarray]) -> None:
        """maj_blocks / min_blocks: three 12-d frame sums (sub-SAE 0/1/2).

        A single-decoder model (s-SAE, chord probe) passes its sums in block 0
        and zeros in blocks 1/2, as the paper's 1-SAE / probe drivers do.
        """
        maj = [[_r6(v) for v in np.asarray(b).tolist()] for b in maj_blocks]
        mn = [[_r6(v) for v in np.asarray(b).tolist()] for b in min_blocks]
        maj_n, min_n = _joint_norms(maj, mn)
        smaj, smin = _joint_scores(maj_n, min_n)
        root, is_min = _joint_decide(smaj, smin)
        self.rows.append(Row(*kept_span(seg), key_label(root, is_min)))
        for x in range(12):
            self.smaj_sum[x] += smaj[x]
            self.smin_sum[x] += smin[x]
        self.n_seg += 1

    def song(self) -> Optional[str]:
        if self.n_seg == 0:
            return None
        smaj_avg = [s / self.n_seg for s in self.smaj_sum]
        smin_avg = [s / self.n_seg for s in self.smin_sum]
        root, is_min = _joint_decide(smaj_avg, smin_avg)
        return key_label(root, is_min)


# ---------------------------------------------------------------------------
# chord recognition
# ---------------------------------------------------------------------------

class ChordCollector:
    def __init__(self) -> None:
        self.rows: List[Row] = []

    def add(self, seg: AudioSegment, chord_rows: Sequence[Tuple[int, int, str, str]]) -> None:
        """chord_rows: output of predict_chord_segments, (cs, ce, ref, est) in frames."""
        keep = seg.keep_from_frame()
        for cs, ce, _ref, est in chord_rows:
            if ce <= keep:
                continue
            cs = max(int(cs), keep)
            self.rows.append(Row(seg.frame_time(cs), seg.frame_time(int(ce)), str(est)))


def load_lab(path: Path) -> List[Tuple[float, float, str]]:
    """Read a 'start end label' chord annotation (.lab) file."""
    return _parse_time_seg_file(Path(path))


def lab_frames(lab: List[Tuple[float, float, str]], seg: AudioSegment) -> np.ndarray:
    """Frame-level labels of one segment, sampled like the dataset label providers."""
    t_len = seg.n_frames
    return _segs_to_frames(lab, seg.start_sec, SEGMENT_SEC, t_len / SEGMENT_SEC, t_len)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def write_key_rows(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# start_sec\tend_sec\tkey\n")
        for r in rows:
            fh.write(f"{r.start_sec:.3f}\t{r.end_sec:.3f}\t{r.label}\n")


def write_lab(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(f"{r.start_sec:.3f}\t{r.end_sec:.3f}\t{r.label}\n")


def write_results(
    out_dir: Path,
    stem: str,
    level: str,
    key_signature: Optional[KeySignatureDecoder] = None,
    key_chord: Optional[KeyChordDecoder] = None,
    chord: Optional[ChordCollector] = None,
    audio_end_sec: float = 0.0,
) -> List[Path]:
    """Write one file per task; returns the written paths."""
    written: List[Path] = []
    for name, dec in ((TASK_KEY_SIGNATURE, key_signature), (TASK_KEY_CHORD, key_chord)):
        if dec is None:
            continue
        if level == "segment":
            rows = dec.rows
        else:
            song = dec.song()
            rows = [Row(0.0, audio_end_sec, song)] if song is not None else []
        p = Path(out_dir) / f"{stem}.{name}.{level}.tsv"
        write_key_rows(p, rows)
        written.append(p)
        print(f"[{stem}] {name} ({level}):")
        for r in rows:
            print(f"    {r.start_sec:8.2f} - {r.end_sec:8.2f}  {r.label}")
    if chord is not None:
        p = Path(out_dir) / f"{stem}.chord.lab"
        write_lab(p, chord.rows)
        written.append(p)
        print(f"[{stem}] chord: {len(chord.rows)} segments -> {p}")
    return written
