#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared key / chord decision helpers used by the step-4 inference drivers, the
linear-probe drivers, the training-time validation metrics and the audio
inference scripts (src/inference).

  * single-ring key-signature estimation  (estimate_key_from_ring)
  * major/minor joint template scoring    (decide_joint_major_minor, App. D.3)
  * segment-level chord decoding          (predict_chord_segments, App. D.1)
  * boundary-peak primitives              (detect_peaks_with_drop, thin_times)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch


NOTE_NAMES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE2PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}


def decode_str_scalar(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def decode_str_array(a: np.ndarray) -> np.ndarray:
    return np.array([decode_str_scalar(v) for v in a], dtype=object)


def parse_feature_groups(path: Path) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    int_pat = re.compile(r"-?\d+")
    tag_pat = re.compile(r"\[([^\[\]]+)\]")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            tags = tag_pat.findall(raw)
            if not tags:
                continue
            group_name = tags[0].strip().lower()
            rhs = raw.split(":", 1)[1] if ":" in raw else raw
            nums = [int(x) for x in int_pat.findall(rhs)]
            if not nums:
                continue
            if len(nums) >= 2 and nums[0] == len(nums) - 1:
                nums = nums[1:]
            if group_name == "chord boundary":
                groups[group_name] = nums
            else:
                seen: Set[int] = set()
                uniq: List[int] = []
                for x in nums:
                    if x not in seen:
                        seen.add(x)
                        uniq.append(x)
                groups[group_name] = uniq
    return groups


def chord_root_pc(chord: str) -> Optional[int]:
    c = chord.strip()
    if not c or c.upper() in {"N", "NOCHORD"}:
        return None
    root = c.split(":", 1)[0].strip()
    m = re.match(r"^([A-Ga-g])([#b]?)$", root)
    if not m:
        return None
    letter = m.group(1).upper()
    acc = m.group(2)
    if acc == "b":
        acc = "B"
    return NOTE2PC.get((letter + acc).upper(), None)


def parse_key_signature_major_pc(label: str) -> Optional[int]:
    s = str(label).strip()
    if not s or s.upper() == "N/A":
        return None
    parts = s.split(":", 1)
    root = parts[0].strip()
    mode = parts[1].strip().lower() if len(parts) > 1 else "maj"
    root_pc = chord_root_pc(root)
    if root_pc is None:
        return None
    if mode.startswith("min"):
        return (int(root_pc) + 3) % 12
    return int(root_pc) % 12


def ref_segment_signature_from_key_frame(
    key_frame: Optional[np.ndarray],
) -> Tuple[Optional[int], bool, bool]:
    if key_frame is None or key_frame.size == 0:
        return None, False, False
    vals = [str(x).strip() for x in key_frame.tolist()]
    vals = [v for v in vals if v and v.upper() != "N"]
    if not vals:
        return None, False, False
    sigs: List[int] = []
    for v in vals:
        sig = parse_key_signature_major_pc(v)
        if sig is not None:
            sigs.append(int(sig))
    if not sigs:
        return None, False, False
    uniq = sorted(set(sigs))
    if len(uniq) > 1:
        return None, True, True
    return uniq[0], False, True


def estimate_ref_key_from_frame(key_frame: Optional[np.ndarray]) -> str:
    if key_frame is None or key_frame.size == 0:
        return "N/A"
    vals = [str(x).strip() for x in key_frame.tolist()]
    vals = [v for v in vals if v and v.upper() != "N"]
    if not vals:
        return "N/A"
    acc: Dict[str, int] = {}
    for v in vals:
        acc[v] = acc.get(v, 0) + 1
    return max(acc.items(), key=lambda kv: kv[1])[0]


# Sentinel written to the ref_key column when a segment's frames span more than
# one distinct key (a within-segment modulation).  estimate_ref_key_from_frame
# collapses such a segment to its majority label, which hides the change from
# the song-level stable-key filter (eval_metrics) and lets key-changing songs
# leak in.  Downstream stable-key filtering treats any song with a CHANGE-tagged
# segment as key-changing, matching eval_skey's "more than one distinct key
# region" criterion at frame resolution.
KEY_CHANGE_TAG = "CHANGE"


def ref_key_label_for_segment(key_frame: Optional[np.ndarray]) -> str:
    """Majority key label for the segment, or KEY_CHANGE_TAG if its frames span
    more than one distinct (root, mode) key.

    Frame labels are piecewise-constant projections of the annotation regions, so
    >1 distinct non-N label inside a segment means a real within-segment key
    change — the segment has no single reference key.
    """
    if key_frame is None or key_frame.size == 0:
        return "N/A"
    vals = [str(x).strip() for x in key_frame.tolist()]
    vals = [v for v in vals if v and v.upper() != "N"]
    if not vals:
        return "N/A"
    acc: Dict[str, int] = {}
    for v in vals:
        acc[v] = acc.get(v, 0) + 1
    if len(acc) > 1:
        return KEY_CHANGE_TAG
    return next(iter(acc))


def segment_time_range_str(g: h5py.Group, seg_idx: int, t_frames: int) -> str:
    start_sec = 0.0
    end_sec = 0.0
    if "start" in g:
        start_sec = float(g["start"][seg_idx])
    if "end" in g:
        end_sec = float(g["end"][seg_idx])
    elif "labels" in g and "segment_sec" in g["labels"].attrs:
        end_sec = start_sec + float(g["labels"].attrs["segment_sec"])
    else:
        end_sec = start_sec + max(1.0, float(max(1, t_frames)))
    return f"{start_sec:.3f}-{end_sec:.3f}"


def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32)
    k = np.ones(int(win), dtype=np.float32) / float(win)
    pad = int(win) // 2
    xpad = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    y = np.convolve(xpad, k, mode="valid")
    return y.astype(np.float32)


def smooth_cols(mat: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return mat.astype(np.float32)
    out = np.zeros_like(mat, dtype=np.float32)
    for k in range(mat.shape[1]):
        out[:, k] = moving_average_1d(mat[:, k], win)
    return out


def thin_times(times: List[int], t_len: int, max_labels: int = 120) -> List[int]:
    if not times:
        return []
    times = sorted(set(int(t) for t in times if 0 <= t < t_len))
    if len(times) <= max_labels:
        return times
    step = max(1, len(times) // max_labels)
    return times[::step]


def detect_peaks_with_drop(
    x: np.ndarray,
    tol: float = 0.2,
    min_rise: float = 0.0,
    min_drop: float = 0.0,
) -> List[int]:
    x = np.asarray(x, dtype=np.float32)
    t_len = int(x.shape[0])
    if t_len < 3:
        return []
    d = x[1:] - x[:-1]
    up = d >= -tol
    down = d <= tol
    peaks: List[int] = []
    t = 0
    while t < t_len - 1:
        while t < t_len - 1 and not up[t]:
            t += 1
        if t >= t_len - 1:
            break
        i = t
        while t < t_len - 1 and up[t]:
            t += 1
        m = t
        if m >= t_len - 1:
            break
        if not down[m]:
            continue
        while t < t_len - 1 and down[t]:
            t += 1
        j = t
        seg = x[i : j + 1]
        p = int(i + np.argmax(seg))
        if (x[p] - x[i] >= min_rise) and (x[p] - x[j] >= min_drop):
            peaks.append(p)
    return peaks


def _segments_from_boundaries(boundaries: List[int], t_len: int) -> List[Tuple[int, int]]:
    if t_len <= 0:
        return []
    b = [0] + sorted(set(int(x) for x in boundaries if 0 < x < t_len)) + [t_len]
    out: List[Tuple[int, int]] = []
    for s, e in zip(b[:-1], b[1:]):
        if e > s:
            out.append((s, e))
    return out


def argmax_id_and_score(mat_tk: np.ndarray, ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    j = mat_tk.argmax(axis=1)
    best_score = mat_tk[np.arange(mat_tk.shape[0]), j]
    best_id = np.array([ids[int(x)] for x in j], dtype=np.int32)
    return best_id, best_score.astype(np.float32)


def weighted_vote_id(ids_arr: np.ndarray, scs_arr: np.ndarray) -> Optional[int]:
    if ids_arr.size == 0:
        return None
    acc: Dict[int, float] = {}
    for fid, sc in zip(ids_arr.tolist(), scs_arr.tolist()):
        fid_i = int(fid)
        acc[fid_i] = acc.get(fid_i, 0.0) + float(sc)
    return int(max(acc.items(), key=lambda kv: kv[1])[0])


def _segment_vote_labels(
    major_id_t: np.ndarray,
    minor_id_t: np.ndarray,
    major_sc_t: np.ndarray,
    minor_sc_t: np.ndarray,
    segs: List[Tuple[int, int]],
    silence_sc_t: Optional[np.ndarray] = None,
    major_mat_t: Optional[np.ndarray] = None,
    minor_mat_t: Optional[np.ndarray] = None,
    qual_mode: str = "sum_peak",
) -> List[Tuple[int, str]]:
    """Vote a (root_id, quality) label per segment.

    Quality (maj/min) decision is controlled by ``qual_mode``:
      - "sum_peak"      : (default, original) compare summed per-frame peak
                          scores  sum(major_sc_t) vs sum(minor_sc_t).
      - "seg_mean_max"  : average each ring's [T,12] activation over the
                          segment frames -> two 12-vectors, compare their max.
      - "seg_mean_mean" : same segment-mean 12-vectors, compare their mean.

    The "seg_mean_*" modes require ``major_mat_t`` / ``minor_mat_t`` (each
    [T, 12], pre- or post-smoothing as chosen by the caller). They only affect
    the maj/min choice; silence detection and root-note voting are unchanged.
    """
    use_seg_mean = (
        qual_mode in ("seg_mean_max", "seg_mean_mean")
        and major_mat_t is not None
        and minor_mat_t is not None
    )
    out: List[Tuple[int, str]] = []
    for s, e in segs:
        maj_sum = float(np.sum(major_sc_t[s:e]))
        min_sum = float(np.sum(minor_sc_t[s:e]))
        sil_sum = float(np.sum(silence_sc_t[s:e])) if silence_sc_t is not None else 0.0
        if silence_sc_t is not None and sil_sum > maj_sum and sil_sum > min_sum:
            out.append((-1, "N"))
            continue
        major_ids = major_id_t[s:e].astype(np.int32)
        major_scs = major_sc_t[s:e].astype(np.float32)
        minor_ids = minor_id_t[s:e].astype(np.int32)
        minor_scs = minor_sc_t[s:e].astype(np.float32)
        if major_ids.size == 0 or minor_ids.size == 0:
            continue

        if use_seg_mean:
            maj_vec = major_mat_t[s:e].mean(axis=0)  # [12]
            min_vec = minor_mat_t[s:e].mean(axis=0)  # [12]
            if qual_mode == "seg_mean_max":
                is_maj = float(maj_vec.max()) >= float(min_vec.max())
            else:  # seg_mean_mean
                is_maj = float(maj_vec.mean()) >= float(min_vec.mean())
        else:
            is_maj = maj_sum >= min_sum

        if is_maj:
            qual, ids, scs = "maj", major_ids, major_scs
        else:
            qual, ids, scs = "min", minor_ids, minor_scs
        best_id = weighted_vote_id(ids, scs)
        if best_id is None:
            continue
        out.append((best_id, qual))
    return out


def _predict_segment_raw_tmpl(
    M: np.ndarray, m: np.ndarray,
    conf_thr: float = 0.18, alpha: float = 0.4,
) -> Tuple[int, str]:
    """Joint root+quality from a segment's major (M) / minor (m) mean 12-vectors.

    Take the direct ring argmax over the 24 (root x maj/min) raw scores; if the
    top-2 margin is confident (>= ``conf_thr``) keep it, otherwise blend the
    normalized raw score with a normalized triad-template score over the
    combined PCP ``M + m`` (weight ``alpha``).  The template for root r sums the
    PCP at the root, third (major +4 / minor +3) and fifth (+7).

    Returns ``(root_pc, "maj"|"min")`` where ``root_pc`` indexes NOTE_NAMES_12.
    """
    pc = M + m
    raw  = np.empty(24, dtype=np.float64)   # [:12] maj roots, [12:] min roots
    tmpl = np.empty(24, dtype=np.float64)
    for r in range(12):
        raw[r]       = M[r]
        raw[12 + r]  = m[r]
        tmpl[r]      = pc[r] + pc[(r + 4) % 12] + pc[(r + 7) % 12]  # major triad
        tmpl[12 + r] = pc[r] + pc[(r + 3) % 12] + pc[(r + 7) % 12]  # minor triad
    top  = np.sort(raw)[::-1]
    conf = (top[0] - top[1]) / (top[0] + 1e-8)
    if conf >= conf_thr:
        final = raw
    else:
        final = raw / (raw.max() + 1e-8) + alpha * (tmpl / (tmpl.max() + 1e-8))
    k = int(final.argmax())
    return k % 12, ("maj" if k < 12 else "min")


def predict_chord_segments(
    z: np.ndarray,
    major_cols: List[int],
    minor_cols: List[int],
    chord_segs: List[Tuple[int, int]],
    *,
    silence_sc_t: Optional[np.ndarray] = None,
    chord_frame_seg: Optional[np.ndarray] = None,
    track: str = "",
    t_z: Optional[int] = None,
    f_chord=None,
    smooth_win: int = 1,
    qual_mode: str = "sum_peak",
    conf_thr: float = 0.18,
    alpha: float = 0.4,
) -> List[Tuple[int, int, str, str]]:
    """Smoothing-free segment-level chord inference shared by every driver.

    Given one decoded segment's activations ``z`` plus its ``chord_segs``
    (boundaries) and ``silence_sc_t`` (silence), this performs the maj/min ring
    argmax, segment voting and output formatting.  Callers differ *only* in how
    they derive ``chord_segs`` (SAE boundary-feature peaks vs. ground-truth
    chord-label changes) and ``silence_sc_t`` (SAE silence feature vs. GT N/X);
    keeping the prediction here is what stops the offline drivers and the
    training-time ``ChordRingMetric`` from drifting apart.

    The ring activations are temporally smoothed with a ``smooth_win`` moving
    average before any decision (``smooth_win <= 1`` is a no-op); mode and root
    are then decided on the smoothed activations.  ``qual_mode``:
      - ``sum_peak`` / ``seg_mean_max`` / ``seg_mean_mean``: two-stage decision
        (maj/min then weighted-vote root) via :func:`_segment_vote_labels`.
      - ``raw_tmpl``: joint root+quality from the segment-mean ring vectors with
        confidence-gated triad-template fallback (:func:`_predict_segment_raw_tmpl`,
        tuned by ``conf_thr`` / ``alpha``).  Silence still wins as in the others.

    Writes full segment chords to ``f_chord`` when provided and always returns
    the rows ``[(cs, ce, ref_chord_str, est_chord), ...]`` so callers can format
    or score them.  Requires ``len(major_cols) >= 12`` and
    ``len(minor_cols) >= 12`` (otherwise an empty list is returned).
    """
    if t_z is None:
        t_z = int(z.shape[0])
    if len(major_cols) < 12 or len(minor_cols) < 12:
        return []
    # Smooth the ring activations first; every downstream mode/root decision
    # (argmax peaks, seg-mean vectors, raw-tmpl) reads the smoothed values.
    major_act = smooth_cols(z[:, major_cols], smooth_win)
    minor_act = smooth_cols(z[:, minor_cols], smooth_win)
    major_best_id, major_best_score = argmax_id_and_score(major_act, major_cols)
    minor_best_id, minor_best_score = argmax_id_and_score(minor_act, minor_cols)

    # Per-segment estimated chord label, aligned with chord_segs.
    est_by_seg: List[str] = []
    if qual_mode == "raw_tmpl":
        for (cs, ce) in chord_segs:
            # Silence rule identical to _segment_vote_labels (per-frame peak sums).
            maj_sum = float(major_best_score[cs:ce].sum())
            min_sum = float(minor_best_score[cs:ce].sum())
            sil_sum = float(silence_sc_t[cs:ce].sum()) if silence_sc_t is not None else 0.0
            if silence_sc_t is not None and sil_sum > maj_sum and sil_sum > min_sum:
                est_by_seg.append("N")
                continue
            M = major_act[cs:ce].mean(axis=0)
            m = minor_act[cs:ce].mean(axis=0)
            r, q = _predict_segment_raw_tmpl(M, m, conf_thr, alpha)
            est_by_seg.append(f"{NOTE_NAMES_12[r]}:{q}")
    else:
        voted = _segment_vote_labels(
            major_id_t=major_best_id, minor_id_t=minor_best_id,
            major_sc_t=major_best_score, minor_sc_t=minor_best_score,
            segs=chord_segs, silence_sc_t=silence_sc_t,
            major_mat_t=major_act, minor_mat_t=minor_act,
            qual_mode=qual_mode,
        )
        for (v_fid, v_qual) in voted:
            if v_qual == "N":
                est_by_seg.append("N")
            else:
                ring_for_chord = major_cols if v_qual == "maj" else minor_cols
                est_by_seg.append(f"{_note_label_for_id(ring_for_chord, int(v_fid))}:{v_qual}")

    results: List[Tuple[int, int, str, str]] = []
    for (cs, ce), est_chord in zip(chord_segs, est_by_seg):
        ref_chord_label = ref_chord_label_for_seg(chord_frame_seg, cs, ce)
        ref_chord_str   = ref_chord_label if ref_chord_label is not None else "N"
        results.append((cs, ce, ref_chord_str, est_chord))
        if f_chord is not None:
            f_chord.write(f"{track}\t{cs}-{ce}\t{ref_chord_str}\t{est_chord}\n")
    return results


def ref_chord_root_for_seg(
    chord_frame: Optional[np.ndarray], s: int, e: int
) -> Optional[int]:
    label = ref_chord_label_for_seg(chord_frame, s, e)
    return chord_root_pc(label) if label is not None else None


def ref_chord_label_for_seg(
    chord_frame: Optional[np.ndarray], s: int, e: int
) -> Optional[str]:
    if chord_frame is None or e <= s:
        return None
    seg = chord_frame[s:e]
    acc: Dict[str, int] = {}
    for v in seg.tolist():
        v = str(v).strip()
        if v and v.upper() not in {"N", "NOCHORD"}:
            acc[v] = acc.get(v, 0) + 1
    if not acc:
        return None
    return max(acc.items(), key=lambda kv: kv[1])[0]


def _note_label_for_id(ring_ids_12: List[int], fid: int) -> str:
    try:
        idx = ring_ids_12.index(int(fid))
        return NOTE_NAMES_12[idx % 12]
    except ValueError:
        return f"[{int(fid)}]"


def standardize_to_BTD_batch(feat: torch.Tensor, t_candidates: List[int]) -> torch.Tensor:
    if feat.ndim == 4 and int(feat.shape[1]) == 1:
        feat = feat.squeeze(1)
    if feat.ndim == 3:
        _, d1, d2 = feat.shape
        for t_ref in t_candidates:
            if d1 == t_ref:
                return feat
            if d2 == t_ref:
                return feat.transpose(1, 2)
        return feat if d1 >= d2 else feat.transpose(1, 2)
    if feat.ndim == 2:
        d1, d2 = feat.shape
        for t_ref in t_candidates:
            if d1 == t_ref:
                return feat.unsqueeze(0)
            if d2 == t_ref:
                return feat.transpose(0, 1).unsqueeze(0)
        return (feat if d1 >= d2 else feat.transpose(0, 1)).unsqueeze(0)
    raise RuntimeError(f"Unsupported feature shape: {feat.shape}")


def rotate_ring12(ids: "List[int]", shift: int) -> "List[int]":
    """Rotate a 12-root ring order: new[p] = ids[(p + shift) % 12].

    Ring orders are sub-SAE-relative (the same id responds on sub-SAE i to a
    root i semitones higher), so converting a ring between sub-SAE views is a
    pure rotation of the id list.  Elements beyond the first 12 pass through.
    """
    ring = list(ids[:12])
    return [ring[(p + shift) % 12] for p in range(12)] + list(ids[12:])


def estimate_key_from_ring(
    z: np.ndarray,
    ring_cols: List[int],
    degree_semitones: int,
) -> Tuple[str, float, np.ndarray]:
    """
    Estimate the key from a single ring (forth / fifth / tonic).

    degree_semitones: semitones above tonic that this ring targets.
      0 = tonic (I), 5 = subdominant (IV / forth), 7 = dominant (V / fifth).

    Ring position p is in chromatic order from C, so:
      tonic_pc = (p - degree_semitones + 12) % 12
    """
    ring_mean = z[:, ring_cols].mean(axis=0)  # (12,) simple time-average

    top1_pos = int(np.argmax(ring_mean))
    tonic_pc = (top1_pos - degree_semitones + 12) % 12
    est_key = f"{NOTE_NAMES_12[tonic_pc]}:maj"
    est_score = float(ring_mean[top1_pos])

    key_scores_12 = np.zeros(12, dtype=np.float32)
    for _i in range(12):
        key_scores_12[(_i - degree_semitones + 12) % 12] = ring_mean[_i]

    return est_key, est_score, key_scores_12


def estimate_minor_key_from_ring(
    z: np.ndarray,
    ring_cols: List[int],
    degree_semitones: int = 0,
) -> Tuple[str, float, np.ndarray]:
    """
    Estimate a minor-mode key from the [Minor Chord] ring.

    The minor-chord ring is reordered (step2) so that position p targets the
    p:min chord.  ``degree_semitones`` says which scale degree of the minor key
    that prominent minor chord is assumed to be — i.e. how many semitones the
    chord root sits above the minor tonic:
      0 = tonic (i),  5 = subdominant (iv / forth),  7 = dominant (v / fifth).
    The minor tonic is therefore  (p - degree_semitones) % 12  and the estimate
    is reported as "<root>:min".

    For song-level accumulation the activations are written in *relative-major*
    pitch-class space — the key-signature space used by the metric (A:min and
    C:maj both map to pc 0).  A minor tonic t is written to slot (t + 3) % 12,
    so position p lands in slot (p - degree_semitones + 3) % 12.
    """
    ring_mean = z[:, ring_cols].mean(axis=0)  # (12,) simple time-average

    top1_pos = int(np.argmax(ring_mean))
    minor_tonic_pc = (top1_pos - degree_semitones + 12) % 12
    est_key = f"{NOTE_NAMES_12[minor_tonic_pc]}:min"
    est_score = float(ring_mean[top1_pos])

    key_scores_12 = np.zeros(12, dtype=np.float32)
    for _i in range(12):
        key_scores_12[(_i - degree_semitones + 3 + 12) % 12] = ring_mean[_i]

    return est_key, est_score, key_scores_12


def estimate_joint_key_from_rings(
    z: np.ndarray,
    ring_specs: List[Tuple[List[int], str, int]],
) -> Tuple[str, float, np.ndarray]:
    """
    Joint key estimation: stack several rings' time-averaged activations and
    take a single argmax across all of them.

    ring_specs: list of (ring_cols, quality, degree_semitones)
      quality "maj": ring position p targets a major chord 'degree' semitones
                     above the tonic → tonic / key-signature pc = (p - degree) % 12
                     (degree 0 = Major Chord ring, 5 = Forth, 7 = Fifth, ...).
      quality "min": ring position p targets the p:min chord → minor tonic pc p,
                     relative-major (key-signature) pc = (p + 3) % 12.

    The winner keeps its true quality, so the estimate may come out as either
    "<root>:maj" or "<root>:min".  ``key_scores_12`` lives in key-signature
    (relative-major) space; when several rings map to the same key-signature
    slot the larger activation is kept (mirroring the global argmax), so that
    song-level argmax over summed activations matches the segment-level winner.
    """
    scores: List[float] = []
    meta: List[Tuple[int, int, str]] = []   # (keysig_pc, root_pc, quality)
    for ring_cols, quality, degree in ring_specs:
        ring_mean = z[:, ring_cols].mean(axis=0)   # (12,) simple time-average
        for p in range(12):
            scores.append(float(ring_mean[p]))
            if quality == "min":
                root_pc = p
                keysig_pc = (p + 3) % 12
            else:
                root_pc = (p - degree + 12) % 12
                keysig_pc = root_pc
            meta.append((keysig_pc, root_pc, quality))

    w = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    keysig_pc, root_pc, quality = meta[w]
    est_key = f"{NOTE_NAMES_12[root_pc]}:{'min' if quality == 'min' else 'maj'}"
    est_score = float(scores[w])

    # key_scores_12 is the max-combined activation in key-signature space (used
    # for the merged score); maj_scores_12 / min_scores_12 keep the two sides
    # apart so callers can pool them separately at the song level.
    key_scores_12 = np.zeros(12, dtype=np.float32)
    maj_scores_12 = np.zeros(12, dtype=np.float32)
    min_scores_12 = np.zeros(12, dtype=np.float32)
    for val, (ks, _root, qual) in zip(scores, meta):
        if val > float(key_scores_12[ks]):
            key_scores_12[ks] = val
        side = min_scores_12 if qual == "min" else maj_scores_12
        if val > float(side[ks]):
            side[ks] = val
    return est_key, est_score, key_scores_12, maj_scores_12, min_scores_12


def decide_maj_min_by_degrees(
    maj_mean: np.ndarray,
    min_mean: np.ndarray,
    maj_degree: int,
) -> Tuple[int, int, float, float]:
    """Decide major vs minor by comparing tonic+subdominant+dominant (I/IV/V) sums.

    maj_mean / min_mean are raw ring time-means in chord-root space (position p =
    chord rooted at pitch class p).  maj_degree is the semitone offset of the
    major-side ring (0 = Major Chord ring, 5 = Forth ring); the minor ring is a
    tonic ring (degree 0).

    For each candidate the tonic stays on its own ring, but the subdominant (IV)
    and dominant (V) roots take the stronger of the minor-ring vs major-side-ring
    activation at that root (the IV/V chord may be either quality):
      xv = maj[x] + max(maj[x+5], min[x+5]) + max(maj[x+7], min[x+7])
      pv = min[p] + max(min[p+5], maj[p+5]) + max(min[p+7], maj[p+7])
    with x = (argmax(maj_mean) - maj_degree) % 12 and p = argmax(min_mean).
    Returns (root_pc, is_minor, xv, pv); major wins ties.
    """
    q = int(np.argmax(maj_mean))
    x = (q - maj_degree + 12) % 12
    p = int(np.argmax(min_mean))
    xv = float(
        maj_mean[x]
        + max(maj_mean[(x + 5) % 12], min_mean[(x + 5) % 12])
        + max(maj_mean[(x + 7) % 12], min_mean[(x + 7) % 12])
    )
    pv = float(
        min_mean[p]
        + max(min_mean[(p + 5) % 12], maj_mean[(p + 5) % 12])
        + max(min_mean[(p + 7) % 12], maj_mean[(p + 7) % 12])
    )
    if xv >= pv:
        return x, 0, xv, pv
    return p, 1, xv, pv


def decide_maj_min_3ring(
    maj_mean: np.ndarray,
    min_mean: np.ndarray,
    forth_mean: np.ndarray,
) -> Tuple[int, int, float, float]:
    """Major vs minor via I+IV+V sums using three rings (all in chord-root space).

    The major and minor hypotheses each keep their tonic/dominant on their own
    chord ring; the major hypothesis takes its subdominant (IV) from the forth
    ring (which specialises in subdominant detection), while the minor hypothesis
    uses its own ring throughout:
      xv = maj[x] + forth[x+5] + maj[x+7]      (major chord ring + forth ring)
      pv = min[p] + min[p+5]   + min[p+7]      (minor chord ring only)
    with x = argmax(maj_mean), p = argmax(min_mean).  The forth ring fires at the
    subdominant pitch class, so the IV root of key x is read directly at x+5.
    Returns (root_pc, is_minor, xv, pv); major wins ties.
    """
    x = int(np.argmax(maj_mean))
    p = int(np.argmax(min_mean))
    xv = float(maj_mean[x] + forth_mean[(x + 5) % 12] + maj_mean[(x + 7) % 12])
    pv = float(min_mean[p] + min_mean[(p + 5) % 12] + min_mean[(p + 7) % 12])
    if xv >= pv:
        return x, 0, xv, pv
    return p, 1, xv, pv


# ── Multi-SAE (0/1/2) tonic voting ────────────────────────────────────────────
# Each frame is decoded by sub-SAEs 0/1/2 simultaneously, giving three ring time-
# means per ring.  The same musical ring is pitch-rotated by the sub-SAE index, so
# the candidate tonic from sub-SAE i is corrected by +i semitones.  Candidates
# (root, mode, activation) from every participating ring are pooled and the most
# frequent (root, mode) wins; ties are broken by the larger summed activation.

def _ring_vote_cands(
    means_per_sae: "List[np.ndarray]",
    degree: int,
    is_minor: bool,
    offsets: "Tuple[int, ...]",
) -> "List[Tuple[int, int, float]]":
    """One (root, mode, activation) candidate per sub-SAE, with the +index shift.

    root = (argmax(ring_mean) - degree + offset) % 12; degree maps the ring's
    peak to the tonic (0 = tonic ring, 5 = forth ring).
    """
    cands: "List[Tuple[int, int, float]]" = []
    for off, m in zip(offsets, means_per_sae):
        a = int(np.argmax(m))
        root = (a - degree + off + 12) % 12
        cands.append((root, 1 if is_minor else 0, float(m[a])))
    return cands


def _sig_vote_cands(
    means_per_sae: "List[np.ndarray]",
    sig_shift: int,
    offsets: "Tuple[int, ...]" = (0, 1, 2),
) -> "List[Tuple[int, int, float]]":
    """One key-signature candidate per sub-SAE (mode slot is a dummy 0).

    Maps a ring's per-SAE tonic into key-signature space (relative-major pc):
    major ring uses sig_shift=0 (tonic == signature), minor ring uses sig_shift=3
    (signature = tonic + 3).  This lets major/minor candidates that imply the same
    signature reinforce each other instead of splitting a (root, mode) vote.
    """
    cands: "List[Tuple[int, int, float]]" = []
    for off, m in zip(offsets, means_per_sae):
        a = int(np.argmax(m))
        sig = (a + off + sig_shift) % 12
        cands.append((sig, 0, float(m[a])))
    return cands


def _tally_vote(cands: "List[Tuple[int, int, float]]") -> Tuple[int, int, float]:
    """Majority vote over (root, mode); ties broken by larger summed activation."""
    cnt: Dict[Tuple[int, int], int] = {}
    act: Dict[Tuple[int, int], float] = {}
    for r, md, a in cands:
        k = (r, md)
        cnt[k] = cnt.get(k, 0) + 1
        act[k] = act.get(k, 0.0) + a
    best = max(cnt.values())
    tied = [k for k in cnt if cnt[k] == best]
    win = tied[0] if len(tied) == 1 else max(tied, key=lambda k: act[k])
    return win[0], win[1], act[win]


def key_from_vote2(
    majside_means: "List[np.ndarray]",
    min_means: "List[np.ndarray]",
    maj_degree: int,
    offsets: "Tuple[int, ...]" = (0, 1, 2),
) -> Tuple[int, int, float]:
    """Two-ring 3-SAE vote: major-side ring + minor ring decide mode AND signature.

    maj_degree is the major-side ring degree (0 = major chord, 5 = forth).
    Returns (tonic_pc, is_minor, score).
    """
    cands = (_ring_vote_cands(majside_means, maj_degree, False, offsets)
             + _ring_vote_cands(min_means, 0, True, offsets))
    return _tally_vote(cands)


def key_from_vote3(
    maj_means: "List[np.ndarray]",
    min_means: "List[np.ndarray]",
    forth_means: "List[np.ndarray]",
    offsets: "Tuple[int, ...]" = (0, 1, 2),
) -> Tuple[int, int, float]:
    """Three-ring 3-SAE vote: forth+minor decide the MODE, major+minor the SIGNATURE.

    The two votes are combined into the final tonic.  Returns (tonic_pc, is_minor,
    score) where score is the signature vote's winning activation.
    """
    _mode_root, mode_is_min, _ = _tally_vote(
        _ring_vote_cands(forth_means, 5, False, offsets)
        + _ring_vote_cands(min_means, 0, True, offsets))
    # Signature vote in key-signature space: major shifts by 0 (tonic == sig),
    # minor by +3, so maj/min candidates implying the same signature combine.
    sig_pc, _sd, sig_score = _tally_vote(
        _sig_vote_cands(maj_means, 0, offsets)
        + _sig_vote_cands(min_means, 3, offsets))
    tonic = sig_pc if mode_is_min == 0 else (sig_pc + 9) % 12
    return tonic, mode_is_min, sig_score


def key_from_vote3_avg(
    maj_means: "List[np.ndarray]",
    min_means: "List[np.ndarray]",
    forth_means: "List[np.ndarray]",
    offsets: "Tuple[int, ...]" = (0, 1, 2),
) -> Tuple[int, int, float]:
    """Three-ring 3-SAE avg: forth-vs-minor decide MODE, the major ring the root.

    Each ring is L1-normalized per sub-SAE, realigned (sub-SAE k rolled by +offset_k)
    and averaged into one 12-vector.  MODE is the forth_minor_tonic comparison: major
    if the forth ring's averaged peak >= the minor ring's (major wins ties).  The key
    signature is the major-chord ring's averaged argmax (degree 0); the tonic is that
    signature for major, or its relative minor (+9) for minor.  In the gt setting
    (mode known) the merged signature therefore depends only on the major ring.
    Returns (tonic_pc, is_minor, score) with score the major ring's averaged peak.
    """
    def _avg(vecs: "List[np.ndarray]") -> "np.ndarray":
        acc = np.zeros(12, dtype=float)
        for k in range(len(offsets)):
            v = np.asarray(vecs[k], dtype=float)
            s = float(np.sum(v))
            if s > 0:
                acc += np.roll(v / s, offsets[k])
        return acc / len(offsets)

    forth_avg = _avg(forth_means)
    min_avg = _avg(min_means)
    maj_avg = _avg(maj_means)
    is_min = 0 if float(forth_avg.max()) >= float(min_avg.max()) else 1
    sig = int(np.argmax(maj_avg))
    tonic = sig if is_min == 0 else (sig + 9) % 12
    return tonic, is_min, float(maj_avg[sig])


# ── Human-readable vote traces (for --debug-vote in step4_infer) ───────────────

def _fmt_cands(means_per_sae, degree, is_minor, offsets=(0, 1, 2)):
    """Return (candidate list, formatted 'sae_i->ROOT:mode(act)' string)."""
    cands = _ring_vote_cands(means_per_sae, degree, is_minor, offsets)
    mode = "min" if is_minor else "maj"
    s = "  ".join(
        f"sae{i}(+{offsets[i]})->{NOTE_NAMES_12[r]}:{mode}({a:.4f})"
        for i, (r, _m, a) in enumerate(cands)
    )
    return cands, s


def _fmt_tally(cands) -> str:
    """Format the vote tally '(ROOT:mode xCount act=sum)' sorted by count then act."""
    cnt: Dict[Tuple[int, int], int] = {}
    act: Dict[Tuple[int, int], float] = {}
    for r, md, a in cands:
        k = (r, md)
        cnt[k] = cnt.get(k, 0) + 1
        act[k] = act.get(k, 0.0) + a
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], -act[kv[0]]))
    return "  ".join(
        f"{NOTE_NAMES_12[k[0]]}:{'min' if k[1] else 'maj'} x{c}(act={act[k]:.4f})"
        for k, c in items
    )


def debug_vote2(maj_means, min_means, maj_degree, offsets=(0, 1, 2)) -> str:
    """Multi-line trace of a vote2 decision: candidates, tally, winner."""
    mc, ms = _fmt_cands(maj_means, maj_degree, False, offsets)
    pc, ps = _fmt_cands(min_means, 0, True, offsets)
    tonic, is_min, score = key_from_vote2(maj_means, min_means, maj_degree, offsets)
    win = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
    return (
        f"    maj cands (deg={maj_degree}): {ms}\n"
        f"    min cands           : {ps}\n"
        f"    tally               : {_fmt_tally(mc + pc)}\n"
        f"    => winner {win} (score={score:.4f})"
    )


def _fmt_sig_cands(means_per_sae, sig_shift, offsets=(0, 1, 2)):
    """Return (sig candidate list, 'sae_i->SIG(act)' string) in key-sig space."""
    cands = _sig_vote_cands(means_per_sae, sig_shift, offsets)
    s = "  ".join(
        f"sae{i}(+{offsets[i]})->sig {NOTE_NAMES_12[sig]}({a:.4f})"
        for i, (sig, _m, a) in enumerate(cands)
    )
    return cands, s


def debug_vote3(maj_means, min_means, forth_means, offsets=(0, 1, 2)) -> str:
    """Multi-line trace of a vote3 decision: mode vote, signature vote, winner."""
    fc, fs = _fmt_cands(forth_means, 5, False, offsets)
    _pc_mode, ps_mode = _fmt_cands(min_means, 0, True, offsets)
    msc, mss = _fmt_sig_cands(maj_means, 0, offsets)   # major -> signature (+0)
    psc, pss = _fmt_sig_cands(min_means, 3, offsets)   # minor -> signature (+3)
    tonic, is_min, score = key_from_vote3(maj_means, min_means, forth_means, offsets)
    win = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
    return (
        f"    [mode] forth cands  : {fs}\n"
        f"    [mode] min   cands  : {ps_mode}\n"
        f"    [mode] tally        : {_fmt_tally(fc + _pc_mode)}\n"
        f"    [sig]  maj   cands  : {mss}\n"
        f"    [sig]  min   cands  : {pss}\n"
        f"    [sig]  tally        : {_fmt_tally(msc + psc)}\n"
        f"    => winner {win} (score={score:.4f})"
    )


def decide_maj_min_multisae(
    maj_vecs: "List[np.ndarray]",
    min_vecs: "List[np.ndarray]",
    offsets: "Tuple[int, ...]" = (0, 1, 2),
    maj_degree: int = 0,
    min_degree: int = 0,
) -> Tuple[int, int, float]:
    """Major vs minor by averaging three sub-SAEs (s0/s1/s2) per ring (avg scheme).

    maj_vecs / min_vecs are length-3 lists of raw (un-normalized) 12-vectors, one
    per sub-SAE, for the major-side ring and the minor-side ring respectively.

    Each of the 6 vectors is L1-normalized (divided by its own sum), then the
    three sub-SAEs of a ring are realigned (sub-SAE k rolled by +offset_k, since
    its tonic reads +k semitones high) and averaged into one 12-vector per ring:
      avg_m[q] = (1/K) Σ_k m_norm[k][(q - offset_k) % 12]
    The global argmax of each averaged ring gives that ring's peak; the larger
    peak wins (major breaks ties).  The winning peak position is mapped to a tonic
    via the ring degree (0 = tonic ring, 5 = forth ring): root = (argmax - degree).
    Unlike per-sub-SAE candidate scoring this searches all 12 roots, so a cross-SAE
    "consensus" root that is no single sub-SAE's argmax can win.  Returns
    (root_pc, is_minor, peak).
    """
    def _avg(vecs: "List[np.ndarray]") -> "np.ndarray":
        acc = np.zeros(12, dtype=float)
        for k in range(len(offsets)):
            v = np.asarray(vecs[k], dtype=float)
            s = float(np.sum(v))
            if s > 0:
                acc += np.roll(v / s, offsets[k])     # align sub-SAE k by +offset_k
        return acc / len(offsets)

    maj_avg = _avg(maj_vecs)
    min_avg = _avg(min_vecs)
    x = int(np.argmax(maj_avg))
    p = int(np.argmax(min_avg))
    xv, pv = float(maj_avg[x]), float(min_avg[p])
    if xv >= pv:
        return (x - maj_degree) % 12, 0, xv
    return (p - min_degree) % 12, 1, pv


# ── Major/minor joint template scoring ───────────────────────────────────────
# Current major_minor_joint model:
#   s1 right-shift 1, s2 right-shift 2 -> raw average -> x**0.50 ->
#   separate L1 normalization -> compact multi-template scoring ->
#   direct 24-class decision.

def _joint_align_avg_raw(
    sae_blocks: "List[np.ndarray]",
    offsets: "Tuple[int, ...]" = (0, 1, 2),
) -> np.ndarray:
    acc = np.zeros(12, dtype=np.float64)
    n = len(offsets)
    for k, off in enumerate(offsets):
        v = np.asarray(sae_blocks[k], dtype=np.float64)
        acc += np.roll(v, int(off))
    return acc / float(n)


def joint_major_minor_norms(
    maj_blocks: "List[np.ndarray]",
    min_blocks: "List[np.ndarray]",
    offsets: "Tuple[int, ...]" = (0, 1, 2),
    power: float = 0.50,
) -> "Tuple[np.ndarray, np.ndarray]":
    maj = _joint_align_avg_raw(maj_blocks, offsets)
    mn = _joint_align_avg_raw(min_blocks, offsets)
    maj = np.power(np.maximum(maj, 0.0), float(power))
    mn = np.power(np.maximum(mn, 0.0), float(power))
    sm = float(np.sum(maj))
    sn = float(np.sum(mn))
    maj = maj / sm if sm > 0 else np.zeros(12, dtype=np.float64)
    mn = mn / sn if sn > 0 else np.zeros(12, dtype=np.float64)
    return maj, mn


def joint_major_minor_scores(
    maj: np.ndarray,
    mn: np.ndarray,
) -> "Tuple[np.ndarray, np.ndarray]":
    W45 = 0.4
    W26 = 0.1

    Smaj = np.zeros(12, dtype=np.float64)
    Smin = np.zeros(12, dtype=np.float64)
    for X in range(12):
        maj_base = (
            1.0 * maj[X]
            + W45 * maj[(X + 5) % 12]   # IV
            + W45 * maj[(X + 7) % 12]   # V
        )
        Smaj[X] = max(
            maj_base + W26 * mn[(X + 9) % 12],  # vi
            maj_base + W26 * mn[(X + 2) % 12],  # ii
        )

        smin_nat = (
            1.0 * mn[X]
            + W45 * mn[(X + 5) % 12]    # iv
            + W45 * mn[(X + 7) % 12]    # v
            + W26 * maj[(X + 8) % 12]   # bVI
        )
        smin_harm = (
            1.0 * mn[X]
            + W45 * mn[(X + 5) % 12]    # iv
            + W45 * maj[(X + 7) % 12]   # V
            + W26 * maj[(X + 8) % 12]   # bVI
        )
        smin_melodic = (
            1.0 * mn[X]
            + W26 * mn[(X + 2) % 12]    # ii
            + W45 * maj[(X + 5) % 12]   # IV
            + W45 * maj[(X + 7) % 12]   # V
        )
        Smin[X] = max(smin_nat, smin_harm, smin_melodic)
    return Smaj, Smin


def joint_major_minor_decide(
    Smaj: np.ndarray,
    Smin: np.ndarray,
) -> "Tuple[int, int, float]":
    scores24 = np.concatenate([Smaj, Smin])
    idx = int(np.argmax(scores24))
    if idx < 12:
        return idx, 0, float(Smaj[idx])
    root = idx - 12
    return root, 1, float(Smin[root])


def joint_major_minor_decide_gt(
    Smaj: np.ndarray,
    Smin: np.ndarray,
    gt_is_min: bool,
) -> "Tuple[int, int, float]":
    if gt_is_min:
        root = int(np.argmax(Smin))
        return root, 1, float(Smin[root])
    root = int(np.argmax(Smaj))
    return root, 0, float(Smaj[root])


def decide_joint_major_minor(
    maj_blocks: "List[np.ndarray]",
    min_blocks: "List[np.ndarray]",
    offsets: "Tuple[int, ...]" = (0, 1, 2),
    gt_is_min: "Optional[bool]" = None,
) -> "Tuple[int, int, float]":
    maj, mn = joint_major_minor_norms(maj_blocks, min_blocks, offsets)
    Smaj, Smin = joint_major_minor_scores(maj, mn)
    if gt_is_min is None:
        return joint_major_minor_decide(Smaj, Smin)
    return joint_major_minor_decide_gt(Smaj, Smin, bool(gt_is_min))

