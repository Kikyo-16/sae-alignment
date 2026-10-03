#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified metrics computation for key and chord inference results.

Key evaluation preserves the original inference and aggregation logic exactly:

    python -m src.analysis.eval_metrics key RESULT_FILE
    python -m src.analysis.eval_metrics key RESULT_FILE \
        --maj-degree 5 --vote2-decision avg

Chord evaluation scores cached predictions against the original frame-level
HDF5 chord annotations:

    python -m src.analysis.eval_metrics chord CACHE_FILE \
        --h5-path DATASET.h5 --split test

For compatibility, chord-only options may also precede the subcommand:

    python -m src.analysis.eval_metrics \
        --h5-path DATASET.h5 --split test chord CACHE_FILE

The program prints one JSON object to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

import h5py
import numpy as np

try:
    import mir_eval
except ImportError:  # Key evaluation itself does not require mir_eval.
    mir_eval = None

from ..evaluation.eval_key_from_tsv import (
    NOTE2PC,
    NOTE_NAMES_12,
    load_rows,
    parse_key_signature_major_pc,
)

# Sentinel value written to the ref_key column for a within-segment key change.
# Kept in sync with src.analysis.infer_key.KEY_CHANGE_TAG (duplicated here so
# this metrics module stays free of infer_key's heavy torch/h5py imports).
KEY_CHANGE_TAG = "CHANGE"


def _parse_key_root_mode(label: str) -> Optional[Tuple[int, int]]:
    """Parse a key label to (root_pc, is_minor) WITHOUT merging relative keys.

    Enharmonics are still merged (C# == Db).  Returns None for N / N/A / junk.
    "C:maj" → (0, 0),  "A:min" → (9, 1).
    """
    s = str(label).strip()
    if not s or s.upper() in {"N", "N/A"}:
        return None
    parts = s.split(":", 1)
    root = parts[0].strip()
    mode = parts[1].strip().lower() if len(parts) > 1 else "maj"
    m = re.match(r"^([A-Ga-g])([#b]?)$", root)
    if not m:
        return None
    letter = m.group(1).upper()
    acc = m.group(2)
    if acc == "b":
        acc = "B"
    root_pc = NOTE2PC.get((letter + acc).upper(), None)
    if root_pc is None:
        return None
    return (int(root_pc) % 12, 1 if mode.startswith("min") else 0)


def _mirex_weighted_score(
    ref: Optional[Tuple[int, int]],
    est: Optional[Tuple[int, int]],
) -> float:
    """MIREX audio key-detection weighted score (mir_eval.key.weighted_score).

    ref / est are (tonic_pc, is_minor).  Returns:
        1.0  same key (tonic + mode)
        0.5  perfect fifth (same mode, est tonic = ref tonic +/- 7)
        0.3  relative major/minor (maj→min at +9, min→maj at +3)
        0.2  parallel major/minor (same tonic, opposite mode)
        0.0  otherwise (or missing estimate)
    """
    if ref is None or est is None:
        return 0.0
    r_tonic, r_min = ref
    e_tonic, e_min = est
    if e_tonic == r_tonic and e_min == r_min:
        return 1.0
    if e_min == r_min and ((e_tonic - r_tonic) % 12 in (7, 5)):
        return 0.5
    if r_min == 0 and e_min == 1 and e_tonic == (r_tonic + 9) % 12:
        return 0.3
    if r_min == 1 and e_min == 0 and e_tonic == (r_tonic + 3) % 12:
        return 0.3
    if e_min != r_min and e_tonic == r_tonic:
        return 0.2
    return 0.0


def _decide_majmin_by_degrees(
    maj_pool: List[float],
    min_pool: List[float],
    maj_degree: int,
) -> Tuple[int, int]:
    """Major-vs-minor by I+IV+V activation sums (see infer_key.decide_maj_min_by_degrees).

    maj_pool / min_pool are raw ring activations in chord-root space; maj_degree
    is the maj-side ring degree (0=major chord, 5=forth).  Returns (root_pc,
    is_minor), major wins ties.
    """
    q = max(range(12), key=lambda i: maj_pool[i])
    x = (q - maj_degree + 12) % 12
    p = max(range(12), key=lambda i: min_pool[i])
    xv = (maj_pool[x]
          + max(maj_pool[(x + 5) % 12], min_pool[(x + 5) % 12])
          + max(maj_pool[(x + 7) % 12], min_pool[(x + 7) % 12]))
    pv = (min_pool[p]
          + max(min_pool[(p + 5) % 12], maj_pool[(p + 5) % 12])
          + max(min_pool[(p + 7) % 12], maj_pool[(p + 7) % 12]))
    return (x, 0) if xv >= pv else (p, 1)


def _decide_majmin_3ring(
    maj_pool: List[float],
    min_pool: List[float],
    forth_pool: List[float],
) -> Tuple[int, int]:
    """3-ring major-vs-minor (see infer_key.decide_maj_min_3ring).

    Major hypothesis takes its IV from the forth ring, minor hypothesis uses its
    own ring throughout (all in chord-root space):
      xv = maj[x] + forth[x+5] + maj[x+7]
      pv = min[p] + min[p+5]   + min[p+7]
    Returns (root_pc, is_minor), major wins ties.
    """
    x = max(range(12), key=lambda i: maj_pool[i])
    p = max(range(12), key=lambda i: min_pool[i])
    xv = maj_pool[x] + forth_pool[(x + 5) % 12] + maj_pool[(x + 7) % 12]
    pv = min_pool[p] + min_pool[(p + 5) % 12] + min_pool[(p + 7) % 12]
    return (x, 0) if xv >= pv else (p, 1)


# ── 3-SAE tonic voting (see infer_key.key_from_vote2 / key_from_vote3) ─────────
# means_per_sae is a list of 3 pooled ring vectors (sub-SAE 0/1/2); the candidate
# tonic from sub-SAE i is corrected by +i semitones.

def _ring_vote_cands(means_per_sae, degree, is_minor, offsets=(0, 1, 2)):
    out = []
    for off, m in zip(offsets, means_per_sae):
        a = max(range(12), key=lambda i: m[i])
        root = (a - degree + off + 12) % 12
        out.append((root, 1 if is_minor else 0, m[a]))
    return out


def _tally_vote(cands) -> Tuple[int, int]:
    cnt: Dict[Tuple[int, int], int] = {}
    act: Dict[Tuple[int, int], float] = {}
    for r, md, a in cands:
        k = (r, md)
        cnt[k] = cnt.get(k, 0) + 1
        act[k] = act.get(k, 0.0) + a
    best = max(cnt.values())
    tied = [k for k in cnt if cnt[k] == best]
    win = tied[0] if len(tied) == 1 else max(tied, key=lambda k: act[k])
    return win[0], win[1]


def _key_from_vote2(majside_means, min_means, maj_degree) -> Tuple[int, int]:
    """Returns (tonic_pc, is_minor); major-side + minor ring decide both."""
    return _tally_vote(_ring_vote_cands(majside_means, maj_degree, False)
                       + _ring_vote_cands(min_means, 0, True))


def _sig_vote_cands(means_per_sae, sig_shift, offsets=(0, 1, 2)):
    """Key-signature-space candidates (major sig_shift=0, minor sig_shift=3)."""
    out = []
    for off, m in zip(offsets, means_per_sae):
        a = max(range(12), key=lambda i: m[i])
        out.append(((a + off + sig_shift) % 12, 0, m[a]))
    return out


def _key_from_vote3(maj_means, min_means, forth_means) -> Tuple[int, int]:
    """Returns (tonic_pc, is_minor); forth+min decide mode, major+min signature.

    The signature vote is in key-signature space so major/minor candidates that
    imply the same signature combine (major shifts +0, minor +3).
    """
    _mr, mode_is_min = _tally_vote(_ring_vote_cands(forth_means, 5, False)
                                   + _ring_vote_cands(min_means, 0, True))
    sig_pc, _sd = _tally_vote(_sig_vote_cands(maj_means, 0)
                              + _sig_vote_cands(min_means, 3))
    tonic = sig_pc if mode_is_min == 0 else (sig_pc + 9) % 12
    return tonic, mode_is_min


def _key_from_vote3_avg(maj_means, min_means, forth_means, offsets=(0, 1, 2)) -> Tuple[int, int]:
    """Avg version (see infer_key.key_from_vote3_avg): forth-vs-minor averaged peaks
    decide MODE (major ties); the major-chord ring's averaged argmax is the key
    signature; tonic = signature (maj) or signature+9 (min)."""
    def _avg(vecs):
        n = len(offsets)
        acc = [0.0] * 12
        for k in range(n):
            v = vecs[k]
            s = float(sum(v))
            if s > 0:
                for i in range(12):
                    acc[(i + offsets[k]) % 12] += v[i] / s
        return [a / n for a in acc]

    forth_avg = _avg(forth_means)
    min_avg = _avg(min_means)
    maj_avg = _avg(maj_means)
    is_min = 0 if max(forth_avg) >= max(min_avg) else 1
    sig = max(range(12), key=lambda i: maj_avg[i])
    return (sig if is_min == 0 else (sig + 9) % 12), is_min


def _decide_majmin_multisae(maj_vecs, min_vecs, offsets=(0, 1, 2),
                            maj_degree=0, min_degree=0) -> Tuple[int, int]:
    """Align-average across sub-SAEs, global argmax (see infer_key.decide_maj_min_multisae).

    maj_vecs / min_vecs: 3 raw per-sub-SAE 12-vectors each.  Each is L1-normalized
    and rolled by +offset_k, then averaged into one vector per ring; the ring with
    the larger averaged global argmax wins (major ties).  The peak position maps to
    a tonic via the ring degree: root = (argmax - degree).  Returns (tonic_pc, is_minor).
    """
    def _avg(vecs):
        n = len(offsets)
        acc = [0.0] * 12
        for k in range(n):
            v = vecs[k]
            s = float(sum(v))
            if s > 0:
                for i in range(12):
                    acc[(i + offsets[k]) % 12] += v[i] / s
        return [a / n for a in acc]

    maj_avg = _avg(maj_vecs)
    min_avg = _avg(min_vecs)
    x = max(range(12), key=lambda i: maj_avg[i])
    p = max(range(12), key=lambda i: min_avg[i])
    if maj_avg[x] >= min_avg[p]:
        return (x - maj_degree) % 12, 0
    return (p - min_degree) % 12, 1


# ---------------------------------------------------------------------------
# Joint scoring (major_minor_joint): refined combined-minor template
# ---------------------------------------------------------------------------
# Current default model:
#   1) roll s1/s2 by +1/+2 and raw-average the three sub-SAEs,
#   2) power-compress with power=0.50,
#   3) L1-normalize major and minor rings separately,
#   4) use compact multi-template scoring:
#        Smaj = max(I + W45*IV + W45*V + W26*vi,
#                   I + W45*IV + W45*V + W26*ii)
#        Smin = max(i + W45*iv + W45*v + W26*bVI,
#                   i + W45*iv + W45*V + W26*bVI,
#                   i + W26*ii + W45*IV + W45*V)
#      with W45=0.4, W26=0.1,
#   5) direct 24-class decision.
# ---------------------------------------------------------------------------

def _align_avg_raw(sae_blocks, offsets=(0, 1, 2)) -> List[float]:
    """Roll each RAW sub-SAE block by +offset_k (cross-SAE pitch correction) and
    average — no per-sub-SAE normalization, so ring magnitude is preserved."""
    n = len(offsets)
    acc = [0.0] * 12
    for k in range(n):
        v = sae_blocks[k]
        for i in range(12):
            acc[(i + offsets[k]) % 12] += float(v[i])
    return [a / n for a in acc]


def _joint_norms(maj_blocks, min_blocks, offsets=(0, 1, 2),
                 power=0.50) -> Tuple[List[float], List[float]]:
    """maj/min ring vectors for the refined joint scoring.

    Merge sub-SAEs (shift-average raw), power-compress (x**0.50 by default),
    then L1-normalize each ring SEPARATELY.
    """
    maj = _align_avg_raw(maj_blocks, offsets)
    mn = _align_avg_raw(min_blocks, offsets)
    maj = [max(x, 0.0) ** power for x in maj]
    mn = [max(x, 0.0) ** power for x in mn]
    sm = float(sum(maj))
    sp = float(sum(mn))
    maj = [x / sm for x in maj] if sm > 0 else [0.0] * 12
    mn = [x / sp for x in mn] if sp > 0 else [0.0] * 12
    return maj, mn


def _joint_scores(maj, mn) -> Tuple[List[float], List[float]]:
    """Per-root major/minor scores for the compact multi-template joint model.

    Major: max(I+IV+V+vi, I+IV+V+ii).
    Minor: max(natural, harmonic, melodic) using only major/minor rings.
    """
    W45 = 0.4     # 4/5 support; same weight everywhere
    W26 = 0.1     # 2/6 support; same weight everywhere

    Smaj = [0.0] * 12
    Smin = [0.0] * 12
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


def _joint_decide(Smaj, Smin) -> Tuple[int, int]:
    """Direct 24-class key decision -> (root_pc, is_minor)."""
    scores24 = list(Smaj) + list(Smin)
    idx = max(range(24), key=lambda i: scores24[i])
    if idx < 12:
        return idx, 0
    return idx - 12, 1


def _joint_decide_gt(Smaj, Smin, gt_is_min: int) -> Tuple[int, int]:
    """gt oracle: the root is the argmax score WITHIN the known gt mode.

    gt major → (argmax(Smaj), maj); gt minor → (argmax(Smin), min). No signature-
    first step — the search is confined to the gt mode's own scores.
    """
    if gt_is_min:
        return max(range(12), key=lambda x: Smin[x]), 1
    return max(range(12), key=lambda x: Smaj[x]), 0

# ---------------------------------------------------------------------------
# Key metrics (computed directly from the TSV)
# ---------------------------------------------------------------------------

def compute_key_metrics(result_file: Path, maj_degree: int = 0,
                        vote2_decision: str = "tally") -> Dict:
    rows = load_rows(result_file)

    # Group valid-reference rows by track so we can drop key-changing songs.
    # A CHANGE-tagged ref_key marks a segment whose frames span >1 distinct key
    # (a within-segment modulation, written by ref_key_label_for_segment).  The
    # per-segment majority labels below would hide such a change, so any track
    # with even one CHANGE segment is recorded as key-changing and dropped.
    track_items: Dict[str, List[Tuple[object, int]]] = defaultdict(list)
    changed_tracks: set = set()
    for r in rows:
        if str(r.ref_label).strip().upper() == KEY_CHANGE_TAG:
            changed_tracks.add(r.track_id)
            continue
        ref_sig = parse_key_signature_major_pc(r.ref_label)
        if ref_sig is None:
            continue
        track_items[r.track_id].append((r, int(ref_sig)))

    # Joint files carry a second (minor-side) activation block; when present the
    # song-level prediction pools maj and min separately and compares the peaks.
    # The 3-ring joint file adds a third (forth-ring) block.
    has_min = any(getattr(r, "min_act_12", None) is not None for r in rows)
    has_forth = any(getattr(r, "forth_act_12", None) is not None for r in rows)
    # 3-SAE vote files store per-sub-SAE ring means (maj_sae always; forth_sae for
    # vote3).  The song level pools each sub-SAE's ring mean across segments, then
    # re-votes.
    has_vote2 = any(getattr(r, "maj_sae", None) is not None for r in rows)
    has_vote3 = any(getattr(r, "forth_sae", None) is not None for r in rows)
    # Template scoring (major + minor ring norms → 48 key/mode candidates) drives
    # both segment and song prediction for the joints major_minor_tonic
    # (vote2 layout + --vote2-decision template) and major_forth_minor_tonic
    # (vote3 layout; forth block ignored).  gt files (vote2 layout + 'avg') keep
    # their own decision.  Segment predictions are recomputed here from the cached
    # ring blocks, so the cached est_key column (old logic) is not used.
    # major_minor_joint: power=0.40 combined-minor joint templates,
    # signature-first decision; song level averages the 24 segment scores. joint_gt
    # is the gt oracle: the root is the argmax within the known gt mode.
    use_joint = has_vote2 and vote2_decision in ("joint", "joint_gt")

    # Segment-level accumulators (stable-key songs only).
    seg_total = seg_hit = seg_hit_strict = 0
    seg_maj = seg_min = 0
    seg_hit_maj = seg_hit_min = 0            # merged-correct, split by gt mode
    seg_predmaj_in_maj = seg_predmin_in_min = 0  # predicted-mode matches gt mode
    seg_mirex_sum = 0.0                      # MIREX weighted score, summed
    # Song-level accumulators.
    song_total = song_hit = song_hit_strict = 0
    song_maj = song_min = 0
    song_hit_maj = song_hit_min = 0
    song_predmaj_in_maj = song_predmin_in_min = 0
    song_mirex_sum = 0.0

    for track, items in track_items.items():
        # Within-segment modulation flagged upstream → key-changing, drop song.
        if track in changed_tracks:
            continue
        # Stable-key songs only: a single *full* reference key (root + mode)
        # across segments.  Do not only check key signature: relative-major/minor
        # changes (e.g. C:maj <-> A:min) must also be treated as key-changing.
        ref_rm_unique = {rm for (r, _sig) in items
                         for rm in [_parse_key_root_mode(r.ref_label)]
                         if rm is not None}
        if len(ref_rm_unique) != 1:
            continue
        if len({sig for (_r, sig) in items}) != 1:
            continue
        ref_sig = items[0][1]

        act_sum = [0.0] * 12      # major-side activation pool (or merged, if no min block)
        min_sum = [0.0] * 12      # minor-side activation pool (joint files only)
        forth_sum = [0.0] * 12    # forth-ring activation pool (3-ring joint only)
        # Per-sub-SAE ring pools (3-SAE vote files), 3 blocks of 12 each.
        maj_sae_sum   = [[0.0] * 12 for _ in range(3)]
        min_sae_sum   = [[0.0] * 12 for _ in range(3)]
        forth_sae_sum = [[0.0] * 12 for _ in range(3)]
        est_mode_votes = [0, 0]   # predicted modes [maj, min]
        ref_mode_counts = [0, 0]  # reference modes  [maj, min]
        ref_rm_set = set()
        smaj_sum = [0.0] * 12                                  # joint: summed Smaj over segs
        smin_sum = [0.0] * 12                                  # joint: summed Smin over segs
        joint_nseg = 0

        for (r, _sig) in items:
            # The joint variant recomputes the segment label from the cached ring
            # blocks (the stored est_key column is from the old logic, unused here).
            if use_joint and getattr(r, "maj_sae", None) is not None \
                    and getattr(r, "min_sae", None) is not None:
                maj_n, min_n = _joint_norms(r.maj_sae, r.min_sae)
                Smaj, Smin = _joint_scores(maj_n, min_n)
                if vote2_decision == "joint_gt":
                    gt_rm = _parse_key_root_mode(r.ref_label)
                    j_root, j_min = _joint_decide_gt(
                        Smaj, Smin, gt_rm[1] if gt_rm is not None else 0)
                else:
                    j_root, j_min = _joint_decide(Smaj, Smin)
                est_label = f"{NOTE_NAMES_12[j_root]}:{'min' if j_min else 'maj'}"
                for X in range(12):
                    smaj_sum[X] += Smaj[X]
                    smin_sum[X] += Smin[X]
                joint_nseg += 1
            else:
                est_label = r.est_label
            est_sig = parse_key_signature_major_pc(est_label)
            merged_ok = est_sig is not None and est_sig == ref_sig
            seg_total += 1
            if merged_ok:
                seg_hit += 1

            # ── Strict: predicted (root, mode) must equal reference (root, mode) ──
            ref_rm = _parse_key_root_mode(r.ref_label)
            est_rm = _parse_key_root_mode(est_label)
            if est_rm is not None and ref_rm is not None and est_rm == ref_rm:
                seg_hit_strict += 1
            if ref_rm is not None:
                ref_mode_counts[ref_rm[1]] += 1
                ref_rm_set.add(ref_rm)
                est_is_min = est_rm[1] if est_rm is not None else None
                if ref_rm[1]:                      # gt-minor segment
                    seg_min += 1
                    if merged_ok:
                        seg_hit_min += 1
                    if est_is_min == 1:
                        seg_predmin_in_min += 1
                else:                              # gt-major segment
                    seg_maj += 1
                    if merged_ok:
                        seg_hit_maj += 1
                    if est_is_min == 0:
                        seg_predmaj_in_maj += 1
            if est_rm is not None:
                est_mode_votes[est_rm[1]] += 1
            seg_mirex_sum += _mirex_weighted_score(ref_rm, est_rm)
            for i, v in enumerate(r.key_act_12):
                act_sum[i] += float(v)
            r_min = getattr(r, "min_act_12", None)
            if r_min is not None:
                for i, v in enumerate(r_min):
                    min_sum[i] += float(v)
            r_forth = getattr(r, "forth_act_12", None)
            if r_forth is not None:
                for i, v in enumerate(r_forth):
                    forth_sum[i] += float(v)
            # Per-sub-SAE ring blocks (3-SAE vote files).
            r_maj_sae = getattr(r, "maj_sae", None)
            if r_maj_sae is not None:
                for s in range(3):
                    for i, v in enumerate(r_maj_sae[s]):
                        maj_sae_sum[s][i] += float(v)
            r_min_sae = getattr(r, "min_sae", None)
            if r_min_sae is not None:
                for s in range(3):
                    for i, v in enumerate(r_min_sae[s]):
                        min_sae_sum[s][i] += float(v)
            r_forth_sae = getattr(r, "forth_sae", None)
            if r_forth_sae is not None:
                for s in range(3):
                    for i, v in enumerate(r_forth_sae[s]):
                        forth_sae_sum[s][i] += float(v)

        # ── Song-level prediction (key signature + mode) ──
        song_total += 1
        if use_joint:
            # Average the per-segment 24 scores, then the same signature-first decision
            # (joint_gt forces the mode to the song's gt mode).
            if joint_nseg > 0:
                Smaj_avg = [s / joint_nseg for s in smaj_sum]
                Smin_avg = [s / joint_nseg for s in smin_sum]
                if vote2_decision == "joint_gt":
                    gt_is_min = 1 if ref_mode_counts[1] > ref_mode_counts[0] else 0
                    root_t, pred_is_min = _joint_decide_gt(Smaj_avg, Smin_avg, gt_is_min)
                else:
                    root_t, pred_is_min = _joint_decide(Smaj_avg, Smin_avg)
            else:
                root_t, pred_is_min = 0, 0
            pred = (root_t + 3) % 12 if pred_is_min else root_t
        elif has_vote3:
            # 3-SAE avg, three rings: pool each sub-SAE's ring block across the song's
            # segments, then forth-vs-minor averaged peaks decide mode, major ring
            # averaged argmax the signature (major_forth_minor_tonic).
            tonic_t, pred_is_min = _key_from_vote3_avg(maj_sae_sum, min_sae_sum, forth_sae_sum)
            pred = (tonic_t + 3) % 12 if pred_is_min else tonic_t
        elif has_vote2:
            # 3-SAE, two rings: pool per-sub-SAE blocks across the song, then decide.
            # 'avg' (major_minor_tonic) = L1-normalize, align-average across sub-SAEs
            # over frame-sum blocks, global argmax; 'tally' = majority re-vote.
            if vote2_decision == "avg":
                # maj_degree handles a forth ring stored in the maj block (single
                # 'forth' variant); major_minor_tonic keeps maj_degree=0.
                tonic_t, pred_is_min = _decide_majmin_multisae(
                    maj_sae_sum, min_sae_sum, maj_degree=maj_degree)
            else:
                tonic_t, pred_is_min = _key_from_vote2(maj_sae_sum, min_sae_sum, maj_degree)
            pred = (tonic_t + 3) % 12 if pred_is_min else tonic_t
        elif has_forth:
            # 3-ring joint: major hypothesis takes its IV from the forth ring,
            # minor hypothesis uses its own ring throughout.  pred is the signature.
            root_t, pred_is_min = _decide_majmin_3ring(act_sum, min_sum, forth_sum)
            pred = (root_t + 3) % 12 if pred_is_min else root_t
        elif has_min:
            # Joint: pool the raw maj/min ring activations, then decide by the
            # I+IV+V (tonic/subdominant/dominant) sums.  pred is the key signature.
            root_t, pred_is_min = _decide_majmin_by_degrees(act_sum, min_sum, maj_degree)
            pred = (root_t + 3) % 12 if pred_is_min else root_t
        else:
            # Merged-activation argmax for the signature, mode by segment vote.
            pred = int(max(range(12), key=lambda i: act_sum[i]))
            pred_is_min = 1 if est_mode_votes[1] > est_mode_votes[0] else 0
        merged_song_ok = pred == ref_sig
        if merged_song_ok:
            song_hit += 1

        song_is_min = 1 if ref_mode_counts[1] > ref_mode_counts[0] else 0
        if song_is_min:
            song_min += 1
            if merged_song_ok:
                song_hit_min += 1
            if pred_is_min == 1:
                song_predmin_in_min += 1
        else:
            song_maj += 1
            if merged_song_ok:
                song_hit_maj += 1
            if pred_is_min == 0:
                song_predmaj_in_maj += 1

        # ── Song-level prediction: root from signature argmax, mode from est vote ──
        root_pred = (pred - 3) % 12 if pred_is_min else pred
        pred_rm_song = (root_pred, pred_is_min)
        if len(ref_rm_set) == 1:
            if pred_rm_song == next(iter(ref_rm_set)):
                song_hit_strict += 1

        # ── Song-level MIREX: reference tonic from signature + majority mode ──
        ref_tonic_song = (ref_sig - 3) % 12 if song_is_min else ref_sig
        song_mirex_sum += _mirex_weighted_score((ref_tonic_song, song_is_min), pred_rm_song)

    def _acc(hit: int, tot: int) -> float:
        return round(100.0 * hit / tot, 4) if tot > 0 else 0.0

    return {
        # ── segment-level ──
        "seg_total":          seg_total,
        "seg_acc":            _acc(seg_hit, seg_total),
        "seg_acc_strict":     _acc(seg_hit_strict, seg_total),
        "seg_maj":            seg_maj,
        "seg_min":            seg_min,
        "seg_hit_maj":        seg_hit_maj,
        "seg_acc_maj":        _acc(seg_hit_maj, seg_maj),
        "seg_hit_min":        seg_hit_min,
        "seg_acc_min":        _acc(seg_hit_min, seg_min),
        "seg_predmaj_in_maj": seg_predmaj_in_maj,
        "seg_predmaj_ratio":  _acc(seg_predmaj_in_maj, seg_maj),
        "seg_predmin_in_min": seg_predmin_in_min,
        "seg_predmin_ratio":  _acc(seg_predmin_in_min, seg_min),
        "seg_mirex":          _acc(seg_mirex_sum, seg_total),
        # fraction of segments whose predicted mode matches the gt mode
        "seg_majmin_acc":     _acc(seg_predmaj_in_maj + seg_predmin_in_min, seg_total),
        # ── song-level ──
        "song_total":          song_total,
        "song_acc":            _acc(song_hit, song_total),
        "song_acc_strict":     _acc(song_hit_strict, song_total),
        "song_maj":            song_maj,
        "song_min":            song_min,
        "song_hit_maj":        song_hit_maj,
        "song_acc_maj":        _acc(song_hit_maj, song_maj),
        "song_hit_min":        song_hit_min,
        "song_acc_min":        _acc(song_hit_min, song_min),
        "song_predmaj_in_maj": song_predmaj_in_maj,
        "song_predmaj_ratio":  _acc(song_predmaj_in_maj, song_maj),
        "song_predmin_in_min": song_predmin_in_min,
        "song_predmin_ratio":  _acc(song_predmin_in_min, song_min),
        "song_mirex":          _acc(song_mirex_sum, song_total),
        "song_majmin_acc":     _acc(song_predmaj_in_maj + song_predmin_in_min, song_total),
    }

# ---------------------------------------------------------------------------
# Chord metrics: frame-level HDF5 references
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
_NO_CHORD = {"N", "NOCHORD"}
_UNKNOWN = {"X", "N/A", "", "UNKNOWN"}


@dataclass(frozen=True)
class CacheRow:
    track_id: str
    start: int
    end: int
    cached_ref: str
    est_label: str
    line_no: int


@dataclass
class CacheSample:
    track_id: str
    rows: List[CacheRow]

    @property
    def length(self) -> int:
        return max((r.end for r in self.rows), default=0)


@dataclass(frozen=True)
class H5Sample:
    index: int
    track_id: str
    ref_labels: np.ndarray

    @property
    def length(self) -> int:
        return int(self.ref_labels.shape[0])


def _decode_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_ref_label(label: Any) -> str:
    text = _decode_str(label).strip()
    upper = text.upper()
    if upper in _UNKNOWN:
        return "X"
    if upper in _NO_CHORD:
        return "N"
    return text


def _normalize_est_label(label: Any) -> str:
    text = _decode_str(label).strip()
    upper = text.upper()
    if upper in _UNKNOWN or upper in _NO_CHORD:
        return "N"
    return text


def parse_cache(path: Path) -> List[CacheRow]:
    """Parse both the legacy four-column cache and a five-column variant."""
    rows: List[CacheRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue

            parts = text.split("\t")
            if len(parts) == 1:
                parts = text.split()

            track_id: str
            start: int
            end: int
            cached_ref: str
            est_label: str

            if len(parts) >= 4 and _RANGE_RE.match(parts[1].strip()):
                match = _RANGE_RE.match(parts[1].strip())
                assert match is not None
                track_id = parts[0].strip()
                start, end = int(match.group(1)), int(match.group(2))
                cached_ref = parts[2].strip()
                est_label = parts[3].strip()
            elif len(parts) >= 5:
                try:
                    start = int(float(parts[1]))
                    end = int(float(parts[2]))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_no}: invalid start/end columns"
                    ) from exc
                track_id = parts[0].strip()
                cached_ref = parts[3].strip()
                est_label = parts[4].strip()
            else:
                raise ValueError(
                    f"{path}:{line_no}: expected four-column start-end cache "
                    "or five-column start/end cache"
                )

            if not track_id:
                raise ValueError(f"{path}:{line_no}: empty track id")
            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}:{line_no}: invalid interval [{start}, {end})"
                )
            rows.append(
                CacheRow(track_id, start, end, cached_ref, est_label, line_no)
            )

    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def group_cache_samples(rows: Sequence[CacheRow]) -> List[CacheSample]:
    """Recover HDF5 sample blocks from legacy local-frame cache intervals.

    Each inference sample starts at local frame 0, and its predicted segments
    tile the sample timeline contiguously.  A new row with start==0 begins the
    next sample.
    """
    samples: List[CacheSample] = []
    current: List[CacheRow] = []
    current_track: str | None = None

    def flush() -> None:
        nonlocal current, current_track
        if not current:
            return
        expected = 0
        for row in current:
            if row.start != expected:
                raise ValueError(
                    f"Cache intervals are not contiguous in sample {current_track!r}: "
                    f"expected start {expected}, got {row.start} at line {row.line_no}"
                )
            expected = row.end
        samples.append(CacheSample(current_track or "", current))
        current = []
        current_track = None

    for row in rows:
        if current and (row.start == 0 or row.track_id != current_track):
            flush()
        if not current:
            if row.start != 0:
                raise ValueError(
                    f"A cache sample must start at frame 0; line {row.line_no} "
                    f"starts at {row.start}"
                )
            current_track = row.track_id
        current.append(row)
    flush()
    return samples


def load_h5_samples(h5_path: Path, split: str) -> List[H5Sample]:
    with h5py.File(h5_path, "r") as handle:
        if split not in handle:
            raise KeyError(f"Split {split!r} not found in {h5_path}")
        group = handle[split]
        if "trackId" not in group:
            raise KeyError(f"{split}/trackId not found in {h5_path}")
        if "labels" not in group or "chord_frame" not in group["labels"]:
            raise KeyError(f"{split}/labels/chord_frame not found in {h5_path}")

        track_ids = group["trackId"][:]
        chord_frames = group["labels"]["chord_frame"]
        if len(track_ids) != len(chord_frames):
            raise ValueError(
                f"trackId count {len(track_ids)} != chord_frame count {len(chord_frames)}"
            )

        samples: List[H5Sample] = []
        for index, raw_track in enumerate(track_ids):
            labels = np.asarray(
                [_decode_str(v) for v in chord_frames[index]], dtype=object
            )
            samples.append(H5Sample(index, _decode_str(raw_track), labels))
    return samples


def _legacy_cached_ref(labels: Sequence[Any], start: int, end: int) -> str:
    """Reproduce infer_key.ref_chord_label_for_seg for cache alignment."""
    counts: Dict[str, int] = {}
    for value in labels[start:end]:
        text = _decode_str(value).strip()
        if text and text.upper() not in _NO_CHORD:
            counts[text] = counts.get(text, 0) + 1
    if not counts:
        return "N"
    return max(counts.items(), key=lambda item: item[1])[0]


def _cache_ref_matches(cache_sample: CacheSample, h5_sample: H5Sample) -> bool:
    if cache_sample.length > h5_sample.length:
        return False
    for row in cache_sample.rows:
        expected = _legacy_cached_ref(h5_sample.ref_labels, row.start, row.end)
        if expected != row.cached_ref:
            return False
    return True


def align_cache_to_h5(
    cache_samples: Sequence[CacheSample],
    h5_samples: Sequence[H5Sample],
) -> List[Tuple[CacheSample, H5Sample]]:
    """Align legacy cache blocks to HDF5 samples in inference order.

    Track id, sample length, and the cached majority reference labels are all
    checked.  The function fails rather than silently scoring a misaligned file.
    """
    aligned: List[Tuple[CacheSample, H5Sample]] = []
    cursor = 0

    for cache_index, cache_sample in enumerate(cache_samples):
        exact_candidates: List[int] = []
        fallback_candidates: List[int] = []

        for h5_index in range(cursor, len(h5_samples)):
            h5_sample = h5_samples[h5_index]
            if h5_sample.track_id != cache_sample.track_id:
                continue
            if cache_sample.length > h5_sample.length:
                continue
            fallback_candidates.append(h5_index)
            if _cache_ref_matches(cache_sample, h5_sample):
                exact_candidates.append(h5_index)
                break  # first compatible sample preserves inference order

        if exact_candidates:
            chosen = exact_candidates[0]
        elif len(fallback_candidates) == 1:
            chosen = fallback_candidates[0]
            sys.stderr.write(
                f"[Warning] cache sample {cache_index} ({cache_sample.track_id}) "
                "matched by order/length, but cached reference labels did not "
                "exactly reproduce the HDF5 labels.\n"
            )
        else:
            raise RuntimeError(
                "Could not safely align cache sample "
                f"{cache_index} (track={cache_sample.track_id!r}, "
                f"length={cache_sample.length}) to {h5_path_hint(h5_samples)}. "
                "Re-run inference with the corrected script if this cache was "
                "produced from a different HDF5 ordering or subset."
            )

        aligned.append((cache_sample, h5_samples[chosen]))
        cursor = chosen + 1

    return aligned


def h5_path_hint(samples: Sequence[H5Sample]) -> str:
    return f"the {len(samples)} HDF5 samples"


def _expand_estimates(cache_sample: CacheSample, length: int) -> np.ndarray:
    estimates = np.full(length, "N", dtype=object)
    covered = np.zeros(length, dtype=bool)
    for row in cache_sample.rows:
        if row.end > length:
            raise ValueError(
                f"Prediction interval [{row.start}, {row.end}) exceeds HDF5 "
                f"sample length {length} for track {row.track_id!r}"
            )
        estimates[row.start : row.end] = _normalize_est_label(row.est_label)
        covered[row.start : row.end] = True
    if not covered.all():
        missing = np.flatnonzero(~covered)
        raise ValueError(
            f"Prediction cache does not cover all frames for track "
            f"{cache_sample.track_id!r}; first missing frame is {int(missing[0])}"
        )
    return estimates


def evaluate_aligned_samples(
    aligned: Sequence[Tuple[CacheSample, H5Sample]],
) -> Dict[str, Any]:
    if mir_eval is None:
        raise RuntimeError(
            "Chord evaluation requires 'mir_eval'. Install it with "
            "'pip install mir_eval'."
        )

    comparators = {
        "root": mir_eval.chord.root,
        "majmin": mir_eval.chord.majmin,
        "mirex": mir_eval.chord.mirex,
    }

    # stats[track][metric] = [correct_weight, valid_weight]
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {name: [0.0, 0.0] for name in comparators}
    )

    for cache_sample, h5_sample in aligned:
        length = cache_sample.length
        ref_labels = [
            _normalize_ref_label(v) for v in h5_sample.ref_labels[:length]
        ]
        est_labels = _expand_estimates(cache_sample, length).tolist()

        for name, comparator in comparators.items():
            comparisons = np.asarray(
                comparator(ref_labels, est_labels), dtype=float
            )
            valid = comparisons >= 0.0
            if not np.any(valid):
                continue
            stats[cache_sample.track_id][name][0] += float(
                comparisons[valid].sum()
            )
            stats[cache_sample.track_id][name][1] += float(valid.sum())

    macro: Dict[str, float] = {}
    micro: Dict[str, float] = {}
    n_songs_by_metric: Dict[str, int] = {}

    for name in comparators:
        song_scores: List[float] = []
        total_correct = 0.0
        total_weight = 0.0
        for track_stats in stats.values():
            correct, weight = track_stats[name]
            if weight <= 0:
                continue
            song_scores.append(correct / weight)
            total_correct += correct
            total_weight += weight
        macro[name] = float(np.mean(song_scores) * 100.0) if song_scores else 0.0
        micro[name] = (
            float(total_correct / total_weight * 100.0) if total_weight > 0 else 0.0
        )
        n_songs_by_metric[name] = len(song_scores)

    n_songs = len({sample.track_id for sample, _ in aligned})
    return {
        "root": round(macro["root"], 2),
        "majmin": round(macro["majmin"], 2),
        "mirex": round(macro["mirex"], 2),
        "song_root": round(macro["root"], 2),
        "song_majmin": round(macro["majmin"], 2),
        "song_mirex": round(macro["mirex"], 2),
        "micro_root": round(micro["root"], 2),
        "micro_majmin": round(micro["majmin"], 2),
        "micro_mirex": round(micro["mirex"], 2),
        "n_songs": n_songs,
        "n_songs_root": n_songs_by_metric["root"],
        "n_songs_majmin": n_songs_by_metric["majmin"],
        "n_songs_mirex": n_songs_by_metric["mirex"],
        "n_cache_samples": len(aligned),
    }


def evaluate_chord_wcsr(
    cache_path: Path,
    h5_path: Path,
    split: str = "test",
) -> Dict[str, Any]:
    cache_rows = parse_cache(cache_path)
    cache_samples = group_cache_samples(cache_rows)
    h5_samples = load_h5_samples(h5_path, split)
    aligned = align_cache_to_h5(cache_samples, h5_samples)
    return evaluate_aligned_samples(aligned)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified key/chord metrics -> one JSON object"
    )

    # These duplicated chord options intentionally support both orders:
    #   eval_metrics --h5-path ... chord FILE
    #   eval_metrics chord FILE --h5-path ...
    parser.add_argument("--h5-path", dest="global_h5_path", type=Path)
    parser.add_argument("--split", dest="global_split", default=None)
    parser.add_argument("--fps", dest="global_fps", type=float, default=None,
                        help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="mode", required=True)

    kp = sub.add_parser("key", help="Compute key metrics")
    kp.add_argument(
        "result_file",
        type=Path,
        help="Key result TSV from infer_key.py / probe_key.py",
    )
    kp.add_argument(
        "--maj-degree",
        type=int,
        default=0,
        help=(
            "Major-side ring degree for joint files "
            "(0=major chord, 5=forth)."
        ),
    )
    kp.add_argument(
        "--vote2-decision",
        choices=["tally", "avg", "joint", "joint_gt"],
        default="tally",
        help=(
            "Decision for two-ring three-SAE files: tally, avg, joint, "
            "or joint_gt."
        ),
    )

    cp = sub.add_parser("chord", help="Compute frame-aligned chord WCSR")
    cp.add_argument("result_file", type=Path, help="Cached chord prediction TSV")
    cp.add_argument("--h5-path", dest="chord_h5_path", type=Path,
                    default=argparse.SUPPRESS)
    cp.add_argument("--split", dest="chord_split",
                    default=argparse.SUPPRESS)
    cp.add_argument("--fps", dest="chord_fps", type=float,
                    default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS)
    cp.add_argument(
        "--raw-file",
        type=Path,
        default=None,
        help=(
            "Optional second chord cache evaluated against the same HDF5; "
            "its fields are returned with a raw_ prefix."
        ),
    )

    args = parser.parse_args(argv)

    if args.mode == "chord":
        args.h5_path = getattr(args, "chord_h5_path", None) or args.global_h5_path
        args.split = getattr(args, "chord_split", None) or args.global_split or "test"
        args.fps = getattr(args, "chord_fps", None) or args.global_fps or 10.0
        if args.h5_path is None:
            parser.error("chord evaluation requires --h5-path")
    else:
        # Reject accidentally supplied chord-only global options rather than
        # silently pretending that key evaluation uses them.
        if args.global_h5_path is not None or args.global_split is not None:
            parser.error("--h5-path/--split are chord-only options")

    return args


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    try:
        _require_file(args.result_file, "result file")

        if args.mode == "key":
            metrics = compute_key_metrics(
                args.result_file,
                maj_degree=args.maj_degree,
                vote2_decision=args.vote2_decision,
            )
        else:
            _require_file(args.h5_path, "HDF5 file")
            metrics = evaluate_chord_wcsr(
                args.result_file,
                args.h5_path,
                args.split,
            )

            if args.raw_file is not None:
                _require_file(args.raw_file, "raw chord cache")
                raw_metrics = evaluate_chord_wcsr(
                    args.raw_file,
                    args.h5_path,
                    args.split,
                )
                for key, value in raw_metrics.items():
                    metrics[f"raw_{key}"] = value

    except Exception as exc:
        # Keep stdout machine-readable for shell callers; diagnostics go to stderr.
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
