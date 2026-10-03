#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

NOTE_NAMES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE2PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}


@dataclass(frozen=True)
class Row:
    track_id: str
    seg_range: str
    ref_label: str
    est_label: str
    score: float
    key_act_12: List[float]
    avg_chord_stability: float = -1.0  # legacy field, may be absent
    # Present only for the "joint" key files: the major-side activations live in
    # key_act_12, the minor-side activations here (both in key-signature space).
    min_act_12: Optional[List[float]] = None
    # Present only for the 3-ring joint file: the forth-ring activations
    # (chord-root space).  Used as the major hypothesis' subdominant (IV) slot.
    forth_act_12: Optional[List[float]] = None
    # Present only for the 3-SAE vote files: per-sub-SAE ring time-means (chord-
    # root space), each a list of 3 blocks (sub-SAE 0/1/2) of 12 floats.
    # vote2 fills maj_sae + min_sae; vote3 also fills forth_sae.
    maj_sae: Optional[List[List[float]]] = None
    min_sae: Optional[List[List[float]]] = None
    forth_sae: Optional[List[List[float]]] = None


def np_isfinite(x: float) -> bool:
    try:
        return x == x and x not in (float("inf"), float("-inf"))
    except Exception:
        return False


def parse_key_signature_major_pc(label: str) -> Optional[int]:
    """
    Map key label to major-signature class (0..11):
    - enharmonic equivalent merged
    - relative major/minor merged
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
    if mode.startswith("min"):
        return (int(root_pc) + 3) % 12
    return int(root_pc) % 12


def parse_row(line: str) -> Optional[Row]:
    txt = line.strip()
    if not txt or txt.startswith("#"):
        return None
    parts = txt.split("\t")
    if len(parts) not in (5, 17, 18, 29, 41, 77, 113):
        parts = txt.split()
    if len(parts) not in (5, 17, 18, 29, 41, 77, 113):
        return None

    track_id = parts[0].strip()
    seg_range = parts[1].strip()
    ref_label = parts[2].strip()
    est_label = parts[3].strip()
    try:
        score = float(parts[4].strip())
    except Exception:
        score = 0.0
    if not np_isfinite(score):
        score = 0.0

    key_act_12 = [0.0] * 12
    if len(parts) >= 17:
        vals: List[float] = []
        for x in parts[5:17]:
            try:
                v = float(str(x).strip())
            except Exception:
                v = 0.0
            if not np_isfinite(v):
                v = 0.0
            vals.append(max(0.0, float(v)))
        if len(vals) == 12:
            key_act_12 = vals

    avg_chord_stability = -1.0
    if len(parts) == 18:
        try:
            avg_chord_stability = float(parts[17].strip())
        except Exception:
            avg_chord_stability = -1.0
        if not np_isfinite(avg_chord_stability):
            avg_chord_stability = -1.0

    def _block(cols: List[str]) -> Optional[List[float]]:
        vals: List[float] = []
        for x in cols:
            try:
                v = float(str(x).strip())
            except Exception:
                v = 0.0
            if not np_isfinite(v):
                v = 0.0
            vals.append(max(0.0, float(v)))
        return vals if len(vals) == 12 else None

    # Joint key files carry a second 12-value block (minor-side activations); the
    # 3-ring joint file adds a third block (forth-ring activations).
    min_act_12: Optional[List[float]] = None
    forth_act_12: Optional[List[float]] = None
    if len(parts) in (29, 41):
        min_act_12 = _block(parts[17:29])
    if len(parts) == 41:
        forth_act_12 = _block(parts[29:41])

    # 3-SAE vote files: per-sub-SAE ring means, 3 blocks of 12 per ring.
    # vote2 (77 cols): majside(0/1/2) + minor(0/1/2).
    # vote3 (113 cols): major(0/1/2) + minor(0/1/2) + forth(0/1/2).
    maj_sae: Optional[List[List[float]]] = None
    min_sae: Optional[List[List[float]]] = None
    forth_sae: Optional[List[List[float]]] = None

    def _blocks(start: int, n_ring: int) -> Optional[List[List[float]]]:
        out: List[List[float]] = []
        for k in range(n_ring):
            b = _block(parts[start + 12 * k: start + 12 * (k + 1)])
            if b is None:
                return None
            out.append(b)
        return out

    if len(parts) in (77, 113):
        maj_sae = _blocks(5, 3)
        min_sae = _blocks(41, 3)
        if len(parts) == 113:
            forth_sae = _blocks(77, 3)

    return Row(
        track_id=track_id,
        seg_range=seg_range,
        ref_label=ref_label,
        est_label=est_label,
        score=score,
        key_act_12=key_act_12,
        avg_chord_stability=avg_chord_stability,
        min_act_12=min_act_12,
        forth_act_12=forth_act_12,
        maj_sae=maj_sae,
        min_sae=min_sae,
        forth_sae=forth_sae,
    )


def load_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            r = parse_row(line)
            if r is not None:
                rows.append(r)
    return rows


def topk_hit(rank: List[int], target: int, k: int) -> int:
    if k <= 0:
        return 0
    return 1 if target in rank[:k] else 0


def genre_from_track_id(track_id: str) -> Optional[str]:
    """
    GTZAN-style ids: ``<genre>/<stem>`` (e.g. ``pop/pop.00000``).
    Returns genre or None if there is no ``/`` prefix.
    """
    s = str(track_id).strip()
    if "/" not in s:
        return None
    g = s.split("/", 1)[0].strip()
    return g or None


def rank12_from_row(r: Row, pred_sig_fallback: Optional[int]) -> List[int]:
    if any(float(v) > 0.0 for v in r.key_act_12):
        return [i for i, _v in sorted(enumerate(r.key_act_12), key=lambda kv: (-float(kv[1]), int(kv[0])))]
    if pred_sig_fallback is not None:
        rest = [i for i in range(12) if i != int(pred_sig_fallback)]
        return [int(pred_sig_fallback)] + rest
    return list(range(12))


def parse_args():
    ap = argparse.ArgumentParser("Evaluate infer_key result with top-1/2/3 hit accuracy.")
    ap.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="infer_key output TSV (5, 17, or 18 columns).",
    )
    ap.add_argument(
        "--pred-out",
        type=str,
        default=None,
        help="Optional TSV output of top3 predictions (note names) for segment and song levels.",
    )
    ap.add_argument(
        "--correct-out",
        type=str,
        default=None,
        help=(
            "If set, write correctly-predicted segments (top-1 hit) with their "
            "avg_chord_stability to this TSV file."
        ),
    )
    ap.add_argument(
        "--incorrect-out",
        type=str,
        default=None,
        help=(
            "If set, write incorrectly-predicted segments (top-1 miss) with their "
            "avg_chord_stability to this TSV file."
        ),
    )
    ap.add_argument(
        "--song-out",
        type=str,
        default=None,
        help="If set, write song-level ref/prediction TSV to this path.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    result_path = Path(args.result_file).resolve()
    rows = load_rows(result_path)
    pred_out = (
        Path(args.pred_out).resolve()
        if args.pred_out
        else result_path.with_name(result_path.stem + ".top3_predictions.tsv")
    )
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    # Segment-level
    seg_eval_total = 0
    seg_top1 = 0
    seg_skip_invalid = 0
    seg_top3_rows: List[str] = []

    # Single-active-key subset: segments where exactly one key_act_12 value > 0
    single_act_eval_total = 0
    single_act_top1 = 0

    # Correctly- and incorrectly-predicted segments
    correct_rows: List[str] = []
    incorrect_rows: List[str] = []

    # Song-level pools: sum key_act_12 across segments, argmax → predicted key
    song_act_sum: Dict[str, List[float]] = defaultdict(lambda: [0.0] * 12)
    song_ref_sigs: Dict[str, List[int]] = defaultdict(list)

    # Per-genre (GTZAN: track_id = genre/stem)
    genres_seen: set[str] = set()
    g_seg_total: Dict[str, int] = defaultdict(int)
    g_seg_top1: Dict[str, int] = defaultdict(int)
    g_seg_skip_invalid: Dict[str, int] = defaultdict(int)

    for r in rows:
        ref_sig = parse_key_signature_major_pc(r.ref_label)
        pred_sig = parse_key_signature_major_pc(r.est_label)
        g = genre_from_track_id(r.track_id)
        if g:
            genres_seen.add(g)
        if ref_sig is None:
            seg_skip_invalid += 1
            if g:
                g_seg_skip_invalid[g] += 1
            continue

        ref = int(ref_sig)
        seg_eval_total += 1
        hit1 = 1 if (pred_sig is not None and pred_sig == ref) else 0
        seg_top1 += hit1
        if hit1 and args.correct_out is not None:
            correct_rows.append(f"{r.track_id}\t{r.seg_range}\t{r.ref_label}\t{r.est_label}\t{r.avg_chord_stability:.6f}")
        elif not hit1 and args.incorrect_out is not None:
            incorrect_rows.append(f"{r.track_id}\t{r.seg_range}\t{r.ref_label}\t{r.est_label}\t{r.avg_chord_stability:.6f}")
        if sum(1 for v in r.key_act_12 if float(v) > 0.0) == 1:
            single_act_eval_total += 1
            single_act_top1 += hit1
        if g:
            g_seg_total[g] += 1
            g_seg_top1[g] += hit1
        seg_top3_rows.append(
            f"segment\t{r.track_id}\t{r.seg_range}\t{NOTE_NAMES_12[ref]}:maj\t{r.est_label}\t-\t-"
        )

        # Song-level accumulation: sum all 12 key_act_12 dims across segments.
        for i, v in enumerate(r.key_act_12):
            song_act_sum[r.track_id][i] += float(v)
        song_ref_sigs[r.track_id].append(ref)

    # Song-level majority-vote accuracy
    song_eval_total = 0
    song_top1 = 0
    song_skip_modulation = 0
    song_skip_no_ref = 0
    song_top3_rows: List[str] = []

    g_song_eval: Dict[str, int] = defaultdict(int)
    g_song_top1: Dict[str, int] = defaultdict(int)
    g_song_skip_mod: Dict[str, int] = defaultdict(int)
    g_song_skip_noref: Dict[str, int] = defaultdict(int)

    all_tracks = sorted(set(list(song_act_sum.keys()) + list(song_ref_sigs.keys())))
    for track in all_tracks:
        sg = genre_from_track_id(track)
        if sg:
            genres_seen.add(sg)
        refs = song_ref_sigs.get(track, [])
        if not refs:
            song_skip_no_ref += 1
            if sg:
                g_song_skip_noref[sg] += 1
            continue
        uniq = sorted(set(int(x) for x in refs))
        if len(uniq) > 1:
            song_skip_modulation += 1
            if sg:
                g_song_skip_mod[sg] += 1
            continue
        ref = int(uniq[0])
        act = song_act_sum.get(track, [0.0] * 12)
        pred_key = int(max(range(12), key=lambda i: act[i]))
        song_eval_total += 1
        hit1 = 1 if pred_key == ref else 0
        song_top1 += hit1
        if sg:
            g_song_eval[sg] += 1
            g_song_top1[sg] += hit1
        song_top3_rows.append(
            f"song\t{track}\t-\t{NOTE_NAMES_12[ref]}:maj\t{NOTE_NAMES_12[pred_key]}:maj\t-\t-"
        )

    def _acc(c: int, n: int) -> float:
        return (100.0 * float(c) / float(n)) if n > 0 else 0.0

    print("[SUMMARY] Segment-level hit accuracy (direct est_key comparison)")
    print("  - compare in key-signature space (enharmonic + relative major/minor merged)")
    print(f"  - evaluated: {seg_eval_total}")
    print(f"  - top1: {seg_top1}/{seg_eval_total} = {_acc(seg_top1, seg_eval_total):.2f}%")
    print(f"  - skipped invalid ref: {seg_skip_invalid}")

    print("[SUMMARY] Single-active-key segment accuracy (exactly one key_act_12 > 0)")
    print(f"  - single-active-key segments: {single_act_eval_total} / {seg_eval_total} total evaluated")
    print(f"  - top1: {single_act_top1}/{single_act_eval_total} = {_acc(single_act_top1, single_act_eval_total):.2f}%")

    print("[SUMMARY] Song-level hit accuracy (argmax of summed key_act_12 across segments)")
    print("  - accumulate key_act_12 across segments, argmax → predicted key")
    print(f"  - evaluated songs: {song_eval_total}")
    print(f"  - top1: {song_top1}/{song_eval_total} = {_acc(song_top1, song_eval_total):.2f}%")
    print(f"  - skipped modulation songs: {song_skip_modulation}")
    print(f"  - skipped no-ref songs: {song_skip_no_ref}")

    if genres_seen:
        print("[SUMMARY] Per-genre (track_id prefix before '/', e.g. GTZAN pop/pop.00000)")
        for genre in sorted(genres_seen):
            gn = g_seg_total.get(genre, 0)
            print(f"  --- genre={genre!r} ---")
            if gn > 0:
                print(
                    f"  [segment] evaluated: {gn}; "
                    f"top1: {g_seg_top1.get(genre, 0)}/{gn} = {_acc(g_seg_top1.get(genre, 0), gn):.2f}%"
                )
            else:
                print("  [segment] evaluated: 0 (no segments with valid ref)")
            inv = g_seg_skip_invalid.get(genre, 0)
            if inv:
                print(f"  [segment] skipped invalid ref: {inv}")
            sn = g_song_eval.get(genre, 0)
            if sn > 0:
                print(
                    f"  [song]   evaluated: {sn}; "
                    f"top1: {g_song_top1.get(genre, 0)}/{sn} = {_acc(g_song_top1.get(genre, 0), sn):.2f}%"
                )
            else:
                print("  [song]   evaluated: 0 (no single-key songs)")
            sm = g_song_skip_mod.get(genre, 0)
            snr = g_song_skip_noref.get(genre, 0)
            if sm or snr:
                print(f"  [song]   skipped modulation: {sm}; skipped no-ref: {snr}")

    with pred_out.open("w", encoding="utf-8") as f:
        f.write("level\ttrack_id\tseg_range\tref_key\tpred_top1\tpred_top2\tpred_top3\n")
        for line in seg_top3_rows:
            f.write(line + "\n")
    print(f"[OUTPUT] Segment-level prediction file: {pred_out}")

    song_out_path = (
        Path(args.song_out).resolve()
        if args.song_out
        else result_path.with_name(result_path.stem + ".song_predictions.tsv")
    )
    song_out_path.parent.mkdir(parents=True, exist_ok=True)
    with song_out_path.open("w", encoding="utf-8") as f:
        f.write("track_id\tref_key\tpred_key\n")
        for line in song_top3_rows:
            # song_top3_rows format: song\ttrack\t-\tref\tpred\t-\t-
            parts = line.split("\t")
            f.write(f"{parts[1]}\t{parts[3]}\t{parts[4]}\n")
    print(f"[OUTPUT] Song-level prediction file: {song_out_path}")

    if args.correct_out is not None:
        correct_out = Path(args.correct_out).resolve()
        correct_out.parent.mkdir(parents=True, exist_ok=True)
        with correct_out.open("w", encoding="utf-8") as f:
            f.write("# track_id\tseg_range\tref_key\test_key\tavg_chord_stability\n")
            for line in correct_rows:
                f.write(line + "\n")
        print(f"[OUTPUT] Correct-segment file: {correct_out} ({len(correct_rows)} rows)")

    if args.incorrect_out is not None:
        incorrect_out = Path(args.incorrect_out).resolve()
        incorrect_out.parent.mkdir(parents=True, exist_ok=True)
        with incorrect_out.open("w", encoding="utf-8") as f:
            f.write("# track_id\tseg_range\tref_key\test_key\tavg_chord_stability\n")
            for line in incorrect_rows:
                f.write(line + "\n")
        print(f"[OUTPUT] Incorrect-segment file: {incorrect_out} ({len(incorrect_rows)} rows)")


if __name__ == "__main__":
    main()

