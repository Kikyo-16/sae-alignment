#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Infer key (forth / fifth / tonic) and optionally chord from SAE features.

Key inference does NOT require [Chord Boundary] — each segment is treated as a
single key region (time-average of ring activations → argmax).

Chord inference requires [Chord Boundary] and [Major Chord] / [Minor Chord].

Usage:
    python -m src.analysis.step4_infer \
        --h5-path          sae-data/rwc/muq/rwc_muq_30s_layer_2.h5 \
        --split            test \
        --ckpt-path        store/.../ckpts/last.ckpt \
        --feature-id-file  store/.../epoch294_feature_ids.txt \
        --data-tag         muq_layer_2 \
        --feature-ds       muq_layer \
        --sae-idx          0 \
        --forth-out-file   store/.../rwc_test_forth.txt \
        --fifth-out-file   store/.../rwc_test_fifth.txt \
        --tonic-out-file   store/.../rwc_test_tonic.txt \
        [--enable-chord --chord-out-file store/.../rwc_test_chord.txt]  # writes *_bndtop1/2/3.txt
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

import re

from .infer_key import (
    NOTE_NAMES_12,
    chord_root_pc,
    decode_str_array,
    decode_str_scalar,
    key_from_vote2,
    key_from_vote3,
    key_from_vote3_avg,
    decide_maj_min_by_degrees,
    decide_maj_min_multisae,
    decide_joint_major_minor,
    debug_vote2,
    debug_vote3,
    estimate_key_from_ring,
    estimate_minor_key_from_ring,
    predict_chord_segments,
    ref_key_label_for_segment,
    segment_time_range_str,
    smooth_cols,
    standardize_to_BTD_batch,
)
from .find_chord_rings import load_sae_and_factory
from .boundary_utils import boundary_segments_from_feature as _boundary_segments_from_feature


def _parse_feature_groups(path: Path) -> Dict[str, List[int]]:
    """Like analysis_archive's parse_feature_groups but does NOT deduplicate ring IDs.

    Probe rings may legitimately contain duplicate SAE feature IDs (two probe
    weight rows mapping to the same nearest encoder feature). Deduplication
    would shrink the list below 12 and cause the ring to be silently skipped.
    Only [Chord Boundary] keeps deduplication (order matters there too but
    boundary_ids can have repeats without breaking anything).
    """
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
            # Strip leading count (e.g. "12 id0 -> id1 -> ...")
            if len(nums) >= 2 and nums[0] == len(nums) - 1:
                nums = nums[1:]
            groups[group_name] = nums   # no deduplication
    return groups


def _derive_boundary_chord_paths(base_path: Path, n: int = 3) -> List[Path]:
    """Return output paths for boundary-feature top-1..top-n chord results.

    Example: chord.txt -> chord_bndtop1.txt, chord_bndtop2.txt, chord_bndtop3.txt.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    suffix = base_path.suffix or ".txt"
    stem = base_path.stem if base_path.suffix else base_path.name
    parent = base_path.parent
    return [parent / f"{stem}_bndtop{i}{suffix}" for i in range(1, n + 1)]



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Step 4: SAE key / chord inference → TSV result files"
    )
    ap.add_argument("--h5-path",          required=True)
    ap.add_argument("--split",            default="test")
    ap.add_argument("--ckpt-path",        required=True)
    ap.add_argument("--feature-id-file",  required=True)
    ap.add_argument("--data-tag",         default=None)
    ap.add_argument("--feature-ds",       default="mel")
    ap.add_argument("--sae-idx",          type=int,   default=0)  # 0 = anchor sub-SAE
    ap.add_argument("--device",           type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",       type=int,   default=32)
    ap.add_argument("--smooth-win",       type=int,   default=9)
    ap.add_argument("--allow-missing-labels", action="store_true")

    # Key output files
    ap.add_argument("--forth-out-file",   default=None)
    # Plain single-ring [Forth] prediction (no 3-SAE averaging), same layout as
    # --fifth-out-file / --tonic-out-file; --forth-out-file stays the vote1_avg one.
    ap.add_argument("--forth-ring-out-file", default=None)
    ap.add_argument("--fifth-out-file",   default=None)
    ap.add_argument("--tonic-out-file",   default=None)
    # Major-mode key prediction from the [Major Chord] ring, sweeping which
    # scale degree the prominent major chord is assumed to be (tonic/forth/fifth)
    ap.add_argument("--major-out-file",       default=None)  # tonic (I),  degree 0
    ap.add_argument("--major-forth-out-file", default=None)  # forth (IV), degree 5
    ap.add_argument("--major-fifth-out-file", default=None)  # fifth (V),  degree 7
    # Minor-mode key prediction from the [Minor Chord] ring, sweeping which
    # scale degree the prominent minor chord is assumed to be (tonic/forth/fifth)
    ap.add_argument("--minor-out-file",       default=None)  # tonic (i),  degree 0
    ap.add_argument("--minor-forth-out-file", default=None)  # forth (iv), degree 5
    ap.add_argument("--minor-fifth-out-file", default=None)  # fifth (v),  degree 7
    # ── Single-SAE (1-1) key predictions ─────────────────────────────────────
    # Counterparts of the vote1_avg / vote2_avg / vote_gt variants below for
    # models with a single sub-SAE (the vote joints need >=3 and are dropped).
    # Plain single rings at degree 0 (17-col key layout):
    ap.add_argument("--major-ring-out-file", default=None)  # [Major Chord] @ 0
    ap.add_argument("--minor-ring-out-file", default=None)  # [Minor Chord] @ 0
    # Two-ring maj/min decision from ONE sub-SAE (infer_key.decide_maj_min_by_
    # degrees, I+IV+V sums); 29-col output (raw maj + min ring time-means).
    ap.add_argument("--major-minor-pair-out-file",    default=None)
    ap.add_argument("--forth-minor-pair-out-file",    default=None)
    # gt oracle: the reference mode picks the ring, then the plain single-ring
    # estimator predicts within it (17-col key layout).
    ap.add_argument("--major-minor-pair-gt-out-file", default=None)
    ap.add_argument("--forth-minor-pair-gt-out-file", default=None)
    # Single-SAE major_minor_joint (same joint template scoring, ONE sub-SAE).
    # Written in the vote2 layout with sub-SAE blocks 1/2 zeroed: the joint merge
    # (align-average → power → per-ring L1) is scale-invariant, so the zero
    # blocks reproduce single-SAE scores exactly and eval_metrics --vote2-decision
    # joint / joint_gt applies unchanged.
    ap.add_argument("--major-minor-joint-1sae-out-file",    default=None)
    ap.add_argument("--major-minor-joint-1sae-gt-out-file", default=None)
    # ── Combination predictions ───────────────────────────────────────────────
    # "vote2"/"vote3": decode each frame with sub-SAEs 0/1/2, take each ring's
    #          tonic per SAE (corrected by +index), and majority-vote across the
    #          rings' candidates.  "gt": per segment pick the ring matching the
    #          reference key mode (major→first ring, minor→minor_tonic ring).
    ap.add_argument("--major-minor-tonic-out-file",    default=None)  # vote2: major + minor ring
    ap.add_argument("--major-minor-tonic-gt-out-file", default=None)  # gt:    major_tonic / minor_tonic
    ap.add_argument("--forth-minor-tonic-out-file",    default=None)  # vote2: forth(deg5) + minor
    ap.add_argument("--forth-minor-tonic-gt-out-file", default=None)  # gt:    forth(deg5) / minor_tonic
    # joint: stores both rings; eval scores refined combined major/minor key templates
    # (power=0.40 + signature-first decision, --vote2-decision joint). joint_gt forces gt mode.
    ap.add_argument("--major-minor-joint-out-file", default=None)
    ap.add_argument("--major-minor-joint-gt-out-file", default=None)
    # Print the per-sub-SAE tonic candidates + vote tally for the vote joints.
    ap.add_argument("--debug-vote", action="store_true",
                    help="print 3-SAE vote candidates/tally during decoding")
    ap.add_argument("--debug-vote-limit", type=int, default=40,
                    help="max vote traces to print per run (0 = unlimited)")

    # Chord output (requires [Chord Boundary] + [Major Chord] in feature-id-file)
    ap.add_argument("--enable-chord",          action="store_true")
    ap.add_argument(
        "--chord-out-file",
        default=None,
        help="Base chord output path. In estimated-boundary mode this writes "
             "three files: <stem>_bndtop1<suffix>, <stem>_bndtop2<suffix>, "
             "and <stem>_bndtop3<suffix>.",
    )
    ap.add_argument("--boundary-score-floor",  type=float, default=0.0)
    ap.add_argument("--boundary-min-gap",      type=int,   default=2,
                    help="Minimum frame gap after boundary peak detection; 0 disables min-gap thinning.")
    ap.add_argument("--boundary-peak-drop",    type=float, default=0.0)
    ap.add_argument(
        "--boundary-peak-tol",
        type=float,
        default=0.2,
        help="Plateau tolerance for boundary peak detection; keep aligned with "
             "step3 --peak-tol.",
    )
    ap.add_argument(
        "--chord-boundary-idx",
        type=int,
        default=0,
        help="Starting index into [Chord Boundary] ids. Default 0 means top1; "
             "with --boundary-top-k 3 this evaluates top1/top2/top3.",
    )
    ap.add_argument(
        "--boundary-top-k",
        type=int,
        default=3,
        help="Number of ranked boundary features to evaluate and write. Default 3.",
    )
    ap.add_argument(
        "--boundary-max-labels",
        type=int,
        default=120,
        help="Maximum number of boundary peaks kept by thin_times; keep identical "
             "to step3 --boundary-max-labels.",
    )
    # Chord decision strategy (shared with the GT-boundary driver).
    # sum-peak reproduces the original behaviour.
    ap.add_argument("--majmin-mode", default="sum-peak",
                    choices=["sum-peak", "seg-mean-max", "seg-mean-mean", "raw-tmpl"])
    ap.add_argument("--majmin-conf-thr", type=float, default=0.18,
                    help="raw-tmpl: top-2 raw-margin below which template fallback kicks in.")
    ap.add_argument("--majmin-alpha", type=float, default=0.4,
                    help="raw-tmpl: weight of the normalized triad-template score.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    h5_path       = Path(args.h5_path).resolve()
    ckpt_path     = Path(args.ckpt_path).resolve()
    feat_id_file  = Path(args.feature_id_file).resolve()

    groups       = _parse_feature_groups(feat_id_file)
    boundary_ids = groups.get("chord boundary", [])

    # (group_key, degree_semitones, out_file_arg, label, quality)
    # quality "min" → estimate_minor_key_from_ring (outputs <root>:min);
    # quality "maj" → estimate_key_from_ring       (outputs <root>:maj).
    # For minor rings the degree is which scale degree the prominent minor
    # chord is assumed to be (0=tonic i, 5=forth iv, 7=fifth v).
    RING_CONFIGS: List[Tuple[str, int, Optional[str], str, str]] = [
        ("forth",       5, args.forth_ring_out_file,  "Forth",      "maj"),
        ("fifth",       7, args.fifth_out_file,       "Fifth",      "maj"),
        ("tonic",       0, args.tonic_out_file,       "Tonic",      "maj"),
        ("major chord", 0, args.major_ring_out_file,  "MajorTonic", "maj"),
        ("minor chord", 0, args.minor_ring_out_file,  "MinorTonic", "min"),
        # --forth-out-file / major_tonic / minor_tonic are the 3-SAE averaging
        # vote1_avg joints below; --forth-ring-out-file above is the plain ring.
        ("major chord", 5, args.major_forth_out_file, "MajorForth", "maj"),
        ("major chord", 7, args.major_fifth_out_file, "MajorFifth", "maj"),
        ("minor chord", 5, args.minor_forth_out_file, "MinorForth", "min"),
        ("minor chord", 7, args.minor_fifth_out_file, "MinorFifth", "min"),
    ]
    active_rings: List[Tuple[str, int, List[int], Path, str]] = []
    for group_key, degree, out_arg, label, quality in RING_CONFIGS:
        if out_arg is None:
            continue
        ids = groups.get(group_key, [])
        if len(ids) < 12:
            print(f"[WARN] Fewer than 12 ids in [{label}]; {label.lower()} key estimation skipped.")
            continue
        out_p = Path(out_arg).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        active_rings.append((label, degree, ids[:12], out_p, quality))
        print(f"[INFO] {label} ring (degree={degree}): {ids[:12]} → {out_p}")

    # ── Combination estimators (joint argmax, or gt-mode ring selection) ──────
    # Each spec ring is (group_key, quality, degree).  For "gt" the first ring is
    # used when the reference key is major, the minor ring when it is minor.
    def _resolve_spec(spec: Tuple[str, str, int]) -> Optional[Tuple[List[int], str, int]]:
        ids = groups.get(spec[0], [])[:12]
        return (ids, spec[1], spec[2]) if len(ids) >= 12 else None

    # Kinds that decode each frame with the 3 sub-SAEs and store per-SAE ring
    # blocks.  vote2/vote3 re-vote (majority); vote1_avg/vote2_avg L1-normalize and
    # align-average across sub-SAEs then global argmax, storing per-SAE frame SUMS
    # so the song level pools by frame count.  vote1_avg is the single-ring case
    # (forth / major_tonic / minor_tonic): the ring is stored in the vote2 layout
    # with the absent block zeroed, so the avg decision returns that ring's averaged
    # argmax.  vote_gt is the gt oracle: the known reference mode selects the ring,
    # which is stored the same single-ring way (so the avg decision reproduces it).
    VOTE_KINDS = ("vote1_avg", "vote2", "vote2_avg", "vote3", "vote3_avg", "vote_gt")
    NEED_ALL_RINGS = ("vote1_avg", "vote2", "vote2_avg", "vote3", "vote3_avg",
                      "pair1", "joint1", "joint1_gt")

    COMBINED_CONFIGS: List[Tuple[str, Optional[str], str, list]] = [
        ("forth",                args.forth_out_file,                "vote1_avg",
            [("forth", "maj", 5)]),
        ("major_tonic",          args.major_out_file,                "vote1_avg",
            [("major chord", "maj", 0)]),
        ("minor_tonic",          args.minor_out_file,                "vote1_avg",
            [("minor chord", "min", 0)]),
        # two-ring tonic-only baselines: compare the aligned-averaged peak of the
        # major-side ring against the aligned-averaged peak of the minor tonic ring.
        ("major_minor_tonic",    args.major_minor_tonic_out_file,    "vote2_avg",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
        ("forth_minor_tonic",    args.forth_minor_tonic_out_file,    "vote2_avg",
            [("forth", "maj", 5), ("minor chord", "min", 0)]),
        # major_minor_joint / joint_gt store the major + minor rings (per-SAE frame
        # sums).  The segment est_key is already computed with the joint template;
        # eval recomputes the same scores from the cached ring blocks for metrics.
        ("major_minor_joint",    args.major_minor_joint_out_file,    "vote2_avg",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
        ("major_minor_joint_gt", args.major_minor_joint_gt_out_file, "vote2_avg",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
        ("major_minor_tonic_gt", args.major_minor_tonic_gt_out_file, "vote_gt",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
        ("forth_minor_tonic_gt", args.forth_minor_tonic_gt_out_file, "vote_gt",
            [("forth", "maj", 5), ("minor chord", "min", 0)]),
        # Single-SAE (1-1) two-ring pairs: maj/min decided by I+IV+V sums on ONE
        # sub-SAE's time-means (no 3-SAE averaging); 29-col maj+min block files.
        ("major_minor_pair",    args.major_minor_pair_out_file,    "pair1",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
        ("forth_minor_pair",    args.forth_minor_pair_out_file,    "pair1",
            [("forth", "maj", 5), ("minor chord", "min", 0)]),
        # Single-SAE gt oracle: reference mode picks the ring (17-col key layout).
        ("major_minor_pair_gt", args.major_minor_pair_gt_out_file, "pair1_gt",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
        ("forth_minor_pair_gt", args.forth_minor_pair_gt_out_file, "pair1_gt",
            [("forth", "maj", 5), ("minor chord", "min", 0)]),
        # Single-SAE major_minor_joint (vote2 layout, blocks 1/2 zeroed).
        ("major_minor_joint_1sae",    args.major_minor_joint_1sae_out_file,    "joint1",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
        ("major_minor_joint_1sae_gt", args.major_minor_joint_1sae_gt_out_file, "joint1_gt",
            [("major chord", "maj", 0), ("minor chord", "min", 0)]),
    ]
    # active_combined: (label, kind, resolved_specs, out_p)
    active_combined: List[Tuple[str, str, list, Path]] = []
    for label, out_arg, kind, specs in COMBINED_CONFIGS:
        if out_arg is None:
            continue
        resolved = [_resolve_spec(s) for s in specs]
        if kind in NEED_ALL_RINGS and any(r is None for r in resolved):
            print(f"[WARN] [{label}] ({kind}) needs all of {[s[0] for s in specs]}; skipped.")
            continue
        if kind in ("vote_gt", "pair1_gt") and all(r is None for r in resolved):
            print(f"[WARN] [{label}] ({kind}) needs at least one of {[s[0] for s in specs]}; skipped.")
            continue
        out_p = Path(out_arg).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        active_combined.append((label, kind, resolved, out_p))
        print(f"[INFO] {label} ({kind}) → {out_p}")

    do_chord = bool(args.enable_chord)
    major_ids: List[int] = []
    minor_ids: List[int] = []
    silence_ids: List[int] = []
    chord_out_file: Optional[Path] = None
    chord_out_files: List[Path] = []
    boundary_eval_ids: List[int] = []

    if do_chord:
        if not args.chord_out_file:
            raise ValueError("--chord-out-file required with --enable-chord")
        if not boundary_ids:
            raise RuntimeError("[Chord Boundary] missing in feature-id-file; cannot do chord inference.")
        if int(args.boundary_top_k) <= 0:
            raise ValueError(f"--boundary-top-k must be > 0, got {args.boundary_top_k}")
        if int(args.chord_boundary_idx) < 0:
            raise ValueError(f"--chord-boundary-idx must be >= 0, got {args.chord_boundary_idx}")
        if int(args.boundary_min_gap) < 0:
            raise ValueError(f"--boundary-min-gap must be >= 0, got {args.boundary_min_gap}; use 0 to disable min-gap thinning")
        if int(args.boundary_max_labels) <= 0:
            raise ValueError(f"--boundary-max-labels must be > 0, got {args.boundary_max_labels}")

        major_ids  = groups.get("major chord", [])[:12]
        minor_ids  = groups.get("minor chord", [])[:12]
        silence_ids = groups.get("silence", [])
        if len(major_ids) < 12:
            raise RuntimeError("[Major Chord] needs at least 12 ids for chord inference.")

        start_idx = int(args.chord_boundary_idx)
        stop_idx = start_idx + int(args.boundary_top_k)
        boundary_eval_ids = [int(x) for x in boundary_ids[start_idx:stop_idx]]
        if len(boundary_eval_ids) < int(args.boundary_top_k):
            raise RuntimeError(
                f"[Chord Boundary] has only {len(boundary_ids)} id(s); need "
                f"indices {start_idx}..{stop_idx - 1} for --boundary-top-k={args.boundary_top_k}."
            )

        chord_out_file = Path(args.chord_out_file).resolve()
        chord_out_files = _derive_boundary_chord_paths(chord_out_file, len(boundary_eval_ids))
        for out_p in chord_out_files:
            out_p.parent.mkdir(parents=True, exist_ok=True)
        for rank, (fid, out_p) in enumerate(zip(boundary_eval_ids, chord_out_files), start=start_idx + 1):
            print(f"[INFO] Chord est-boundary top{rank}: feature={fid} -> {out_p}")

    lit, sae, model_factory = load_sae_and_factory(ckpt_path, args.device)
    module  = sae.sae
    use_tag = args.data_tag or str(lit.data_tag)
    n_sub   = int(getattr(module, "n_saes", 0))
    use_idx = max(0, min(args.sae_idx, n_sub - 1))
    print(f"[INFO] use_tag={use_tag}  sae_idx={use_idx}  split={args.split}")

    # The vote2/vote3 joints decode each frame with sub-SAEs 0/1/2 and correct the
    # candidate tonic from sub-SAE i by +i semitones (cross-SAE pitch rotation).
    vote_sae_idxs = (0, 1, 2)
    vote_offsets  = vote_sae_idxs
    if any(kind in VOTE_KINDS for _, kind, _, _ in active_combined) and n_sub < 3:
        print(f"[WARN] vote joints need >=3 sub-SAEs but n_saes={n_sub}; dropping them.")
        active_combined = [c for c in active_combined if c[1] not in VOTE_KINDS]
    # Sub-SAE indices whose sparse codes are needed this run.
    needed_sae_idxs = sorted(
        {use_idx} | (set(vote_sae_idxs)
                     if any(k in VOTE_KINDS for _, k, _, _ in active_combined) else set())
    )

    key_act_header = "\t".join([f"act_{n}:maj" for n in NOTE_NAMES_12])

    @contextlib.contextmanager
    def _open_opt(path_or_none: Optional[Path], header: str):
        if path_or_none is None:
            yield None
        else:
            with path_or_none.open("w", encoding="utf-8") as fh:
                fh.write(header + "\n")
                yield fh

    key_header = f"# track_id\tseg_range\tref_key\test_key\tscore\t{key_act_header}"
    # Vote joints store per-sub-SAE raw ring means (chord-root space), one 12-wide
    # block per sub-SAE per ring, so the song level can pool across segments and
    # re-vote.  vote2: major-side(0/1/2) + minor(0/1/2) = 72 cols.  vote3 adds the
    # forth ring (major(0/1/2) + minor(0/1/2) + forth(0/1/2)) = 108 cols.
    def _ring_sae_header(ring: str) -> str:
        return "\t".join(f"{ring}_s{i}_{n}" for i in vote_sae_idxs for n in NOTE_NAMES_12)

    vote2_header = (
        f"# track_id\tseg_range\tref_key\test_key\tscore"
        f"\t{_ring_sae_header('majring')}\t{_ring_sae_header('minring')}"
    )
    vote3_header = (
        f"# track_id\tseg_range\tref_key\test_key\tscore"
        f"\t{_ring_sae_header('majring')}\t{_ring_sae_header('minring')}"
        f"\t{_ring_sae_header('forthring')}"
    )
    # Single-SAE pair layout: one raw maj block + one raw min block (29 cols),
    # the joint layout eval_key_from_tsv parses as key_act_12 + min_act_12.
    pair_header = (
        f"# track_id\tseg_range\tref_key\test_key\tscore"
        "\t" + "\t".join(f"majring_{n}" for n in NOTE_NAMES_12)
        + "\t" + "\t".join(f"minring_{n}" for n in NOTE_NAMES_12)
    )

    def _combined_header(kind: str) -> str:
        if kind in ("vote1_avg", "vote2", "vote2_avg", "vote_gt", "joint1", "joint1_gt"):
            return vote2_header
        if kind in ("vote3", "vote3_avg"):
            return vote3_header
        if kind == "pair1":
            return pair_header
        return key_header

    n_vote = len(vote_sae_idxs)
    ring_file_ctxs = [
        _open_opt(out_p, key_header) for _, _, _, out_p, _ in active_rings
    ]
    combined_file_ctxs = [
        _open_opt(out_p, _combined_header(kind))
        for _, kind, _, out_p in active_combined
    ]
    # Column width of the activation block per combined file
    # (12 / 12*2 pair1 / 12*3*2 vote2 / 12*3*3 vote3).
    combined_zeros = [
        "\t".join(["0.000000"] * (
            12 * n_vote * 3 if kind in ("vote3", "vote3_avg") else
            12 * n_vote * 2 if kind in ("vote1_avg", "vote2", "vote2_avg", "vote_gt",
                                        "joint1", "joint1_gt") else
            12 * 2 if kind == "pair1" else 12))
        for _, kind, _, _ in active_combined
    ]

    chord_file_ctxs = [
        _open_opt(out_p, "# track_id	seg_range	ref_chord	test_chord")
        for out_p in chord_out_files
    ]

    with h5py.File(h5_path, "r") as f, \
         contextlib.ExitStack() as stack:

        ring_fhs     = [stack.enter_context(ctx) for ctx in ring_file_ctxs]
        combined_fhs = [stack.enter_context(ctx) for ctx in combined_file_ctxs]
        chord_fhs    = [stack.enter_context(ctx) for ctx in chord_file_ctxs]

        if args.split not in f:
            raise KeyError(f"Split '{args.split}' not found in {h5_path}.")
        g = f[args.split]
        if args.feature_ds not in g:
            raise KeyError(f"Feature dataset '{args.feature_ds}' not in split '{args.split}'.")

        has_key_labels   = ("labels" in g) and ("key_frame"   in g["labels"])
        has_chord_labels = ("labels" in g) and ("chord_frame" in g["labels"])
        if not has_key_labels and not args.allow_missing_labels:
            raise RuntimeError("Missing key_frame labels; pass --allow-missing-labels.")

        ds_feat = g[args.feature_ds]
        n_total = int(ds_feat.shape[0])
        t_feat  = int(ds_feat.shape[-1])
        bs      = max(1, args.batch_size)

        dbg_printed = 0  # number of vote traces printed so far (--debug-vote)
        pbar = tqdm(total=n_total, desc="Infer", unit="seg")
        for s0 in range(0, n_total, bs):
            s1   = min(n_total, s0 + bs)
            idxs = list(range(s0, s1))

            x_np = np.array(ds_feat[s0:s1], copy=True)
            x    = torch.from_numpy(x_np).float().to(args.device)
            with torch.no_grad():
                feat_batch: Dict[str, torch.Tensor] = model_factory({use_tag: x}, t=0)

            tracks: List[str] = (
                [decode_str_scalar(v) for v in g["trackId"][s0:s1]]
                if "trackId" in g else [f"seg{i:06d}" for i in idxs]
            )

            zero12 = "\t".join(["0.000000"] * 12)

            if use_tag not in feat_batch:
                for j, seg_idx in enumerate(idxs):
                    ref_key   = ref_key_label_for_segment(
                        decode_str_array(g["labels"]["key_frame"][seg_idx])
                    ) if has_key_labels else "N/A"
                    seg_range = segment_time_range_str(g, seg_idx, t_feat)
                    for fh in ring_fhs:
                        if fh is not None:
                            fh.write(f"{tracks[j]}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zero12}\n")
                    for ci, fh in enumerate(combined_fhs):
                        if fh is not None:
                            fh.write(f"{tracks[j]}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{combined_zeros[ci]}\n")
                pbar.update(len(idxs))
                continue

            feat   = feat_batch[use_tag]
            x_sae  = standardize_to_BTD_batch(feat, t_candidates=[t_feat, int(x_np.shape[-1])])
            z_by_idx: Dict[int, np.ndarray] = {}
            with torch.no_grad():
                for si in needed_sae_idxs:
                    sp = module.inference(x_sae, idx=si, topk_wide=0)
                    z_by_idx[si] = sp.detach().float().cpu().numpy()  # [B, T, D]
            z_batch = z_by_idx[use_idx]
            if z_batch.ndim != 3:
                pbar.update(len(idxs))
                continue

            key_frames_batch: List[Optional[np.ndarray]] = (
                [decode_str_array(g["labels"]["key_frame"][i]) for i in idxs]
                if has_key_labels else [None] * len(idxs)
            )
            chord_frames_batch: List[Optional[np.ndarray]] = (
                [decode_str_array(g["labels"]["chord_frame"][i]) for i in idxs]
                if (do_chord and has_chord_labels) else [None] * len(idxs)
            )

            for j, seg_idx in enumerate(idxs):
                track = tracks[j]
                z     = z_batch[j]
                if z.ndim != 2 or z.shape[0] <= 1:
                    seg_range = segment_time_range_str(g, seg_idx, t_feat)
                    ref_key   = ref_key_label_for_segment(key_frames_batch[j])
                    for fh in ring_fhs:
                        if fh is not None:
                            fh.write(f"{track}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zero12}\n")
                    for ci, fh in enumerate(combined_fhs):
                        if fh is not None:
                            fh.write(f"{track}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{combined_zeros[ci]}\n")
                    pbar.update(1)
                    continue

                t_z, d_z  = int(z.shape[0]), int(z.shape[1])
                seg_range = segment_time_range_str(g, seg_idx, t_z)

                key_frame = key_frames_batch[j]
                if key_frame is not None:
                    key_frame = key_frame[:t_z]
                ref_key   = ref_key_label_for_segment(key_frame)

                # ── Key estimation (whole segment, no boundary needed) ─────────
                for ri, (label, degree, ring_ids, _, quality) in enumerate(active_rings):
                    fh = ring_fhs[ri]
                    if fh is None:
                        continue
                    ring_cols = [i for i in ring_ids if 0 <= i < d_z]
                    if len(ring_cols) < 12:
                        fh.write(f"{track}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zero12}\n")
                        continue
                    if quality == "min":
                        est_key, est_score, key_scores_12 = estimate_minor_key_from_ring(z, ring_cols, degree)
                    else:
                        est_key, est_score, key_scores_12 = estimate_key_from_ring(z, ring_cols, degree)
                    act12 = "\t".join([f"{float(v):.6f}" for v in key_scores_12.tolist()])
                    fh.write(f"{track}\t{seg_range}\t{ref_key}\t{est_key}\t{est_score:.6f}\t{act12}\n")

                # ── Combination estimators (joint argmax / gt-mode selection) ──
                def _seg_cols(resolved_spec):
                    """Filter a resolved (ids, quality, degree) spec to this seg."""
                    if resolved_spec is None:
                        return None
                    ids, quality, degree = resolved_spec
                    cols = [i for i in ids if 0 <= i < d_z]
                    return (cols, quality, degree) if len(cols) >= 12 else None

                def _est_one(cols, quality, degree):
                    if quality == "min":
                        return estimate_minor_key_from_ring(z, cols, degree)
                    return estimate_key_from_ring(z, cols, degree)

                # Per-sub-SAE time-mean of a ring (chord-root space): one [12]
                # vector per voting sub-SAE, all using the same ring columns.
                def _sae_means(cols):
                    return [z_by_idx[si][j][:, cols].mean(axis=0) for si in vote_sae_idxs]

                # Per-sub-SAE frame SUM of a ring (vote2_avg stores these so the
                # song level pools by true frame count = whole-song time-average).
                def _sae_sums(cols):
                    return [z_by_idx[si][j][:, cols].sum(axis=0) for si in vote_sae_idxs]

                ref_is_minor = "min" in ref_key.lower()
                for ci, (label, kind, resolved, _) in enumerate(active_combined):
                    fh = combined_fhs[ci]
                    if fh is None:
                        continue
                    seg_specs = [_seg_cols(r) for r in resolved]
                    if kind == "vote1_avg":
                        # Single-ring 3-SAE averaging (major_tonic / minor_tonic):
                        # L1-normalize each per-SAE ring vector, align-average across
                        # sub-SAEs, global argmax.  Stored in the vote2 layout with
                        # the absent ring zeroed so the avg decision returns this
                        # ring's averaged argmax; mode is fixed by the ring quality.
                        spec = seg_specs[0]
                        if spec is None:
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            ring_sums = _sae_sums(spec[0])
                            zeros = [np.zeros(12) for _ in vote_sae_idxs]
                            if spec[1] == "maj":
                                maj_sums, min_sums = ring_sums, zeros
                                deg_kw = {"maj_degree": spec[2]}
                            else:
                                maj_sums, min_sums = zeros, ring_sums
                                deg_kw = {"min_degree": spec[2]}
                            tonic, is_min, est_score = decide_maj_min_multisae(
                                maj_sums, min_sums, vote_offsets, **deg_kw
                            )
                            est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                            act = "\t".join(
                                [f"{float(v):.6f}" for m in maj_sums for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in min_sums for v in m.tolist()]
                            )
                    elif kind == "vote2_avg":
                        # 3-SAE major+minor ring output.  For major_minor_joint tags,
                        # compute the segment est_key with the current joint template
                        # (power=0.40 refined combined-minor + signature-first); for older vote2_avg
                        # users, fall back to the original align-average argmax.  Store
                        # per-SAE frame SUMS (maj 0/1/2 then min 0/1/2) so eval can
                        # recompute segment scores and average 24 scores at song level.
                        if any(s is None for s in seg_specs):
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            maj_spec  = next(s for s in seg_specs if s[1] == "maj")
                            min_spec  = next(s for s in seg_specs if s[1] == "min")
                            maj_sums  = _sae_sums(maj_spec[0])
                            min_sums  = _sae_sums(min_spec[0])
                            if label == "major_minor_joint":
                                tonic, is_min, est_score = decide_joint_major_minor(
                                    maj_sums, min_sums, vote_offsets, gt_is_min=None
                                )
                            elif label == "major_minor_joint_gt":
                                tonic, is_min, est_score = decide_joint_major_minor(
                                    maj_sums, min_sums, vote_offsets, gt_is_min=ref_is_minor
                                )
                            else:
                                tonic, is_min, est_score = decide_maj_min_multisae(
                                    maj_sums, min_sums, vote_offsets,
                                    maj_degree=maj_spec[2], min_degree=min_spec[2],
                                )
                            est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                            act = "\t".join(
                                [f"{float(v):.6f}" for m in maj_sums for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in min_sums for v in m.tolist()]
                            )
                    elif kind == "vote2":
                        # 3-SAE vote: major-side ring + minor ring decide mode and
                        # signature.  Store per-SAE ring means: majside(0/1/2) then
                        # minor(0/1/2) so the song level can pool and re-vote.
                        if any(s is None for s in seg_specs):
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            maj_spec  = next(s for s in seg_specs if s[1] == "maj")
                            min_spec  = next(s for s in seg_specs if s[1] == "min")
                            maj_means = _sae_means(maj_spec[0])
                            min_means = _sae_means(min_spec[0])
                            tonic, is_min, est_score = key_from_vote2(
                                maj_means, min_means, maj_spec[2], vote_offsets
                            )
                            est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                            act = "\t".join(
                                [f"{float(v):.6f}" for m in maj_means for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in min_means for v in m.tolist()]
                            )
                            if args.debug_vote and (
                                args.debug_vote_limit <= 0 or dbg_printed < args.debug_vote_limit
                            ):
                                tqdm.write(
                                    f"[VOTE2 {label}] {track} {seg_range} "
                                    f"ref={ref_key} est={est_key}\n"
                                    + debug_vote2(maj_means, min_means, maj_spec[2], vote_offsets)
                                )
                                dbg_printed += 1
                    elif kind == "vote3":
                        # 3-SAE vote, three rings: forth+minor decide MODE, major+
                        # minor decide SIGNATURE.  Store maj(0/1/2)+min(0/1/2)+
                        # forth(0/1/2) per-SAE means.
                        if any(s is None for s in seg_specs):
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            maj_spec    = next(s for s in seg_specs if s[1] == "maj")
                            min_spec    = next(s for s in seg_specs if s[1] == "min")
                            forth_spec  = next(s for s in seg_specs if s[1] == "forth")
                            maj_means   = _sae_means(maj_spec[0])
                            min_means   = _sae_means(min_spec[0])
                            forth_means = _sae_means(forth_spec[0])
                            tonic, is_min, est_score = key_from_vote3(
                                maj_means, min_means, forth_means, vote_offsets
                            )
                            est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                            act = "\t".join(
                                [f"{float(v):.6f}" for m in maj_means for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in min_means for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in forth_means for v in m.tolist()]
                            )
                            if args.debug_vote and (
                                args.debug_vote_limit <= 0 or dbg_printed < args.debug_vote_limit
                            ):
                                tqdm.write(
                                    f"[VOTE3 {label}] {track} {seg_range} "
                                    f"ref={ref_key} est={est_key}\n"
                                    + debug_vote3(maj_means, min_means, forth_means, vote_offsets)
                                )
                                dbg_printed += 1
                    elif kind == "vote3_avg":
                        # 3-SAE avg, three rings (major_forth_minor_tonic): forth-vs-
                        # minor averaged peaks decide MODE; the major ring's averaged
                        # argmax is the signature; tonic = sig (maj) or sig+9 (min).
                        # Store per-SAE frame SUMS maj(0/1/2)+min(0/1/2)+forth(0/1/2).
                        if any(s is None for s in seg_specs):
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            maj_spec    = next(s for s in seg_specs if s[1] == "maj")
                            min_spec    = next(s for s in seg_specs if s[1] == "min")
                            forth_spec  = next(s for s in seg_specs if s[1] == "forth")
                            maj_sums    = _sae_sums(maj_spec[0])
                            min_sums    = _sae_sums(min_spec[0])
                            forth_sums  = _sae_sums(forth_spec[0])
                            tonic, is_min, est_score = key_from_vote3_avg(
                                maj_sums, min_sums, forth_sums, vote_offsets
                            )
                            est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                            act = "\t".join(
                                [f"{float(v):.6f}" for m in maj_sums for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in min_sums for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in forth_sums for v in m.tolist()]
                            )
                    elif kind in ("joint1", "joint1_gt"):
                        # Single-SAE major_minor_joint: this sub-SAE's frame sums in
                        # block s0, blocks s1/s2 zeroed.  The joint merge L1-norm-
                        # alizes after align-averaging, so the zero blocks give
                        # exactly the single-SAE joint scores; eval's vote2-layout
                        # joint / joint_gt path applies unchanged.
                        if any(s is None for s in seg_specs):
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            maj_spec = next(s for s in seg_specs if s[1] == "maj")
                            min_spec = next(s for s in seg_specs if s[1] == "min")
                            maj_sums = [z[:, maj_spec[0]].sum(axis=0),
                                        np.zeros(12), np.zeros(12)]
                            min_sums = [z[:, min_spec[0]].sum(axis=0),
                                        np.zeros(12), np.zeros(12)]
                            tonic, is_min, est_score = decide_joint_major_minor(
                                maj_sums, min_sums, vote_offsets,
                                gt_is_min=(ref_is_minor if kind == "joint1_gt" else None),
                            )
                            est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                            act = "\t".join(
                                [f"{float(v):.6f}" for m in maj_sums for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in min_sums for v in m.tolist()]
                            )
                    elif kind == "pair1":
                        # Single-SAE two-ring maj/min decision (1-1 models): I+IV+V
                        # sums on this sub-SAE's ring time-means.  Store the raw maj
                        # + min means (chord-root space); eval's has_min song path
                        # pools them and re-applies decide_maj_min_by_degrees.
                        if any(s is None for s in seg_specs):
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            maj_spec = next(s for s in seg_specs if s[1] == "maj")
                            min_spec = next(s for s in seg_specs if s[1] == "min")
                            maj_mean = z[:, maj_spec[0]].mean(axis=0)
                            min_mean = z[:, min_spec[0]].mean(axis=0)
                            root, is_min, xv, pv = decide_maj_min_by_degrees(
                                maj_mean, min_mean, maj_spec[2]
                            )
                            est_key = f"{NOTE_NAMES_12[root]}:{'min' if is_min else 'maj'}"
                            est_score = float(max(xv, pv))
                            act = "\t".join(
                                [f"{float(v):.6f}" for v in maj_mean.tolist()]
                                + [f"{float(v):.6f}" for v in min_mean.tolist()]
                            )
                    elif kind == "pair1_gt":
                        # Single-SAE gt oracle: the reference mode picks the ring and
                        # the plain single-ring estimator predicts within it (17-col
                        # key layout, activations in key-signature space).
                        maj_spec, min_spec = seg_specs[0], seg_specs[1]
                        chosen = (min_spec if ref_is_minor else maj_spec) or maj_spec or min_spec
                        if chosen is None:
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            est_key, est_score, key_scores_12 = _est_one(*chosen)
                            act = "\t".join(f"{float(v):.6f}" for v in key_scores_12.tolist())
                    else:  # vote_gt: the gt mode picks the ring; predict its tonic by
                        # single-ring 3-SAE averaging.  Stored in the vote2 layout with
                        # the unused block zeroed so the avg decision reproduces the
                        # gt-mode selection.  resolved = [maj-branch spec, min-branch spec].
                        maj_spec, min_spec = seg_specs[0], seg_specs[1]
                        chosen = (min_spec if ref_is_minor else maj_spec) or maj_spec or min_spec
                        if chosen is None:
                            est_key, est_score, act = "N/A", 0.0, combined_zeros[ci]
                        else:
                            ring_sums = _sae_sums(chosen[0])
                            zeros = [np.zeros(12) for _ in vote_sae_idxs]
                            if chosen[1] == "maj":
                                maj_sums, min_sums = ring_sums, zeros
                                deg_kw = {"maj_degree": chosen[2]}
                            else:
                                maj_sums, min_sums = zeros, ring_sums
                                deg_kw = {"min_degree": chosen[2]}
                            tonic, is_min, est_score = decide_maj_min_multisae(
                                maj_sums, min_sums, vote_offsets, **deg_kw
                            )
                            est_key = f"{NOTE_NAMES_12[tonic]}:{'min' if is_min else 'maj'}"
                            act = "\t".join(
                                [f"{float(v):.6f}" for m in maj_sums for v in m.tolist()]
                                + [f"{float(v):.6f}" for m in min_sums for v in m.tolist()]
                            )
                    fh.write(f"{track}\t{seg_range}\t{ref_key}\t{est_key}\t{est_score:.6f}\t{act}\n")

                # ── Chord inference (requires boundary) ───────────────────────
                if not do_chord:
                    pbar.update(1)
                    continue

                chord_frame_seg = chord_frames_batch[j]
                if chord_frame_seg is not None:
                    chord_frame_seg = chord_frame_seg[:t_z]

                major_cols          = [i for i in major_ids  if 0 <= i < d_z]
                minor_cols          = [i for i in minor_ids  if 0 <= i < d_z]
                silence_cols        = [i for i in silence_ids if 0 <= i < d_z]

                if len(major_cols) < 12:   # major ring required; minor checked downstream
                    pbar.update(1)
                    continue

                # Silence from the SAE silence feature (source-specific; smoothed
                # before taking the per-frame min, then passed to the shared core).
                # Compute once and reuse for boundary top1/top2/top3 outputs.
                silence_sc_t: Optional[np.ndarray] = None
                if silence_cols:
                    sil_sm       = smooth_cols(z[:, silence_cols], args.smooth_win)
                    silence_sc_t = sil_sm.min(axis=1)

                # Estimated-boundary chord inference: write one result file per
                # ranked boundary feature, reusing the same z / rings / silence.
                for chord_bnd_id, f_chord in zip(boundary_eval_ids, chord_fhs):
                    chord_segs = _boundary_segments_from_feature(
                        z=z,
                        feature_id=int(chord_bnd_id),
                        t_z=t_z,
                        smooth_win=args.smooth_win,
                        peak_tol=float(args.boundary_peak_tol),
                        peak_drop=float(args.boundary_peak_drop),
                        min_gap=int(args.boundary_min_gap),
                        score_floor=float(args.boundary_score_floor),
                        max_labels=int(args.boundary_max_labels),
                    )

                    predict_chord_segments(
                        z, major_cols, minor_cols, chord_segs,
                        silence_sc_t=silence_sc_t,
                        chord_frame_seg=chord_frame_seg,
                        track=track, t_z=t_z,
                        f_chord=f_chord,
                        smooth_win=args.smooth_win,
                        qual_mode=args.majmin_mode.replace("-", "_"),
                        conf_thr=args.majmin_conf_thr,
                        alpha=args.majmin_alpha,
                    )

                pbar.update(1)

        pbar.close()
    print("[DONE] Step 4 inference complete.")


if __name__ == "__main__":
    main()
