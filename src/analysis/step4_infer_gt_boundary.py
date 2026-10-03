#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 (GT-boundary variant): Chord inference using ground-truth chord boundaries
and silence labels instead of SAE boundary/silence feature IDs.

Key inference is identical to step4_infer (no boundary needed).
Chord inference replaces SAE boundary/silence with GT labels:
  - Boundaries : frame indices where chord_frame label changes
  - Silence    : frames where chord_frame is "N" or "X"

Requires an existing feature_ids.txt (from step2_rings / run_dir) for
ring IDs ([Major Chord] / [Minor Chord]).

Usage:
    python -m src.analysis.step4_infer_gt_boundary \\
        --h5-path          sae-data/pop909/muq/pop909_muq_30s_layer_2.h5 \\
        --split            train \\
        --ckpt-path        store/.../ckpts/last.ckpt \\
        --feature-id-file  store/.../epoch294_feature_ids.txt \\
        --data-tag         muq_layer_2 \\
        --feature-ds       muq_layer \\
        --sae-idx          0 \\
        --enable-chord \\
        --chord-out-file   store/.../pop909_train_chord_gtbnd.txt
"""
from __future__ import annotations

import argparse
import contextlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..analysis.infer_key import (
    NOTE_NAMES_12,
    _segments_from_boundaries,
    decode_str_array,
    decode_str_scalar,
    estimate_key_from_ring,
    estimate_ref_key_from_frame,
    predict_chord_segments,
    segment_time_range_str,
    standardize_to_BTD_batch,
)
from ..analysis.find_chord_rings import load_sae_and_factory


# ---------------------------------------------------------------------------
# GT boundary / silence helpers
# ---------------------------------------------------------------------------

def _gt_chord_boundaries(chord_frame: Optional[np.ndarray], t_z: int) -> List[int]:
    """Frame indices where the GT chord label changes."""
    if chord_frame is None or len(chord_frame) == 0:
        return []
    boundaries: List[int] = []
    prev = str(chord_frame[0]).strip()
    for t in range(1, min(t_z, len(chord_frame))):
        cur = str(chord_frame[t]).strip()
        if cur != prev:
            boundaries.append(t)
            prev = cur
    return boundaries


def _gt_silence_score(chord_frame: Optional[np.ndarray], t_z: int) -> Optional[np.ndarray]:
    """Per-frame silence score derived from GT (1.0 where label is N/X/empty, else 0.0).

    The resulting array is compatible with _segment_vote_labels: sil_sum for a
    fully-silent segment equals the segment length, which dominates typical
    maj_sum / min_sum values, so the segment is correctly classified as 'N'.
    """
    if chord_frame is None:
        return None
    score = np.zeros(t_z, dtype=np.float32)
    for t in range(min(t_z, len(chord_frame))):
        if str(chord_frame[t]).strip().upper() in ("N", "X", ""):
            score[t] = 1.0
    return score


# ---------------------------------------------------------------------------
# Feature group parser (identical to step4_infer)
# ---------------------------------------------------------------------------

def _parse_feature_groups(path: Path) -> Dict[str, List[int]]:
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
            groups[group_name] = nums
    return groups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Step 4 (GT-boundary): chord inference with ground-truth boundaries"
    )
    ap.add_argument("--h5-path",              required=True)
    ap.add_argument("--split",                default="test")
    ap.add_argument("--ckpt-path",            required=True)
    ap.add_argument("--feature-id-file",      required=True)
    ap.add_argument("--data-tag",             default=None)
    ap.add_argument("--feature-ds",           default="mel")
    ap.add_argument("--sae-idx",              type=int,   default=0)
    ap.add_argument("--device",               type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",           type=int,   default=32)
    ap.add_argument("--smooth-win",           type=int,   default=9,
                    help="Moving-average window applied to the chord ring "
                         "activations before mode/root decision (<=1 disables).")
    ap.add_argument("--allow-missing-labels", action="store_true")

    # Key outputs (optional)
    ap.add_argument("--forth-out-file",       default=None)
    ap.add_argument("--fifth-out-file",       default=None)
    ap.add_argument("--tonic-out-file",       default=None)

    # Chord output
    ap.add_argument("--enable-chord",         action="store_true")
    ap.add_argument("--chord-out-file",       default=None)

    # Chord decision strategy
    ap.add_argument("--majmin-mode", default="sum-peak",
                    choices=["sum-peak", "seg-mean-max", "seg-mean-mean", "raw-tmpl"],
                    help="Per-segment chord decision. sum-peak: original (sum of "
                         "per-frame peak scores). seg-mean-max/seg-mean-mean: "
                         "segment-mean 12-vector per ring, compare max / mean. "
                         "raw-tmpl: joint root+quality with triad-template fallback.")
    ap.add_argument("--majmin-conf-thr", type=float, default=0.18,
                    help="raw-tmpl: top-2 raw-margin below which the template "
                         "fallback kicks in.")
    ap.add_argument("--majmin-alpha", type=float, default=0.4,
                    help="raw-tmpl: weight of the normalized triad-template score.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    h5_path      = Path(args.h5_path).resolve()
    ckpt_path    = Path(args.ckpt_path).resolve()
    feat_id_file = Path(args.feature_id_file).resolve()

    groups = _parse_feature_groups(feat_id_file)

    RING_CONFIGS: List[Tuple[str, int, Optional[str]]] = [
        ("forth", 5, args.forth_out_file),
        ("fifth", 7, args.fifth_out_file),
        ("tonic", 0, args.tonic_out_file),
    ]
    active_rings: List[Tuple[str, int, List[int], Path]] = []
    for group_key, degree, out_arg in RING_CONFIGS:
        ids = groups.get(group_key, [])
        if len(ids) < 12:
            print(f"[WARN] Fewer than 12 ids in [{group_key.capitalize()}]; skipped.")
            continue
        if out_arg is None:
            continue
        out_p = Path(out_arg).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        active_rings.append((group_key.capitalize(), degree, ids[:12], out_p))
        print(f"[INFO] {group_key} ring: {ids[:12]} → {out_p}")

    do_chord = bool(args.enable_chord)
    major_ids: List[int] = []
    minor_ids: List[int] = []
    chord_out_file: Optional[Path] = None

    if do_chord:
        if not args.chord_out_file:
            raise ValueError("--chord-out-file required with --enable-chord")
        major_ids = groups.get("major chord", [])[:12]
        minor_ids = groups.get("minor chord", [])[:12]
        if len(major_ids) < 12:
            raise RuntimeError("[Major Chord] needs at least 12 ids for chord inference.")
        chord_out_file = Path(args.chord_out_file).resolve()
        chord_out_file.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] Chord mode: GT boundaries (chord_frame label changes)")
        print("[INFO] Silence   : GT labels N/X → silence_sc_t=1.0")

    lit, sae, model_factory = load_sae_and_factory(ckpt_path, args.device)
    module  = sae.sae
    use_tag = args.data_tag or str(lit.data_tag)
    n_sub   = int(getattr(module, "n_saes", 0))
    use_idx = max(0, min(args.sae_idx, n_sub - 1))
    print(f"[INFO] use_tag={use_tag}  sae_idx={use_idx}  split={args.split}")

    key_act_header = "\t".join([f"act_{n}:maj" for n in NOTE_NAMES_12])

    @contextlib.contextmanager
    def _open_opt(path_or_none: Optional[Path], header: str):
        if path_or_none is None:
            yield None
        else:
            with path_or_none.open("w", encoding="utf-8") as fh:
                fh.write(header + "\n")
                yield fh

    ring_file_ctxs = [
        _open_opt(out_p, f"# track_id\tseg_range\tref_key\test_key\tscore\t{key_act_header}")
        for _, _, _, out_p in active_rings
    ]

    with h5py.File(h5_path, "r") as f, \
         _open_opt(chord_out_file,     "# track_id\tseg_range\tref_chord\test_chord") as f_chord, \
         contextlib.ExitStack() as stack:

        ring_fhs = [stack.enter_context(ctx) for ctx in ring_file_ctxs]

        if args.split not in f:
            raise KeyError(f"Split '{args.split}' not found in {h5_path}.")
        g = f[args.split]
        if args.feature_ds not in g:
            raise KeyError(f"Feature dataset '{args.feature_ds}' not in split '{args.split}'.")

        has_key_labels   = ("labels" in g) and ("key_frame"   in g["labels"])
        has_chord_labels = ("labels" in g) and ("chord_frame" in g["labels"])
        if not has_key_labels and not args.allow_missing_labels:
            raise RuntimeError("Missing key_frame labels; pass --allow-missing-labels.")
        if do_chord and not has_chord_labels:
            raise RuntimeError(
                "chord_frame labels are required for GT-boundary chord inference "
                f"but are missing in {h5_path} split={args.split}."
            )

        ds_feat = g[args.feature_ds]
        n_total = int(ds_feat.shape[0])
        t_feat  = int(ds_feat.shape[-1])
        bs      = max(1, args.batch_size)

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
                    ref_key   = (
                        estimate_ref_key_from_frame(
                            decode_str_array(g["labels"]["key_frame"][seg_idx])
                        ) if has_key_labels else "N/A"
                    )
                    seg_range = segment_time_range_str(g, seg_idx, t_feat)
                    for fh in ring_fhs:
                        if fh is not None:
                            fh.write(f"{tracks[j]}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zero12}\n")
                pbar.update(len(idxs))
                continue

            feat   = feat_batch[use_tag]
            x_sae  = standardize_to_BTD_batch(feat, t_candidates=[t_feat, int(x_np.shape[-1])])
            with torch.no_grad():
                sparse_z = module.inference(x_sae, idx=use_idx, topk_wide=0)
            z_batch = sparse_z.detach().float().cpu().numpy()  # [B, T, D]
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
                    ref_key   = estimate_ref_key_from_frame(key_frames_batch[j])
                    for fh in ring_fhs:
                        if fh is not None:
                            fh.write(f"{track}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zero12}\n")
                    pbar.update(1)
                    continue

                t_z, d_z  = int(z.shape[0]), int(z.shape[1])
                seg_range = segment_time_range_str(g, seg_idx, t_z)

                key_frame = key_frames_batch[j]
                if key_frame is not None:
                    key_frame = key_frame[:t_z]
                ref_key = estimate_ref_key_from_frame(key_frame)

                # ── Key estimation (identical to step4_infer) ─────────────────
                for ri, (label, degree, ring_ids, _) in enumerate(active_rings):
                    fh = ring_fhs[ri]
                    if fh is None:
                        continue
                    ring_cols = [i for i in ring_ids if 0 <= i < d_z]
                    if len(ring_cols) < 12:
                        fh.write(f"{track}\t{seg_range}\t{ref_key}\tN/A\t0.000000\t{zero12}\n")
                        continue
                    est_key, est_score, key_scores_12 = estimate_key_from_ring(z, ring_cols, degree)
                    act12 = "\t".join([f"{float(v):.6f}" for v in key_scores_12.tolist()])
                    fh.write(f"{track}\t{seg_range}\t{ref_key}\t{est_key}\t{est_score:.6f}\t{act12}\n")

                # ── Chord inference (GT boundaries + GT silence) ──────────────
                if not do_chord:
                    pbar.update(1)
                    continue

                chord_frame_seg = chord_frames_batch[j]
                if chord_frame_seg is not None:
                    chord_frame_seg = chord_frame_seg[:t_z]

                major_cols       = [i for i in major_ids if 0 <= i < d_z]
                minor_cols       = [i for i in minor_ids if 0 <= i < d_z]
                if len(major_cols) < 12:   # major ring required; minor checked downstream
                    pbar.update(1)
                    continue

                # ── GT-specific sources: boundaries + silence ─────────────────
                # Boundaries from GT chord label changes; silence from GT N/X.
                chord_boundaries = _gt_chord_boundaries(chord_frame_seg, t_z)
                chord_segs   = _segments_from_boundaries(chord_boundaries, t_z) or [(0, t_z)]
                silence_sc_t = _gt_silence_score(chord_frame_seg, t_z)

                # Shared (smoothing-free) chord inference + output
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
    print("[DONE] GT-boundary inference complete.")


if __name__ == "__main__":
    main()
