#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Classify size-12 rings/sequences as Major, Minor, Forth, Fifth, Tonic
and write feature_ids.txt (which step3_boundary.py will later append to).

Reads rings_raw.txt produced by step1_graph.py, runs SAE inference on the
first few songs of an H5 dataset, then uses musical alignment to assign roles.

Major/Minor: scored by chord-label alignment across all songs (concatenated).
Forth/Fifth/Tonic: per-song, per-segment dominant-feature voting, then
  cross-song majority vote for degree assignment.

Usage:
    python -m src.analysis.step2_rings \\
        --ckpt-path    store/.../ckpts/last.ckpt \\
        --rings-file   store/.../analysis/rings_raw.txt \\
        --h5-path      sae-data/pop909/muq/pop909_muq_30s_layer_2.h5 \\
        --split        train \\
        --feature-ds   muq_layer \\
        --out-file     store/.../analysis/feature_ids.txt \\
        [--sae-idx 1] [--device cuda] [--batch-size 32] [--max-songs 1]
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .find_chord_rings import (
    NOTE_NAMES_12,
    chord_quality,
    chord_root_pc,
    format_ring_line,
    get_song_key_tonic,
    identify_maj_min_rings_global,
    load_sae_and_factory,
    load_songs,
    standardize_to_BTD_batch,
)


def parse_rings_file(path: Path) -> List[List[int]]:
    """
    Parse rings_raw.txt produced by step1_graph.py.
    Returns list of size-12 node-ID lists (Ring and Seq entries both included).
    """
    rings: List[List[int]] = []
    header_re = re.compile(r'^\[(.+?)\]:\s*(.+)$')
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = header_re.match(line)
        if not m:
            continue
        rest = m.group(2).strip()
        if '->' in rest:
            parts = rest.split(None, 1)
            ids_str = parts[1] if (len(parts) == 2 and parts[0].isdigit()) else rest
            ids = [int(x.strip()) for x in ids_str.split('->')]
        else:
            tokens = rest.split()
            if tokens and tokens[0].isdigit():
                tokens = tokens[1:]
            try:
                ids = [int(t) for t in tokens]
            except ValueError:
                continue
        if len(ids) == 12:
            rings.append(ids)
    return rings


def _infer_songs_segmented(
    songs,
    module,
    model_factory,
    use_tag: str,
    use_idx: int,
    device: str,
    batch_size: int,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Run SAE inference per song.
    Returns list of (z_segs, chord_segs, key_segs) one entry per song.
      z_segs:    [N_valid, T, D]
      chord_segs: [N_valid, T]  (string labels)
      key_segs:   [N_valid, T]  (string labels) or None
    Only segments that pass through the model successfully are included;
    chord_segs/key_segs rows are aligned to those same segments.
    """
    result = []
    for tid, feat_np, chord_2d, key_2d in songs:
        n_segs = feat_np.shape[0]
        t_feat = int(feat_np.shape[-1])
        bs     = max(1, int(batch_size))

        z_parts:      List[np.ndarray] = []
        valid_indices: List[int]       = []

        for s0 in range(0, n_segs, bs):
            s1  = min(n_segs, s0 + bs)
            x   = torch.from_numpy(feat_np[s0:s1]).float().to(device)
            with torch.no_grad():
                feature_batch = model_factory({use_tag: x}, t=0)
            if use_tag not in feature_batch:
                continue
            x_for_sae = standardize_to_BTD_batch(
                feature_batch[use_tag], t_candidates=[t_feat, feat_np.shape[-1]]
            )
            with torch.no_grad():
                sparse_z = module.inference(x_for_sae, idx=use_idx, topk_wide=0)
            z_batch = sparse_z.detach().float().cpu().numpy()   # [B, T, D]
            if z_batch.ndim != 3:
                continue
            for j in range(z_batch.shape[0]):
                z_parts.append(z_batch[j])
                valid_indices.append(s0 + j)

        if not z_parts:
            print(f"[WARN] Song {tid}: no valid segments, skipped.")
            continue

        vi        = np.array(valid_indices, dtype=np.int64)
        z_segs    = np.stack(z_parts, axis=0)        # [N_valid, T, D]
        chord_out = chord_2d[vi]                     # [N_valid, T]
        key_out   = key_2d[vi] if key_2d is not None else None

        print(f"[INFO] Song {tid}: {z_segs.shape[0]} segments  "
              f"T={z_segs.shape[1]}  D={z_segs.shape[2]}")
        result.append((z_segs, chord_out, key_out))

    return result


def _vote_forth_fifth_tonic(
    candidate_rings: List[Tuple[int, List[int]]],
    songs_inferred:  List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    semitone_step:   int = 1,
) -> Tuple[
    Tuple[List[int], int],
    Tuple[List[int], int],
    Tuple[List[int], int],
]:
    """
    Per-song, per-segment voting to identify Forth/Fifth/Tonic rings.

    Each song is processed independently (supports different keys across songs):
      1. Determine the song's key tonic; keep only same-key segments.
      2. Per segment: find the ring feature with the highest mean activation
         → that segment's vote for the ring's dominant feature.
      3. Elected feature (majority across same-key segments) → root PC from
         major-chord activation → degree = (root - key_tonic) % 12.

    After all songs, each ring's degree = majority vote across songs.
    When multiple rings share a degree, a per-segment activation tournament
    (across all same-key segments of songs where both rings were classified
    with that degree) decides the winner.

    Returns (forth, fifth, tonic) as ((ring, c_offset), ...).
    """
    inv_step  = pow(semitone_step, -1, 12)
    _DEG_NAME = {0: "Tonic", 5: "Forth", 7: "Fifth"}

    rings_lookup: Dict[int, List[int]] = {ridx: ring for ridx, ring in candidate_rings}

    def reorder(ring: List[int], c_offset: int) -> List[int]:
        return [ring[(c_offset + p * inv_step) % 12] for p in range(12)]

    def _fmt_root_acts(root_sums: np.ndarray, root_count: np.ndarray, top: int = 5) -> str:
        """Format top-N root activations as 'F=15.3(23f) G=12.1(18f) ...'"""
        ranked = sorted(
            [(i, root_sums[i], root_count[i]) for i in range(12) if root_sums[i] > 0],
            key=lambda x: -x[1],
        )[:top]
        return "  ".join(f"{NOTE_NAMES_12[i]}={v:.1f}({n}f)" for i, v, n in ranked) or "(none)"

    def _fmt_votes(vote_counts: Dict[int, int], top: int = 3) -> str:
        """Format top-N feature vote counts as 'feat1234×6  feat5678×1  ...'"""
        ranked = sorted(vote_counts.items(), key=lambda x: -x[1])[:top]
        return "  ".join(f"feat{f}×{c}" for f, c in ranked)

    # Per ring accumulate per-song records when a valid degree is found.
    # record keys: song_idx, voted_feat, voted_pos, degree, c_offset, sk_idx
    ring_records:      Dict[int, List[Dict]] = {ridx: [] for ridx, _ in candidate_rings}
    ring_degree_votes: Dict[int, Counter]    = {ridx: Counter() for ridx, _ in candidate_rings}

    n_songs = len(songs_inferred)
    for song_idx, (z_segs, chord_segs, key_segs) in enumerate(songs_inferred):
        if key_segs is None:
            print(f"\n── Song {song_idx}/{n_songs} ── no key labels, skipped")
            continue

        N_segs, T, _ = z_segs.shape

        song_key_tonic_i = get_song_key_tonic(key_segs.ravel())
        if song_key_tonic_i is None:
            print(f"\n── Song {song_idx}/{n_songs} ── cannot determine key, skipped")
            continue

        # Discard songs with any modulation — only single-key songs are reliable.
        seg_keys = [get_song_key_tonic(key_segs[s]) for s in range(N_segs)]
        n_off = sum(1 for k in seg_keys if k != song_key_tonic_i)
        if n_off > 0:
            print(f"\n── Song {song_idx}/{n_songs}  key={NOTE_NAMES_12[song_key_tonic_i]}"
                  f"  [{n_off}/{N_segs} segs differ] → skipped (multi-key)")
            continue

        N_sk = N_segs
        print(f"\n── Song {song_idx}/{n_songs}  key={NOTE_NAMES_12[song_key_tonic_i]}"
              f"  segs: {N_sk} (all same-key)")

        sk_idx   = np.arange(N_segs, dtype=np.int64)
        z_sk     = z_segs       # [N_sk, T, D]
        chord_sk = chord_segs   # [N_sk, T]

        # Precompute major-chord root PC per frame (−1 if not a major chord).
        major_root = np.full((N_sk, T), -1, dtype=np.int64)
        for si in range(N_sk):
            for t in range(T):
                rpc  = chord_root_pc(str(chord_sk[si, t]))
                qual = chord_quality(str(chord_sk[si, t]))
                if rpc is not None and qual == "maj":
                    major_root[si, t] = rpc
        n_maj_frames = int((major_root >= 0).sum())
        print(f"   major-chord frames: {n_maj_frames}/{N_sk * T}")

        for ridx, ring in candidate_rings:
            ring_arr = np.array(ring, dtype=np.int64)

            # --- Step 1+2: per-segment vote for the ring's dominant feature ---
            vote_counts: Dict[int, int] = {}
            for si in range(N_sk):
                mean_act = z_sk[si][:, ring_arr].mean(axis=0)   # [12]
                dom_feat = ring[int(mean_act.argmax())]
                vote_counts[dom_feat] = vote_counts.get(dom_feat, 0) + 1

            voted_feat = max(vote_counts, key=vote_counts.__getitem__)
            voted_pos  = ring.index(voted_feat)
            n_votes    = vote_counts[voted_feat]

            # --- Step 3: root PC from major-chord activation (vectorised) ---
            feat_act_sk = z_sk[:, :, voted_feat]   # [N_sk, T]
            root_sums   = np.zeros(12, dtype=np.float64)
            root_count  = np.zeros(12, dtype=np.int64)
            for rpc_val in range(12):
                m = (major_root == rpc_val)
                if m.any():
                    root_sums[rpc_val]  = float(feat_act_sk[m].sum())
                    root_count[rpc_val] = int(m.sum())

            # Print per-ring analysis block
            vote_str = _fmt_votes(vote_counts)
            print(f"   Ring {ridx:2d}: vote [{vote_str}] → elected feat{voted_feat}"
                  f" ({n_votes}/{N_sk} segs)")

            if root_sums.sum() == 0:
                print(f"           root acts: (no major-chord activation) → skipped")
                continue

            root_str  = _fmt_root_acts(root_sums, root_count)
            best_root = int(root_sums.argmax())
            diff      = (best_root - song_key_tonic_i) % 12
            degree    = diff if diff in (0, 5, 7) else -1
            c_offset  = int((voted_pos - best_root * inv_step) % 12)
            deg_str   = _DEG_NAME.get(degree, f"? (diff={diff})")

            print(f"           root acts: {root_str}")
            print(f"           best root: {NOTE_NAMES_12[best_root]}"
                  f"  diff=(root-key)%12={diff}  → {deg_str}")

            if degree != -1:
                ring_degree_votes[ridx][degree] += 1
                ring_records[ridx].append({
                    "song_idx":       song_idx,
                    "song_key_tonic": song_key_tonic_i,
                    "voted_feat":     voted_feat,
                    "voted_pos":      voted_pos,
                    "degree":         degree,
                    "c_offset":       c_offset,
                    "sk_idx":         sk_idx,
                    "vote_frac":      n_votes / N_sk,
                })

    # --- Determine each ring's final degree by cross-song majority vote ---
    print(f"\n{'═'*60}")
    print(f"Degree summary across {n_songs} song(s)")
    print(f"{'═'*60}")

    ring_degree:  Dict[int, int] = {}
    ring_coffset: Dict[int, int] = {}

    for ridx, _ in candidate_rings:
        votes = ring_degree_votes[ridx]
        if not votes:
            print(f"  Ring {ridx:2d}: no valid degree in any song → unclassified")
            continue
        best_deg  = votes.most_common(1)[0][0]
        ring_degree[ridx]  = best_deg
        c_offsets = [r["c_offset"] for r in ring_records[ridx] if r["degree"] == best_deg]
        ring_coffset[ridx] = Counter(c_offsets).most_common(1)[0][0]

        vote_detail = "  ".join(
            f"{_DEG_NAME.get(d,'?')}×{c}" for d, c in votes.most_common()
        )
        print(f"  Ring {ridx:2d}: {vote_detail}  →  {_DEG_NAME.get(best_deg,'?')}"
              f"  C_offset={ring_coffset[ridx]}")

    # --- Assign best ring per degree (a ring may serve multiple degrees) ---
    print()
    result_ring: Dict[int, List[int]] = {}
    result_c:    Dict[int, int]       = {}

    for degree in (5, 7, 0):
        cands = [r for r, d in ring_degree.items() if d == degree]
        if not cands:
            print(f"[WARN] {_DEG_NAME[degree]} ring not found.")
            continue

        if len(cands) == 1:
            best = cands[0]
            print(f"[{_DEG_NAME[degree]}] single candidate: Ring {best}")
        else:
            print(f"[{_DEG_NAME[degree]}] {len(cands)} candidates: rings {cands}")
            # Score each ring by its mean vote_frac across songs where it was
            # classified as this degree.
            scores: Dict[int, float] = {}
            for r in cands:
                # Songs with no record are excluded (ring not analysed there).
                # Songs where ring was classified as a DIFFERENT degree score 0.
                fracs = [rec["vote_frac"] if rec["degree"] == degree else 0.0
                         for rec in ring_records[r]]
                scores[r] = float(np.mean(fracs)) if fracs else 0.0
                n_match = sum(1 for rec in ring_records[r] if rec["degree"] == degree)
                print(f"  Ring {r}: mean vote_frac={scores[r]:.3f}"
                      f"  ({n_match}/{len(ring_records[r])} song(s) matched degree)")

            best = max(cands, key=lambda r: scores[r])
            print(f"  → Ring {best} wins (score={scores[best]:.3f})")

        result_ring[degree] = reorder(rings_lookup[best], ring_coffset[best])
        result_c[degree]    = ring_coffset[best]
        print(f"  → {_DEG_NAME[degree]} ring: idx={best}  C_offset={ring_coffset[best]}")

    return (
        (result_ring.get(5, []), result_c.get(5, -1)),
        (result_ring.get(7, []), result_c.get(7, -1)),
        (result_ring.get(0, []), result_c.get(0, -1)),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Step 2: classify size-12 rings → feature_ids.txt"
    )
    ap.add_argument("--ckpt-path",   required=True)
    ap.add_argument("--rings-file",  required=True,
                    help="orbits_raw.txt from step1_graph.py")
    ap.add_argument("--h5-path",     required=True,
                    help="H5 dataset for alignment (e.g. pop909 train)")
    ap.add_argument("--split",       default="train")
    ap.add_argument("--feature-ds",  required=True,
                    help="Feature dataset key inside the H5 (e.g. muq_layer)")
    ap.add_argument("--out-file",    required=True,
                    help="Output feature_ids.txt path")
    ap.add_argument("--sae-idx",     type=int, default=0)  # 0 = anchor sub-SAE
    ap.add_argument("--device",      type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",  type=int, default=32)
    ap.add_argument("--max-songs",   type=int, default=5,
                    help="Songs to use for musical alignment (default: 5)")
    ap.add_argument("--semitone-step", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    rings_path = Path(args.rings_file)
    out_path   = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rings = parse_rings_file(rings_path)
    if not rings:
        raise RuntimeError(f"No size-12 structures found in {rings_path}")
    print(f"[INFO] Loaded {len(rings)} size-12 ring(s)/seq(s) from {rings_path}")

    lit, sae, model_factory = load_sae_and_factory(Path(args.ckpt_path), args.device)
    module  = sae.sae
    use_tag = str(lit.data_tag)
    n_sub   = int(getattr(module, "n_saes", 0))
    use_idx = max(0, min(args.sae_idx, n_sub - 1))
    print(f"[INFO] use_tag={use_tag}  sae_idx={use_idx}  split={args.split}")

    max_songs: Optional[int] = args.max_songs if args.max_songs > 0 else None
    songs = load_songs(
        Path(args.h5_path), args.split, args.feature_ds, max_songs=max_songs
    )

    # Run inference per song (enables per-song key handling).
    songs_inferred = _infer_songs_segmented(
        songs, module, model_factory, use_tag, use_idx, args.device, args.batch_size,
    )
    if not songs_inferred:
        raise RuntimeError("No valid segments produced from any song.")

    # Concatenate across songs for major/minor chord-alignment scoring.
    z_t         = np.concatenate([z.reshape(-1, z.shape[-1]) for z, _, _ in songs_inferred])
    chord_frame = np.concatenate([c.ravel() for _, c, _ in songs_inferred])
    print(f"[INFO] Total frames for major/minor alignment: {z_t.shape[0]}  D={z_t.shape[1]}")

    # --- Major / Minor rings (shared global-argmax voting + root correction) ---
    print(f"[INFO] Identifying major/minor rings (global-argmax voting) over {len(rings)} ring(s) ...")
    sel = identify_maj_min_rings_global(rings, z_t, chord_frame, args.semitone_step)
    maj_ridx = int(sel["major_ridx"])
    major_ring: List[int] = sel["major_ring"]
    print(f"[INFO] Major ring idx={maj_ridx}  score(maj-min)={sel['major_score']:.0f}  "
          f"C_offset={sel['major_off']}  (n_marks={sel['n_marks']})")

    min_ridx = int(sel["minor_ridx"])
    minor_ring: List[int] = sel["minor_ring"]
    if minor_ring:
        print(f"[INFO] Minor ring idx={min_ridx}  score(min-maj)={sel['minor_score']:.0f}  "
              f"C_offset={sel['minor_off']}")

    # --- Forth / Fifth / Tonic: per-song voting ---
    # Major ring is kept as a candidate (can double as a degree ring).
    # Minor ring is excluded to avoid confusion with minor-mode keys.
    _minor_excl: Set[int] = {min_ridx} if minor_ring else set()
    candidate_rings = [(ridx, ring) for ridx, ring in enumerate(rings)
                       if ridx not in _minor_excl]

    print(f"\n{'─'*60}")
    print(f"Candidate rings/seqs for Forth/Fifth/Tonic ({len(candidate_rings)} total)")
    print(f"{'─'*60}")
    for ridx, ring in candidate_rings:
        print(f"  Ring {ridx:2d}: {' -> '.join(str(x) for x in ring)}")
    print()
    forth_ring: List[int] = []
    fifth_ring: List[int] = []
    tonic_ring: List[int] = []

    if not candidate_rings:
        print("[WARN] No candidate rings left for Forth/Fifth/Tonic.")
    else:
        has_key = any(ks is not None for _, _, ks in songs_inferred)
        if not has_key:
            print("[WARN] No key_frame labels; skipping Forth/Fifth/Tonic identification.")
        else:
            (forth_ring, _), (fifth_ring, _), (tonic_ring, _) = _vote_forth_fifth_tonic(
                candidate_rings, songs_inferred, args.semitone_step,
            )

    lines: List[str] = []
    if major_ring:
        lines.append(format_ring_line("Major Chord", major_ring))
    if minor_ring:
        lines.append(format_ring_line("Minor Chord", minor_ring))
    if forth_ring:
        lines.append(format_ring_line("Forth", forth_ring))
    if fifth_ring:
        lines.append(format_ring_line("Fifth", fifth_ring))
    if tonic_ring:
        lines.append(format_ring_line("Tonic", tonic_ring))

    if not lines:
        raise RuntimeError("No rings identified; cannot write feature_ids.txt.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[DONE] feature_ids.txt → {out_path}")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
