#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for labelling recovered SAE rings with musical concepts.

Used by step2_rings.py (offline anchor-based labelling) and by the
training-time validation metric (music_sae.chord_ring_metric):

  * checkpoint loading            (load_sae_and_factory)
  * chord / key label parsing     (chord_root_pc, chord_quality, ...)
  * song-wise H5 iteration        (iter_songs, load_songs, count_songs)
  * major / minor ring selection  (identify_maj_min_rings_global)
  * feature_ids.txt formatting    (format_ring_line)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..music_sae.sae_lit import LitSAE  # type: ignore
from .draw_graph import size12_structures_from_module

# ---------------------------------------------------------------------------
# Note / chord helpers
# ---------------------------------------------------------------------------

NOTE_NAMES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE2PC: Dict[str, int] = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}


def chord_root_pc(label: str) -> Optional[int]:
    c = str(label).strip()
    if not c or c.upper() in {"N", "NOCHORD", "X"}:
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


def chord_quality(label: str) -> Optional[str]:
    """Return 'maj', 'min', or None."""
    c = str(label).strip()
    if not c or c.upper() in {"N", "NOCHORD", "X"}:
        return None
    parts = c.split(":", 1)
    if len(parts) < 2:
        return None
    q = parts[1].strip().lower()
    if q.startswith("maj") or q in {"major", ""}:
        return "maj"
    if q.startswith("min") or q in {"minor"}:
        return "min"
    if q == "7":
        return "dom7"
    return None


def parse_key_signature_major_pc(label: str) -> Optional[int]:
    """
    Map key label to major-key-signature tonic (0..11).
      C:maj -> 0,  A:min -> 0 (relative major),  G:maj -> 7, etc.
    """
    s = str(label).strip()
    if not s or s.upper() in {"N", "N/A", "X"}:
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


def decode_str_scalar(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def decode_str_array(a: np.ndarray) -> np.ndarray:
    return np.array([decode_str_scalar(v) for v in a], dtype=object)


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def load_sae_and_factory(ckpt_path: Path, device: str):
    ckpt_path = ckpt_path.resolve()
    print(f"[INFO] Loading LitSAE from {ckpt_path}")
    lit: LitSAE = LitSAE.load_from_checkpoint(str(ckpt_path), map_location=device, strict=False)
    lit.eval().to(device)
    if hasattr(lit, "model_factory"):
        lit.model_factory.to(device)
    sae = lit.sae
    sae.eval().to(device)
    if not hasattr(sae, "sae"):
        raise RuntimeError("Expected GroupSAE to have attribute `sae` (MultiSAE).")
    return lit, sae, lit.model_factory


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


# ---------------------------------------------------------------------------
# Size-12 rings via the directed-graph pipeline (shared core in draw_graph.py).
# ---------------------------------------------------------------------------

def compute_size12_rings(
    module,
    device: str,
    *,
    topk: int = 5,
    candidate_topk: int = 20,
    block: int = 512,
    score_mode: str = "avg",
    relative_factor: float = 1.5,
    score_threshold: float = 0.2,
    min_wcc_size: int = 10,
    edge_disjoint: bool = True,
) -> List[List[int]]:
    """Return size-12 directed cycles + sequences (feature-id lists).

    Uses the directed-graph core in draw_graph.py (build_edges → prune →
    threshold/WCC filter → size-12 cycles/sequences → edge-disjoint filter).
    Edge scores are cycle-consistency distances (lower is better).
    """
    structures, _rank1_self, _K = size12_structures_from_module(
        module, device,
        topk=topk, candidate_topk=candidate_topk, block=block, score_mode=score_mode,
        relative_factor=relative_factor, score_threshold=score_threshold,
        min_wcc_size=min_wcc_size, edge_disjoint=edge_disjoint,
    )
    rings = [list(s["nodes"]) for s in structures]
    n_ring = sum(1 for s in structures if s["kind"] == "cycle")
    n_seq = sum(1 for s in structures if s["kind"] != "cycle")
    print(f"[INFO] Found {len(rings)} size-12 structures "
          f"({n_ring} rings, {n_seq} sequences).")
    return rings


# ---------------------------------------------------------------------------
# H5 data loading: song-by-song iterator
# ---------------------------------------------------------------------------

def _discover_song_segments(
    g,
    max_songs: Optional[int],
    track_ids: Optional[Set[str]],
    orig_only: bool,
) -> Tuple[List[str], Dict[str, List[int]], bool, bool]:
    """Group segment indices by trackId (optionally filtered).

    Returns (song_order, song_segs, has_chord, has_key).
    """
    has_track_id = "trackId" in g
    has_chord = "labels" in g and "chord_frame" in g["labels"]
    has_key = "labels" in g and "key_frame" in g["labels"]
    has_orig_flag = "semitone" in g or "aug" in g

    # Bulk-read the small string/meta columns once (per-element h5 reads in a
    # python loop over n_total are themselves slow on large splits).
    track_col = decode_str_array(g["trackId"][:]) if has_track_id else None
    # orig mask: prefer numeric `semitone` (==0), fall back to legacy `aug` string.
    orig_mask = None
    if orig_only and has_orig_flag:
        if "semitone" in g:
            orig_mask = (np.asarray(g["semitone"][:]) == 0)
        else:
            aug_col = decode_str_array(g["aug"][:])
            orig_mask = np.array([str(a) == "orig" for a in aug_col], dtype=bool)
    n_total = len(track_col) if track_col is not None else (
        len(orig_mask) if orig_mask is not None else 0
    )

    song_order: List[str] = []
    song_segs: Dict[str, List[int]] = {}
    for seg_idx in range(n_total):
        if orig_mask is not None and not orig_mask[seg_idx]:
            continue
        tid = str(track_col[seg_idx]) if track_col is not None else "all"
        if track_ids is not None and tid not in track_ids:
            continue
        if tid not in song_segs:
            song_segs[tid] = []
            song_order.append(tid)
            if max_songs is not None and len(song_order) > max_songs:
                break
        song_segs[tid].append(seg_idx)

    if max_songs is not None:
        song_order = song_order[:max_songs]
    return song_order, song_segs, has_chord, has_key


def _read_song(
    g, feature_ds: str, segs: List[int], has_chord: bool, has_key: bool
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    feat_np = np.stack([np.array(g[feature_ds][i], copy=True) for i in segs])
    t_len = int(feat_np.shape[-1])
    if has_chord:
        chord_2d = np.stack([decode_str_array(g["labels"]["chord_frame"][i]) for i in segs])
    else:
        chord_2d = np.full((len(segs), t_len), "N", dtype=object)
    if has_key:
        key_2d: Optional[np.ndarray] = np.stack(
            [decode_str_array(g["labels"]["key_frame"][i]) for i in segs]
        )
    else:
        key_2d = None
    return feat_np, chord_2d, key_2d


def iter_songs(
    h5_path: Path,
    split: str,
    feature_ds: str,
    max_songs: Optional[int] = None,
    track_ids: Optional[Set[str]] = None,
    orig_only: bool = False,
):
    """Stream songs one at a time as (track_id, feat_np, chord_2d, key_2d_or_None).

    Memory stays bounded to a single song; the caller is expected to consume and
    discard each song before requesting the next. Same filtering semantics as
    load_songs.
    """
    with h5py.File(h5_path, "r") as f:
        if split not in f:
            raise KeyError(f"Split '{split}' not found.")
        g = f[split]
        if feature_ds not in g:
            raise KeyError(f"Feature dataset '{feature_ds}' not in split.")
        song_order, song_segs, has_chord, has_key = _discover_song_segments(
            g, max_songs, track_ids, orig_only
        )
        for tid in song_order:
            feat_np, chord_2d, key_2d = _read_song(
                g, feature_ds, song_segs[tid], has_chord, has_key
            )
            yield tid, feat_np, chord_2d, key_2d


def count_songs(
    h5_path: Path,
    split: str,
    feature_ds: str,
    max_songs: Optional[int] = None,
    track_ids: Optional[Set[str]] = None,
    orig_only: bool = False,
) -> int:
    """Number of songs :func:`iter_songs` will yield (same filtering semantics).

    Only reads the small string/meta columns, not the features, so it is cheap
    enough to call before streaming to give a progress bar its total.
    """
    with h5py.File(h5_path, "r") as f:
        if split not in f:
            raise KeyError(f"Split '{split}' not found.")
        g = f[split]
        if feature_ds not in g:
            raise KeyError(f"Feature dataset '{feature_ds}' not in split.")
        song_order, _segs, _hc, _hk = _discover_song_segments(
            g, max_songs, track_ids, orig_only
        )
        return len(song_order)


def load_songs(
    h5_path: Path,
    split: str,
    feature_ds: str,
    max_songs: Optional[int] = None,
    track_ids: Optional[Set[str]] = None,
    orig_only: bool = False,
) -> List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Return list of (track_id, feat_np, chord_labels_2d, key_labels_2d_or_None)
    grouped by song. feat_np: [N_segs, ...], chord/key_labels: [N_segs, T_frames].

    If ``track_ids`` is given, only songs whose trackId is in the set are loaded.
    If ``orig_only`` is True and the split carries augmentation info (``semitone``,
    or a legacy ``aug`` column), only the non-augmented (semitone == 0 / aug ==
    'orig') segments are kept — required when reading a
    training H5 that contains pitch-shift augmented copies.

    Loads everything into memory; for large splits prefer iter_songs.
    """
    with h5py.File(h5_path, "r") as f:
        if split not in f:
            raise KeyError(f"Split '{split}' not found.")
        g = f[split]
        if feature_ds not in g:
            raise KeyError(f"Feature dataset '{feature_ds}' not in split.")
        song_order, song_segs, has_chord, has_key = _discover_song_segments(
            g, max_songs, track_ids, orig_only
        )
        n_segs_total = sum(len(song_segs[t]) for t in song_order)
        print(f"[INFO] Loading {len(song_order)} song(s) / {n_segs_total} segment(s) "
              f"from split='{split}' (orig_only={orig_only}).", flush=True)

        result = []
        for tid in tqdm(song_order, desc="load_songs", unit="song"):
            feat_np, chord_2d, key_2d = _read_song(
                g, feature_ds, song_segs[tid], has_chord, has_key
            )
            result.append((tid, feat_np, chord_2d, key_2d))

    return result


# ---------------------------------------------------------------------------
# Ring role scoring: major / minor (chord-label alignment)
# ---------------------------------------------------------------------------

def precompute_chord_pc_qual(chord_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parse chord labels once into per-frame (root_pc, quality_code) arrays.

    root_pc: int64, -1 where there is no parseable root.
    qual:    int64, 0=maj, 1=min, -1=other/none.
    Parsing each label is done a single time here so ring scoring can be fully
    vectorised over frames and offsets (the per-frame regex parse used to run
    12 x n_rings x 2 times, which dominated runtime on large frame sets).
    """
    T = len(chord_frame)
    root_pc = np.full(T, -1, dtype=np.int64)
    qual = np.full(T, -1, dtype=np.int64)
    for t in range(T):
        label = str(chord_frame[t])
        rp = chord_root_pc(label)
        if rp is None:
            continue
        root_pc[t] = int(rp)
        q = chord_quality(label)
        if q == "maj":
            qual[t] = 0
        elif q == "min":
            qual[t] = 1
    return root_pc, qual


# ---------------------------------------------------------------------------
# Major / Minor ring identification (global-argmax voting)
# ---------------------------------------------------------------------------
#
# Shared by the offline driver (step2_rings) and the training-time validation
# metric (music_sae.chord_ring_metric) so the code paths can never drift to
# different maj/min rules.
#
# Logic:
#   Step 1 — all rings pooled into one global candidate set. For every frame
#            with a parseable GT root, the single highest-activation pooled
#            feature wins and is tagged with that frame's (root, quality).
#            Counts accumulate per (winning feature, root): ``tot`` (any
#            quality), ``maj`` (quality 0), ``min_`` (quality 1).
#   Step 2 — per ring, pick the rotation offset maximising root-aligned marks
#            (correction is by root only — quality is ignored here); marks whose
#            root does not match the position's expected root are discarded.
#   Step 3 — at that fixed offset count the kept maj / min marks. The major ring
#            maximises (maj − min); the minor ring maximises (min − maj) among
#            the remaining rings.


def pooled_ring_cols(rings: List[List[int]]) -> Tuple[List[int], Dict[int, int]]:
    """Union of all feature columns across rings (the global-argmax candidate set).

    Returns ``(pooled, feat_to_k)`` where ``pooled`` is the sorted column list
    and ``feat_to_k`` maps a feature column id to its index into the per-feature
    accumulator arrays.
    """
    pooled = sorted({int(c) for ring in rings for c in ring})
    feat_to_k = {c: k for k, c in enumerate(pooled)}
    return pooled, feat_to_k


def accumulate_global_marks(
    z_t: np.ndarray,        # [N, D] frame activations
    root_pc: np.ndarray,    # [N]   -1 where no parseable root
    qual: np.ndarray,       # [N]    0=maj 1=min -1=other
    pooled: List[int],
    tot: np.ndarray,        # [K, 12] in/out (any quality)
    maj: np.ndarray,        # [K, 12] in/out (quality 0)
    min_: np.ndarray,       # [K, 12] in/out (quality 1)
) -> int:
    """Step 1 of global-argmax ring ID, accumulated over a frame chunk.

    Only frames with a valid GT root (``root_pc >= 0``) and a positive pooled
    activation contribute. The single highest-activation pooled feature wins the
    frame and is tagged with the frame's ``(root, quality)``. Updates ``tot`` /
    ``maj`` / ``min_`` in place and returns the number of contributing frames.
    """
    if z_t.shape[0] == 0:
        return 0
    pooled_arr = np.asarray(pooled, dtype=np.int64)
    K = len(pooled_arr)
    sub = z_t[:, pooled_arr]                            # [N, K]
    win = sub.argmax(axis=1)                            # [N] pooled-local index
    win_val = sub[np.arange(sub.shape[0]), win]
    m = (root_pc >= 0) & (win_val > 0)
    if not m.any():
        return 0
    kf = win[m].astype(np.int64)
    r = root_pc[m].astype(np.int64)
    q = qual[m]
    lin = kf * 12 + r
    tot += np.bincount(lin, minlength=K * 12).reshape(K, 12)
    mlin = lin[q == 0]
    if mlin.size:
        maj += np.bincount(mlin, minlength=K * 12).reshape(K, 12)
    nlin = lin[q == 1]
    if nlin.size:
        min_ += np.bincount(nlin, minlength=K * 12).reshape(K, 12)
    return int(m.sum())


def select_maj_min_rings(
    rings: List[List[int]],
    tot: np.ndarray,        # [K, 12]
    maj: np.ndarray,        # [K, 12]
    min_: np.ndarray,       # [K, 12]
    feat_to_k: Dict[int, int],
    semitone_step: int = 1,
) -> Dict[str, object]:
    """Step 2 (root correction → per-ring offset) + Step 3 (maj−min selection).

    Returns a dict with keys: ``major_ridx``, ``major_ring``, ``major_off``,
    ``major_score``, ``minor_ridx``, ``minor_ring``, ``minor_off``,
    ``minor_score`` and ``ring_stats`` (list of per-ring
    ``(ridx, off, matched, total, maj_cnt, min_cnt)``). Reordered rings are in
    pitch-class order (position ``p`` ↦ root ``p``); the minor fields are empty /
    ``-1`` when there is only one ring.
    """
    inv_step = pow(semitone_step, -1, 12)
    q_idx = np.arange(12)

    def reorder(ring: List[int], off: int) -> List[int]:
        return [ring[(off + p * inv_step) % 12] for p in range(12)]

    # Per ring: best offset (max root-aligned marks) + maj/min counts there.
    ring_stats: List[Tuple[int, int, int, int, int, int]] = []
    for ridx, ring in enumerate(rings):
        ks = np.array([feat_to_k[int(c)] for c in ring], dtype=np.int64)  # position q ↦ accumulator row
        tot_r = tot[ks]      # [12 (q), 12 (root)]
        total = int(tot_r.sum())
        best_off, best_matched = 0, -1
        for off in range(12):
            # position q ↦ expected root r = ((q - off) * step) % 12
            roots = ((q_idx - off) * semitone_step) % 12
            matched = int(tot_r[q_idx, roots].sum())
            if matched > best_matched:
                best_matched, best_off = matched, off
        roots = ((q_idx - best_off) * semitone_step) % 12
        maj_cnt = int(maj[ks][q_idx, roots].sum())
        min_cnt = int(min_[ks][q_idx, roots].sum())
        ring_stats.append((ridx, best_off, best_matched, total, maj_cnt, min_cnt))

    n = len(rings)
    maj_idx = max(range(n), key=lambda i: ring_stats[i][4] - ring_stats[i][5])
    others = [i for i in range(n) if i != maj_idx]
    min_idx = max(others, key=lambda i: ring_stats[i][5] - ring_stats[i][4]) if others else -1

    mi = ring_stats[maj_idx]
    out: Dict[str, object] = {
        "major_ridx": maj_idx,
        "major_ring": reorder(rings[maj_idx], mi[1]),
        "major_off": mi[1],
        "major_score": float(mi[4] - mi[5]),
        "minor_ridx": -1,
        "minor_ring": [],
        "minor_off": -1,
        "minor_score": 0.0,
        "ring_stats": ring_stats,
    }
    if min_idx >= 0:
        ni = ring_stats[min_idx]
        out.update({
            "minor_ridx": min_idx,
            "minor_ring": reorder(rings[min_idx], ni[1]),
            "minor_off": ni[1],
            "minor_score": float(ni[5] - ni[4]),
        })
    return out


def identify_maj_min_rings_global(
    rings: List[List[int]],
    z_t: np.ndarray,
    chord_frame: np.ndarray,
    semitone_step: int = 1,
) -> Dict[str, object]:
    """Convenience wrapper: Step 1+2+3 over a single in-memory frame matrix.

    Parses ``chord_frame`` once, runs the global-argmax accumulation over all
    frames, then selects the major/minor rings. See :func:`select_maj_min_rings`
    for the returned dict (with an extra ``n_marks`` key).
    """
    pooled, feat_to_k = pooled_ring_cols(rings)
    K = len(pooled)
    tot = np.zeros((K, 12), dtype=np.int64)
    maj = np.zeros((K, 12), dtype=np.int64)
    min_ = np.zeros((K, 12), dtype=np.int64)
    root_pc, qual = precompute_chord_pc_qual(chord_frame)
    n_marks = accumulate_global_marks(z_t, root_pc, qual, pooled, tot, maj, min_)
    out = select_maj_min_rings(rings, tot, maj, min_, feat_to_k, semitone_step)
    out["n_marks"] = n_marks
    return out


# ---------------------------------------------------------------------------
# Key helper for forth / fifth / tonic ring identification (step2_rings)
# ---------------------------------------------------------------------------

def get_song_key_tonic(key_frame: np.ndarray) -> Optional[int]:
    """Most frequent relative-major tonic across all frames."""
    counts = np.zeros(12, dtype=np.int64)
    for k in key_frame:
        pc = parse_key_signature_major_pc(str(k))
        if pc is not None:
            counts[pc] += 1
    return int(counts.argmax()) if counts.sum() > 0 else None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_ring_line(group_name: str, ring: List[int]) -> str:
    ids_str = " -> ".join(str(x) for x in ring)
    return f"[{group_name}]: {len(ring)} {ids_str}"

