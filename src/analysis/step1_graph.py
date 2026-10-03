#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Build the SAE pitch-shift *directed graph* and emit four files:
  orbits_raw.txt  — human-readable orbit records with source/type metadata:
                    [Ring12] raw directed 12-cycles, [Seq12] exact 12-node path
                    components, [Chain] leftover greedy chains, and
                    [GreedyRing12] leftover greedy 12-cycles.
  orbits_raw_score.txt — same orbits, but arrow-joined per-edge scores
                    (nodes[i] -> nodes[i+1], + closing edge for rings) instead
                    of feature ids.
  timbre.txt      — rank-1 self-loop node IDs (best neighbour is itself;
                    pitch-invariant / timbre)
  invalid.txt     — kept for output-file compatibility (empty under the
                    directed-graph pipeline; there is no single "invalid" set)

Ring/sequence detection reuses the directed-graph core in draw_graph.py:
  build_edges (top-k forward+backward average-cosine ghost-rescue edges over
    sub-SAEs 0/1/2; higher = better)
    → relative per-source prune (drop dist > relative_factor * top1_dist)
    → distance threshold (score >= score_threshold) + small-WCC removal
    → size-12 cycles (nx.simple_cycles) + exact 12-node directed sequences
    → drop higher-distance structures that share directed edges.

Note: this is the offline analysis path only. The training-time ring metric /
callback still use the lightweight functional-map core in
music_sae/self_graph.py (no networkx).

Usage:
    python -m src.analysis.step1_graph \
        --ckpt-path  store/.../ckpts/last.ckpt \
        --out-dir    store/.../analysis \
        [--sae-idx 0] [--device cuda] \
        [--topk 5] [--candidate-topk 20] [--block 512] [--score-mode avg] \
        [--relative-factor 1.5] [--score-threshold 0.2] [--min-wcc-size 10] \
        [--allow-shared-size12-edges]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

from .find_chord_rings import load_sae_and_factory
from .draw_graph import OrbitRecord, orbits_from_module


def compute_graph(
    module,
    device: str,
    *,
    topk: int = 6,
    candidate_topk: int = 6,
    block: int = 512,
    score_mode: str = "avg",
    relative_factor: float = 1.5,
    score_threshold: float = 0.8,
    min_wcc_size: int = 10,
    edge_disjoint: bool = True,
    min_chain_len: int = 11,
    jump_threshold: float = 0.8,
    replacement_pool: str = "all",
    overlap_mode: str = "edge",
    only_same_component_jump: bool = False,
    debug_outdir: str | None = None,
) -> Tuple[List[OrbitRecord], Set[int], Set[int]]:
    """Build the directed pitch-shift graph; extract size-12 structures first,
    then decompose the remaining graph into simple chains + size-12 rings.

    Returns
    -------
    orbits : list of OrbitRecord
        Size-12 cycles/sequences plus leftover greedy chains/rings with explicit
        source/type metadata for interpretable output.
    timbre : set of int
        Rank-1 self-loop nodes (best neighbour is themselves; pitch-invariant).
    invalid : set of int
        Empty — retained only so the emitted output-file set is unchanged.
    """
    orbits, rank1_self_nodes, K = orbits_from_module(
        module, device,
        topk=topk, candidate_topk=candidate_topk, block=block, score_mode=score_mode,
        relative_factor=relative_factor, score_threshold=score_threshold,
        min_wcc_size=min_wcc_size, edge_disjoint=edge_disjoint, include_leftover=True,
        min_chain_len=min_chain_len, return_records=True,
        jump_threshold=jump_threshold, replacement_pool=replacement_pool,
        overlap_mode=overlap_mode, only_same_component_jump=only_same_component_jump,
        debug_outdir=debug_outdir,
    )
    timbre: Set[int] = set(rank1_self_nodes)
    invalid: Set[int] = set()

    by_kind: Dict[str, int] = {}
    for rec in orbits:
        by_kind[rec.label] = by_kind.get(rec.label, 0) + 1
    counts = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())) or "none"
    print(
        f"[INFO] K={K} features | orbits: {counts} | "
        f"timbre(rank-1 self)={len(timbre)}"
    )
    return orbits, timbre, invalid


def _ids_str(ids: List[int]) -> str:
    return " -> ".join(str(x) for x in ids)


def _fmt_float(x) -> str:
    return "NA" if x is None else f"{float(x):.6f}"


def write_orbits(orbits: List[OrbitRecord], path: Path, *, legacy: bool = False) -> None:
    counters: Dict[str, int] = {}
    lines = []
    if not legacy:
        lines.extend([
            "# orbits_raw.txt",
            "# Tags:",
            "#   Ring12       = raw enumerated directed 12-cycle",
            "#   Seq12        = raw exact 12-node directed path component",
            "#   Chain        = leftover greedy path-cover chain",
            "#   GreedyRing12 = leftover greedy path-cover cycle of length 12",
            "# Metadata fields: source, kind, len, mean/min/max edge score, original structure id, ranks.",
            "",
        ])
    for rec in orbits:
        if legacy:
            label = "Ring" if rec.is_ring else "Seq"
            idx = counters.get(label, 0)
            counters[label] = idx + 1
            lines.append(f"[{label} {idx:02d}]: {len(rec.nodes)} {_ids_str(rec.nodes)}")
            continue

        idx = counters.get(rec.label, 0)
        counters[rec.label] = idx + 1
        meta = [
            f"source={rec.source}",
            f"kind={rec.kind}",
            f"len={len(rec.nodes)}",
        ]
        if rec.mean_score is not None:
            meta.extend([
                f"mean={_fmt_float(rec.mean_score)}",
                f"min={_fmt_float(rec.min_score)}",
                f"max={_fmt_float(rec.max_score)}",
            ])
        if rec.original_structure_id:
            meta.append(f"orig={rec.original_structure_id}")
        if rec.all_ranks:
            meta.append(f"ranks={rec.all_ranks}")
        lines.append(f"[{rec.label} {idx:02d} | {' | '.join(meta)}]: {len(rec.nodes)} {_ids_str(rec.nodes)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    fmt = "legacy" if legacy else "explained"
    print(f"[DONE] orbits_raw ({fmt}) → {path}  ({len(orbits)} orbits)")


def write_orbits_scores(orbits: List[OrbitRecord], path: Path, *, legacy: bool = False) -> None:
    """orbits_raw_score.txt — line-for-line mirror of orbits_raw.txt, but each
    arrow-joined entry is the edge score nodes[i] -> nodes[i+1] instead of the
    feature id (rings include the closing edge, so they have 12 scores; a
    sequence/chain of n nodes has n-1)."""
    counters: Dict[str, int] = {}
    lines = []
    if not legacy:
        lines.extend([
            "# orbits_raw_score.txt",
            "# Same orbits as orbits_raw.txt, but arrow-joined *edge scores*",
            "# (cycle-consistency distance, higher = better) instead of feature",
            "# ids: entry i is the score of edge nodes[i] -> nodes[i+1]; rings",
            "# also include the closing edge nodes[-1] -> nodes[0].",
            "",
        ])
    for rec in orbits:
        scores = rec.edge_scores or []
        scores_str = " -> ".join(f"{s:.4f}" for s in scores)
        if legacy:
            label = "Ring" if rec.is_ring else "Seq"
            idx = counters.get(label, 0)
            counters[label] = idx + 1
            lines.append(f"[{label} {idx:02d}]: {len(scores)} {scores_str}")
            continue

        idx = counters.get(rec.label, 0)
        counters[rec.label] = idx + 1
        meta = [
            f"source={rec.source}",
            f"kind={rec.kind}",
            f"len={len(rec.nodes)}",
        ]
        if rec.mean_score is not None:
            meta.extend([
                f"mean={_fmt_float(rec.mean_score)}",
                f"min={_fmt_float(rec.min_score)}",
                f"max={_fmt_float(rec.max_score)}",
            ])
        if rec.original_structure_id:
            meta.append(f"orig={rec.original_structure_id}")
        if rec.all_ranks:
            meta.append(f"ranks={rec.all_ranks}")
        lines.append(f"[{rec.label} {idx:02d} | {' | '.join(meta)}]: {len(scores)} {scores_str}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[DONE] orbits_raw_score → {path}  ({len(orbits)} orbits)")


def write_node_set(label: str, nodes: Set[int], path: Path) -> None:
    ids_str = " ".join(str(x) for x in sorted(nodes))
    line = f"[{label}]: {ids_str}" if ids_str else f"[{label}]:"
    path.write_text(line + "\n", encoding="utf-8")
    print(f"[DONE] {label.lower()} → {path}  ({len(nodes)} nodes)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Step 1: SAE directed pitch-shift graph → orbits_raw.txt, timbre.txt, invalid.txt"
    )
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--out-dir",   required=True)
    ap.add_argument("--sae-idx",   type=int, default=0,
                    help="Accepted for CLI compatibility; the graph always uses sub-SAEs 0,1,2.")
    ap.add_argument("--device",    type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    # Ghost-rescue average-cosine graph parameters.
    ap.add_argument("--topk",            type=int,   default=6,
                    help="Per-source high-neighbour top-k by average cosine score.")
    ap.add_argument("--candidate-topk",  type=int,   default=6,
                    help="Accepted for compatibility; ghost-rescue mode uses --topk.")
    ap.add_argument("--block",           type=int,   default=512)
    ap.add_argument("--score-mode",      choices=["avg"], default="avg",
                    help="Ghost-rescue mode uses avg cosine score only.")
    ap.add_argument("--relative-factor", type=float, default=1.5,
                    help="Accepted for compatibility; not used by ghost-rescue mode.")
    ap.add_argument("--score-threshold", type=float, default=0.7,
                    help="Keep top-k real edges with avg cosine score >= this.")
    ap.add_argument("--jump-threshold",  type=float, default=0.8,
                    help="Insert ghost k->GHOST->j when dot(W2[k], W0[j]) >= this and j is unreachable from k.")
    ap.add_argument("--replacement-pool", choices=["all", "topk"], default="all",
                    help="Candidate pool used to replace ghost nodes after size-12 structures are found.")
    ap.add_argument("--overlap-mode", choices=["edge", "node"], default="edge",
                    help="Overlap filter for resolved structures.")
    ap.add_argument("--only-same-component-jump", action="store_true",
                    help="Only insert ghosts when jump source and destination are in the same weak component.")
    ap.add_argument("--debug-graph-dir", type=str, default=None,
                    help="Optional directory for detailed ghost-rescue CSV debug outputs.")
    ap.add_argument("--min-wcc-size",    type=int,   default=10)
    ap.add_argument("--min-chain-len",   type=int,   default=11,
                    help="Keep leftover chains with length >= this (rings are always size 12).")
    ap.add_argument("--allow-shared-size12-edges", action="store_true",
                    help="Keep size-12 structures even if they share directed edges.")
    ap.add_argument("--legacy-orbits-format", action="store_true",
                    help="Write the old [Ring]/[Seq] orbits_raw.txt format without metadata.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lit, sae, _ = load_sae_and_factory(Path(args.ckpt_path), args.device)
    module = sae.sae
    n_sub = int(getattr(module, "n_saes", 0))
    print(f"[INFO] n_saes={n_sub} (graph uses sub-SAEs 0, 1, 2)")

    orbits, timbre, invalid = compute_graph(
        module, args.device,
        topk=args.topk,
        candidate_topk=args.candidate_topk,
        block=args.block,
        score_mode=args.score_mode,
        relative_factor=args.relative_factor,
        score_threshold=args.score_threshold,
        min_wcc_size=args.min_wcc_size,
        edge_disjoint=not args.allow_shared_size12_edges,
        min_chain_len=args.min_chain_len,
        jump_threshold=args.jump_threshold,
        replacement_pool=args.replacement_pool,
        overlap_mode=args.overlap_mode,
        only_same_component_jump=args.only_same_component_jump,
        debug_outdir=args.debug_graph_dir,
    )

    write_orbits(orbits,                out_dir / "orbits_raw.txt", legacy=args.legacy_orbits_format)
    write_orbits_scores(orbits,         out_dir / "orbits_raw_score.txt", legacy=args.legacy_orbits_format)
    write_node_set("Timbre",  timbre,   out_dir / "timbre.txt")
    write_node_set("Invalid", invalid,  out_dir / "invalid.txt")


if __name__ == "__main__":
    main()
