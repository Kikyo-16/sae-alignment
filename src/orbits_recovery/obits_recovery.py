#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover rings and sequences from a MultiSAE successor map.

Successor confidence
--------------------
For exactly two sub-SAEs, SAE 0 is the reference space:

    v(i, j) = cos(w_i^(1), w_j^(0)).

For three or more sub-SAEs, an interior SAE s is the reference space:

    v_s(i, j) = 0.5 * [cos(w_i^(s+1), w_j^(s))
                     + cos(w_i^(s),   w_j^(s-1))].

Each feature keeps only its highest-confidence destination. A destination is
rejected when its score is below ``--prob``. Repeatedly following the accepted
successors recovers rings, fixed points, and open sequences.

Outputs
-------
  orbits_raw.txt  all recovered rings and sequences
  timbre.txt      fixed-point feature IDs
  invalid.txt     feature IDs whose best successor is below threshold
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Decoder extraction and top-1 matching are implemented locally; only checkpoint
# loading is shared with the analysis package.
from ..analysis.find_chord_rings import load_sae_and_factory


def _decoder_vectors(sae_module, device: str) -> torch.Tensor:
    """Return decoder feature vectors with shape [n_features, d_model].

    The current SAE implementation uses ``fc2.weight`` with shape
    [d_model, n_features], so decoder columns are transposed here. A few common
    alternative attribute names are supported to keep the analysis script
    compatible with older checkpoints/code revisions.
    """
    weight: Optional[torch.Tensor] = None
    n_features: Optional[int] = None

    fc1 = getattr(sae_module, "fc1", None)
    if fc1 is not None:
        n_features = getattr(fc1, "out_features", None)

    if n_features is None:
        hidden = getattr(sae_module, "hidden", None)
        if hidden is not None:
            n_features = int(hidden)

    fc2 = getattr(sae_module, "fc2", None)
    if fc2 is not None and hasattr(fc2, "weight"):
        weight = fc2.weight
        if n_features is None:
            n_features = getattr(fc2, "in_features", None)

    if weight is None:
        decoder = getattr(sae_module, "decoder", None)
        if decoder is not None and hasattr(decoder, "weight"):
            weight = decoder.weight
            if n_features is None:
                n_features = getattr(decoder, "in_features", None)

    if weight is None:
        for attr in ("W_dec", "w_dec", "decoder_weight"):
            candidate = getattr(sae_module, attr, None)
            if isinstance(candidate, torch.Tensor):
                weight = candidate
                break

    if weight is None:
        raise AttributeError(
            "Cannot locate decoder weights. Expected one of "
            "sae.fc2.weight, sae.decoder.weight, sae.W_dec, sae.w_dec, or "
            "sae.decoder_weight."
        )

    if weight.ndim != 2:
        raise ValueError(
            f"Decoder weight must be two-dimensional, got {tuple(weight.shape)}"
        )

    if n_features is not None:
        n_features = int(n_features)
        if weight.shape[1] == n_features:
            vectors = weight.transpose(0, 1)
        elif weight.shape[0] == n_features:
            vectors = weight
        else:
            raise ValueError(
                "Cannot infer decoder orientation: "
                f"weight shape={tuple(weight.shape)}, n_features={n_features}."
            )
    else:
        # In SAE decoders, the feature dimension is normally the larger axis.
        vectors = weight.transpose(0, 1) if weight.shape[1] >= weight.shape[0] else weight

    return vectors.detach().to(device=device, dtype=torch.float32)


@torch.no_grad()
def _top1_cosine_map(
    source: torch.Tensor,
    destination: torch.Tensor,
    block: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the highest-cosine destination for every source vector."""
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    if source.ndim != 2 or destination.ndim != 2:
        raise ValueError("source and destination must both have shape [M, D]")
    if source.shape[1] != destination.shape[1]:
        raise ValueError(
            "Decoder dimensions do not match: "
            f"source={tuple(source.shape)}, destination={tuple(destination.shape)}"
        )

    source_n = F.normalize(source, dim=1)
    destination_n = F.normalize(destination, dim=1)
    destination_t = destination_n.transpose(0, 1).contiguous()

    n_source = int(source_n.shape[0])
    best_indices = torch.empty(n_source, dtype=torch.long, device=source.device)
    best_scores = torch.empty(n_source, dtype=torch.float32, device=source.device)

    for start in range(0, n_source, block):
        end = min(n_source, start + block)
        scores = source_n[start:end] @ destination_t
        score, index = scores.max(dim=1)
        best_indices[start:end] = index
        best_scores[start:end] = score

    best_indices[best_scores < threshold] = -1
    return (
        best_indices.cpu().numpy().astype(np.int64),
        best_scores.cpu().numpy().astype(np.float32),
    )


@torch.no_grad()
def _bidirectional_top1_map(
    upward_source: torch.Tensor,
    reference: torch.Tensor,
    downward_destination: torch.Tensor,
    block: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find successors using the paper's bidirectional confidence."""
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")

    n_features = int(reference.shape[0])
    tensors = (upward_source, reference, downward_destination)
    if any(x.ndim != 2 for x in tensors):
        raise ValueError("All decoder matrices must have shape [M, D]")
    if any(int(x.shape[0]) != n_features for x in tensors):
        raise ValueError(
            "All sub-SAEs must have the same number of features: "
            f"up={upward_source.shape[0]}, ref={reference.shape[0]}, "
            f"down={downward_destination.shape[0]}"
        )
    if len({int(x.shape[1]) for x in tensors}) != 1:
        raise ValueError(
            "All sub-SAEs must have the same decoder dimension: "
            f"up={upward_source.shape[1]}, ref={reference.shape[1]}, "
            f"down={downward_destination.shape[1]}"
        )

    up_n = F.normalize(upward_source, dim=1)
    ref_n = F.normalize(reference, dim=1)
    down_n = F.normalize(downward_destination, dim=1)

    ref_t = ref_n.transpose(0, 1).contiguous()
    down_t = down_n.transpose(0, 1).contiguous()

    best_indices = torch.empty(n_features, dtype=torch.long, device=reference.device)
    best_scores = torch.empty(n_features, dtype=torch.float32, device=reference.device)

    for start in range(0, n_features, block):
        end = min(n_features, start + block)

        # f_i^(s+1) should match candidate f_j^(s).
        upward_score = up_n[start:end] @ ref_t

        # f_i^(s) should match candidate f_j^(s-1).
        downward_score = ref_n[start:end] @ down_t

        scores_avg = 0.5*(upward_score + downward_score)
        _, index = scores_avg.max(dim=1)

        scores_max = torch.maximum(upward_score, downward_score)
        best_score_max = scores_max.gather(1, index.unsqueeze(1)).squeeze(1)

        best_indices[start:end] = index
        best_scores[start:end] = best_score_max

    best_indices[best_scores < threshold] = -1
    return (
        best_indices.cpu().numpy().astype(np.int64),
        best_scores.cpu().numpy().astype(np.float32),
    )


def _resolve_reference_index(n_saes: int, requested: Optional[int]) -> int:
    if n_saes < 2:
        raise RuntimeError(
            f"Need at least two sub-SAEs for successor recovery, got {n_saes}."
        )

    if n_saes == 2:
        if requested not in (None, 0):
            raise ValueError(
                "With exactly two sub-SAEs, --sae-idx must be 0."
            )
        return 0

    reference_index = n_saes // 2 if requested is None else int(requested)
    if not 0 < reference_index < n_saes - 1:
        raise ValueError(
            f"With {n_saes} sub-SAEs, --sae-idx must be an interior index "
            f"in [1, {n_saes - 2}], got {reference_index}."
        )
    return reference_index


def _build_successor_map(
    module,
    device: str,
    reference_index: int,
    block: int,
    threshold: float,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Set[int], Set[int], int]:
    sub_saes = list(module.sae_modules)
    vectors = [_decoder_vectors(sae, device) for sae in sub_saes]

    if len(sub_saes) == 2:
        successor, scores = _top1_cosine_map(
            source=vectors[1],
            destination=vectors[0],
            block=block,
            threshold=threshold,
        )
        score_type = "one-direction"
    else:
        s = reference_index
        successor, scores = _bidirectional_top1_map(
            upward_source=vectors[s + 1],
            reference=vectors[s],
            downward_destination=vectors[s - 1],
            block=block,
            threshold=threshold,
        )
        score_type = "bidirectional"

    n_features = int(successor.shape[0])
    forward: Dict[int, int] = {}
    reverse: Dict[int, List[int]] = {}
    timbre: Set[int] = set()
    invalid: Set[int] = set()

    # In-degree de-duplication: when several predecessors point to the same
    # successor, keep only the highest-confidence transition (Sec. 3.3).
    best_pred: Dict[int, Tuple[int, float]] = {}
    rejected_preds: Set[int] = set()

    for source_index, destination_index_raw in enumerate(successor):
        destination_index = int(destination_index_raw)
        if destination_index < 0:
            invalid.add(source_index)
        elif destination_index == source_index:
            timbre.add(source_index)
        else:
            score_i = float(scores[source_index])
            if destination_index not in best_pred:
                best_pred[destination_index] = (source_index, score_i)
            else:
                prev_source, prev_score = best_pred[destination_index]
                if score_i > prev_score:
                    rejected_preds.add(prev_source)
                    best_pred[destination_index] = (source_index, score_i)
                else:
                    rejected_preds.add(source_index)

    for destination_index, (source_index, _) in best_pred.items():
        forward[source_index] = destination_index
        reverse[destination_index] = [source_index]

    invalid.update(rejected_preds)

    accepted = scores[successor >= 0]
    if accepted.size:
        score_summary = (
            f"min={accepted.min():.4f}, mean={accepted.mean():.4f}, "
            f"max={accepted.max():.4f}"
        )
    else:
        score_summary = "no accepted transitions"

    print(
        f"[INFO] score={score_type} reference_sae={reference_index} "
        f"threshold={threshold:.4f} {score_summary}"
    )
    return forward, reverse, timbre, invalid, n_features


def _trace_orbits(
    forward: Dict[int, int],
    reverse: Dict[int, List[int]],
) -> List[Tuple[List[int], bool]]:
    """Trace rings and open sequences from a partial functional map."""

    def trace_from(start: int) -> List[Tuple[List[int], bool]]:
        current = start
        path = [start]

        while current in forward:
            current = forward[current]
            if current in path:
                cycle_start = path.index(current)
                cycle = path[cycle_start:]
                parts: List[Tuple[List[int], bool]] = [(cycle, True)]
                if cycle_start > 0:
                    parts.append((path[: cycle_start + 1], False))
                return parts
            path.append(current)

        return [(path, False)]

    visited: Set[int] = set()
    orbits: List[Tuple[List[int], bool]] = []

    # Start with roots of open trajectories.
    for start in (node for node in forward if node not in reverse):
        if start in visited:
            continue
        for nodes, is_ring in trace_from(start):
            visited.update(nodes)
            orbits.append((nodes, is_ring))

    # Remaining unvisited nodes belong to cycles or trajectories feeding cycles.
    for start in forward:
        if start in visited:
            continue
        for nodes, is_ring in trace_from(start):
            visited.update(nodes)
            orbits.append((nodes, is_ring))

    return orbits


def compute_graph(
    module,
    device: str,
    block: int = 1024,
    prob: float = 0.5,
    sae_idx: Optional[int] = None,
) -> Tuple[List[Tuple[List[int], bool]], Set[int], Set[int]]:
    sub_saes = list(module.sae_modules)
    reference_index = _resolve_reference_index(len(sub_saes), sae_idx)

    forward, reverse, timbre, invalid, n_features = _build_successor_map(
        module=module,
        device=device,
        reference_index=reference_index,
        block=block,
        threshold=prob,
    )
    orbits = _trace_orbits(forward, reverse)

    size_12 = [(nodes, is_ring) for nodes, is_ring in orbits if len(nodes) == 12]
    n_rings = sum(is_ring for _, is_ring in size_12)
    n_sequences = len(size_12) - n_rings
    n_pitch = n_features - len(timbre) - len(invalid)

    print(
        f"[INFO] {n_features} features: {len(timbre)} timbre, "
        f"{len(invalid)} invalid, {n_pitch} pitch nodes"
    )
    print(
        f"[INFO] Found {len(orbits)} structures total "
        f"({len(size_12)} size-12: {n_rings} rings, "
        f"{n_sequences} sequences)"
    )
    return orbits, timbre, invalid


def _ids_string(ids: List[int]) -> str:
    return " -> ".join(str(value) for value in ids)


def write_orbits(orbits: List[Tuple[List[int], bool]], path: Path) -> None:
    ring_index = 0
    sequence_index = 0
    lines: List[str] = []

    for ids, is_ring in orbits:
        if is_ring:
            tag = f"Ring {ring_index:02d}"
            ring_index += 1
        else:
            tag = f"Seq {sequence_index:02d}"
            sequence_index += 1
        lines.append(f"[{tag}]: {len(ids)} {_ids_string(ids)}")

    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    print(f"[DONE] orbits_raw -> {path} ({len(lines)} structures)")


def write_node_set(label: str, nodes: Set[int], path: Path) -> None:
    values = " ".join(str(value) for value in sorted(nodes))
    line = f"[{label}]: {values}" if values else f"[{label}]:"
    path.write_text(line + "\n", encoding="utf-8")
    print(f"[DONE] {label.lower()} -> {path} ({len(nodes)} nodes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover MultiSAE successor-map structures and write "
            "orbits_raw.txt, timbre.txt, and invalid.txt."
        )
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--sae-idx",
        type=int,
        default=None,
        help=(
            "Reference sub-SAE. For two SAEs this must be 0. For three or "
            "more SAEs the default is the middle interior SAE."
        ),
    )
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument(
        "--prob",
        "--threshold",
        dest="prob",
        type=float,
        default=0.5,
        help="Minimum accepted successor confidence.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, sae, _ = load_sae_and_factory(Path(args.ckpt_path), args.device)
    module = sae.sae
    n_saes = len(list(module.sae_modules))
    reference_index = _resolve_reference_index(n_saes, args.sae_idx)
    print(f"[INFO] sae_idx={reference_index} n_saes={n_saes}")

    orbits, timbre, invalid = compute_graph(
        module=module,
        device=args.device,
        block=args.block,
        prob=args.prob,
        sae_idx=reference_index,
    )

    write_orbits(orbits, output_dir / "orbits_raw.txt")
    write_node_set("Timbre", timbre, output_dir / "timbre.txt")
    write_node_set("Invalid", invalid, output_dir / "invalid.txt")


if __name__ == "__main__":
    main()