#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize a Multi-SAE predecessor/successor graph with edge cap ``ne``.

This script follows the score used by ``obits_recovery.py``:

* Two sub-SAEs::

      score(i, j) = cos(w_i^(1), w_j^(0))

* Three or more sub-SAEs, with reference SAE ``s``::

      score(i, j) = cos(w_i^(s+1), w_j^(s))
                    * cos(w_i^(s), w_j^(s-1))

For each feature ``i``, candidate successors ``j`` below ``--threshold`` are
removed. The script computes the ne highest-scoring outgoing edges of every source and the
ne highest-scoring incoming edges of every destination without materializing the full score
matrix on CPU.

Default edge policy: ``mutual``
    Keep i -> j only when it is both one of the ne highest-scoring successors of i and one of the ne highest-scoring
    predecessors of j. Therefore every node has at most ne retained predecessors
    and at most ne retained successors.

Attention-node colors:
    Forth       black
    Major Chord red
    Minor Chord green

Outputs
-------
``ne_orbit_edges.tsv``
    Retained directed edges, scores, and incoming/outgoing ranks.
``ne_orbit_graph.graphml``
    Graph for later interactive analysis.
``ne_orbit_components.tsv``
    Weakly connected-component statistics.
``ne_orbit_graph_all.<format>``
    All retained components; only attention nodes are labelled by default.
``ne_orbit_graph_focus.<format>``
    Components containing Forth/Major/Minor nodes; all nodes are labelled when
    the focused graph is not too large.

Recommended invocation from the project root::

    python -m src.analysis.visualize_ne_orbit_graph \
      --ckpt-path /path/to/model.ckpt \
      --feature-ids /path/to/epoch109_feature_ids.txt \
      --out-dir store/results_latest/analysis/ne_orbit_graph/epoch109 \
      --ne 3 --threshold 0.5 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Patch
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_VERSION = "2.0-ne-labels"
TAG_RE = re.compile(r"\[([^\[\]]+)\]")
INT_RE = re.compile(r"-?\d+")

ATTENTION_LABELS: Mapping[str, str] = {
    "forth": "forth",
    "major": "major chord",
    "minor": "minor chord",
}
ATTENTION_COLORS: Mapping[str, str] = {
    "forth": "black",
    "major": "red",
    "minor": "green",
}
# A deterministic priority for the unlikely case that one feature belongs to
# multiple annotated groups.
ATTENTION_PRIORITY: Tuple[str, ...] = ("forth", "major", "minor")


@dataclass(frozen=True)
class EdgeRecord:
    source: int
    destination: int
    score: float
    out_rank: int
    in_rank: int
    selected_by: str


@dataclass(frozen=True)
class ComponentRecord:
    component_id: int
    n_nodes: int
    n_edges: int
    attention_groups: str
    attention_nodes: str


def _norm_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def parse_feature_groups(path: Path) -> Dict[str, List[int]]:
    """Parse ``[Label]: count id -> id ...`` feature-group files."""
    groups: Dict[str, List[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            tags = TAG_RE.findall(raw)
            if not tags:
                continue
            label = _norm_label(tags[0])
            rhs = raw.split(":", 1)[1] if ":" in raw else raw
            values = [int(x) for x in INT_RE.findall(rhs)]
            if len(values) >= 2 and values[0] == len(values) - 1:
                values = values[1:]
            groups[label] = values
    return groups


def attention_groups_from_file(path: Path) -> Dict[str, List[int]]:
    groups = parse_feature_groups(path)
    result: Dict[str, List[int]] = {}
    for short_name, file_label in ATTENTION_LABELS.items():
        result[short_name] = list(groups.get(_norm_label(file_label), []))
    return result


def _decoder_vectors(sae_module, device: str) -> torch.Tensor:
    """Return decoder feature vectors as ``[n_features, d_model]``."""
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
            "Cannot locate decoder weights. Expected sae.fc2.weight, "
            "sae.decoder.weight, sae.W_dec, sae.w_dec, or sae.decoder_weight."
        )
    if weight.ndim != 2:
        raise ValueError(f"Decoder weight must be 2-D, got {tuple(weight.shape)}")

    if n_features is not None:
        n_features = int(n_features)
        if weight.shape[1] == n_features:
            vectors = weight.transpose(0, 1)
        elif weight.shape[0] == n_features:
            vectors = weight
        else:
            raise ValueError(
                "Cannot infer decoder orientation: "
                f"weight={tuple(weight.shape)}, n_features={n_features}"
            )
    else:
        vectors = weight.transpose(0, 1) if weight.shape[1] >= weight.shape[0] else weight

    return vectors.detach().to(device=device, dtype=torch.float32)


def _resolve_reference_index(n_saes: int, requested: Optional[int]) -> int:
    if n_saes < 2:
        raise RuntimeError(f"Need at least two sub-SAEs, got {n_saes}")
    if n_saes == 2:
        if requested not in (None, 0):
            raise ValueError("With exactly two sub-SAEs, --sae-idx must be 0")
        return 0

    reference_index = n_saes // 2 if requested is None else int(requested)
    if not 0 < reference_index < n_saes - 1:
        raise ValueError(
            f"With {n_saes} sub-SAEs, --sae-idx must be in "
            f"[1, {n_saes - 2}], got {reference_index}"
        )
    return reference_index


def load_decoder_vectors(
    ckpt_path: Path,
    device: str,
    sae_idx: Optional[int],
) -> Tuple[List[torch.Tensor], int]:
    """Load the checkpoint with the same project helper as obits_recovery."""
    # Prefer the project loader when available. Fall back to reading decoder
    # weights directly from a Lightning checkpoint, which makes this script
    # runnable as a standalone analysis utility.
    try:
        from src.analysis.find_chord_rings import load_sae_and_factory
    except ImportError:
        load_sae_and_factory = None

    if load_sae_and_factory is not None:
        _, sae, _ = load_sae_and_factory(ckpt_path, device)
        module = sae.sae
        sub_saes = list(module.sae_modules)
        reference_index = _resolve_reference_index(len(sub_saes), sae_idx)
        vectors = [_decoder_vectors(sub_sae, device) for sub_sae in sub_saes]
        return vectors, reference_index

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    pattern = re.compile(r"(?:^|\.)sae_modules\.(\d+)\.fc2\.weight$")
    found: Dict[int, torch.Tensor] = {}
    for key, value in state_dict.items():
        match = pattern.search(key)
        if match and isinstance(value, torch.Tensor):
            found[int(match.group(1))] = value
    if len(found) < 2:
        raise RuntimeError(
            "Could not find at least two decoder matrices matching "
            "sae_modules.<index>.fc2.weight in the checkpoint."
        )
    indices = sorted(found)
    if indices != list(range(len(indices))):
        raise RuntimeError(f"Non-contiguous SAE indices in checkpoint: {indices}")
    vectors = [found[index].transpose(0, 1).to(device=device, dtype=torch.float32)
               for index in indices]
    reference_index = _resolve_reference_index(len(vectors), sae_idx)
    print("[LOAD] standalone checkpoint state_dict decoder extraction")
    return vectors, reference_index


def _validate_vector_shapes(vectors: Sequence[torch.Tensor]) -> Tuple[int, int]:
    if len(vectors) < 2:
        raise ValueError("At least two decoder matrices are required")
    if any(value.ndim != 2 for value in vectors):
        raise ValueError("Every decoder matrix must have shape [n_features, d_model]")
    n_features = int(vectors[0].shape[0])
    d_model = int(vectors[0].shape[1])
    for index, value in enumerate(vectors[1:], start=1):
        if tuple(value.shape) != (n_features, d_model):
            raise ValueError(
                f"Decoder shape mismatch at SAE {index}: expected "
                f"{(n_features, d_model)}, got {tuple(value.shape)}"
            )
    return n_features, d_model


def _normalized_scoring_vectors(
    vectors: Sequence[torch.Tensor], reference_index: int
) -> Tuple[torch.Tensor, ...]:
    if len(vectors) == 2:
        return F.normalize(vectors[1], dim=1), F.normalize(vectors[0], dim=1)

    s = reference_index
    return (
        F.normalize(vectors[s + 1], dim=1),
        F.normalize(vectors[s], dim=1),
        F.normalize(vectors[s - 1], dim=1),
    )


def _score_block(
    normalized: Tuple[torch.Tensor, ...],
    start: int,
    end: int,
) -> torch.Tensor:
    """Compute the current obits_recovery score for source rows [start:end]."""
    if len(normalized) == 2:
        source, destination = normalized
        return source[start:end] @ destination.transpose(0, 1)

    upward_source, reference, downward_destination = normalized
    upward_score = upward_source[start:end] @ reference.transpose(0, 1)
    downward_score = reference[start:end] @ downward_destination.transpose(0, 1)
    return upward_score * downward_score


@torch.no_grad()
def compute_directional_ne(
    vectors: Sequence[torch.Tensor],
    reference_index: int,
    ne: int,
    threshold: float,
    block: int,
    include_self_loops: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ne outgoing and incoming candidates blockwise.

    Returns
    -------
    outgoing_destinations, outgoing_scores, incoming_sources, incoming_scores
        Four arrays with shape ``[n_features, ne]``. Missing entries use -1
        for indices and ``-inf`` for scores.
    """
    n_features, _ = _validate_vector_shapes(vectors)
    if ne <= 0:
        raise ValueError(f"--ne must be positive, got {ne}")
    if block <= 0:
        raise ValueError(f"--block must be positive, got {block}")

    k = min(int(ne), n_features)
    device = vectors[0].device
    normalized = _normalized_scoring_vectors(vectors, reference_index)

    outgoing_destinations = np.full((n_features, k), -1, dtype=np.int64)
    outgoing_scores = np.full((n_features, k), -np.inf, dtype=np.float32)

    incoming_sources_t = torch.full(
        (n_features, k), -1, dtype=torch.long, device=device
    )
    incoming_scores_t = torch.full(
        (n_features, k), -torch.inf, dtype=torch.float32, device=device
    )

    for start in range(0, n_features, block):
        end = min(n_features, start + block)
        scores = _score_block(normalized, start, end)

        if not include_self_loops:
            local_rows = torch.arange(end - start, device=device)
            global_rows = torch.arange(start, end, device=device)
            scores[local_rows, global_rows] = -torch.inf

        scores = scores.masked_fill(scores < threshold, -torch.inf)

        # ne successors for each source row.
        out_values, out_indices = torch.topk(scores, k=k, dim=1)
        valid_out = torch.isfinite(out_values)
        out_indices = out_indices.masked_fill(~valid_out, -1)
        outgoing_destinations[start:end] = out_indices.cpu().numpy().astype(np.int64)
        outgoing_scores[start:end] = out_values.cpu().numpy().astype(np.float32)

        # ne predecessors for each destination column, merged across blocks.
        block_k = min(k, end - start)
        block_values, block_local_sources = torch.topk(
            scores.transpose(0, 1), k=block_k, dim=1
        )
        block_sources = block_local_sources + start
        block_sources = block_sources.masked_fill(~torch.isfinite(block_values), -1)

        merged_scores = torch.cat((incoming_scores_t, block_values), dim=1)
        merged_sources = torch.cat((incoming_sources_t, block_sources), dim=1)
        incoming_scores_t, selected = torch.topk(merged_scores, k=k, dim=1)
        incoming_sources_t = merged_sources.gather(1, selected)
        incoming_sources_t = incoming_sources_t.masked_fill(
            ~torch.isfinite(incoming_scores_t), -1
        )

        print(f"[SCORE] rows {start}:{end}/{n_features}")

    return (
        outgoing_destinations,
        outgoing_scores,
        incoming_sources_t.cpu().numpy().astype(np.int64),
        incoming_scores_t.cpu().numpy().astype(np.float32),
    )


def _candidate_maps(
    outgoing_destinations: np.ndarray,
    outgoing_scores: np.ndarray,
    incoming_sources: np.ndarray,
    incoming_scores: np.ndarray,
) -> Tuple[
    Dict[Tuple[int, int], Tuple[float, int]],
    Dict[Tuple[int, int], Tuple[float, int]],
]:
    out_map: Dict[Tuple[int, int], Tuple[float, int]] = {}
    in_map: Dict[Tuple[int, int], Tuple[float, int]] = {}

    for source in range(outgoing_destinations.shape[0]):
        for rank0, destination_raw in enumerate(outgoing_destinations[source]):
            destination = int(destination_raw)
            score = float(outgoing_scores[source, rank0])
            if destination >= 0 and math.isfinite(score):
                out_map[(source, destination)] = (score, rank0 + 1)

    for destination in range(incoming_sources.shape[0]):
        for rank0, source_raw in enumerate(incoming_sources[destination]):
            source = int(source_raw)
            score = float(incoming_scores[destination, rank0])
            if source >= 0 and math.isfinite(score):
                in_map[(source, destination)] = (score, rank0 + 1)

    return out_map, in_map


def select_edges(
    outgoing_destinations: np.ndarray,
    outgoing_scores: np.ndarray,
    incoming_sources: np.ndarray,
    incoming_scores: np.ndarray,
    ne: int,
    policy: str,
) -> List[EdgeRecord]:
    """Select retained edges under mutual, union, or greedy degree policy."""
    out_map, in_map = _candidate_maps(
        outgoing_destinations,
        outgoing_scores,
        incoming_sources,
        incoming_scores,
    )

    if policy == "mutual":
        keys = set(out_map) & set(in_map)
    elif policy == "union":
        keys = set(out_map) | set(in_map)
    elif policy == "greedy":
        candidate_keys = set(out_map) | set(in_map)
        ranked = sorted(
            candidate_keys,
            key=lambda key: max(
                out_map.get(key, (-math.inf, -1))[0],
                in_map.get(key, (-math.inf, -1))[0],
            ),
            reverse=True,
        )
        out_degree: Dict[int, int] = {}
        in_degree: Dict[int, int] = {}
        keys = set()
        for source, destination in ranked:
            if out_degree.get(source, 0) >= ne:
                continue
            if in_degree.get(destination, 0) >= ne:
                continue
            keys.add((source, destination))
            out_degree[source] = out_degree.get(source, 0) + 1
            in_degree[destination] = in_degree.get(destination, 0) + 1
    else:
        raise ValueError(f"Unknown edge policy: {policy}")

    records: List[EdgeRecord] = []
    for source, destination in keys:
        out_item = out_map.get((source, destination))
        in_item = in_map.get((source, destination))
        scores = [item[0] for item in (out_item, in_item) if item is not None]
        score = max(scores)
        if out_item is not None and in_item is not None:
            selected_by = "both"
        elif out_item is not None:
            selected_by = "outgoing"
        else:
            selected_by = "incoming"
        records.append(
            EdgeRecord(
                source=source,
                destination=destination,
                score=score,
                out_rank=out_item[1] if out_item is not None else -1,
                in_rank=in_item[1] if in_item is not None else -1,
                selected_by=selected_by,
            )
        )

    return sorted(records, key=lambda item: (-item.score, item.source, item.destination))


def _attention_membership(
    attention_groups: Mapping[str, Sequence[int]],
) -> Dict[int, Set[str]]:
    membership: Dict[int, Set[str]] = {}
    for group_name, nodes in attention_groups.items():
        for node in nodes:
            membership.setdefault(int(node), set()).add(group_name)
    return membership


def build_graph(
    n_features: int,
    edges: Sequence[EdgeRecord],
    attention_groups: Mapping[str, Sequence[int]],
    include_isolated: bool,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    membership = _attention_membership(attention_groups)

    if include_isolated:
        nodes: Set[int] = set(range(n_features))
    else:
        # Draw only nodes incident to at least one retained edge. Attention
        # annotations do not force isolated nodes back into the graph.
        nodes = {edge.source for edge in edges} | {edge.destination for edge in edges}

    for node in sorted(nodes):
        groups = sorted(membership.get(node, set()))
        graph.add_node(
            node,
            feature_id=int(node),
            attention=",".join(groups),
            is_attention=bool(groups),
        )

    for edge in edges:
        graph.add_edge(
            edge.source,
            edge.destination,
            score=float(edge.score),
            out_rank=int(edge.out_rank),
            in_rank=int(edge.in_rank),
            selected_by=edge.selected_by,
        )
    return graph


def _component_attention(
    nodes: Iterable[int], membership: Mapping[int, Set[str]]
) -> Tuple[str, str]:
    groups: Set[str] = set()
    attention_nodes: List[int] = []
    for node in nodes:
        node_groups = membership.get(int(node), set())
        if node_groups:
            groups.update(node_groups)
            attention_nodes.append(int(node))
    return ",".join(sorted(groups)), " ".join(str(x) for x in sorted(attention_nodes))


def prune_small_components(
    graph: nx.DiGraph,
    min_component_size: int,
) -> Tuple[nx.DiGraph, int, int]:
    """Remove weakly connected components smaller than ``min_component_size``."""
    if min_component_size <= 1:
        return graph.copy(), 0, 0
    small = [
        set(component)
        for component in nx.weakly_connected_components(graph)
        if len(component) < min_component_size
    ]
    removed_nodes = set().union(*small) if small else set()
    pruned = graph.copy()
    pruned.remove_nodes_from(removed_nodes)
    return pruned, len(small), len(removed_nodes)


def component_records(
    graph: nx.DiGraph,
    attention_groups: Mapping[str, Sequence[int]],
) -> Tuple[List[ComponentRecord], Dict[int, int]]:
    membership = _attention_membership(attention_groups)
    components = sorted(
        nx.weakly_connected_components(graph),
        key=lambda values: (-len(values), min(values) if values else -1),
    )
    records: List[ComponentRecord] = []
    node_to_component: Dict[int, int] = {}
    for component_id, nodes in enumerate(components):
        subgraph = graph.subgraph(nodes)
        groups, attention_nodes = _component_attention(nodes, membership)
        records.append(
            ComponentRecord(
                component_id=component_id,
                n_nodes=subgraph.number_of_nodes(),
                n_edges=subgraph.number_of_edges(),
                attention_groups=groups,
                attention_nodes=attention_nodes,
            )
        )
        for node in nodes:
            node_to_component[int(node)] = component_id
    return records, node_to_component


def write_edges(path: Path, edges: Sequence[EdgeRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("source", "destination", "score", "out_rank", "in_rank", "selected_by")
        )
        for edge in edges:
            writer.writerow(
                (
                    edge.source,
                    edge.destination,
                    f"{edge.score:.8f}",
                    edge.out_rank,
                    edge.in_rank,
                    edge.selected_by,
                )
            )


def write_components(path: Path, records: Sequence[ComponentRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("component_id", "n_nodes", "n_edges", "attention_groups", "attention_nodes")
        )
        for item in records:
            writer.writerow(
                (
                    item.component_id,
                    item.n_nodes,
                    item.n_edges,
                    item.attention_groups,
                    item.attention_nodes,
                )
            )


def _graphviz_component_layout(
    graph: nx.Graph,
    engine: str,
    seed: int,
) -> Dict[int, np.ndarray]:
    if graph.number_of_nodes() == 1:
        node = next(iter(graph.nodes))
        return {int(node): np.array([0.0, 0.0], dtype=float)}
    if graph.number_of_nodes() == 2:
        first, second = list(graph.nodes)
        return {
            int(first): np.array([-0.5, 0.0], dtype=float),
            int(second): np.array([0.5, 0.0], dtype=float),
        }

    if engine != "spring":
        try:
            args = "-Goverlap=prism -Gsep=+30 -GK=2.0"
            raw = nx.nx_agraph.graphviz_layout(graph, prog=engine, args=args)
            return {int(node): np.asarray(value, dtype=float) for node, value in raw.items()}
        except Exception as exc:  # pragma: no cover - environment fallback
            print(f"[WARN] graphviz {engine} layout failed; using spring: {exc}")

    raw = nx.spring_layout(
        graph,
        seed=seed,
        k=max(0.35, 2.5 / math.sqrt(max(graph.number_of_nodes(), 2))),
        iterations=300,
    )
    return {int(node): np.asarray(value, dtype=float) for node, value in raw.items()}


def _normalize_component_positions(
    positions: Mapping[int, np.ndarray],
    n_nodes: int,
    node_gap: float,
) -> Tuple[Dict[int, np.ndarray], float, float]:
    nodes = list(positions)
    values = np.vstack([positions[node] for node in nodes])
    values -= values.mean(axis=0, keepdims=True)

    spans = np.ptp(values, axis=0)
    spans = np.maximum(spans, 1e-6)
    target_long_side = node_gap * max(2.0, 2.2 * math.sqrt(max(n_nodes, 1)))
    scale = target_long_side / float(max(spans))
    values *= scale

    normalized = {node: values[index] for index, node in enumerate(nodes)}
    spans = np.ptp(values, axis=0)
    width = max(float(spans[0]), node_gap) + 2.0 * node_gap
    height = max(float(spans[1]), node_gap) + 2.0 * node_gap
    return normalized, width, height


def clustered_component_layout(
    graph: nx.DiGraph,
    engine: str,
    seed: int,
    node_gap: float,
    component_gap: float,
) -> Dict[int, np.ndarray]:
    """Lay out components independently, then pack them into separated rows."""
    if graph.number_of_nodes() == 0:
        return {}

    components = sorted(
        nx.weakly_connected_components(graph),
        key=lambda values: (-len(values), min(values) if values else -1),
    )
    boxes: List[Tuple[Dict[int, np.ndarray], float, float]] = []
    for index, nodes in enumerate(components):
        undirected = graph.subgraph(nodes).to_undirected()
        raw = _graphviz_component_layout(undirected, engine=engine, seed=seed + index)
        boxes.append(_normalize_component_positions(raw, len(nodes), node_gap=node_gap))

    total_area = sum(
        (width + component_gap) * (height + component_gap)
        for _, width, height in boxes
    )
    target_row_width = max(
        max(width for _, width, _ in boxes),
        math.sqrt(max(total_area, 1.0)) * 1.35,
    )

    packed: Dict[int, np.ndarray] = {}
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    for local_positions, width, height in boxes:
        if x_cursor > 0.0 and x_cursor + width > target_row_width:
            x_cursor = 0.0
            y_cursor += row_height + component_gap
            row_height = 0.0

        offset = np.array([x_cursor + width / 2.0, y_cursor + height / 2.0])
        for node, value in local_positions.items():
            packed[int(node)] = value + offset
        x_cursor += width + component_gap
        row_height = max(row_height, height)

    values = np.vstack(list(packed.values()))
    center = values.mean(axis=0)
    return {node: value - center for node, value in packed.items()}


def _node_attention_group(node: int, membership: Mapping[int, Set[str]]) -> Optional[str]:
    groups = membership.get(int(node), set())
    for group_name in ATTENTION_PRIORITY:
        if group_name in groups:
            return group_name
    return None


def _edge_rad_groups(graph: nx.DiGraph) -> Dict[float, List[Tuple[int, int]]]:
    groups: Dict[float, List[Tuple[int, int]]] = {0.025: [], 0.12: [], -0.12: []}
    for source, destination in graph.edges:
        if graph.has_edge(destination, source) and source != destination:
            radius = 0.12 if source < destination else -0.12
        else:
            radius = 0.025
        groups[radius].append((source, destination))
    return groups


def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], pad: float = 0.0) -> bool:
    return not (
        a[2] + pad <= b[0] or b[2] + pad <= a[0]
        or a[3] + pad <= b[1] or b[3] + pad <= a[1]
    )


def _point_in_rect(x: float, y: float, rect: Tuple[float, float, float, float]) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float(np.cross(q - p, r - p))

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    eps = 1e-8
    return (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    )


def _segment_hits_rect(
    a: np.ndarray,
    b: np.ndarray,
    rect: Tuple[float, float, float, float],
    pad: float = 0.0,
) -> bool:
    expanded = (rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad)
    if max(a[0], b[0]) < expanded[0] or min(a[0], b[0]) > expanded[2]:
        return False
    if max(a[1], b[1]) < expanded[1] or min(a[1], b[1]) > expanded[3]:
        return False
    if _point_in_rect(float(a[0]), float(a[1]), expanded) or _point_in_rect(float(b[0]), float(b[1]), expanded):
        return True
    corners = [
        np.array([expanded[0], expanded[1]]),
        np.array([expanded[2], expanded[1]]),
        np.array([expanded[2], expanded[3]]),
        np.array([expanded[0], expanded[3]]),
    ]
    return any(
        _segments_intersect(a, b, corners[index], corners[(index + 1) % 4])
        for index in range(4)
    )


def _edge_display_segments(
    graph: nx.DiGraph,
    positions: Mapping[int, np.ndarray],
    ax,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for radius, edge_list in _edge_rad_groups(graph).items():
        for source, destination in edge_list:
            p0, p2 = positions[source], positions[destination]
            delta = p2 - p0
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                continue
            perpendicular = np.array([-delta[1], delta[0]]) / length
            control = (p0 + p2) / 2.0 + perpendicular * radius * length
            ts = np.linspace(0.0, 1.0, 15)
            curve = np.asarray([
                (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * control + t ** 2 * p2
                for t in ts
            ])
            display = ax.transData.transform(curve)
            segments.extend((display[index], display[index + 1]) for index in range(len(display) - 1))
    return segments


def _place_labels_around_nodes(
    ax,
    fig,
    graph: nx.DiGraph,
    positions: Mapping[int, np.ndarray],
    label_nodes: Sequence[int],
    node_sizes: Mapping[int, float],
    font_size: float,
) -> None:
    """Place labels just outside node boundaries while avoiding graph geometry."""
    if not label_nodes:
        return

    fig.canvas.draw()
    dpi = float(fig.dpi)
    node_display = {
        int(node): np.asarray(ax.transData.transform(positions[node]), dtype=float)
        for node in graph.nodes
    }
    node_radii = {
        int(node): math.sqrt(max(float(node_sizes[node]), 1.0) / math.pi) * dpi / 72.0 + 2.0
        for node in graph.nodes
    }
    edge_segments = _edge_display_segments(graph, positions, ax)
    placed: List[Tuple[float, float, float, float]] = []

    # Place high-degree and attention nodes first because they have fewer clean sectors.
    membership_order = sorted(
        (int(node) for node in label_nodes),
        key=lambda node: (-(graph.in_degree(node) + graph.out_degree(node)), node),
    )
    angles = [
        math.radians(value)
        for value in (0, 180, 90, 270, 45, 135, 315, 225, 22.5, 157.5, 337.5, 202.5, 67.5, 112.5, 292.5, 247.5)
    ]

    for node in membership_order:
        text_value = str(node)
        glyph_box = TextPath(
            (0.0, 0.0), text_value, size=font_size, prop=FontProperties()
        ).get_extents()
        width = max(12.0, float(glyph_box.width) * dpi / 72.0 * 1.16)
        height = max(font_size * 1.18 * dpi / 72.0, float(glyph_box.height) * dpi / 72.0 * 1.22)
        center = node_display[node]
        radius = node_radii[node]

        neighbor_vectors: List[np.ndarray] = []
        for neighbor in set(graph.predecessors(node)) | set(graph.successors(node)):
            vector = node_display[int(neighbor)] - center
            norm_value = float(np.linalg.norm(vector))
            if norm_value > 1e-9:
                neighbor_vectors.append(vector / norm_value)
        if neighbor_vectors:
            preferred = -np.sum(neighbor_vectors, axis=0)
            preferred_norm = float(np.linalg.norm(preferred))
            preferred = preferred / preferred_norm if preferred_norm > 1e-9 else np.array([0.0, 1.0])
        else:
            preferred = np.array([0.0, 1.0])

        best = None
        found_clean = False
        for distance_level, extra_distance in enumerate((0.0, height * 1.15, height * 2.3)):
            for rank, angle in enumerate(angles):
                direction = np.array([math.cos(angle), math.sin(angle)])
                projected_half = 0.5 * (abs(direction[0]) * width + abs(direction[1]) * height)
                label_center = center + direction * (
                    radius + 5.0 + projected_half + extra_distance
                )
                rect = (
                    label_center[0] - width / 2.0,
                    label_center[1] - height / 2.0,
                    label_center[0] + width / 2.0,
                    label_center[1] + height / 2.0,
                )

                label_hits = sum(_rects_overlap(rect, other, pad=3.0) for other in placed)
                node_hits = 0
                node_clearance = float("inf")
                for other_node, other_center in node_display.items():
                    if other_node == node:
                        continue
                    nearest_x = min(max(other_center[0], rect[0]), rect[2])
                    nearest_y = min(max(other_center[1], rect[1]), rect[3])
                    distance = math.hypot(other_center[0] - nearest_x, other_center[1] - nearest_y)
                    clearance = distance - node_radii[other_node]
                    node_clearance = min(node_clearance, clearance)
                    if clearance < 2.0:
                        node_hits += 1
                edge_hits = sum(_segment_hits_rect(a, b, rect, pad=1.5) for a, b in edge_segments)
                preference = float(np.dot(direction, preferred))
                score = (
                    -100000.0 * label_hits
                    -100000.0 * node_hits
                    -2500.0 * edge_hits
                    +25.0 * preference
                    +min(node_clearance, 30.0)
                    -0.2 * rank
                    -8.0 * distance_level
                )
                candidate = (score, label_center, rect)
                if best is None or candidate[0] > best[0]:
                    best = candidate
                if label_hits == 0 and node_hits == 0 and edge_hits == 0 and preference > -0.25:
                    found_clean = True
                    break
            if found_clean:
                break

        assert best is not None
        _, label_center, rect = best
        data_position = ax.transData.inverted().transform(label_center)
        text = ax.text(
            float(data_position[0]),
            float(data_position[1]),
            text_value,
            fontsize=font_size,
            color="black",
            ha="center",
            va="center",
            zorder=6,
            clip_on=False,
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=2.2, foreground="white"),
            path_effects.Normal(),
        ])
        placed.append(rect)


def draw_graph(
    graph: nx.DiGraph,
    attention_groups: Mapping[str, Sequence[int]],
    output_stem: Path,
    formats: Sequence[str],
    engine: str,
    seed: int,
    node_gap: float,
    component_gap: float,
    label_all: bool,
    max_labels: int,
    title: str,
    focus_mode: bool = False,
) -> None:
    if graph.number_of_nodes() == 0:
        print(f"[WARN] skip empty graph: {output_stem.name}")
        return

    positions = clustered_component_layout(
        graph,
        engine=engine,
        seed=seed,
        node_gap=node_gap,
        component_gap=component_gap,
    )
    membership = _attention_membership(attention_groups)
    n_nodes = graph.number_of_nodes()

    values = np.vstack(list(positions.values()))
    span = np.maximum(np.ptp(values, axis=0), 1.0)
    if focus_mode:
        ratio = float(np.clip(span[0] / span[1], 0.75, 1.9))
        height = 15.5
        width = float(np.clip(height * ratio, 16.0, 28.0))
        fig, ax = plt.subplots(figsize=(width, height))
        title_size = 27
        legend_size = 16
    else:
        side = min(54.0, max(14.0, 8.0 + math.sqrt(n_nodes) * 0.42))
        fig, ax = plt.subplots(figsize=(side, side))
        title_size = max(20, min(28, side * 0.68))
        legend_size = max(12, min(17, side * 0.42))

    ax.set_title(title, fontsize=title_size, pad=22, fontweight="semibold")
    ax.set_axis_off()

    scores = [float(data["score"]) for _, _, data in graph.edges(data=True)]
    if scores:
        score_min = min(scores)
        score_max = max(scores)
        if math.isclose(score_min, score_max):
            score_min -= 1e-6
            score_max += 1e-6
        norm = Normalize(vmin=score_min, vmax=score_max)
        cmap = LinearSegmentedColormap.from_list(
            "orbit_blue",
            ("#dcefff", "#86c8f2", "#2f7fb8", "#03182b"),
        )

        arrow_size = max(5, min(11, int(150 / math.sqrt(max(n_nodes, 1)))))
        if focus_mode:
            arrow_size = max(9, arrow_size + 2)
        node_shrink = max(1.5, min(4.0, 38.0 / math.sqrt(max(n_nodes, 1))))
        if focus_mode:
            node_shrink = 5.8
        for radius, edge_list in _edge_rad_groups(graph).items():
            for source, destination in edge_list:
                value = float(graph.edges[source, destination]["score"])
                color = cmap(norm(value))
                width_value = 0.35 + 1.4 * norm(value)
                if focus_mode:
                    width_value *= 1.12
                arrow = FancyArrowPatch(
                    posA=positions[source],
                    posB=positions[destination],
                    arrowstyle="-|>",
                    mutation_scale=arrow_size,
                    connectionstyle=f"arc3,rad={radius}",
                    linewidth=width_value,
                    color=color,
                    alpha=0.82,
                    shrinkA=node_shrink,
                    shrinkB=node_shrink,
                    capstyle="round",
                    joinstyle="miter",
                    zorder=1,
                )
                ax.add_patch(arrow)

        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.025, pad=0.012)
        colorbar.set_label(
            "Edge score (dark navy = larger)",
            fontsize=15 if focus_mode else 13,
            labelpad=10,
        )
        colorbar.ax.tick_params(labelsize=13 if focus_mode else 11)

    if focus_mode:
        ordinary_size = 220.0
        attention_size = 360.0
        ordinary_linewidth = 0.75
        attention_linewidth = 1.5
    else:
        # Keep the original all-graph node drawing unchanged.
        ordinary_size = max(7, min(34, 500 / math.sqrt(max(n_nodes, 1))))
        attention_size = max(65, min(180, 1200 / math.sqrt(max(n_nodes, 1))))
        ordinary_linewidth = 0.25
        attention_linewidth = 1.0

    node_sizes: Dict[int, float] = {}
    ordinary_nodes = [
        node for node in graph.nodes if _node_attention_group(node, membership) is None
    ]
    for node in ordinary_nodes:
        node_sizes[int(node)] = ordinary_size
    if ordinary_nodes:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=ordinary_nodes,
            node_color="#d9d9d9",
            edgecolors="#777777",
            linewidths=ordinary_linewidth,
            node_size=ordinary_size,
            alpha=0.94,
            ax=ax,
        )

    for group_name in ATTENTION_PRIORITY:
        nodes = [
            node
            for node in graph.nodes
            if _node_attention_group(node, membership) == group_name
        ]
        for node in nodes:
            node_sizes[int(node)] = attention_size
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodes,
            node_color=ATTENTION_COLORS[group_name],
            edgecolors="white" if group_name == "forth" else "black",
            linewidths=attention_linewidth,
            node_size=attention_size,
            alpha=1.0,
            ax=ax,
        )

    # Stable limits are needed before collision-aware display-coordinate labels.
    margin_x = max(span[0] * (0.07 if focus_mode else 0.045), node_gap * 1.2)
    margin_y = max(span[1] * (0.08 if focus_mode else 0.045), node_gap * 1.2)
    ax.set_xlim(float(values[:, 0].min() - margin_x), float(values[:, 0].max() + margin_x))
    ax.set_ylim(float(values[:, 1].min() - margin_y), float(values[:, 1].max() + margin_y))

    should_label_all = label_all or n_nodes <= max_labels
    if should_label_all:
        label_nodes = list(graph.nodes)
    else:
        label_nodes = [node for node in graph.nodes if node in membership]
    label_font_size = 10.5 if focus_mode else max(7.0, min(10.0, 105 / math.sqrt(max(len(label_nodes), 1))))

    legend_handles = [
        Patch(facecolor=ATTENTION_COLORS["forth"], edgecolor="white", label="Forth"),
        Patch(facecolor=ATTENTION_COLORS["major"], edgecolor="black", label="Major Chord"),
        Patch(facecolor=ATTENTION_COLORS["minor"], edgecolor="black", label="Minor Chord"),
        Patch(facecolor="#d9d9d9", edgecolor="#777777", label="Other feature"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=legend_size,
        borderpad=0.8,
        labelspacing=0.55,
        handlelength=1.8,
    )
    legend.get_frame().set_alpha(0.94)
    # Finalize axes geometry before collision-aware placement; moving the axes
    # after labels are placed would invalidate all display-coordinate checks.
    fig.tight_layout(pad=1.2)
    _place_labels_around_nodes(
        ax=ax,
        fig=fig,
        graph=graph,
        positions=positions,
        label_nodes=label_nodes,
        node_sizes=node_sizes,
        font_size=label_font_size,
    )

    for extension in formats:
        output = output_stem.with_suffix(f".{extension}")
        fig.savefig(output, dpi=240, bbox_inches="tight", pad_inches=0.12)
        print(f"[DONE] figure -> {output}")
    plt.close(fig)

def focused_graph(
    graph: nx.DiGraph,
    attention_groups: Mapping[str, Sequence[int]],
) -> nx.DiGraph:
    attention_nodes = set(_attention_membership(attention_groups))
    selected_nodes: Set[int] = set()
    for component in nx.weakly_connected_components(graph):
        if component & attention_nodes:
            selected_nodes.update(component)
    return graph.subgraph(selected_nodes).copy()


def parse_formats(value: str) -> List[str]:
    allowed = {"png", "pdf", "svg"}
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported formats {invalid}; choose from png,pdf,svg"
        )
    if not values:
        raise argparse.ArgumentTypeError("At least one output format is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and visualize a Multi-SAE predecessor/successor graph with per-node edge cap ne.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--feature-ids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ne", type=int, default=3, help="Maximum retained incoming and outgoing edges per node.")
    parser.add_argument(
        "--threshold",
        "--prob",
        dest="threshold",
        type=float,
        default=0.5,
        help="Remove candidate edges below this score before ne selection.",
    )
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--sae-idx", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--edge-policy",
        choices=("mutual", "union", "greedy"),
        default="mutual",
        help=(
            "mutual: ne from both directions; union: selected by either "
            "direction; greedy: globally descending edges with in/out degree caps."
        ),
    )
    parser.add_argument(
        "--include-self-loops",
        action="store_true",
        help="Keep self transitions. By default they are excluded as timbre-like nodes.",
    )
    parser.add_argument(
        "--include-isolated",
        action="store_true",
        help="Include every feature node, even when it has no retained edge.",
    )
    parser.add_argument(
        "--layout-engine",
        choices=("sfdp", "neato", "spring"),
        default="sfdp",
    )
    parser.add_argument("--node-gap", type=float, default=2.5)
    parser.add_argument("--component-gap", type=float, default=8.0)
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=1,
        help=(
            "Remove weakly connected components smaller than this size. "
            "Use 3 to omit one- and two-node components."
        ),
    )
    parser.add_argument(
        "--focus-node-gap",
        type=float,
        default=None,
        help="Node gap for the focus figure; default: --node-gap.",
    )
    parser.add_argument(
        "--focus-component-gap",
        type=float,
        default=None,
        help="Component gap for the focus figure; default: --component-gap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", type=parse_formats, default=parse_formats("png,pdf"))
    parser.add_argument(
        "--label-all",
        action="store_true",
        help="Label every feature ID in both figures. Usually too dense for the full graph.",
    )
    parser.add_argument(
        "--max-focus-labels",
        type=int,
        default=250,
        help="Automatically label every node when a graph has at most this many nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[SCRIPT] visualize_ne_orbit_graph.py v{SCRIPT_VERSION}")

    ckpt_path = args.ckpt_path.expanduser().resolve()
    feature_ids_path = args.feature_ids.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not feature_ids_path.is_file():
        raise FileNotFoundError(f"Feature IDs file not found: {feature_ids_path}")
    if args.ne <= 0:
        raise ValueError(f"--ne must be positive, got {args.ne}")
    if args.block <= 0:
        raise ValueError(f"--block must be positive, got {args.block}")
    if args.node_gap <= 0 or args.component_gap <= 0:
        raise ValueError("--node-gap and --component-gap must be positive")
    if args.min_component_size <= 0:
        raise ValueError("--min-component-size must be positive")
    if args.focus_node_gap is not None and args.focus_node_gap <= 0:
        raise ValueError("--focus-node-gap must be positive")
    if args.focus_component_gap is not None and args.focus_component_gap <= 0:
        raise ValueError("--focus-component-gap must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    attention_groups = attention_groups_from_file(feature_ids_path)
    for group_name in ATTENTION_PRIORITY:
        print(
            f"[ATTENTION] {group_name}: "
            f"{len(attention_groups.get(group_name, []))} nodes"
        )

    vectors, reference_index = load_decoder_vectors(
        ckpt_path=ckpt_path,
        device=args.device,
        sae_idx=args.sae_idx,
    )
    n_features, d_model = _validate_vector_shapes(vectors)
    print(
        f"[MODEL] n_saes={len(vectors)} reference_sae={reference_index} "
        f"n_features={n_features} d_model={d_model}"
    )
    score_name = "cosine" if len(vectors) == 2 else "bidirectional cosine product"
    print(
        f"[CONFIG] score={score_name} ne={args.ne} "
        f"threshold={args.threshold} policy={args.edge_policy}"
    )

    for group_name, nodes in attention_groups.items():
        invalid = [node for node in nodes if not 0 <= node < n_features]
        if invalid:
            raise ValueError(
                f"{group_name} contains feature IDs outside [0, {n_features - 1}]: {invalid}"
            )

    outgoing_destinations, outgoing_scores, incoming_sources, incoming_scores = (
        compute_directional_ne(
            vectors=vectors,
            reference_index=reference_index,
            ne=args.ne,
            threshold=args.threshold,
            block=args.block,
            include_self_loops=args.include_self_loops,
        )
    )
    edges = select_edges(
        outgoing_destinations=outgoing_destinations,
        outgoing_scores=outgoing_scores,
        incoming_sources=incoming_sources,
        incoming_scores=incoming_scores,
        ne=args.ne,
        policy=args.edge_policy,
    )
    graph = build_graph(
        n_features=n_features,
        edges=edges,
        attention_groups=attention_groups,
        include_isolated=args.include_isolated,
    )
    graph, removed_components, removed_nodes = prune_small_components(
        graph, args.min_component_size
    )
    retained_pairs = set(graph.edges())
    edges = [
        edge for edge in edges
        if (edge.source, edge.destination) in retained_pairs
    ]
    print(
        f"[FILTER] min_component_size={args.min_component_size} "
        f"removed_components={removed_components} removed_nodes={removed_nodes}"
    )

    records, node_to_component = component_records(graph, attention_groups)
    for node, component_id in node_to_component.items():
        graph.nodes[node]["component_id"] = int(component_id)

    edge_path = out_dir / "ne_orbit_edges.tsv"
    component_path = out_dir / "ne_orbit_components.tsv"
    graphml_path = out_dir / "ne_orbit_graph.graphml"
    write_edges(edge_path, edges)
    write_components(component_path, records)
    nx.write_graphml(graph, graphml_path)

    print(f"[DONE] edges -> {edge_path} ({len(edges)} edges)")
    print(f"[DONE] components -> {component_path} ({len(records)} components)")
    print(f"[DONE] graphml -> {graphml_path}")
    print(
        f"[GRAPH] nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
        f"components={len(records)}"
    )

    title_suffix = (
        f"ne={args.ne}, threshold={args.threshold:g}, "
        f"policy={args.edge_policy}, component size >= {args.min_component_size}"
    )
    draw_graph(
        graph=graph,
        attention_groups=attention_groups,
        output_stem=out_dir / "ne_orbit_graph_all",
        formats=args.formats,
        engine=args.layout_engine,
        seed=args.seed,
        node_gap=args.node_gap,
        component_gap=args.component_gap,
        label_all=args.label_all,
        max_labels=0 if not args.label_all else 10**9,
        title=f"Multi-SAE orbit graph — all retained components\n{title_suffix}",
        focus_mode=False,
    )

    focus = focused_graph(graph, attention_groups)
    draw_graph(
        graph=focus,
        attention_groups=attention_groups,
        output_stem=out_dir / "ne_orbit_graph_focus",
        formats=args.formats,
        engine=args.layout_engine,
        seed=args.seed,
        node_gap=(args.focus_node_gap if args.focus_node_gap is not None else max(args.node_gap, 5.0)),
        component_gap=(
            args.focus_component_gap
            if args.focus_component_gap is not None
            else min(args.component_gap, 3.2)
        ),
        label_all=args.label_all,
        max_labels=args.max_focus_labels,
        title=(
            "Multi-SAE orbit graph — Forth / Major / Minor components\n"
            f"{title_suffix}"
        ),
        focus_mode=True,
    )
    print(
        f"[FOCUS] nodes={focus.number_of_nodes()} edges={focus.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
