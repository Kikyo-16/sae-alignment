#!/usr/bin/env python3
"""
End-to-end directed sparse SAE pitch-shift graph pipeline.

Edge scores are cycle-consistency *distances* (matching high_score in
music_sae.score): dist = 0.5*((1 - s_forward) + (1 - s_backward)). Lower is
better throughout this module.

This script goes from a checkpoint to the final directed graph visualization:
  1. Load SAE-0/1/2 decoder vectors from a MultiSAE checkpoint.
  2. Build a directed weighted sparse graph on SAE-1 features.
  3. Apply per-source relative pruning (keep dist <= relative_factor * best).
  4. Apply a fixed distance threshold (keep dist <= score_threshold).
  5. Remove weakly connected components smaller than a threshold.
  6. Keep/mark rank-1 self-loop nodes.
  7. Detect directed size-12 cycles and exact size-12 path-like sequences.
  8. If size-12 structures share directed edges, keep only the one with the
     smaller mean edge distance.
  9. Draw the graph while preserving edge distance color/width, plus size-12
     multicolor halo highlights and feature-id labels.

Default settings match the current analysis run:
  --topk 5 --candidate-topk 20 --relative-factor 1.5 --score-threshold 0.2
  --min-wcc-size 10 --score-mode avg
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    import matplotlib as mpl
    return plt, LineCollection, PolyCollection, Line2D, mpl


def ensure_networkx():
    import networkx as nx
    return nx


@dataclass
class Edge:
    src: int
    dst: int
    score: float
    s_forward: float
    s_backward: float
    rank: int


@dataclass
class OrbitRecord:
    """Human-readable orbit metadata used by step1_graph output.

    ``source`` separates raw size-12 structures from leftover greedy cleanup,
    while ``label`` controls the readable tag written to orbits_raw.txt.
    """
    nodes: List[int]
    is_ring: bool
    label: str
    source: str
    kind: str
    original_structure_id: str = ""
    mean_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    all_ranks: str = ""
    # Per-edge scores along ``nodes`` order: nodes[i] -> nodes[i+1], plus the
    # closing edge nodes[-1] -> nodes[0] for rings. NaN where the edge is
    # missing from the graph.
    edge_scores: Optional[List[float]] = None


def torch_load_any(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def import_symbol(spec: str):
    if ":" not in spec:
        raise ValueError("Expected symbol spec like 'package.module:function'.")
    module_name, name = spec.split(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, name)


def maybe_unwrap_module(obj):
    if hasattr(obj, "module") and hasattr(obj.module, "sae_modules"):
        return obj.module
    return obj


def decoder_vectors_from_module(module, device: str, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, str]]:
    module = maybe_unwrap_module(module)
    if not hasattr(module, "sae_modules"):
        raise RuntimeError("Loaded object does not expose .sae_modules")
    saes = list(module.sae_modules)
    if len(saes) < 3:
        raise RuntimeError(f"Need at least 3 sub-SAEs, got {len(saes)}")
    Ws = []
    for idx in range(3):
        sae = saes[idx]
        if not hasattr(sae, "fc2") or not hasattr(sae.fc2, "weight"):
            raise RuntimeError(f"sub-SAE {idx} does not expose fc2.weight")
        weight = sae.fc2.weight.detach().to(device)
        W = weight.t().contiguous()
        W = W / (W.norm(dim=1, keepdim=True) + eps)
        Ws.append(W)
    info = {"loader": "module.sae_modules", "keys": "module object"}
    return Ws[0], Ws[1], Ws[2], info


def find_state_dict(ckpt, state_dict_key: Optional[str] = None):
    if state_dict_key:
        cur = ckpt
        for part in state_dict_key.split("."):
            cur = cur[part]
        return cur
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "module", "net", "sae"]:
            val = ckpt.get(key)
            if isinstance(val, dict) and any(torch.is_tensor(v) for v in val.values()):
                return val
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    return None


def clean_state_key(k: str) -> str:
    # Remove common distributed wrappers repeatedly, but keep nested model names otherwise.
    for prefix in ["module.", "model."]:
        while k.startswith(prefix):
            k = k[len(prefix):]
    return k


def extract_decoder_weights_from_state_dict(
    state: Dict[str, torch.Tensor],
    device: str,
    decoder_weight_keys: Optional[str] = None,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, str]]:
    if decoder_weight_keys:
        keys = [x.strip() for x in decoder_weight_keys.split(",") if x.strip()]
        if len(keys) != 3:
            raise ValueError("--decoder-weight-keys must contain exactly 3 comma-separated keys.")
        weights = []
        for key in keys:
            if key not in state:
                raise KeyError(f"Decoder key not found in state_dict: {key}")
            weights.append(state[key])
        used = keys
    else:
        # Match nested paths like:
        #   sae.sae.sae_modules.0.fc2.weight
        #   module.sae_modules.1.fc2.weight
        #   saes.2.fc2.weight
        patterns = [
            re.compile(r"(?:^|\.)(?:sae_modules|saes|sub_saes|subsaes)\.(\d+)\.fc2\.weight$"),
            re.compile(r"(?:^|\.)(?:sae_modules|saes|sub_saes|subsaes)\.(\d+)\.decoder\.weight$"),
        ]
        found: Dict[int, str] = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            k_clean = clean_state_key(str(key))
            for pat in patterns:
                m = pat.search(k_clean)
                if m:
                    idx = int(m.group(1))
                    if idx not in found:
                        found[idx] = str(key)
                    break
        missing = [i for i in range(3) if i not in found]
        if missing:
            sample = [str(k) for k in list(state.keys())[:80]]
            raise RuntimeError(
                "Could not find decoder fc2 weights for SAE indices 0, 1, 2 in state_dict.\n"
                f"Missing indices: {missing}\n"
                "Use --decoder-weight-keys key0,key1,key2 if your checkpoint uses custom names.\n"
                "First state_dict keys:\n  " + "\n  ".join(sample)
            )
        used = [found[i] for i in range(3)]
        weights = [state[k] for k in used]

    Ws = []
    for i, weight in enumerate(weights):
        if not torch.is_tensor(weight):
            raise TypeError(f"Decoder weight {used[i]} is not a tensor")
        if weight.ndim != 2:
            raise ValueError(f"Decoder weight {used[i]} must be 2D, got shape {tuple(weight.shape)}")
        # fc2.weight is expected to be (d_model, hidden); decoder directions are (hidden, d_model).
        W = weight.detach().to(device).t().contiguous()
        W = W / (W.norm(dim=1, keepdim=True) + eps)
        Ws.append(W)
    if Ws[0].shape != Ws[1].shape or Ws[2].shape != Ws[1].shape:
        raise RuntimeError(f"SAE decoder shapes differ after transpose: {Ws[0].shape}, {Ws[1].shape}, {Ws[2].shape}")
    info = {"loader": "state_dict_decoder_weights", "keys": ";".join(used)}
    return Ws[0], Ws[1], Ws[2], info


def load_decoder_vectors(
    ckpt_path: str,
    device: str,
    model_factory: Optional[str] = None,
    state_dict_key: Optional[str] = None,
    decoder_weight_keys: Optional[str] = None,
    strict: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, str]]:
    ckpt = torch_load_any(ckpt_path, map_location=device)

    # Full module checkpoint.
    if hasattr(ckpt, "sae_modules") or (hasattr(ckpt, "module") and hasattr(ckpt.module, "sae_modules")):
        return decoder_vectors_from_module(ckpt, device)

    if isinstance(ckpt, dict):
        for key in ["model", "module", "net", "sae", "multi_sae", "model_obj"]:
            val = ckpt.get(key)
            if hasattr(val, "sae_modules") or (hasattr(val, "module") and hasattr(val.module, "sae_modules")):
                return decoder_vectors_from_module(val, device)

    # Direct state_dict extraction is preferred for plotting; no model class required.
    state = find_state_dict(ckpt, state_dict_key=state_dict_key)
    if isinstance(state, dict):
        try:
            return extract_decoder_weights_from_state_dict(state, device, decoder_weight_keys=decoder_weight_keys)
        except Exception as state_exc:
            if model_factory is None:
                raise
            print(f"Direct decoder extraction failed, trying --model-factory. Reason: {state_exc}", flush=True)

    # Optional model factory fallback.
    if model_factory is not None:
        factory = import_symbol(model_factory)
        model = maybe_unwrap_module(factory())
        if state is None:
            state = find_state_dict(ckpt, state_dict_key=state_dict_key)
        if state is None:
            raise RuntimeError("Could not find a state_dict to load into --model-factory model.")
        model.load_state_dict(state, strict=strict)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return decoder_vectors_from_module(model, device)

    raise RuntimeError("Could not load decoder vectors from checkpoint.")


@torch.no_grad()
def build_edges(
    W0: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    candidate_topk: int,
    final_topk: int,
    block: int,
    score_mode: str,
) -> List[Edge]:
    if block <= 0:
        raise ValueError("block must be positive")
    if candidate_topk < final_topk:
        raise ValueError("candidate_topk must be >= topk")
    if final_topk <= 0:
        raise ValueError("topk must be positive")
    if score_mode not in {"avg", "sum"}:
        raise ValueError("score_mode must be avg or sum")
    if W0.shape != W1.shape or W2.shape != W1.shape:
        raise RuntimeError(f"W0/W1/W2 shapes differ: {W0.shape}, {W1.shape}, {W2.shape}")

    K, D = W1.shape
    candidate_topk = min(candidate_topk, K)
    final_topk = min(final_topk, candidate_topk)
    W0 = F.normalize(W0.float(), dim=1)
    W1 = F.normalize(W1.float(), dim=1)
    W2 = F.normalize(W2.float(), dim=1)
    W1T = W1.t().contiguous()

    edges: List[Edge] = []
    for s in range(0, K, block):
        e = min(K, s + block)
        B = e - s
        sim_forward_all = W2[s:e] @ W1T
        cand_s_forward, cand_idx = torch.topk(sim_forward_all, k=candidate_topk, dim=1, largest=True)
        del sim_forward_all

        W0_cand = W0[cand_idx.reshape(-1)].view(B, candidate_topk, D)
        s_backward = (W0_cand * W1[s:e].view(B, 1, D)).sum(dim=-1)
        del W0_cand

        # Cycle-consistency distance (matches high_score in music_sae.score):
        #   dist = 0.5 * ((1 - s_forward) + (1 - s_backward)); sum-mode keeps the
        #   un-halved form. Lower distance = better alignment, so keep the
        #   final_topk *smallest* distances (rank 1 = nearest neighbour).
        if score_mode == "sum":
            total = (1.0 - cand_s_forward) + (1.0 - s_backward)
        else:
            total = 0.5 * ((1.0 - cand_s_forward) + (1.0 - s_backward))
        keep_scores, keep_pos = torch.topk(total, k=final_topk, dim=1, largest=False)
        keep_dst = torch.gather(cand_idx, 1, keep_pos)
        keep_sf = torch.gather(cand_s_forward, 1, keep_pos)
        keep_sb = torch.gather(s_backward, 1, keep_pos)

        scores_np = keep_scores.detach().cpu().numpy()
        dst_np = keep_dst.detach().cpu().numpy()
        sf_np = keep_sf.detach().cpu().numpy()
        sb_np = keep_sb.detach().cpu().numpy()
        for bi in range(B):
            src = s + bi
            for r in range(final_topk):
                score = float(scores_np[bi, r])
                if not math.isfinite(score):
                    continue
                edges.append(Edge(
                    src=src,
                    dst=int(dst_np[bi, r]),
                    score=score,
                    s_forward=float(sf_np[bi, r]),
                    s_backward=float(sb_np[bi, r]),
                    rank=r + 1,
                ))
        print(f"processed nodes {e}/{K}; raw edges {len(edges)}", flush=True)
    return edges


def write_edges_csv(edges: Sequence[Edge], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "score", "s_forward", "s_backward", "rank"])
        for ed in edges:
            w.writerow([ed.src, ed.dst, ed.score, ed.s_forward, ed.s_backward, ed.rank])


def prune_edges_relative(edges: Sequence[Edge], relative_factor: float) -> List[Edge]:
    """Per-source relative prune in distance space.

    For each source keep only neighbours whose cycle-consistency distance is
    within ``relative_factor`` times the source's best (smallest) distance,
    i.e. drop an edge when ``dist > relative_factor * top1_dist``. Lower is
    better; ``relative_factor=1.5`` keeps neighbours no worse than 1.5x the top-1.
    """
    by_src: Dict[int, List[Edge]] = defaultdict(list)
    for ed in edges:
        by_src[ed.src].append(ed)
    out: List[Edge] = []
    for src, rows in by_src.items():
        rows = sorted(rows, key=lambda x: (x.rank, x.score))
        best = min(float(x.score) for x in rows)
        threshold = relative_factor * best
        for ed in rows:
            if ed.score <= threshold:
                out.append(ed)
    out.sort(key=lambda e: (e.src, e.rank, e.score, e.dst))
    return out


def edges_to_records(edges: Sequence[Edge]) -> List[dict]:
    return [
        {
            "src": ed.src,
            "dst": ed.dst,
            "score": float(ed.score),
            "s_forward": float(ed.s_forward),
            "s_backward": float(ed.s_backward),
            "rank": int(ed.rank),
        }
        for ed in edges
    ]


def write_dict_csv(rows: Sequence[dict], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_filtered_graphs(
    edges: Sequence[Edge],
    score_threshold: float,
    min_wcc_size: int,
    keep_rank1_self_loops: bool,
):
    nx = ensure_networkx()

    nonself0 = [ed for ed in edges if ed.src != ed.dst and ed.score <= score_threshold]
    G0 = nx.DiGraph()
    for ed in nonself0:
        G0.add_edge(ed.src, ed.dst, score=ed.score, s_forward=ed.s_forward, s_backward=ed.s_backward, rank=ed.rank)

    wcc_rows = []
    large_nodes = set()
    for cid, comp in enumerate(sorted(nx.weakly_connected_components(G0), key=len, reverse=True)):
        comp = set(map(int, comp))
        comp_edges = G0.subgraph(comp).number_of_edges()
        keep = len(comp) >= min_wcc_size
        wcc_rows.append({
            "component_id_by_size": cid,
            "size_nodes": len(comp),
            "size_edges_nonself": comp_edges,
            "kept": bool(keep),
        })
        if keep:
            large_nodes.update(comp)

    nonself = [ed for ed in nonself0 if ed.src in large_nodes and ed.dst in large_nodes]
    G_non = nx.DiGraph()
    for ed in nonself:
        G_non.add_edge(ed.src, ed.dst, score=ed.score, s_forward=ed.s_forward, s_backward=ed.s_backward, rank=ed.rank)

    rank1_self_edges_all = [ed for ed in edges if ed.rank == 1 and ed.src == ed.dst]
    rank1_self_nodes_all = {ed.src for ed in rank1_self_edges_all}
    sources_with_nonself_out = {ed.src for ed in nonself}
    self_loop_edges: List[Edge] = []
    if keep_rank1_self_loops:
        for ed in rank1_self_edges_all:
            if ed.src in large_nodes and ed.src in sources_with_nonself_out and ed.score <= score_threshold:
                self_loop_edges.append(ed)

    G_show = G_non.copy()
    for u, v in G_show.edges():
        G_show[u][v]["self_rank1"] = False
    for ed in self_loop_edges:
        G_show.add_edge(ed.src, ed.dst, score=ed.score, s_forward=ed.s_forward, s_backward=ed.s_backward, rank=ed.rank, self_rank1=True)

    return G_non, G_show, nonself, self_loop_edges, rank1_self_nodes_all, wcc_rows


def layout_graph_sfdp_or_spring(G_non, outdir: Path, layout_scale: float, seed: int) -> Dict[int, Tuple[float, float]]:
    nx = ensure_networkx()
    if G_non.number_of_nodes() == 0:
        return {}
    sfdp = shutil.which("sfdp")
    pos = {}
    if sfdp is not None:
        dot = outdir / "layout_sfdp.dot"
        with dot.open("w") as f:
            f.write("digraph G {\n")
            f.write('graph [layout=sfdp, overlap=scale, K=4.5, repulsiveforce=5.5, sep="+58", outputorder=edgesfirst];\n')
            f.write('node [shape=point, width=0.018, label=""];\n')
            f.write('edge [arrowsize=0.01];\n')
            for n in G_non.nodes:
                f.write(f'"{int(n)}";\n')
            for u, v, d in G_non.edges(data=True):
                w = max(0.05, float(d.get("score", 1.0)))
                f.write(f'"{int(u)}" -> "{int(v)}" [weight={w:.6f}];\n')
            f.write("}\n")
        try:
            plain = subprocess.check_output([sfdp, "-Goverlap=scale", "-Tplain", str(dot)], text=True, timeout=300)
            (outdir / "layout_sfdp.plain").write_text(plain)
            for line in plain.splitlines():
                parts = line.split()
                if len(parts) >= 4 and parts[0] == "node":
                    pos[int(parts[1].strip('"'))] = (float(parts[2]), float(parts[3]))
        except Exception as exc:
            print(f"sfdp layout failed; falling back to spring_layout: {exc}", flush=True)
            pos = {}
    if len(pos) != G_non.number_of_nodes():
        print("using networkx spring_layout", flush=True)
        pos = nx.spring_layout(G_non, seed=seed, weight="score", k=None, iterations=400)
        pos = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}

    nodes = sorted(pos)
    xy = np.array([pos[n] for n in nodes], dtype=float)
    xy[:, 0] -= xy[:, 0].mean()
    xy[:, 1] -= xy[:, 1].mean()
    scale = float(np.percentile(np.sqrt((xy ** 2).sum(axis=1)), 99))
    if scale <= 1e-12:
        scale = 1.0
    xy = xy / scale * layout_scale
    return {n: (float(xy[i, 0]), float(xy[i, 1])) for i, n in enumerate(nodes)}


def canonical_cycle(c: Sequence[int]) -> Tuple[int, ...]:
    c = list(map(int, c))
    m = min(range(len(c)), key=lambda i: c[i])
    return tuple(c[m:] + c[:m])


def find_size12_structures(G_non):
    nx = ensure_networkx()
    cycles12 = []
    try:
        cyc_iter = nx.simple_cycles(G_non, length_bound=12)
        cycles12 = [c for c in cyc_iter if len(c) == 12]
    except TypeError:
        cycles12 = [c for c in nx.simple_cycles(G_non) if len(c) == 12]
    cycles12 = sorted({canonical_cycle(c) for c in cycles12})

    seqs12 = []
    for comp in nx.weakly_connected_components(G_non):
        comp = set(map(int, comp))
        if len(comp) != 12:
            continue
        H = G_non.subgraph(comp).copy()
        if H.number_of_edges() != 11:
            continue
        indeg = dict(H.in_degree())
        outdeg = dict(H.out_degree())
        sources = [n for n in H.nodes if indeg[n] == 0 and outdeg[n] == 1]
        sinks = [n for n in H.nodes if indeg[n] == 1 and outdeg[n] == 0]
        middles_ok = all((indeg[n] == 1 and outdeg[n] == 1) for n in H.nodes if n not in sources + sinks)
        if len(sources) == 1 and len(sinks) == 1 and middles_ok:
            seq = [int(sources[0])]
            cur = int(sources[0])
            ok = True
            while cur != int(sinks[0]):
                succ = list(H.successors(cur))
                if len(succ) != 1:
                    ok = False
                    break
                cur = int(succ[0])
                seq.append(cur)
                if len(seq) > 12:
                    ok = False
                    break
            if ok and len(seq) == 12 and seq[-1] == int(sinks[0]):
                seqs12.append(tuple(seq))
    seqs12 = sorted(set(seqs12))

    structures = []
    for cid, cyc in enumerate(cycles12, start=1):
        edges = tuple((cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc)))
        structures.append({
            "structure_id": f"C{cid:02d}",
            "kind": "cycle",
            "index": cid,
            "nodes": tuple(cyc),
            "edges": edges,
            "sequence": " -> ".join(map(str, cyc)) + f" -> {cyc[0]}",
        })
    for sid, seq in enumerate(seqs12, start=1):
        edges = tuple((seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        structures.append({
            "structure_id": f"S{sid:02d}",
            "kind": "sequence",
            "index": sid,
            "nodes": tuple(seq),
            "edges": edges,
            "sequence": " -> ".join(map(str, seq)),
        })
    for s in structures:
        scores = []
        ranks = []
        for u, v in s["edges"]:
            d = G_non.get_edge_data(u, v, default={})
            if "score" in d:
                scores.append(float(d["score"]))
            if "rank" in d:
                ranks.append(int(d["rank"]))
        s["mean_score"] = float(np.mean(scores)) if scores else float("nan")
        s["min_score"] = float(np.min(scores)) if scores else float("nan")
        s["max_score"] = float(np.max(scores)) if scores else float("nan")
        s["all_ranks"] = ";".join(map(str, sorted(set(ranks))))
    return cycles12, seqs12, structures


def filter_structures_edge_disjoint(structures: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    # Lower mean edge distance = better; on ties prefer cycles, then smaller index.
    ordered = sorted(
        structures,
        key=lambda s: (
            float(s.get("mean_score", float("inf"))),
            0 if s.get("kind") == "cycle" else 1,
            int(s.get("index", 0)),
        ),
    )
    used_edges = set()
    kept = []
    rejected = []
    for s in ordered:
        edges = set((int(u), int(v)) for u, v in s["edges"])
        overlap = sorted(edges & used_edges)
        if overlap:
            rejected.append({
                "structure_id": s["structure_id"],
                "kind": s["kind"],
                "index": s["index"],
                "mean_score": s.get("mean_score", float("nan")),
                "overlap_edge_count": len(overlap),
                "overlap_edges": ";".join(f"{u}->{v}" for u, v in overlap),
                "sequence": s["sequence"],
            })
        else:
            kept.append(s)
            used_edges.update(edges)

    renumbered = []
    c_i = 0
    s_i = 0
    for display_order, item in enumerate(kept, start=1):
        item = dict(item)
        item["original_structure_id"] = item["structure_id"]
        if item["kind"] == "cycle":
            c_i += 1
            item["structure_id"] = f"C{c_i:02d}"
            item["index"] = c_i
        else:
            s_i += 1
            item["structure_id"] = f"S{s_i:02d}"
            item["index"] = s_i
        item["display_order"] = display_order
        renumbered.append(item)
    return renumbered, rejected


def build_pitch_graph(
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
):
    """Decoder vectors → top-k forward/backward edges → relative prune →
    distance threshold + small-WCC removal. Returns ``(G_non, rank1_self_nodes, K)``.
    Edge scores are cycle-consistency distances (lower is better).
    """
    W0, W1, W2, _info = decoder_vectors_from_module(module, device)
    K = int(W1.shape[0])
    edges = build_edges(W0, W1, W2, candidate_topk, topk, block, score_mode)
    pruned = prune_edges_relative(edges, relative_factor)
    G_non, _G_show, _nonself, _self_loops, rank1_self_nodes, _wcc = build_filtered_graphs(
        pruned,
        score_threshold=score_threshold,
        min_wcc_size=min_wcc_size,
        keep_rank1_self_loops=True,
    )
    return G_non, {int(n) for n in rank1_self_nodes}, K


def _size12_structures(G_non, edge_disjoint: bool) -> List[dict]:
    _cycles12, _seqs12, structures_before = find_size12_structures(G_non)
    if edge_disjoint:
        structures, _rejected = filter_structures_edge_disjoint(structures_before)
    else:
        structures = [dict(s) for s in structures_before]
    return structures


def size12_structures_from_module(
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
) -> Tuple[List[dict], set, int]:
    """Size-12 structures only (cycles + exact 12-node sequences, edge-disjoint
    filtered unless edge_disjoint=False). Used by
    find_chord_rings.compute_size12_rings. Returns
    ``(structures, rank1_self_nodes, K)``.
    """
    G_non, rank1_self_nodes, K = build_pitch_graph(
        module, device, topk=topk, candidate_topk=candidate_topk, block=block,
        score_mode=score_mode, relative_factor=relative_factor,
        score_threshold=score_threshold, min_wcc_size=min_wcc_size,
    )
    return _size12_structures(G_non, edge_disjoint), rank1_self_nodes, K


def decompose_remaining_chains(
    G_non, used_nodes, min_chain_len: int = 10
) -> List[Tuple[List[int], bool]]:
    """Greedily reduce the subgraph induced on the nodes NOT in ``used_nodes`` to
    vertex-disjoint simple paths and size-12 cycles.

    For every node we keep at most one incoming and one outgoing edge, choosing
    edges in ascending distance order; so at a branch like A->B->C vs A->D->E->C the
    lower-distance path is kept and the other's edges are dropped. A cycle is only
    closed when it has exactly 12 nodes (otherwise its closing edge is left out and
    the component stays a chain).

    Returns a list of ``(node_ids, is_ring)``; rings are size 12. Chains shorter
    than ``min_chain_len`` (and isolated nodes) are dropped.
    """
    nodes = [int(n) for n in G_non.nodes if int(n) not in used_nodes]
    nodeset = set(nodes)
    edges = [
        (int(u), int(v), float(d.get("score", 0.0)))
        for u, v, d in G_non.edges(data=True)
        if int(u) in nodeset and int(v) in nodeset and int(u) != int(v)
    ]
    edges.sort(key=lambda e: e[2])  # smallest distance (best) first

    out_used: Dict[int, int] = {}
    in_used: Dict[int, int] = {}
    parent: Dict[int, int] = {n: n for n in nodes}
    length: Dict[int, int] = {n: 1 for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, _s in edges:
        if u in out_used or v in in_used:          # would branch at u or v
            continue
        ru, rv = find(u), find(v)
        if ru == rv:
            # u is the tail and v the head of the same path → closing a cycle.
            if length[ru] == 12:
                out_used[u] = v
                in_used[v] = u
            continue                                # non-12 cycle: leave it open
        out_used[u] = v                             # extend / merge two paths
        in_used[v] = u
        parent[ru] = rv
        length[rv] = length[ru] + length[rv]

    out: List[Tuple[List[int], bool]] = []
    visited: set = set()
    # Simple paths: start at a head (no chosen in-edge) that has a chosen out-edge.
    for n in nodes:
        if n in visited or n in in_used or n not in out_used:
            continue
        path = [n]
        cur = n
        while cur in out_used:
            cur = out_used[cur]
            path.append(cur)
        visited.update(path)
        out.append((path, False))
    # Remaining nodes with a chosen out-edge belong to size-12 cycles.
    for n in nodes:
        if n in visited or n not in out_used:
            continue
        cyc = [n]
        cur = out_used[n]
        while cur != n:
            cyc.append(cur)
            cur = out_used[cur]
        visited.update(cyc)
        out.append((cyc, True))
    # Drop chains shorter than min_chain_len (rings are size 12, always kept).
    return [(ids, ir) for ids, ir in out if ir or len(ids) >= min_chain_len]


def edge_scores_for_nodes(G_non, nodes: List[int], is_ring: bool) -> List[float]:
    """Per-edge scores along a node path: nodes[i] -> nodes[i+1] (+ the closing
    edge for rings). NaN where the consecutive pair is not an edge in G_non."""
    pairs = list(zip(nodes, nodes[1:]))
    if is_ring and len(nodes) > 1:
        pairs.append((nodes[-1], nodes[0]))
    out: List[float] = []
    for u, v in pairs:
        d = G_non.get_edge_data(int(u), int(v), default={}) or {}
        out.append(float(d["score"]) if "score" in d else float("nan"))
    return out


def orbits_from_module(
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
    include_leftover: bool = True,
    min_chain_len: int = 10,
    return_records: bool = False,
):
    """Full orbit extraction for step1_graph.

    The extraction is intentionally two-stage:
      1. Raw size-12 structures from the filtered directed graph:
         - ``Ring12``: an enumerated directed 12-cycle.
         - ``Seq12``: an exact 12-node directed path component.
      2. Optional leftover cleanup after removing nodes used by stage 1:
         - ``Chain``: greedy path-cover chain, length >= ``min_chain_len``.
         - ``GreedyRing12``: size-12 cycle produced by the greedy leftover cover.

    By default this keeps the historical API and returns
    ``(List[(node_ids, is_ring)], rank1_self_nodes, K)``.  If
    ``return_records=True``, the first item is ``List[OrbitRecord]`` with
    human-readable metadata for more interpretable ``orbits_raw.txt`` output.
    """
    G_non, rank1_self_nodes, K = build_pitch_graph(
        module, device, topk=topk, candidate_topk=candidate_topk, block=block,
        score_mode=score_mode, relative_factor=relative_factor,
        score_threshold=score_threshold, min_wcc_size=min_wcc_size,
    )
    structures = _size12_structures(G_non, edge_disjoint)

    records: List[OrbitRecord] = []
    for s in structures:
        is_cycle = s["kind"] == "cycle"
        nodes = list(map(int, s["nodes"]))
        records.append(OrbitRecord(
            nodes=nodes,
            is_ring=is_cycle,
            label="Ring12" if is_cycle else "Seq12",
            source="raw_size12",
            kind="directed_12_cycle" if is_cycle else "exact_12_node_path_component",
            original_structure_id=str(s.get("original_structure_id", s.get("structure_id", ""))),
            mean_score=s.get("mean_score"),
            min_score=s.get("min_score"),
            max_score=s.get("max_score"),
            all_ranks=str(s.get("all_ranks", "")),
            edge_scores=edge_scores_for_nodes(G_non, nodes, is_cycle),
        ))

    if include_leftover:
        used_nodes = {int(n) for s in structures for n in s["nodes"]}
        for ids, is_ring in decompose_remaining_chains(G_non, used_nodes, min_chain_len=min_chain_len):
            ids = list(map(int, ids))
            records.append(OrbitRecord(
                nodes=ids,
                is_ring=is_ring,
                label="GreedyRing12" if is_ring else "Chain",
                source="leftover_greedy",
                kind="greedy_leftover_12_cycle" if is_ring else "greedy_leftover_chain",
                edge_scores=edge_scores_for_nodes(G_non, ids, is_ring),
            ))

    if return_records:
        return records, rank1_self_nodes, K
    return [(r.nodes, r.is_ring) for r in records], rank1_self_nodes, K


def assign_structure_colors(structures: Sequence[dict]):
    _, _, _, _, mpl = ensure_matplotlib()
    n = len(structures)
    if n <= 20:
        colors = [mpl.colors.to_hex(mpl.cm.tab20(i)) for i in range(n)]
    else:
        colors = [mpl.colors.to_hex(mpl.cm.hsv(i / max(1, n))) for i in range(n)]
    out = []
    for s, color in zip(structures, colors):
        s = dict(s)
        s["color"] = color
        out.append(s)
    return out


def write_size12_tables(outdir: Path, G_non, structures_before: Sequence[dict], structures: Sequence[dict], rejected: Sequence[dict], pos: Dict[int, Tuple[float, float]], rank1_self_nodes_all: set):
    _, _, _, _, mpl = ensure_matplotlib()
    before_rows = []
    for s in structures_before:
        before_rows.append({
            "structure_id": s["structure_id"],
            "kind": s["kind"],
            "index": s["index"],
            "node_count": len(s["nodes"]),
            "edge_count": len(s["edges"]),
            "mean_score": s.get("mean_score", np.nan),
            "min_score": s.get("min_score", np.nan),
            "max_score": s.get("max_score", np.nan),
            "all_ranks": s.get("all_ranks", ""),
            "sequence": s["sequence"],
        })
    write_dict_csv(before_rows, outdir / "size12_structures_before_edge_disjoint_filter.csv")
    write_dict_csv(list(rejected), outdir / "size12_structures_rejected_shared_edges.csv")

    structure_rows = []
    node_to_structs = defaultdict(list)
    edge_to_structs = defaultdict(list)
    struct_color = {}
    for s in structures:
        struct_color[s["structure_id"]] = s["color"]
        structure_rows.append({
            "structure_id": s["structure_id"],
            "original_structure_id": s.get("original_structure_id", s["structure_id"]),
            "kind": s["kind"],
            "index": s["index"],
            "display_order": s.get("display_order", s["index"]),
            "color": s["color"],
            "node_count": len(s["nodes"]),
            "edge_count": len(s["edges"]),
            "mean_score": s.get("mean_score", np.nan),
            "min_score": s.get("min_score", np.nan),
            "max_score": s.get("max_score", np.nan),
            "all_ranks": s.get("all_ranks", ""),
            "sequence": s["sequence"],
        })
        for n in s["nodes"]:
            node_to_structs[int(n)].append(s["structure_id"])
        for u, v in s["edges"]:
            edge_to_structs[(int(u), int(v))].append(s["structure_id"])
    write_dict_csv(structure_rows, outdir / "size12_structures_edge_disjoint.csv")

    cycle_rows = []
    seq_rows = []
    for s in structures:
        if s["kind"] == "cycle":
            cyc = s["nodes"]
            for p, u in enumerate(cyc):
                v = cyc[(p + 1) % len(cyc)]
                d = G_non.get_edge_data(u, v, default={})
                cycle_rows.append({
                    "structure_id": s["structure_id"],
                    "cycle_id": s["index"],
                    "color": s["color"],
                    "position": p + 1,
                    "feature_id": u,
                    "next_feature_id": v,
                    "edge_score": d.get("score", np.nan),
                    "edge_rank": d.get("rank", np.nan),
                    "cycle_as_sequence": s["sequence"],
                })
        else:
            seq = s["nodes"]
            for p, u in enumerate(seq):
                v = seq[p + 1] if p + 1 < len(seq) else None
                d = G_non.get_edge_data(u, v, default={}) if v is not None else {}
                seq_rows.append({
                    "structure_id": s["structure_id"],
                    "sequence_id": s["index"],
                    "color": s["color"],
                    "position": p + 1,
                    "feature_id": u,
                    "next_feature_id": v,
                    "edge_score": d.get("score", np.nan),
                    "edge_rank": d.get("rank", np.nan),
                    "sequence": s["sequence"],
                })
    write_dict_csv(cycle_rows, outdir / "directed_cycles_size12_edge_disjoint.csv")
    write_dict_csv(seq_rows, outdir / "directed_sequences_size12_edge_disjoint.csv")

    hnode_rows = []
    for n in sorted(node_to_structs):
        mids = node_to_structs[n]
        primary = mids[0]
        x, y = pos.get(n, (np.nan, np.nan))
        hnode_rows.append({
            "feature_id": n,
            "primary_structure_id": primary,
            "all_structure_ids": ";".join(mids),
            "primary_color": struct_color[primary],
            "membership_count": len(mids),
            "rank1_is_self_loop": n in rank1_self_nodes_all,
            "x": x,
            "y": y,
        })
    write_dict_csv(hnode_rows, outdir / "highlighted_size12_nodes_edge_disjoint.csv")

    hedge_rows = []
    for u, v in sorted(edge_to_structs):
        mids = edge_to_structs[(u, v)]
        primary = mids[0]
        d = G_non.get_edge_data(u, v, default={})
        hedge_rows.append({
            "src": u,
            "dst": v,
            "primary_structure_id": primary,
            "all_structure_ids": ";".join(mids),
            "primary_color": struct_color[primary],
            "membership_count": len(mids),
            "score": d.get("score", np.nan),
            "rank": d.get("rank", np.nan),
        })
    write_dict_csv(hedge_rows, outdir / "highlighted_size12_edges_edge_disjoint.csv")
    return node_to_structs, edge_to_structs, struct_color


def export_node_edge_tables(
    outdir: Path,
    G_non,
    G_show,
    edges: Sequence[Edge],
    nonself_edges: Sequence[Edge],
    self_loop_edges: Sequence[Edge],
    rank1_self_nodes_all: set,
    pos: Dict[int, Tuple[float, float]],
    wcc_rows: Sequence[dict],
):
    shown_edges = []
    for u, v, d in G_show.edges(data=True):
        shown_edges.append({
            "src": int(u),
            "dst": int(v),
            "score": float(d.get("score", np.nan)),
            "s_forward": float(d.get("s_forward", np.nan)),
            "s_backward": float(d.get("s_backward", np.nan)),
            "rank": int(d.get("rank", -1)),
            "edge_type": "rank1_self_loop" if u == v else "nonself",
        })
    shown_edges.sort(key=lambda r: (r["src"], r["rank"], r["dst"]))
    write_dict_csv(shown_edges, outdir / "shown_edges_with_rank1_self_loops.csv")
    write_edges_csv(nonself_edges, outdir / "valid_edges_nonself_score_threshold_wcc.csv")
    write_edges_csv(self_loop_edges, outdir / "shown_rank1_self_loop_edges.csv")
    write_dict_csv(list(wcc_rows), outdir / "wcc_size_filter_components.csv")

    rows = []
    nodes = sorted(G_non.nodes())
    for n in nodes:
        x, y = pos.get(int(n), (np.nan, np.nan))
        rows.append({
            "feature_id": int(n),
            "x": x,
            "y": y,
            "rank1_is_self_loop": int(n) in rank1_self_nodes_all,
            "in_degree_nonself": int(G_non.in_degree(n)),
            "out_degree_nonself": int(G_non.out_degree(n)),
            "degree_nonself": int(G_non.degree(n)),
            "in_weight_nonself": sum(float(d.get("score", 0.0)) for _, _, d in G_non.in_edges(n, data=True)),
            "out_weight_nonself": sum(float(d.get("score", 0.0)) for _, _, d in G_non.out_edges(n, data=True)),
        })
    write_dict_csv(rows, outdir / "valid_nodes_score_threshold_wcc_rank1self.csv")


def rgba_from_hex(mpl, hex_color: str, alpha: float):
    r, g, b = mpl.colors.to_rgb(hex_color)
    return (r, g, b, alpha)


def draw_graph(
    outdir: Path,
    G_non,
    G_show,
    pos: Dict[int, Tuple[float, float]],
    rank1_self_nodes_all: set,
    node_to_structs,
    edge_to_structs,
    struct_color,
    structures: Sequence[dict],
    score_threshold: float,
    min_wcc_size: int,
    output_stem: str,
    title_extra: str = "",
):
    plt, LineCollection, PolyCollection, Line2D, mpl = ensure_matplotlib()
    if G_non.number_of_nodes() == 0 or G_non.number_of_edges() == 0:
        raise RuntimeError("No non-self graph remains after filtering; nothing to draw.")

    shown_edges = list(G_show.edges(data=True))
    scores = np.array([float(d.get("score", 0.0)) for _, _, d in shown_edges], dtype=float)
    vmin, vmax = float(scores.min()), float(scores.max())
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-6
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis_r  # low distance (better) purple, high distance yellow.

    def edge_style(score: float):
        # `score` is a distance (lower = better); map to goodness so better edges
        # stay purple/opaque/thick, matching the previous similarity-based look.
        t = 1.0 - float(norm(score))
        rgba = cmap(t)
        alpha = 0.26 + 0.68 * t
        color = (rgba[0], rgba[1], rgba[2], alpha)
        lw = 0.12 + 2.20 * t
        return t, color, lw

    def arrow_geometry(u, v, score, halo=False):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = float((dx * dx + dy * dy) ** 0.5)
        if dist <= 1e-9:
            return None
        ux, uy = dx / dist, dy / dist
        px, py = -uy, ux
        t, color, lw = edge_style(score)
        if halo:
            head_len = 0.029 + 0.040 * t
            head_w = 0.024 + 0.040 * t
            node_gap = 0.0060
            seg_lw = lw * 2.60 + 5.7
        else:
            head_len = 0.011 + 0.020 * t
            head_w = 0.006 + 0.016 * t
            node_gap = 0.0075
            seg_lw = lw
        tip = np.array([x1 - ux * node_gap, y1 - uy * node_gap])
        base = tip - np.array([ux, uy]) * head_len
        left = base + np.array([px, py]) * head_w * 0.5
        right = base - np.array([px, py]) * head_w * 0.5
        seg = [(x0 + ux * node_gap, y0 + uy * node_gap), tuple(base)]
        return seg, [tip, left, right], color, seg_lw

    segments = []
    line_colors = []
    line_widths = []
    arrow_polys = []
    arrow_colors = []
    halo_segments = []
    halo_colors = []
    halo_widths = []
    halo_arrow_polys = []
    halo_arrow_colors = []
    loop_segments = []
    loop_colors = []
    loop_widths = []
    loop_arrow_polys = []
    loop_arrow_colors = []

    for u, v, d in G_show.edges(data=True):
        u, v = int(u), int(v)
        if u not in pos or v not in pos:
            continue
        score = float(d.get("score", 0.0))
        if u == v:
            continue
        if (u, v) in edge_to_structs:
            primary = edge_to_structs[(u, v)][0]
            halo_color = rgba_from_hex(mpl, struct_color[primary], 0.80)
            geom_h = arrow_geometry(u, v, score, halo=True)
            if geom_h is not None:
                seg, poly, _, lw_h = geom_h
                halo_segments.append(seg)
                halo_colors.append(halo_color)
                halo_widths.append(lw_h)
                halo_arrow_polys.append(poly)
                halo_arrow_colors.append(halo_color)
        geom = arrow_geometry(u, v, score, halo=False)
        if geom is not None:
            seg, poly, color, lw = geom
            segments.append(seg)
            line_colors.append(color)
            line_widths.append(lw)
            arrow_polys.append(poly)
            arrow_colors.append(color)

    self_loop_edges = [(u, v, d) for u, v, d in G_show.edges(data=True) if int(u) == int(v)]
    base_loop_r = 0.025
    for idx, (u, v, d) in enumerate(sorted(self_loop_edges, key=lambda x: int(x[0]))):
        n = int(u)
        if n not in pos:
            continue
        x, y = pos[n]
        s = float(d.get("score", 0.0))
        t, color, lw = edge_style(s)
        radius = base_loop_r + 0.004 * ((idx % 3) - 1)
        cx = x + radius * 0.26
        cy = y + radius * 0.28
        angles = np.linspace(np.deg2rad(35), np.deg2rad(325), 30)
        pts = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
        loop_segments.extend([[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)])
        loop_colors.extend([color] * (len(pts) - 1))
        loop_widths.extend([lw] * (len(pts) - 1))
        tip = pts[-1]
        prev = pts[-2]
        vec = tip - prev
        dist = float(np.sqrt((vec ** 2).sum()))
        if dist > 1e-12:
            uvec = vec / dist
            pvec = np.array([-uvec[1], uvec[0]])
            head_len = 0.010 + 0.014 * t
            head_w = 0.006 + 0.013 * t
            base = tip - uvec * head_len
            left = base + pvec * head_w * 0.5
            right = base - pvec * head_w * 0.5
            loop_arrow_polys.append([tip, left, right])
            loop_arrow_colors.append(color)

    nodes = sorted(pos)
    xy = np.array([pos[n] for n in nodes], dtype=float)
    deg = dict(G_non.degree())
    node_sizes = np.array([5.5 + 3.2 * np.log1p(deg.get(n, 0)) for n in nodes])
    special_mask = np.array([n in rank1_self_nodes_all for n in nodes], dtype=bool)
    normal_mask = ~special_mask
    highlight_nodes = set(node_to_structs.keys())
    highlight_mask = np.array([n in highlight_nodes for n in nodes], dtype=bool)
    highlight_edgecolors = [struct_color[node_to_structs[n][0]] for n in nodes if n in highlight_nodes]

    fig, ax = plt.subplots(figsize=(40, 40), dpi=240)
    ax.set_aspect("equal")
    ax.axis("off")

    if halo_segments:
        ax.add_collection(LineCollection(halo_segments, colors=halo_colors, linewidths=halo_widths, zorder=0.6))
    if halo_arrow_polys:
        ax.add_collection(PolyCollection(halo_arrow_polys, facecolors=halo_arrow_colors, edgecolors="none", zorder=0.7))
    if segments:
        ax.add_collection(LineCollection(segments, colors=line_colors, linewidths=line_widths, zorder=1.0))
    if arrow_polys:
        ax.add_collection(PolyCollection(arrow_polys, facecolors=arrow_colors, edgecolors="none", zorder=2.0))
    if loop_segments:
        ax.add_collection(LineCollection(loop_segments, colors=loop_colors, linewidths=loop_widths, zorder=3.0))
    if loop_arrow_polys:
        ax.add_collection(PolyCollection(loop_arrow_polys, facecolors=loop_arrow_colors, edgecolors="none", zorder=4.0))

    ax.scatter(xy[normal_mask, 0], xy[normal_mask, 1], s=node_sizes[normal_mask],
               c="#222222", alpha=0.70, linewidths=0, zorder=5)
    ax.scatter(xy[special_mask, 0], xy[special_mask, 1], s=node_sizes[special_mask] * 1.22,
               c="#d95f02", alpha=0.92, linewidths=0.18, edgecolors="white", zorder=6)

    if highlight_mask.any():
        hxy = xy[highlight_mask]
        hsizes = node_sizes[highlight_mask]
        ax.scatter(hxy[:, 0], hxy[:, 1], s=hsizes * 5.6,
                   facecolors="none", edgecolors=highlight_edgecolors, linewidths=1.85, alpha=0.99, zorder=8)
        ax.scatter(hxy[:, 0], hxy[:, 1], s=hsizes * 2.6,
                   facecolors="none", edgecolors="white", linewidths=0.55, alpha=0.93, zorder=7.5)

    xy_mean = xy.mean(axis=0)
    for n in sorted(highlight_nodes):
        if n not in pos:
            continue
        x, y = pos[n]
        vec = np.array([x, y]) - xy_mean
        norm_vec = np.linalg.norm(vec)
        offset = np.array([0.012, 0.012]) if norm_vec < 1e-9 else vec / norm_vec * 0.018
        primary = node_to_structs[n][0]
        label_color = struct_color[primary]
        ax.text(x + offset[0], y + offset[1], str(n), fontsize=7.3, color="#111111",
                ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.78,
                          edgecolor=label_color, linewidth=0.70))

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    mx, my = xmax - xmin, ymax - ymin
    if mx <= 1e-12:
        mx = 1.0
    if my <= 1e-12:
        my = 1.0
    ax.set_xlim(xmin - 0.12 * mx, xmax + 0.12 * mx)
    ax.set_ylim(ymin - 0.12 * my, ymax + 0.12 * my)

    title = (
        f"Directed SAE feature graph, distance <= {score_threshold:g}, WCC size >= {min_wcc_size}\n"
        f"edge-disjoint size-12 cycles/sequences highlighted; edge color/width encodes distance (lower=better)"
    )
    if title_extra:
        title += "\n" + title_extra
    ax.set_title(title, fontsize=18, pad=20)

    text = (
        f"drawn graph: {G_non.number_of_nodes()} nodes, {G_non.number_of_edges()} non-self edges; "
        f"displayed edges incl. self-loops: {G_show.number_of_edges()}\n"
        f"colored halo/ring: membership in a directed size-12 cycle or exact 12-node directed sequence\n"
        f"distance colormap: lower purple (better), higher yellow; rank-1 self-loop nodes are orange"
    )
    ax.text(0.01, 0.01, text, transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="none"))

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label="normal node", markerfacecolor="#222222", markersize=6),
        Line2D([0], [0], marker="o", color="none", label="rank-1 successor is self-loop", markerfacecolor="#d95f02", markeredgecolor="white", markersize=7),
    ]
    for s in structures[:12]:
        legend_handles.append(Line2D([0], [0], color=s["color"], lw=4.2, alpha=0.85, label=f"{s['structure_id']} {s['kind']}"))
    if len(structures) > 12:
        legend_handles.append(Line2D([0], [0], color="gray", lw=4.2, alpha=0.85, label=f"+{len(structures)-12} more; see CSV"))
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.82, fontsize=10.2, ncol=1)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.012, shrink=0.78)
    cbar.set_label("edge distance (lower=better)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout(pad=0.35)
    png = outdir / f"{output_stem}.png"
    pdf = outdir / f"{output_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png, pdf


def write_summary(outdir: Path, rows: Sequence[dict]) -> None:
    write_dict_csv(rows, outdir / "summary.csv", fieldnames=["metric", "value"])


def zip_output(outdir: Path, zip_path: Optional[Path] = None) -> Path:
    if zip_path is None:
        zip_path = outdir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.iterdir()):
            if p.is_file():
                z.write(p, arcname=f"{outdir.name}/{p.name}")
    return zip_path


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end SAE directed sparse graph pipeline.")
    p.add_argument("ckpt_path", type=str, help="Path to checkpoint.")
    p.add_argument("--output-dir", type=str, default="sae_directed_graph_final_out")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-factory", type=str, default=None, help="Optional module:function returning model, if needed.")
    p.add_argument("--state-dict-key", type=str, default=None, help="Optional nested checkpoint key for state_dict, e.g. state_dict.")
    p.add_argument("--decoder-weight-keys", type=str, default=None, help="Explicit key0,key1,key2 for SAE-0/1/2 fc2 weights.")
    p.add_argument("--non-strict", action="store_true", help="Use strict=False when loading via --model-factory.")

    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--candidate-topk", type=int, default=20)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--score-mode", choices=["avg", "sum"], default="avg")
    p.add_argument("--relative-factor", type=float, default=1.5, help="Per-source pruning: keep dist <= factor*best_dist (drop if beyond 1.5x top-1).")
    p.add_argument("--score-threshold", type=float, default=0.2, help="Fixed distance threshold: keep edges with dist <= this.")
    p.add_argument("--min-wcc-size", type=int, default=10, help="Hide weakly connected components smaller than this.")
    p.add_argument("--no-rank1-self-loops", action="store_true", help="Do not draw rank-1 self-loops or special self-loop node color.")

    p.add_argument("--no-size12-highlight", action="store_true", help="Do not detect/highlight size-12 structures.")
    p.add_argument("--allow-shared-size12-edges", action="store_true", help="Do not remove lower-scoring size-12 structures that share edges.")
    p.add_argument("--layout-scale", type=float, default=1.80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-stem", type=str, default="valid_directed_graph_final")
    p.add_argument("--no-zip", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint: {args.ckpt_path}", flush=True)
    W0, W1, W2, load_info = load_decoder_vectors(
        args.ckpt_path,
        device=args.device,
        model_factory=args.model_factory,
        state_dict_key=args.state_dict_key,
        decoder_weight_keys=args.decoder_weight_keys,
        strict=not args.non_strict,
    )
    K, D = W1.shape
    print(f"loaded decoder vectors: K={K}, d_model={D}", flush=True)
    print(f"loader info: {load_info}", flush=True)

    print("building raw top-k edges", flush=True)
    raw_edges = build_edges(W0, W1, W2, args.candidate_topk, args.topk, args.block, args.score_mode)
    write_edges_csv(raw_edges, outdir / "raw_topk_edges.csv")

    print(f"applying relative pruning: factor={args.relative_factor}", flush=True)
    pruned_edges = prune_edges_relative(raw_edges, args.relative_factor)
    write_edges_csv(pruned_edges, outdir / "pruned_edges_relative.csv")

    print(f"filtering graph: dist <= {args.score_threshold}, WCC size >= {args.min_wcc_size}", flush=True)
    G_non, G_show, nonself_edges, self_loop_edges, rank1_self_nodes_all, wcc_rows = build_filtered_graphs(
        pruned_edges,
        score_threshold=args.score_threshold,
        min_wcc_size=args.min_wcc_size,
        keep_rank1_self_loops=not args.no_rank1_self_loops,
    )
    if G_non.number_of_nodes() == 0 or G_non.number_of_edges() == 0:
        raise RuntimeError("No non-self graph remains after filtering. Raise --score-threshold or lower --min-wcc-size.")

    print(f"drawing graph has {G_non.number_of_nodes()} nodes, {G_non.number_of_edges()} non-self edges, {len(self_loop_edges)} rank-1 self-loops", flush=True)
    print("computing layout", flush=True)
    pos = layout_graph_sfdp_or_spring(G_non, outdir, layout_scale=args.layout_scale, seed=args.seed)

    export_node_edge_tables(outdir, G_non, G_show, pruned_edges, nonself_edges, self_loop_edges, rank1_self_nodes_all, pos, wcc_rows)

    structures_before = []
    structures = []
    rejected = []
    cycles12 = []
    seqs12 = []
    if not args.no_size12_highlight:
        print("detecting size-12 cycles/sequences", flush=True)
        cycles12, seqs12, structures_before = find_size12_structures(G_non)
        if args.allow_shared_size12_edges:
            structures = [dict(s) for s in structures_before]
            rejected = []
        else:
            structures, rejected = filter_structures_edge_disjoint(structures_before)
        structures = assign_structure_colors(structures)
        print(f"size-12 found: cycles={len(cycles12)}, sequences={len(seqs12)}, kept={len(structures)}, rejected={len(rejected)}", flush=True)
    else:
        print("size-12 highlighting disabled", flush=True)

    node_to_structs, edge_to_structs, struct_color = write_size12_tables(
        outdir, G_non, structures_before, structures, rejected, pos, rank1_self_nodes_all
    )

    print("drawing final graph", flush=True)
    png, pdf = draw_graph(
        outdir=outdir,
        G_non=G_non,
        G_show=G_show,
        pos=pos,
        rank1_self_nodes_all=rank1_self_nodes_all if not args.no_rank1_self_loops else set(),
        node_to_structs=node_to_structs,
        edge_to_structs=edge_to_structs,
        struct_color=struct_color,
        structures=structures,
        score_threshold=args.score_threshold,
        min_wcc_size=args.min_wcc_size,
        output_stem=args.output_stem,
        title_extra=f"topk={args.topk}, candidate_topk={args.candidate_topk}, relative_factor={args.relative_factor:g}",
    )

    summary_rows = [
        {"metric": "ckpt_path", "value": str(args.ckpt_path)},
        {"metric": "loader", "value": load_info.get("loader", "")},
        {"metric": "decoder_keys", "value": load_info.get("keys", "")},
        {"metric": "K", "value": K},
        {"metric": "d_model", "value": D},
        {"metric": "raw_edges", "value": len(raw_edges)},
        {"metric": "pruned_edges_relative", "value": len(pruned_edges)},
        {"metric": "relative_factor", "value": args.relative_factor},
        {"metric": "score_threshold", "value": args.score_threshold},
        {"metric": "min_wcc_size", "value": args.min_wcc_size},
        {"metric": "drawn_nodes", "value": G_non.number_of_nodes()},
        {"metric": "drawn_nonself_edges", "value": G_non.number_of_edges()},
        {"metric": "drawn_rank1_self_loops", "value": len(self_loop_edges)},
        {"metric": "displayed_edges_total", "value": G_show.number_of_edges()},
        {"metric": "rank1_self_nodes_in_graph", "value": len(set(G_non.nodes()) & rank1_self_nodes_all)},
        {"metric": "directed_cycles_size12_found_before_filter", "value": len(cycles12)},
        {"metric": "directed_sequences_size12_found_before_filter", "value": len(seqs12)},
        {"metric": "size12_structures_kept", "value": len(structures)},
        {"metric": "size12_structures_rejected_shared_edges", "value": len(rejected)},
        {"metric": "highlighted_nodes_unique", "value": len(node_to_structs)},
        {"metric": "highlighted_edges_unique", "value": len(edge_to_structs)},
        {"metric": "png", "value": str(png)},
        {"metric": "pdf", "value": str(pdf)},
    ]
    write_summary(outdir, summary_rows)

    zip_path = None
    if not args.no_zip:
        zip_path = zip_output(outdir)
        print(f"wrote zip: {zip_path}", flush=True)
    print(f"wrote png: {png}", flush=True)
    print(f"wrote pdf: {pdf}", flush=True)
    print(f"outputs are in: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Ghost-rescue AVG-cosine orbit extraction override
# -----------------------------------------------------------------------------
# This block intentionally appears after the historical implementation so that
# step1_graph imports the updated `orbits_from_module` and
# `size12_structures_from_module` without changing the surrounding API.

from dataclasses import asdict as _ghost_asdict
from typing import Any as _GhostAny, Set as _GhostSet


@dataclass
class GhostEdgeRec:
    src: int
    dst: int
    score: float
    s_forward: float
    s_backward: float
    rank: int
    edge_type: str = "topk"


@dataclass
class GhostRec:
    ghost_id: str
    jump_src: int
    jump_dst: int
    jump_score: float
    src_component_id: int
    dst_component_id: int
    same_component: bool


@torch.no_grad()
def build_topk_avg_cos_edges(W0: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, topk: int, block: int) -> List[GhostEdgeRec]:
    """For each SAE-1 feature, keep top-k neighbours by AVG cosine score.

    score(i,j) = 0.5 * (dot(W2[i], W1[j]) + dot(W1[i], W0[j])).
    Higher is better.
    """
    if topk <= 0:
        raise ValueError("topk must be positive")
    K, _D = W1.shape
    topk = min(topk, K)
    W0 = F.normalize(W0.float(), dim=1)
    W1 = F.normalize(W1.float(), dim=1)
    W2 = F.normalize(W2.float(), dim=1)
    W0T = W0.t().contiguous()
    W1T = W1.t().contiguous()
    out: List[GhostEdgeRec] = []
    for s in range(0, K, block):
        e = min(K, s + block)
        sf = W2[s:e] @ W1T
        sb = W1[s:e] @ W0T
        score = 0.5 * (sf + sb)
        vals, idxs = torch.topk(score, k=topk, dim=1, largest=True)
        sf_keep = torch.gather(sf, 1, idxs)
        sb_keep = torch.gather(sb, 1, idxs)
        vals_np = vals.detach().cpu().numpy()
        idxs_np = idxs.detach().cpu().numpy()
        sf_np = sf_keep.detach().cpu().numpy()
        sb_np = sb_keep.detach().cpu().numpy()
        for bi in range(e - s):
            src = s + bi
            for r in range(topk):
                out.append(GhostEdgeRec(
                    src=src,
                    dst=int(idxs_np[bi, r]),
                    score=float(vals_np[bi, r]),
                    s_forward=float(sf_np[bi, r]),
                    s_backward=float(sb_np[bi, r]),
                    rank=r + 1,
                    edge_type="topk",
                ))
        print(f"avg-cos topk edges: processed {e}/{K}; edges={len(out)}", flush=True)
    return out


def _ghost_edge_map(edges: Sequence[GhostEdgeRec]) -> Dict[Tuple[int, int], GhostEdgeRec]:
    return {(int(e.src), int(e.dst)): e for e in edges}


def _ghost_write_rows(path: Path, rows: Sequence[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _ghost_build_threshold_graph(edges: Sequence[GhostEdgeRec], threshold: float, include_self: bool = False):
    nx = ensure_networkx()
    G = nx.DiGraph()
    for ed in edges:
        if not include_self and ed.src == ed.dst:
            continue
        if float(ed.score) >= threshold:
            G.add_edge(
                int(ed.src), int(ed.dst),
                score=float(ed.score), s_forward=float(ed.s_forward), s_backward=float(ed.s_backward),
                rank=int(ed.rank), edge_type=ed.edge_type, is_ghost=False, is_resolved=False,
            )
    return G


def _ghost_component_map_weak(G):
    nx = ensure_networkx()
    comp_id: Dict[_GhostAny, int] = {}
    comps: Dict[int, _GhostSet[_GhostAny]] = {}
    for cid, comp in enumerate(nx.weakly_connected_components(G)):
        cset = set(comp)
        comps[cid] = cset
        for n in cset:
            comp_id[n] = cid
    return comp_id, comps


@torch.no_grad()
def _ghost_add_rescues(G, W0: torch.Tensor, W2: torch.Tensor, jump_threshold: float, only_same_component: bool = False):
    """Insert ghost nodes for high-jump unreachable pairs in the thresholded graph.

    A ghost is inserted as k -> GHOST_k_to_j -> j. No real middle feature is
    selected here; real replacement happens only after a size-12 structure is
    found.
    """
    nx = ensure_networkx()
    H = G.copy()
    real_nodes = sorted(int(n) for n in G.nodes if isinstance(n, (int, np.integer)))
    if not real_nodes:
        return H, [], []
    comp_id, _comps = _ghost_component_map_weak(G)
    reach = {int(n): set(nx.descendants(G, n)) for n in real_nodes}  # frozen reachability
    W0 = F.normalize(W0.float(), dim=1)
    W2 = F.normalize(W2.float(), dim=1)
    node_t = torch.tensor(real_nodes, device=W2.device, dtype=torch.long)
    jump_mat = (W2[node_t] @ W0[node_t].t()).detach().cpu().numpy()
    ghosts: List[GhostRec] = []
    rows: List[dict] = []
    ghost_counter = 0
    for ai, k in enumerate(real_nodes):
        for bj, j in enumerate(real_nodes):
            if k == j:
                continue
            jump = float(jump_mat[ai, bj])
            if jump < jump_threshold:
                continue
            same_comp = comp_id.get(k, -2) == comp_id.get(j, -3)
            base_row = {
                "jump_src": k, "jump_dst": j, "jump_score": jump,
                "src_component_id": comp_id.get(k, -1), "dst_component_id": comp_id.get(j, -1),
                "same_component": same_comp,
            }
            if j in reach.get(k, set()):
                rows.append({**base_row, "applied": False, "reason": "already_reachable", "ghost_id": ""})
                continue
            if only_same_component and not same_comp:
                rows.append({**base_row, "applied": False, "reason": "different_component", "ghost_id": ""})
                continue
            ghost_id = f"GHOST_{ghost_counter:05d}_{k}_to_{j}"
            ghost_counter += 1
            H.add_node(ghost_id, is_ghost=True, jump_src=k, jump_dst=j, jump_score=jump)
            for u, v in [(k, ghost_id), (ghost_id, j)]:
                H.add_edge(
                    u, v, score=jump, s_forward=float("nan"), s_backward=float("nan"), rank=-1,
                    edge_type="ghost", is_ghost=True, is_resolved=False, ghost_id=ghost_id,
                    jump_src=k, jump_dst=j, jump_score=jump,
                )
            gr = GhostRec(ghost_id, k, j, jump, comp_id.get(k, -1), comp_id.get(j, -1), same_comp)
            ghosts.append(gr)
            rows.append({**base_row, "applied": True, "reason": "insert_ghost", "ghost_id": ghost_id})
    return H, ghosts, rows


def _ghost_node_sort_key(n):
    return (0, int(n)) if isinstance(n, (int, np.integer)) else (1, str(n))


def _ghost_canonical_cycle(cyc: Sequence[_GhostAny]) -> Tuple[_GhostAny, ...]:
    c = list(cyc)
    m = min(range(len(c)), key=lambda i: _ghost_node_sort_key(c[i]))
    return tuple(c[m:] + c[:m])


def _ghost_edge_pairs(nodes: Sequence[_GhostAny], is_cycle: bool) -> List[Tuple[_GhostAny, _GhostAny]]:
    pairs = list(zip(nodes, nodes[1:]))
    if is_cycle and len(nodes) > 1:
        pairs.append((nodes[-1], nodes[0]))
    return pairs


def _ghost_seq_str(nodes: Sequence[_GhostAny], is_cycle: bool = False) -> str:
    s = " -> ".join(str(x) for x in nodes)
    if is_cycle and nodes:
        s += " -> " + str(nodes[0])
    return s


def _ghost_find_size_structures(G, length: int = 12) -> List[dict]:
    nx = ensure_networkx()
    structures: List[dict] = []
    cycles = []
    try:
        it = nx.simple_cycles(G, length_bound=length)
        cycles = [_ghost_canonical_cycle(c) for c in it if len(c) == length]
    except TypeError:
        cycles = [_ghost_canonical_cycle(c) for c in nx.simple_cycles(G) if len(c) == length]
    cycles = sorted(set(cycles), key=lambda x: tuple(str(n) for n in x))
    for idx, cyc in enumerate(cycles, 1):
        pairs = _ghost_edge_pairs(cyc, True)
        scores = [float(G[u][v].get("score", np.nan)) for u, v in pairs if G.has_edge(u, v)]
        ghosts = [n for n in cyc if isinstance(n, str) and n.startswith("GHOST_")]
        structures.append({
            "structure_id": f"C{idx:04d}", "kind": "cycle", "nodes": list(cyc), "is_cycle": True,
            "node_count": len(cyc), "edge_count": len(pairs), "sequence": _ghost_seq_str(cyc, True),
            "raw_total_score": float(np.nansum(scores)),
            "raw_mean_score": float(np.nanmean(scores)) if scores else float("nan"),
            "ghost_node_count": len(ghosts), "ghost_nodes": ";".join(map(str, ghosts)),
        })

    seqs = []
    for comp in nx.weakly_connected_components(G):
        comp = set(comp)
        if len(comp) != length:
            continue
        H = G.subgraph(comp).copy()
        if H.number_of_edges() != length - 1:
            continue
        indeg = dict(H.in_degree())
        outdeg = dict(H.out_degree())
        heads = [n for n in H.nodes if indeg[n] == 0 and outdeg[n] == 1]
        tails = [n for n in H.nodes if indeg[n] == 1 and outdeg[n] == 0]
        middles_ok = all(indeg[n] == 1 and outdeg[n] == 1 for n in H.nodes if n not in heads + tails)
        if len(heads) != 1 or len(tails) != 1 or not middles_ok:
            continue
        seq = [heads[0]]
        cur = heads[0]
        ok = True
        while cur != tails[0]:
            succ = list(H.successors(cur))
            if len(succ) != 1:
                ok = False
                break
            cur = succ[0]
            seq.append(cur)
            if len(seq) > length:
                ok = False
                break
        if ok and len(seq) == length:
            seqs.append(tuple(seq))
    seqs = sorted(set(seqs), key=lambda x: tuple(str(n) for n in x))
    for idx, seq in enumerate(seqs, 1):
        pairs = _ghost_edge_pairs(seq, False)
        scores = [float(G[u][v].get("score", np.nan)) for u, v in pairs if G.has_edge(u, v)]
        ghosts = [n for n in seq if isinstance(n, str) and str(n).startswith("GHOST_")]
        structures.append({
            "structure_id": f"S{idx:04d}", "kind": "chain", "nodes": list(seq), "is_cycle": False,
            "node_count": len(seq), "edge_count": len(pairs), "sequence": _ghost_seq_str(seq, False),
            "raw_total_score": float(np.nansum(scores)),
            "raw_mean_score": float(np.nanmean(scores)) if scores else float("nan"),
            "ghost_node_count": len(ghosts), "ghost_nodes": ";".join(map(str, ghosts)),
        })
    return structures


@torch.no_grad()
def _ghost_score_vector_src_to_all(src: int, W0: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((W1 @ W2[int(src)]) + (W0 @ W1[int(src)]))


@torch.no_grad()
def _ghost_score_vector_all_to_dst(dst: int, W0: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((W2 @ W1[int(dst)]) + (W1 @ W0[int(dst)]))


@torch.no_grad()
def _ghost_edge_score(src: int, dst: int, W0: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> Tuple[float, float, float]:
    sf = float(torch.dot(W2[int(src)], W1[int(dst)]).detach().cpu())
    sb = float(torch.dot(W1[int(src)], W0[int(dst)]).detach().cpu())
    return 0.5 * (sf + sb), sf, sb


@torch.no_grad()
def _ghost_resolve_structure(s: dict, W0: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, K: int, topk_edges: Optional[Dict[Tuple[int, int], GhostEdgeRec]] = None, replacement_pool: str = "all") -> dict:
    nodes = list(s["nodes"])
    is_cycle = bool(s["is_cycle"])
    original_real_nodes = {int(n) for n in nodes if isinstance(n, (int, np.integer))}
    used = set(original_real_nodes)
    resolved_nodes: List[_GhostAny] = []
    replacement_rows: List[dict] = []
    for idx, n in enumerate(nodes):
        if not (isinstance(n, str) and n.startswith("GHOST_")):
            resolved_nodes.append(n)
            continue
        prev_n = nodes[idx - 1] if idx > 0 else (nodes[-1] if is_cycle else None)
        next_n = nodes[(idx + 1) % len(nodes)] if (idx + 1 < len(nodes) or is_cycle) else None
        if prev_n is None or next_n is None or isinstance(prev_n, str) or isinstance(next_n, str):
            replacement_rows.append({"ghost_id": n, "replacement": "", "reason": "bad_neighbors"})
            resolved_nodes.append(n)
            continue
        prev_i, next_i = int(prev_n), int(next_n)
        total = _ghost_score_vector_src_to_all(prev_i, W0, W1, W2) + _ghost_score_vector_all_to_dst(next_i, W0, W1, W2)
        mask = torch.ones(K, dtype=torch.bool, device=total.device)
        if used:
            mask[torch.tensor(sorted(used), dtype=torch.long, device=total.device)] = False
        if replacement_pool == "topk":
            pool = torch.zeros(K, dtype=torch.bool, device=total.device)
            if topk_edges is not None:
                for (u, v), _ed in topk_edges.items():
                    if u == prev_i:
                        pool[int(v)] = True
                    if v == next_i:
                        pool[int(u)] = True
            mask &= pool
        total_masked = total.masked_fill(~mask, -1e9)
        best = int(torch.argmax(total_masked).detach().cpu())
        best_total = float(total_masked[best].detach().cpu())
        if best_total <= -1e8:
            replacement_rows.append({"ghost_id": n, "replacement": "", "reason": "no_candidate"})
            resolved_nodes.append(n)
            continue
        used.add(best)
        score_left, sf_left, sb_left = _ghost_edge_score(prev_i, best, W0, W1, W2)
        score_right, sf_right, sb_right = _ghost_edge_score(best, next_i, W0, W1, W2)
        replacement_rows.append({
            "ghost_id": n, "replacement": best, "prev": prev_i, "next": next_i,
            "score_prev_repl": score_left, "sf_prev_repl": sf_left, "sb_prev_repl": sb_left,
            "score_repl_next": score_right, "sf_repl_next": sf_right, "sb_repl_next": sb_right,
            "two_edge_score": score_left + score_right, "reason": "ok",
        })
        resolved_nodes.append(best)

    unresolved = any(isinstance(n, str) and str(n).startswith("GHOST_") for n in resolved_nodes)
    final_pairs = _ghost_edge_pairs(resolved_nodes, is_cycle) if not unresolved else []
    final_edge_rows: List[dict] = []
    total_score = 0.0
    for pos, (u, v) in enumerate(final_pairs):
        if isinstance(u, str) or isinstance(v, str):
            continue
        sc, sf, sb = _ghost_edge_score(int(u), int(v), W0, W1, W2)
        total_score += sc
        final_edge_rows.append({"position": pos, "src": int(u), "dst": int(v), "score": sc, "s_forward": sf, "s_backward": sb})
    out = dict(s)
    out.update({
        "resolved_nodes": resolved_nodes,
        "resolved_sequence": _ghost_seq_str(resolved_nodes, is_cycle),
        "resolved_unresolved_ghost_count": int(unresolved),
        "replacement_rows": replacement_rows,
        "replacement_count": len([r for r in replacement_rows if r.get("reason") == "ok"]),
        "final_edge_rows": final_edge_rows,
        "final_edges": [(int(u), int(v)) for u, v in final_pairs if not isinstance(u, str) and not isinstance(v, str)],
        "final_real_nodes": [int(n) for n in resolved_nodes if not isinstance(n, str)],
        "final_total_score": float(total_score),
        "final_mean_score": float(total_score / len(final_pairs)) if final_pairs else float("nan"),
    })
    return out


def _ghost_filter_structures_overlap(structures: Sequence[dict], mode: str = "edge") -> Tuple[List[dict], List[dict]]:
    ordered = sorted(structures, key=lambda s: (float(s.get("final_total_score", -1e9)), float(s.get("final_mean_score", -1e9))), reverse=True)
    used_edges: _GhostSet[Tuple[int, int]] = set()
    used_nodes: _GhostSet[int] = set()
    kept: List[dict] = []
    rejected: List[dict] = []
    for s in ordered:
        if int(s.get("resolved_unresolved_ghost_count", 0)) > 0:
            item = dict(s); item["reject_reason"] = "unresolved_ghost"; rejected.append(item); continue
        edges = set((int(u), int(v)) for u, v in s.get("final_edges", []))
        nodes = set(int(n) for n in s.get("final_real_nodes", []))
        if mode == "node":
            overlap = nodes & used_nodes
            overlap_text = ";".join(map(str, sorted(overlap)))
        else:
            overlap = edges & used_edges
            overlap_text = ";".join(f"{u}->{v}" for u, v in sorted(overlap))
        if overlap:
            item = dict(s); item["reject_reason"] = f"{mode}_overlap"; item["overlap"] = overlap_text; rejected.append(item)
        else:
            item = dict(s); item["kept_index"] = len(kept) + 1; kept.append(item)
            used_edges.update(edges); used_nodes.update(nodes)
    return kept, rejected


def _ghost_materialize_kept_graph(G_base, kept_structures: Sequence[dict]):
    H = G_base.copy()
    for s in kept_structures:
        for e in s.get("final_edge_rows", []):
            u, v = int(e["src"]), int(e["dst"])
            if not H.has_edge(u, v):
                H.add_edge(u, v, score=float(e["score"]), s_forward=float(e["s_forward"]), s_backward=float(e["s_backward"]), rank=-1, edge_type="resolved_from_ghost", is_ghost=False, is_resolved=True)
    return H


def _ghost_decompose_remaining_chains(G, used_nodes: _GhostSet[int], min_chain_len: int = 11) -> List[dict]:
    nodes = [int(n) for n in G.nodes if isinstance(n, (int, np.integer)) and int(n) not in used_nodes]
    nodeset = set(nodes)
    edges = []
    for u, v, d in G.edges(data=True):
        if not isinstance(u, (int, np.integer)) or not isinstance(v, (int, np.integer)):
            continue
        u, v = int(u), int(v)
        if u == v or u not in nodeset or v not in nodeset:
            continue
        edges.append((u, v, float(d.get("score", 0.0))))
    edges.sort(key=lambda e: e[2], reverse=True)
    out_used: Dict[int, int] = {}
    in_used: Dict[int, int] = {}
    parent = {n: n for n in nodes}
    length = {n: 1 for n in nodes}
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v, _s in edges:
        if u in out_used or v in in_used:
            continue
        ru, rv = find(u), find(v)
        if ru == rv:
            if length[ru] == 12:
                out_used[u] = v; in_used[v] = u
            continue
        out_used[u] = v; in_used[v] = u
        parent[ru] = rv; length[rv] = length[ru] + length[rv]
    results: List[dict] = []
    visited: _GhostSet[int] = set()
    for n in nodes:
        if n in visited or n in in_used or n not in out_used:
            continue
        path = [n]; cur = n
        while cur in out_used:
            cur = out_used[cur]; path.append(cur)
        visited.update(path)
        if len(path) >= min_chain_len:
            pairs = _ghost_edge_pairs(path, False)
            scores = [float(G[u][v].get("score", np.nan)) for u, v in pairs]
            results.append({"kind": "leftover_chain", "is_cycle": False, "nodes": path, "edge_scores": scores, "total_score": float(np.nansum(scores)), "mean_score": float(np.nanmean(scores)) if scores else float("nan")})
    for n in nodes:
        if n in visited or n not in out_used:
            continue
        cyc = [n]; cur = out_used[n]
        while cur != n:
            cyc.append(cur); cur = out_used[cur]
            if len(cyc) > 12:
                break
        visited.update(cyc)
        if len(cyc) == 12:
            pairs = _ghost_edge_pairs(cyc, True)
            scores = [float(G[u][v].get("score", np.nan)) for u, v in pairs]
            results.append({"kind": "leftover_cycle12", "is_cycle": True, "nodes": cyc, "edge_scores": scores, "total_score": float(np.nansum(scores)), "mean_score": float(np.nanmean(scores)) if scores else float("nan")})
    results.sort(key=lambda r: (r["kind"], -len(r["nodes"]), -float(r["total_score"])))
    for i, r in enumerate(results, 1):
        r["leftover_id"] = f"L{i:04d}"
    return results


def _ghost_flatten_structure_rows(structures: Sequence[dict]) -> List[dict]:
    rows = []
    for s in structures:
        rows.append({
            "structure_id": s.get("structure_id"), "kept_index": s.get("kept_index", ""), "kind": s.get("kind"),
            "node_count": s.get("node_count"), "edge_count": s.get("edge_count"),
            "ghost_node_count": s.get("ghost_node_count"), "replacement_count": s.get("replacement_count", ""),
            "raw_total_score": s.get("raw_total_score"), "raw_mean_score": s.get("raw_mean_score"),
            "final_total_score": s.get("final_total_score"), "final_mean_score": s.get("final_mean_score"),
            "resolved_unresolved_ghost_count": s.get("resolved_unresolved_ghost_count", ""),
            "reject_reason": s.get("reject_reason", ""), "overlap": s.get("overlap", ""),
            "sequence": s.get("sequence"), "resolved_sequence": s.get("resolved_sequence", ""), "ghost_nodes": s.get("ghost_nodes", ""),
        })
    return rows


def _ghost_extract_orbits_from_vectors(
    W0: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    *,
    topk: int = 6,
    block: int = 512,
    score_threshold: float = 0.8,
    jump_threshold: float = 0.8,
    min_wcc_size: int = 10,
    structure_size: int = 12,
    replacement_pool: str = "all",
    overlap_mode: str = "edge",
    include_leftover: bool = True,
    min_chain_len: int = 11,
    only_same_component_jump: bool = False,
    debug_outdir: Optional[Path] = None,
):
    nx = ensure_networkx()
    K, _D = W1.shape
    W0 = F.normalize(W0.float(), dim=1)
    W1 = F.normalize(W1.float(), dim=1)
    W2 = F.normalize(W2.float(), dim=1)
    topk_edges = build_topk_avg_cos_edges(W0, W1, W2, topk=topk, block=block)
    topk_map = _ghost_edge_map(topk_edges)
    rank1_self_nodes = {int(e.src) for e in topk_edges if int(e.rank) == 1 and int(e.src) == int(e.dst)}

    G0 = _ghost_build_threshold_graph(topk_edges, score_threshold, include_self=False)
    Gghost, ghosts, proposal_rows = _ghost_add_rescues(G0, W0, W2, jump_threshold, only_same_component=only_same_component_jump)

    Gfind = Gghost
    wcc_rows = []
    if min_wcc_size and min_wcc_size > 1:
        keep_nodes = set()
        for cid, comp in enumerate(sorted(nx.weakly_connected_components(Gghost), key=len, reverse=True)):
            comp = set(comp)
            keep = len(comp) >= min_wcc_size
            wcc_rows.append({"component_id": cid, "node_count": len(comp), "edge_count": Gghost.subgraph(comp).number_of_edges(), "kept": keep})
            if keep:
                keep_nodes.update(comp)
        Gfind = Gghost.subgraph(keep_nodes).copy()

    raw_structures = _ghost_find_size_structures(Gfind, length=structure_size)
    resolved = [_ghost_resolve_structure(s, W0, W1, W2, K, topk_edges=topk_map, replacement_pool=replacement_pool) for s in raw_structures]
    kept, rejected = _ghost_filter_structures_overlap(resolved, mode=overlap_mode)

    Gmaterial = _ghost_materialize_kept_graph(G0, kept)
    used_nodes = {int(n) for s in kept for n in s.get("final_real_nodes", [])}
    leftovers = _ghost_decompose_remaining_chains(Gmaterial, used_nodes, min_chain_len=min_chain_len) if include_leftover else []

    if debug_outdir is not None:
        debug_outdir.mkdir(parents=True, exist_ok=True)
        _ghost_write_rows(debug_outdir / "topk_raw_edges_all.csv", [_ghost_asdict(e) for e in topk_edges])
        _ghost_write_rows(debug_outdir / "topk_edges_ge_threshold_before_ghost.csv", [dict(src=u, dst=v, **d) for u, v, d in G0.edges(data=True)])
        _ghost_write_rows(debug_outdir / "ghost_rescue_pairs.csv", proposal_rows)
        _ghost_write_rows(debug_outdir / "ghost_nodes.csv", [_ghost_asdict(g) for g in ghosts])
        _ghost_write_rows(debug_outdir / "wcc_components_after_ghost.csv", wcc_rows)
        _ghost_write_rows(debug_outdir / "ghost_structures_before_resolution.csv", _ghost_flatten_structure_rows(raw_structures))
        _ghost_write_rows(debug_outdir / "resolved_structures_before_overlap.csv", _ghost_flatten_structure_rows(resolved))
        _ghost_write_rows(debug_outdir / "size12_structures_kept_overlap_filtered.csv", _ghost_flatten_structure_rows(kept))
        _ghost_write_rows(debug_outdir / "size12_structures_rejected_overlap.csv", _ghost_flatten_structure_rows(rejected))
        repl_rows = []
        edge_rows = []
        for s in resolved:
            for rr in s.get("replacement_rows", []):
                repl_rows.append({**rr, "structure_id": s.get("structure_id"), "resolved_sequence": s.get("resolved_sequence", "")})
            for er in s.get("final_edge_rows", []):
                edge_rows.append({**er, "structure_id": s.get("structure_id"), "kind": s.get("kind")})
        _ghost_write_rows(debug_outdir / "ghost_replacements.csv", repl_rows)
        _ghost_write_rows(debug_outdir / "resolved_structure_edges.csv", edge_rows)
        _ghost_write_rows(debug_outdir / "leftover_chains_gt10_and_cycles12.csv", [
            {"leftover_id": r.get("leftover_id"), "kind": r.get("kind"), "node_count": len(r.get("nodes", [])), "total_score": r.get("total_score"), "mean_score": r.get("mean_score"), "sequence": _ghost_seq_str(r.get("nodes", []), bool(r.get("is_cycle")))} for r in leftovers
        ])
        _ghost_write_rows(debug_outdir / "summary.csv", [
            {"metric": "score_mode", "value": "avg_cosine"},
            {"metric": "topk", "value": topk},
            {"metric": "score_threshold", "value": score_threshold},
            {"metric": "jump_threshold", "value": jump_threshold},
            {"metric": "raw_topk_edges", "value": len(topk_edges)},
            {"metric": "initial_threshold_graph_nodes", "value": G0.number_of_nodes()},
            {"metric": "initial_threshold_graph_edges", "value": G0.number_of_edges()},
            {"metric": "ghost_pairs_applied", "value": len(ghosts)},
            {"metric": "structure_search_nodes", "value": Gfind.number_of_nodes()},
            {"metric": "structure_search_edges", "value": Gfind.number_of_edges()},
            {"metric": "size12_structures_before_overlap", "value": len(raw_structures)},
            {"metric": "structures_with_ghost_before_overlap", "value": sum(1 for s in raw_structures if int(s.get("ghost_node_count", 0)) > 0)},
            {"metric": "kept_structures", "value": len(kept)},
            {"metric": "leftover_items", "value": len(leftovers)},
        ], fieldnames=["metric", "value"])

    records: List[OrbitRecord] = []
    for idx, s in enumerate(kept):
        is_cycle = bool(s.get("is_cycle"))
        nodes = [int(n) for n in s.get("final_real_nodes", [])]
        scores = [float(e.get("score", float("nan"))) for e in s.get("final_edge_rows", [])]
        records.append(OrbitRecord(
            nodes=nodes,
            is_ring=is_cycle,
            label="Ring12" if is_cycle else "Seq12",
            source="ghost_rescue_avg",
            kind="ghost_resolved_12_cycle" if is_cycle else "ghost_resolved_12_chain",
            original_structure_id=str(s.get("structure_id", "")),
            mean_score=float(s.get("final_mean_score", float("nan"))),
            min_score=float(np.nanmin(scores)) if scores else None,
            max_score=float(np.nanmax(scores)) if scores else None,
            all_ranks="",
            edge_scores=scores,
        ))
    for r in leftovers:
        nodes = [int(n) for n in r.get("nodes", [])]
        is_cycle = bool(r.get("is_cycle"))
        scores = [float(x) for x in r.get("edge_scores", [])]
        records.append(OrbitRecord(
            nodes=nodes,
            is_ring=is_cycle,
            label="GreedyRing12" if is_cycle else "Chain",
            source="leftover_greedy",
            kind="greedy_leftover_12_cycle" if is_cycle else "greedy_leftover_chain",
            mean_score=float(r.get("mean_score", float("nan"))),
            min_score=float(np.nanmin(scores)) if scores else None,
            max_score=float(np.nanmax(scores)) if scores else None,
            edge_scores=scores,
        ))
    info = {
        "topk_edges": topk_edges,
        "G0": G0,
        "Gghost": Gghost,
        "Gfind": Gfind,
        "kept": kept,
        "rejected": rejected,
        "leftovers": leftovers,
        "ghosts": ghosts,
    }
    return records, rank1_self_nodes, K, info


# Override historical API with ghost-rescue average-cosine orbit extraction.
def orbits_from_module(
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
    include_leftover: bool = True,
    min_chain_len: int = 11,
    return_records: bool = False,
    jump_threshold: float = 0.8,
    replacement_pool: str = "all",
    overlap_mode: str = "edge",
    only_same_component_jump: bool = False,
    debug_outdir: Optional[str] = None,
):
    """Ghost-rescue AVG-cosine orbit extraction.

    Kept for API compatibility with step1_graph. `candidate_topk`,
    `relative_factor`, and `edge_disjoint` are accepted but not used by the new
    logic. `score_mode` must be avg-like; this implementation always uses AVG
    cosine score, higher is better.
    """
    if score_mode not in {"avg", "cos_avg", "avg_cosine"}:
        raise ValueError("ghost-rescue extraction uses avg cosine score; set --score-mode avg")
    W0, W1, W2, _info = decoder_vectors_from_module(module, device)
    records, rank1_self_nodes, K, _info2 = _ghost_extract_orbits_from_vectors(
        W0, W1, W2,
        topk=topk,
        block=block,
        score_threshold=score_threshold,
        jump_threshold=jump_threshold,
        min_wcc_size=min_wcc_size,
        structure_size=12,
        replacement_pool=replacement_pool,
        overlap_mode=overlap_mode,
        include_leftover=include_leftover,
        min_chain_len=min_chain_len,
        only_same_component_jump=only_same_component_jump,
        debug_outdir=Path(debug_outdir) if debug_outdir else None,
    )
    if return_records:
        return records, rank1_self_nodes, K
    return [(r.nodes, r.is_ring) for r in records], rank1_self_nodes, K


def size12_structures_from_module(
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
    jump_threshold: float = 0.8,
    replacement_pool: str = "all",
    overlap_mode: str = "edge",
):
    records, rank1_self_nodes, K = orbits_from_module(
        module, device, topk=topk, candidate_topk=candidate_topk, block=block,
        score_mode=score_mode, relative_factor=relative_factor, score_threshold=score_threshold,
        min_wcc_size=min_wcc_size, edge_disjoint=edge_disjoint, include_leftover=False,
        return_records=True, jump_threshold=jump_threshold, replacement_pool=replacement_pool,
        overlap_mode=overlap_mode,
    )
    structures = []
    for i, rec in enumerate(records, 1):
        if rec.label not in {"Ring12", "Seq12"}:
            continue
        nodes = tuple(rec.nodes)
        is_cycle = rec.is_ring
        structures.append({
            "structure_id": f"C{i:02d}" if is_cycle else f"S{i:02d}",
            "kind": "cycle" if is_cycle else "sequence",
            "index": i,
            "nodes": nodes,
            "edges": tuple((nodes[j], nodes[(j + 1) % len(nodes)]) for j in range(len(nodes))) if is_cycle else tuple((nodes[j], nodes[j + 1]) for j in range(len(nodes) - 1)),
            "sequence": _ghost_seq_str(nodes, is_cycle),
            "mean_score": rec.mean_score,
            "min_score": rec.min_score,
            "max_score": rec.max_score,
        })
    return structures, rank1_self_nodes, K
