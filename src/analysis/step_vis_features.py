#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step visualization: 2D SAE feature map with semantic coloring.

Reads run_dir.sh outputs:
    orbits_raw.txt              (step1_graph.py)
    epoch{N}_feature_ids.txt    (step2_rings.py + step3_boundary.py)

Plot 1  (<out-dir>/features_all.png)
    All SAE features coloured by category:
      [Forth]       → black
      [Major Chord] → red
      [Minor Chord] → green
      Other orbits  → yellow  (any orbit in orbits_raw not named above)
      Others        → blue

Plot 2  (<out-dir>/features_forth.png)
    [Forth] ring only, with directed arrows and pitch-class labels
    (C, C#, D, ... — ring is already reordered chromatically by step2).

Usage:
    python -m src.analysis.step_vis_features \\
        --ckpt-path   store/.../ckpts/last.ckpt \\
        --orbits      store/.../analysis/<run>/orbits_raw.txt \\
        --feature-ids store/.../analysis/<run>/epoch42_feature_ids.txt \\
        --out-dir     store/.../analysis/<run> \\
        [--vec dec] [--method tsne] [--sae-idx 1] [--device cuda]
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..music_sae.sae_lit import LitSAE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

C_FORTH  = '#111111'   # black  — [Forth]
C_MAJOR  = '#E83030'   # red    — [Major Chord]
C_MINOR  = '#3DB461'   # green  — [Minor Chord]
# C_OTHER  = '#FDAF13'   # yellow — other orbits
C_BG     = '#89BCFE'   # blue   — ungrouped

VEC_CHOICES = ('enc', 'dec', 'enc_raw', 'dec_raw')


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Group:
    tag: str
    kind: str               # 'ring' | 'chain' | 'scatter'
    node_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File parsing  (same format as vis_ring_seq.py)
# ---------------------------------------------------------------------------

def parse_ids_file(path: Path) -> List[Group]:
    groups: List[Group] = []
    header_re = re.compile(r'^\[(.+?)\]:\s*(.+)$')
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line:
            continue
        m = header_re.match(line)
        if not m:
            continue
        tag  = m.group(1).strip()
        rest = m.group(2).strip()
        if '->' in rest:
            parts = rest.split(None, 1)
            ids_str = parts[1] if (len(parts) == 2 and parts[0].isdigit()) else rest
            node_ids = [int(x.strip()) for x in ids_str.split('->')]
            kind = 'chain' if tag.startswith('Seq') else 'ring'
        else:
            tokens = rest.split()
            if not tokens:
                continue
            try:
                node_ids = [int(t) for t in tokens]
            except ValueError:
                continue
            kind = 'scatter'
        groups.append(Group(tag=tag, kind=kind, node_ids=node_ids))
    return groups


def _find_group(groups: List[Group], tag: str) -> Optional[Group]:
    for g in groups:
        if g.tag == tag:
            return g
    return None


# ---------------------------------------------------------------------------
# SAE loading / vector extraction
# ---------------------------------------------------------------------------

def load_module(ckpt_path: Path, device: str, sae_idx: int):
    lit: LitSAE = LitSAE.load_from_checkpoint(
        str(ckpt_path.resolve()), map_location=device, strict=False
    )
    lit.eval().to(device)
    sae = lit.sae
    sae.eval().to(device)
    if not hasattr(sae, 'sae'):
        raise RuntimeError("Expected GroupSAE to have .sae (MultiSAE).")
    modules = list(sae.sae.sae_modules)
    idx = max(0, min(sae_idx, len(modules) - 1))
    print(f"[INFO] Using sub-SAE idx={idx} / {len(modules)}")
    return modules[idx]


def extract_all_vectors(module, vec_type: str, device: str, eps: float = 1e-8) -> torch.Tensor:
    enc = module.fc1.weight.detach().to(device)
    dec = module.fc2.weight.detach().to(device).t()
    if vec_type == 'enc':
        v = enc / (enc.norm(dim=1, keepdim=True) + eps)
    elif vec_type == 'dec':
        v = dec / (dec.norm(dim=1, keepdim=True) + eps)
    elif vec_type == 'enc_raw':
        v = enc
    elif vec_type == 'dec_raw':
        v = dec
    else:
        raise ValueError(f"--vec must be one of {VEC_CHOICES}, got {vec_type!r}")
    return v.float().cpu()


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce(vecs: np.ndarray, method: str, n_components: int = 2,
           perplexity: float = 30.0, pca_pre_dims: int = 50,
           random_state: int = 42) -> np.ndarray:
    N, D = vecs.shape
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        coords = reducer.fit_transform(vecs)
        evr = reducer.explained_variance_ratio_
        print(f"[INFO] PCA EVR: {' '.join(f'PC{i+1}={v:.3f}' for i, v in enumerate(evr))}")
        return coords
    elif method == 'tsne':
        work = vecs
        if D > pca_pre_dims:
            n_pre = min(pca_pre_dims, N - 1)
            print(f"[INFO] PCA pre-reduction {D}→{n_pre} before t-SNE")
            work = PCA(n_components=n_pre, random_state=random_state).fit_transform(vecs)
        perp = min(perplexity, (N - 1) / 3.0)
        print(f"[INFO] t-SNE  N={N}  perplexity={perp:.1f}")
        tsne = TSNE(n_components=n_components, perplexity=perp, max_iter=1000,
                    init='pca' if N >= 4 else 'random', random_state=random_state)
        return tsne.fit_transform(work)
    else:
        raise ValueError(f"--method must be 'tsne' or 'pca', got {method!r}")


# ---------------------------------------------------------------------------
# Plot 1: all features, coloured by category
# ---------------------------------------------------------------------------

def plot_all_features(
    coords: np.ndarray,
    n_feat: int,
    forth_group:  Optional[Group],
    major_group:  Optional[Group],
    minor_group:  Optional[Group],
    # other_orbits: List[Group],
    out_path: Path,
) -> None:
    forth_ids  = set(forth_group.node_ids) if forth_group else set()
    major_ids  = set(major_group.node_ids) if major_group else set()
    minor_ids  = set(minor_group.node_ids) if minor_group else set()
    named_ids  = forth_ids | major_ids | minor_ids
    other_ids: Set[int] = set()
    # for o in other_orbits:
    #     other_ids.update(o.node_ids)
    bg_ids = [i for i in range(n_feat)
              if i not in named_ids and i not in other_ids
              and not np.isnan(coords[i, 0])]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect('equal', adjustable='datalim')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def _scatter_ids(ids, color, alpha=0.4, zorder=1):
        if not ids:
            return
        pts = coords[list(ids)]
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=80, alpha=alpha,
                   zorder=zorder, linewidths=0)

    def _draw_group(g: Group, color: str, z_line: int, z_pt: int) -> None:
        ids = g.node_ids
        pts = np.array([coords[fid] for fid in ids])
        if g.kind == 'ring':
            ring_xs = np.append(pts[:, 0], pts[0, 0])
            ring_ys = np.append(pts[:, 1], pts[0, 1])
            ax.plot(ring_xs, ring_ys, color=color, lw=0.6, alpha=0.7, zorder=z_line)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.6, alpha=0.7, zorder=z_line)
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=80, zorder=z_pt,
                   alpha=0.92, edgecolors='white', linewidths=0.5)

    # Back-to-front: blue → yellow → green → red → black
    _scatter_ids(bg_ids, C_BG, alpha=0.4, zorder=1)
    # for o in other_orbits:
    #     _draw_group(o, C_OTHER, z_line=2, z_pt=3)
    if minor_group:
        _draw_group(minor_group, C_MINOR, z_line=4, z_pt=5)
    if major_group:
        _draw_group(major_group, C_MAJOR, z_line=4, z_pt=5)
    if forth_group:
        _draw_group(forth_group, C_FORTH, z_line=6, z_pt=7)

    ax.margins(0.02)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()

    # Two versions: <stem>_nolegend saved first, then the legend is added and
    # the main figure saved.
    nolegend_path = out_path.with_name(f"{out_path.stem}_nolegend{out_path.suffix}")
    plt.savefig(str(nolegend_path), dpi=150, bbox_inches='tight', transparent=True)
    print(f"[DONE] Saved → {nolegend_path}")

    handles = []
    if forth_group:
        handles.append(mpatches.Patch(color=C_FORTH, label='Relative-major\nsubdominant'))
    if major_group:
        handles.append(mpatches.Patch(color=C_MAJOR, label='Major Chord'))
    if minor_group:
        handles.append(mpatches.Patch(color=C_MINOR, label='Minor Chord'))
    handles += [
        # mpatches.Patch(color=C_OTHER, label='Other nodes in\nchains and rings'),
        mpatches.Patch(color=C_BG,    label='Others'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=20,
              framealpha=0.88, ncol=1)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"[DONE] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: [Forth] ring only, with directed arrows and note labels
# ---------------------------------------------------------------------------

def _arrow2d(ax, x0: float, y0: float, x1: float, y1: float,
             color: str, lw: float = 4) -> None:
    if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
        return
    ax.annotate(
        '',
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=18, shrinkA=5, shrinkB=5),
        zorder=4,
    )


def _pt_seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Distance from point P to segment AB, and the closest point on AB."""
    AB = B - A
    t  = np.dot(P - A, AB) / max(float(np.dot(AB, AB)), 1e-12)
    t  = max(0.0, min(1.0, t))
    Q  = A + t * AB
    return float(np.hypot(P[0] - Q[0], P[1] - Q[1])), Q


def _repel_label_positions(
    fig, ax,
    pts: np.ndarray,
    node_pts: np.ndarray | None = None,
    edges: list | None = None,
    init_r_pts: float = 55.0,
    min_dist_pts: float = 80.0,
    min_node_pts: float = 55.0,
    min_edge_pts: float = 55.0,
    n_iter: int = 1000,
    tighten_iter: int = 300,
) -> np.ndarray:
    """
    Return label positions (data coords, shape [n, 2]) that:
      1. do not overlap each other
      2. do not cover any node (scatter circle)
      3. do not cover any edge (arrow line segment)
      4. are as close as possible to their own node (tightening phase)

    pts          : n node positions being labeled
    node_pts     : all obstacle node positions; defaults to pts
    edges        : list of (A_data, B_data) segment pairs to avoid
    init_r_pts   : initial radial push from own node, in typographic points
    min_dist_pts : minimum label-to-label centre distance, in points
    min_node_pts : minimum label-to-any-node centre distance, in points
    min_edge_pts : minimum label-centre-to-edge distance, in points
    """
    if node_pts is None:
        node_pts = pts

    fig.canvas.draw()
    pts_to_px  = fig.dpi / 72.0
    tr         = ax.transData
    tr_inv     = tr.inverted()

    own_px     = tr.transform(pts).astype(float)
    all_nd_px  = tr.transform(node_pts).astype(float)
    cx, cy     = own_px.mean(axis=0)
    init_px    = init_r_pts  * pts_to_px
    min_px     = min_dist_pts * pts_to_px
    min_nd_px  = min_node_pts * pts_to_px
    min_ed_px  = min_edge_pts * pts_to_px

    edges_px: list = []
    if edges is not None:
        for A_d, B_d in edges:
            edges_px.append((
                tr.transform([A_d])[0].astype(float),
                tr.transform([B_d])[0].astype(float),
            ))

    def _feasible(pos: np.ndarray, skip: int) -> bool:
        for j in range(len(pts)):
            if j == skip:
                continue
            if np.hypot(pos[0] - lbl_px[j][0], pos[1] - lbl_px[j][1]) < min_px:
                return False
        for nd in all_nd_px:
            if np.hypot(pos[0] - nd[0], pos[1] - nd[1]) < min_nd_px:
                return False
        for A, B in edges_px:
            d, _ = _pt_seg_dist(pos, A, B)
            if d < min_ed_px:
                return False
        return True

    # Seed labels radially outward from ring centre
    lbl_px = own_px.copy()
    for i in range(len(pts)):
        dv   = own_px[i] - [cx, cy]
        norm = max(np.hypot(dv[0], dv[1]), 1e-9)
        lbl_px[i] += dv / norm * init_px

    # Main repulsion: label↔label, label↔node, label↔edge
    for _ in range(n_iter):
        moved = False

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dv   = lbl_px[i] - lbl_px[j]
                dist = np.hypot(dv[0], dv[1])
                if dist < min_px and dist > 1e-9:
                    push        = (min_px - dist) * 0.5 * dv / dist
                    lbl_px[i] += push
                    lbl_px[j] -= push
                    moved = True

        for i in range(len(pts)):
            for nd in all_nd_px:
                dv   = lbl_px[i] - nd
                dist = np.hypot(dv[0], dv[1])
                if dist < min_nd_px and dist > 1e-9:
                    lbl_px[i] += (min_nd_px - dist) * dv / dist
                    moved = True

        for i in range(len(pts)):
            for A, B in edges_px:
                dist, Q = _pt_seg_dist(lbl_px[i], A, B)
                if dist < min_ed_px and dist > 1e-9:
                    dv = lbl_px[i] - Q
                    lbl_px[i] += (min_ed_px - dist) * dv / dist
                    moved = True

        if not moved:
            break

    # Tightening: pull each label toward its own node while staying feasible
    for _ in range(tighten_iter):
        for i in range(len(pts)):
            dv   = own_px[i] - lbl_px[i]
            dist = float(np.hypot(dv[0], dv[1]))
            if dist < 1.0:
                continue
            step    = min(2.0, dist * 0.05)
            new_pos = lbl_px[i] + dv / dist * step
            if _feasible(new_pos, skip=i):
                lbl_px[i] = new_pos

    return tr_inv.transform(lbl_px)


def _bbox_edge_point(
    node_disp: np.ndarray,
    label_disp: np.ndarray,
    bbox,
) -> np.ndarray:
    """
    Return the display-coord point on the bbox boundary that lies on the
    ray from label_disp toward node_disp (smallest positive parametric t).
    Falls back to label_disp if no intersection is found.
    """
    lx, ly = float(label_disp[0]), float(label_disp[1])
    nx, ny = float(node_disp[0]),  float(node_disp[1])
    dx, dy = nx - lx, ny - ly
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    eps = 1e-9
    t_hits = []

    if abs(dx) > eps:
        for ex in (x0, x1):
            t = (ex - lx) / dx
            if t > eps:
                yh = ly + t * dy
                if y0 - eps <= yh <= y1 + eps:
                    t_hits.append(t)

    if abs(dy) > eps:
        for ey in (y0, y1):
            t = (ey - ly) / dy
            if t > eps:
                xh = lx + t * dx
                if x0 - eps <= xh <= x1 + eps:
                    t_hits.append(t)

    if not t_hits:
        return np.array(label_disp, dtype=float)
    t_min = min(t_hits)
    return np.array([lx + t_min * dx, ly + t_min * dy])


def plot_ring(
    coords: np.ndarray,
    group: Group,
    color: str,
    out_path: Path,
) -> None:
    """Draw a single ring with directed arrows and pitch-class note labels."""
    ids   = group.node_ids
    pts   = np.array([coords[fid] for fid in ids])
    n     = len(pts)

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect('equal', adjustable='datalim')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        _arrow2d(ax, x0, y0, x1, y1, color=color)

    ax.scatter(pts[:, 0], pts[:, 1],
               facecolors='none', edgecolors=color,
               s=220, linewidths=4, zorder=3)

    ax.margins(0.02)
    plt.tight_layout()

    edges_data = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    label_pts = _repel_label_positions(fig, ax, pts, node_pts=pts, edges=edges_data)

    # Pass 1: draw text boxes at zorder=9, collect artist refs for bbox query
    text_artists = []
    for i in range(n):
        lx, ly = label_pts[i]
        ta = ax.text(
            lx, ly, NOTES[i % len(NOTES)],
            fontsize=44, fontweight='bold', color='#7B2FBE',
            ha='center', va='center', zorder=9,
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      alpha=0.90, ec='#7B2FBE', lw=2),
        )
        text_artists.append(ta)

    # Commit layout so text bboxes are fully rendered
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tr        = ax.transData

    # Pass 2: connector lines at zorder=11 — from exact bbox edge to node center
    for i, (px, py) in enumerate(pts):
        node_disp  = tr.transform([[px, py]])[0]
        lbl_disp   = tr.transform([label_pts[i]])[0]
        win_bbox   = text_artists[i].get_window_extent(renderer)
        start_disp = _bbox_edge_point(node_disp, lbl_disp, win_bbox)
        start_data = tr.inverted().transform([start_disp])[0]
        ax.plot([start_data[0], px], [start_data[1], py],
                '-', color='#7B2FBE', lw=2, zorder=11)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"[DONE] Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Visualize SAE features with semantic coloring.'
    )
    ap.add_argument('--ckpt-path',    required=True,
                    help='LitSAE checkpoint (.ckpt)')
    ap.add_argument('--orbits',       default=None,
                    help='orbits_raw.txt from step1_graph.py '
                         '(omit for mode=1-1 where no ring structure exists)')
    ap.add_argument('--feature-ids',  required=True,
                    help='epoch*_feature_ids.txt from step2_rings.py')
    ap.add_argument('--out-dir',          required=True,
                    help='Directory to write features_all.png and features_forth.png')
    ap.add_argument('--act-counts-file',  default=None,
                    help='activation_counts.npy from step_activation_counts.py '
                         '(features below --min-act-count are excluded from t-SNE)')
    ap.add_argument('--min-act-count',    type=int, default=1,
                    help='Minimum activation count to include a feature in t-SNE '
                         '(ring/orbit features are always included). Default: 1')
    ap.add_argument('--vec',              default='dec', choices=list(VEC_CHOICES))
    ap.add_argument('--method',           default='tsne', choices=['tsne', 'pca'])
    ap.add_argument('--sae-idx',          type=int, default=0)  # 0 = anchor sub-SAE
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--perplexity',       type=float, default=30.0)
    ap.add_argument('--pca-pre',          type=int,   default=50)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    raw_orbits  = parse_ids_file(Path(args.orbits)) if args.orbits else []
    feat_groups = parse_ids_file(Path(args.feature_ids))

    forth_group = _find_group(feat_groups, 'Forth')
    major_group = _find_group(feat_groups, 'Major Chord')
    minor_group = _find_group(feat_groups, 'Minor Chord')

    named_sets = {
        frozenset(g.node_ids)
        for g in (forth_group, major_group, minor_group)
        if g is not None
    }
    # other_orbits = [o for o in raw_orbits
    #                 if frozenset(o.node_ids) not in named_sets]

    print(f"[INFO] [Forth]       : {len(forth_group.node_ids) if forth_group else 0} nodes")
    print(f"[INFO] [Major Chord] : {len(major_group.node_ids) if major_group else 0} nodes")
    print(f"[INFO] [Minor Chord] : {len(minor_group.node_ids) if minor_group else 0} nodes")
    # print(f"[INFO] Other orbits  : {len(other_orbits)} orbit(s), "
    #       f"{sum(len(o.node_ids) for o in other_orbits)} nodes")

    module   = load_module(Path(args.ckpt_path), args.device, args.sae_idx)
    all_vecs = extract_all_vectors(module, args.vec, args.device)
    n_feat   = all_vecs.shape[0]
    print(f"[INFO] Total SAE features: {n_feat}")

    # Collect all group feature IDs — these are always included in t-SNE
    all_group_ids: Set[int] = set()
    for g in ([forth_group, major_group, minor_group]):
        if g is not None:
            all_group_ids.update(g.node_ids)

    if args.act_counts_file:
        counts = np.load(args.act_counts_file)
        if len(counts) != n_feat:
            raise ValueError(f"activation_counts length {len(counts)} != n_feat {n_feat}")
        active_mask = counts >= args.min_act_count
        for gid in all_group_ids:          # ring features bypass the threshold
            active_mask[gid] = True
        active_idx = np.where(active_mask)[0]
        n_active   = len(active_idx)
        print(f"[INFO] Activation filter: min_count={args.min_act_count}, "
              f"kept={n_active}/{n_feat} ({n_feat - n_active} excluded from t-SNE)")
        print(f"\n[INFO] Reducing {n_active} features → 2D ({args.method}) ...")
        coords_sub = reduce(all_vecs.numpy()[active_idx], args.method, n_components=2,
                            perplexity=args.perplexity, pca_pre_dims=args.pca_pre)
        coords = np.full((n_feat, 2), np.nan)
        coords[active_idx] = coords_sub
    else:
        print(f"\n[INFO] Reducing {n_feat} features → 2D ({args.method}) ...")
        coords = reduce(all_vecs.numpy(), args.method, n_components=2,
                        perplexity=args.perplexity, pca_pre_dims=args.pca_pre)

    out_dir = Path(args.out_dir)

    plot_all_features(
        coords       = coords,
        n_feat       = n_feat,
        forth_group  = forth_group,
        major_group  = major_group,
        minor_group  = minor_group,
        # other_orbits = other_orbits,
        out_path     = out_dir / 'features_all.png',
    )

    for group, color, name in (
        (forth_group,  C_FORTH, 'forth'),
        (major_group,  C_MAJOR, 'major'),
        (minor_group,  C_MINOR, 'minor'),
    ):
        if group is None:
            print(f"[WARN] No [{name.capitalize()}] group found; skipping ring plot.")
            continue
        plot_ring(
            coords   = coords,
            group    = group,
            color    = color,
            out_path = out_dir / f'features_{name}.png',
        )


if __name__ == '__main__':
    main()
