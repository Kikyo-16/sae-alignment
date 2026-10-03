#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step P: Use trained linear-probe checkpoints to find SAE encoder features for
musical rings, then write them to probe_feature_ids.txt.

Chord probe (``--chord-probe``) → ``[Major Chord]`` / ``[Minor Chord]``.
Mirrors ``ProbeChordMetric._select_chord_feature_ids``:
  - The chord probe ``linear.weight`` is (25, d_model): rows 0-11 = C..B major,
    rows 12-23 = C..B minor, row 24 = N (unused).
  - For each of the 24 maj/min rows, pick the SAE encoder feature (cosine
    similarity) with the highest score.
  - ring position p (pitch class p, root pc p) ↦ selected SAE feature id.
    No reordering — downstream chord recognition assumes ring[p] is the
    feature for a chord with root pitch-class p.

Key scale-degree probes (``--forth-probe`` / ``--fifth-probe`` /
``--tonic-probe``) → ``[Forth]`` / ``[Fifth]`` / ``[Tonic]``.
Mirrors ``ProbeKeyMetric._build_replaced_probe``:
  - Each key probe ``linear.weight`` is (12, d_model): row p = the scale-degree
    note at absolute pitch class p (tonic pc = (p - degree_semitones) % 12).
  - Row p's SAE encoder nearest neighbour becomes ring position p — the same
    convention downstream key inference expects for these ring groups.

At least one probe must be given; only lines for the given probes are written.

Usage:
    python -m src.analysis.step_probe_rings \\
        --ckpt-path     store/.../ckpts/last.ckpt \\
        --chord-probe   store/.../chord_probe.ckpt \\
        [--forth-probe  store/.../key_forth.ckpt] \\
        [--fifth-probe  store/.../key_fifth.ckpt] \\
        [--tonic-probe  store/.../key_tonic.ckpt] \\
        --out-file      store/.../analysis/probe_feature_ids.txt \\
        [--sae-idx 0] [--device cuda]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .find_chord_rings import format_ring_line, load_sae_and_factory
from ..linear_probe.probe_lit import LitLinearProbe


def find_chord_rings_from_probe(
    probe_ckpt: Path,
    enc_w_norm: torch.Tensor,   # (N_feat, d_model) — already L2-normalised
    device: str,
) -> Tuple[List[int], List[int]]:
    """Return ``(major_ids, minor_ids)``, each a length-12 list of SAE feature
    ids in pitch-class order (position p ↦ root pc p).

    probe_w rows 0-11 predict the 12 major chords (C..B), rows 12-23 the 12
    minor chords; best_feat[r] is the SAE encoder feature nearest probe_w[r].
    """
    probe: LitLinearProbe = LitLinearProbe.load_from_checkpoint(
        str(probe_ckpt), map_location=device, strict=False
    )
    probe.eval()
    probe_w = probe.linear.weight.data.to(device)   # (25, d_model)
    if probe_w.shape[0] < 24:
        raise ValueError(
            f"Expected a 25-class chord probe (>=24 maj/min rows), got "
            f"{probe_w.shape[0]} rows in {probe_ckpt.name}."
        )

    chord_w = F.normalize(probe_w[:24], dim=1)       # rows 0-23 (drop N row 24)
    cos_sim = chord_w @ enc_w_norm.T                 # (24, N_feat)
    best_feat = cos_sim.argmax(dim=1)                # (24,)
    best_scores = cos_sim[torch.arange(24, device=device), best_feat]

    major_ids = [int(best_feat[i].item()) for i in range(12)]
    minor_ids = [int(best_feat[i].item()) for i in range(12, 24)]
    maj_sc = [round(float(v), 4) for v in best_scores[:12].tolist()]
    min_sc = [round(float(v), 4) for v in best_scores[12:24].tolist()]
    print(f"[INFO] chord probe = {probe_ckpt.name}")
    print(f"       major SAE ids : {major_ids}")
    print(f"       major cos     : {maj_sc}")
    print(f"       minor SAE ids : {minor_ids}")
    print(f"       minor cos     : {min_sc}")
    return major_ids, minor_ids


def find_degree_ring_from_probe(
    probe_ckpt: Path,
    enc_w_norm: torch.Tensor,   # (N_feat, d_model) — already L2-normalised
    device: str,
    label: str,
) -> List[int]:
    """Return a length-12 list of SAE feature ids (position p ↦ degree-note pc p).

    Mirrors ``ProbeKeyMetric._build_replaced_probe``: the key probe
    ``linear.weight`` is (12, d_model), row p = scale-degree note at pitch
    class p; each row maps to its nearest SAE encoder feature (cosine).
    """
    probe: LitLinearProbe = LitLinearProbe.load_from_checkpoint(
        str(probe_ckpt), map_location=device, strict=False
    )
    probe.eval()
    probe_w = probe.linear.weight.data.to(device)   # (12, d_model)
    if probe_w.shape[0] != 12:
        raise ValueError(
            f"Expected a 12-class key probe for [{label}], got "
            f"{probe_w.shape[0]} rows in {probe_ckpt.name}."
        )

    key_w = F.normalize(probe_w, dim=1)
    cos_sim = key_w @ enc_w_norm.T                   # (12, N_feat)
    best_feat = cos_sim.argmax(dim=1)                # (12,)
    best_scores = cos_sim[torch.arange(12, device=device), best_feat]

    ids = [int(v) for v in best_feat.tolist()]
    sc = [round(float(v), 4) for v in best_scores.tolist()]
    print(f"[INFO] {label} probe = {probe_ckpt.name}")
    print(f"       SAE ids : {ids}")
    print(f"       cos     : {sc}")
    return ids


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Step P: linear-probe weights → SAE ring IDs → probe_feature_ids.txt"
    )
    ap.add_argument("--ckpt-path",   required=True,
                    help="LitSAE checkpoint (.ckpt)")
    ap.add_argument("--chord-probe", default=None,
                    help="LitLinearProbe chord checkpoint (25-class)")
    ap.add_argument("--forth-probe", default=None,
                    help="LitLinearProbe key forth-degree checkpoint (12-class)")
    ap.add_argument("--fifth-probe", default=None,
                    help="LitLinearProbe key fifth-degree checkpoint (12-class)")
    ap.add_argument("--tonic-probe", default=None,
                    help="LitLinearProbe key tonic-degree checkpoint (12-class)")
    ap.add_argument("--out-file",    required=True,
                    help="Output path for probe_feature_ids.txt (overwritten)")
    ap.add_argument("--sae-idx",     type=int, default=0)
    ap.add_argument("--device",      type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    degree_probes = [
        ("Forth", args.forth_probe),
        ("Fifth", args.fifth_probe),
        ("Tonic", args.tonic_probe),
    ]
    if args.chord_probe is None and not any(p for _, p in degree_probes):
        raise ValueError(
            "At least one probe is required "
            "(--chord-probe / --forth-probe / --fifth-probe / --tonic-probe)."
        )
    for label, p in [("Chord", args.chord_probe)] + degree_probes:
        if p is not None and not Path(p).exists():
            raise FileNotFoundError(f"{label} probe checkpoint not found: {p}")

    lit, sae, _ = load_sae_and_factory(Path(args.ckpt_path), args.device)
    module  = sae.sae
    n_sub   = int(getattr(module, "n_saes", 0))
    use_idx = max(0, min(args.sae_idx, n_sub - 1))
    print(f"[INFO] sae_idx={use_idx}  n_saes={n_sub}")

    sae_module = module.sae_modules[use_idx]
    enc_w_raw  = sae_module.fc1.weight.data.to(args.device)   # (N_feat, d_model)
    enc_w_norm = F.normalize(enc_w_raw, dim=1)                # pre-normalise once

    lines: List[str] = []
    if args.chord_probe is not None:
        major_ids, minor_ids = find_chord_rings_from_probe(
            Path(args.chord_probe), enc_w_norm, args.device
        )
        lines.append(format_ring_line("Major Chord", major_ids))
        lines.append(format_ring_line("Minor Chord", minor_ids))

    for label, p in degree_probes:
        if p is None:
            continue
        ids = find_degree_ring_from_probe(Path(p), enc_w_norm, args.device, label)
        lines.append(format_ring_line(label, ids))

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[DONE] probe_feature_ids.txt → {out_path}")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
