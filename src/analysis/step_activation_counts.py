#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Activation count statistics: for each SAE feature, count the number of audio
segments in which the feature appears in the top-k activations of at least one
frame.

Per-segment rule
    For a segment with T frames, each frame has exactly k non-zero entries in
    the sparse SAE activation (top-k selection).  The activated feature set for
    the segment is the union of those k-index sets across all T frames.
    activation_counts[f] = number of segments where feature f was in the union.

Outputs
    <out-dir>/activation_counts.npy   int64 array shape [D]
    <out-dir>/activation_counts.txt   TSV sorted by count descending

Usage:
    python -m src.analysis.step_activation_counts \
        --ckpt-path         store/.../ckpts/last.ckpt \
        --h5-path           sae-data/pop909/muq/pop909_muq_30s_layer_2.h5 \
        --split             train \
        --feature-ds        muq_layer \
        --out-dir           store/.../analysis \
        [--sae-idx 1] [--device cuda] [--batch-size 32]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from ..analysis.find_chord_rings import load_sae_and_factory, standardize_to_BTD_batch


def _run_sae(
    module,
    model_factory,
    use_tag: str,
    use_idx: int,
    x_np: np.ndarray,
    t_feat: int,
    device: str,
) -> Optional[np.ndarray]:
    """Return [B, T, D] float32 sparse activations, or None on failure."""
    x = torch.from_numpy(x_np).float().to(device)
    with torch.no_grad():
        feat_batch: Dict[str, torch.Tensor] = model_factory({use_tag: x}, t=0)
    if use_tag not in feat_batch:
        return None
    feat = feat_batch[use_tag]
    x_sae = standardize_to_BTD_batch(feat, t_candidates=[t_feat, int(x_np.shape[-1])])
    with torch.no_grad():
        z = module.inference(x_sae, idx=use_idx, topk_wide=1)
    z_np = z.detach().float().cpu().numpy()
    return z_np if z_np.ndim == 3 else None


def compute_activation_counts(
    module,
    model_factory,
    use_tag: str,
    use_idx: int,
    h5_path: Path,
    split: str,
    feature_ds: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Returns int64 array of shape [D] where counts[f] = number of segments in
    which feature f was activated (top-k) in at least one frame.
    """
    counts: Optional[np.ndarray] = None
    n_segments = 0

    with h5py.File(h5_path, "r") as f:
        g = f[split]
        ds_feat = g[feature_ds]
        n_total = int(ds_feat.shape[0])
        t_feat = int(ds_feat.shape[-1])

        bs = max(1, batch_size)
        for b0 in tqdm(range(0, n_total, bs), desc="Activation counts", unit="batch"):
            idxs = list(range(b0, min(b0 + bs, n_total)))
            x_np = np.stack([np.array(ds_feat[i], copy=True) for i in idxs])
            z_batch = _run_sae(module, model_factory, use_tag, use_idx,
                               x_np, t_feat, device)
            if z_batch is None:
                continue

            if counts is None:
                D = z_batch.shape[2]
                counts = np.zeros(D, dtype=np.int64)

            # z_batch: [B, T, D]; non-zero positions are the top-k activated features.
            # Union across frames per segment: any(z[b, :, f] != 0) over T.
            activated = (z_batch != 0).any(axis=1)  # [B, D] bool
            counts += activated.sum(axis=0).astype(np.int64)
            n_segments += len(idxs)

    if counts is None:
        raise RuntimeError("No segments processed — check h5_path and split.")

    print(f"[INFO] Processed {n_segments} segments, D={counts.shape[0]}")
    print(f"[INFO] Features activated in >0 segments: {int((counts > 0).sum())}")
    return counts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute per-feature activation counts across a dataset."
    )
    ap.add_argument("--ckpt-path",   required=True)
    ap.add_argument("--h5-path",     required=True)
    ap.add_argument("--split",       default="train")
    ap.add_argument("--feature-ds",  required=True)
    ap.add_argument("--out-dir",     required=True,
                    help="Directory to write activation_counts.npy and .txt")
    ap.add_argument("--sae-idx",     type=int, default=0)  # 0 = anchor sub-SAE
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",  type=int, default=32)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    lit, sae, model_factory = load_sae_and_factory(Path(args.ckpt_path), args.device)
    module = sae.sae
    use_tag = str(lit.data_tag)
    n_sub = int(getattr(module, "n_saes", 0))
    use_idx = max(0, min(args.sae_idx, n_sub - 1))
    print(f"[INFO] use_tag={use_tag}  sae_idx={use_idx}  split={args.split}")

    counts = compute_activation_counts(
        module, model_factory, use_tag, use_idx,
        Path(args.h5_path), args.split, args.feature_ds, args.device,
        args.batch_size,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "activation_counts.npy"
    np.save(npy_path, counts)
    print(f"[DONE] Saved {npy_path}")

    txt_path = out_dir / "activation_counts.txt"
    order = np.argsort(-counts)
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("feature_id\tactivation_count\n")
        for fid in order:
            fh.write(f"{int(fid)}\t{int(counts[fid])}\n")
    print(f"[DONE] Saved {txt_path}")


if __name__ == "__main__":
    main()
