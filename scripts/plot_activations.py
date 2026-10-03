#!/usr/bin/env python3
"""
Plot SAE feature activations of audio files, grouped by orbit.

One heatmap per 30 s segment: x = time, y = feature ids, one block per line of
the --feature-ids file (red lines separate blocks). No labels are drawn.
Plotting follows src/analysis/vis_activation.py (same SAE inference, same
block-wise normalisation, same heatmap style).

--feature-ids accepts orbits_raw.txt from scripts/find_orbits.py directly, or
any file with one group per line, e.g. "[Ring 03]: 12 a -> b -> ..." or
"[Major Chord]: 12 a -> b -> ...". Lines starting with '#' are ignored.

Outputs: <out-dir>/<audio stem>/<stem>_<segment index>_<start>s.png

Example:
    python scripts/plot_activations.py --ckpt o-sae.ckpt \\
        --feature-ids results/muq-2/orbits_raw.txt --min-orbit-size 10 \\
        --audio song.mp3 --out-dir plots
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import List

from _common import collect_audio, default_device

import numpy as np
import torch

from src.analysis.vis_activation import (
    load_sae_and_factory,
    make_heatmap_time_x,
    parse_feature_id_groups_file,
    standardize_to_BTD,
)
from src.inference.audio import LAST_SEGMENT_CHOICES, SEGMENT_SEC, FeatureExtractor, extract_segments
from src.inference.runners import parse_data_tag


def load_groups(path: Path, min_size: int) -> List[List[int]]:
    """Feature-id groups, one per non-comment line (vis_activation parsing rules)."""
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "groups.txt"
        clean.write_text("\n".join(lines) + "\n", encoding="utf-8")
        groups = parse_feature_id_groups_file(clean)
    return [g for g in groups if len(g) >= min_size]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="LitSAE checkpoint (.ckpt).")
    ap.add_argument("--feature-ids", required=True,
                    help="orbits_raw.txt or feature_ids.txt: one feature-id group per line.")
    ap.add_argument("--audio", required=True, nargs="+", help="Audio files and/or directories.")
    ap.add_argument("--out-dir", default="plots", help="Output directory. Default: plots.")
    ap.add_argument("--min-orbit-size", type=int, default=1,
                    help="Only plot groups with at least this many features "
                         "(e.g. 10 to keep long orbits from orbits_raw.txt). Default: 1.")
    ap.add_argument("--sae-idx", type=int, default=0, help="Sub-SAE to decode with. Default: 0.")
    ap.add_argument("--last-segment", default="drop", choices=LAST_SEGMENT_CHOICES,
                    help="Trailing audio shorter than 30 s: drop it (default) or plot one more "
                         "30 s window ending at the last sample.")
    ap.add_argument("--feature-cell-height", type=float, default=1.0)
    ap.add_argument("--model-id", default=None, help="Override the Hugging Face id of the foundation model.")
    ap.add_argument("--device", default=default_device())
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    groups = load_groups(Path(args.feature_ids), args.min_orbit_size)
    if not groups:
        raise SystemExit(f"No feature-id groups with >= {args.min_orbit_size} features in {args.feature_ids}.")
    print(f"[INFO] plotting {len(groups)} group(s), {sum(map(len, groups))} features")

    lit, sae, model_factory, norm_fn = load_sae_and_factory(Path(args.ckpt).resolve(), device=args.device)
    module = sae.sae
    use_tag = str(lit.data_tag)
    n_sub = int(getattr(module, "n_saes", 0))
    if not (0 <= args.sae_idx < n_sub):
        raise ValueError(f"--sae-idx must be in [0, {n_sub - 1}], got {args.sae_idx}")
    extractor = FeatureExtractor(*parse_data_tag(use_tag), args.device, args.model_id)

    for audio in collect_audio(args.audio):
        segments = extract_segments(audio, extractor, args.last_segment)
        if not segments:
            print(f"[SKIP] {audio}: shorter than one 30 s segment.")
            continue
        out_dir = Path(args.out_dir) / audio.stem
        for si, seg in enumerate(segments):
            x_np = seg.feat
            x = torch.from_numpy(np.array(x_np, copy=True)).float().to(args.device).unsqueeze(0)
            with torch.no_grad():
                feature_batch = norm_fn(model_factory({use_tag: x}, t=0))
                x_for_sae = standardize_to_BTD(feature_batch[use_tag], t_candidates=[x_np.shape[-1]])
                z = module.inference(x_for_sae, idx=args.sae_idx, topk_wide=0).squeeze(0)
            t_z, d_z = z.shape

            cols_grouped = [[f for f in g if 0 <= f < d_z] for g in groups]
            cols_grouped = [c for c in cols_grouped if c]
            group_sizes = [len(c) for c in cols_grouped]
            all_cols = [f for c in cols_grouped for f in c]
            mat_feat = z[:, all_cols].transpose(0, 1).detach().cpu().numpy().astype(np.float32)

            # Block-wise normalisation with a global floor (as in vis_activation.py).
            global_max = float(mat_feat.max()) if mat_feat.size else 0.0
            min_block_scale = 0.1 * global_max
            offset = 0
            for gsz in group_sizes:
                block = mat_feat[offset:offset + gsz, :]
                scale = max(float(block.max()) if block.size else 0.0, min_block_scale)
                if scale > 0:
                    block /= scale
                offset += gsz

            out_path = out_dir / f"{audio.stem}_{si:03d}_{seg.start_sec:.0f}s.png"
            make_heatmap_time_x(
                out_path=out_path,
                mat_feat=mat_feat,
                feat_labels=[str(f) for f in all_cols],
                feat_group_sizes=group_sizes,
                pitch_vals=np.zeros((128, t_z), dtype=np.float32),
                melody_vals=None,
                x_label_frame=[""] * t_z,
                chord_spans=[(0, t_z)],
                seg_start_sec=seg.start_sec,
                seg_duration_sec=SEGMENT_SEC,
                title="",
                all_midi_span=0,
                melody_span=0,
                feature_cell_height=args.feature_cell_height,
                pianoroll_cell_height=1.0,
            )
            print(f"[{audio.stem}] segment {si} ({seg.start_sec:.1f}s) -> {out_path}")


if __name__ == "__main__":
    main()
