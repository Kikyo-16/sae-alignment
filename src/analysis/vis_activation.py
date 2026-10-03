#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise SAE feature activations over time as stacked heatmaps.

Version: 8.0
Changes in v8.0:
  - Add independent relative cell-height controls for SAE feature rows and
    piano-roll rows via --feature-cell-height and --pianoroll-cell-height.

Changes in v7.0:
  - Add --hide-midi-reference to hide both the all-MIDI and melody blocks.

Changes in v6.0:
  - Omit the all-MIDI or melody block when the corresponding MIDI reference
    contains no active pitches; fixed-span rows are used only when reference
    activity is present.

Earlier changes:
  - Use fixed-height, activity-localized pitch windows for the all-MIDI and
    melody blocks. Their spans are configurable with --all-midi-span and
    --melody-span (defaults: 72 and 36 semitone rows).
  - GTZAN: plot at most five songs per genre by default.
  - Feature blocks: divide by max(block_peak, 0.1 * global_feature_peak).
  - Heatmap: use the ``cividis`` colour map with range [0, 1].
  - SAE activations: call MultiSAE.inference(), which computes
    ReLU(encoder(x - shared_pre_bias)).

Inputs : a LitSAE checkpoint, an H5 feature dataset, and a feature_ids.txt
         (the [Ring]/[Major Chord]/... groups produced by the ring pipeline).
Output : one PNG per segment (first --max-plots segments of the split).

Layout (same as analysis_archive/infer_vis.py):
  rows  = a fixed-span all-MIDI window → a fixed-span melody window →
          grouped feature activations (one block per feature_ids group, blocks
          separated by red horizontal lines)
  cols  = feature time frames
  bottom x-axis = chord/key label at each chord-segment centre
  top   x-axis = time in seconds
  red vertical lines mark chord-segment boundaries.

Feature activations come from module.inference(x, idx, topk_wide=0), which
subtracts the shared SAE pre_bias before applying the selected encoder and
ReLU. Each feature block is normalised by max(block_peak, 0.1 * global_peak).

Usage:
    python -m src.analysis.vis_activation \\
        --ckpt-path        store/.../ckpts/last.ckpt \\
        --h5-path          sae-data/pop909/muq/pop909_muq_30s_layer_2.h5 \\
        --split            test \\
        --feature-id-file  store/.../epoch{N}_feature_ids.txt \\
        --feature-ds       muq_layer \\
        --output-dir       store/.../vis \\
        [--sae-idx 0] [--max-plots 30] [--all-midi-span 72]
        [--melody-span 36] [--hide-midi-reference]
        [--feature-cell-height 1.0] [--pianoroll-cell-height 1.0]
        [--device cuda] [--allow-missing-labels]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..music_sae.sae_lit import LitSAE


SCRIPT_VERSION = "8.0"


# ── string / note helpers ────────────────────────────────────────────────────

def decode_str_scalar(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def decode_str_array(a: np.ndarray) -> np.ndarray:
    return np.array([decode_str_scalar(v) for v in a], dtype=object)


def midi_to_note_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if midi < 0 or midi > 127:
        return f"midi{midi}"
    return f"{names[midi % 12]}{(midi // 12) - 1}"


NOTE2PC = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "FB": 4, "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7,
    "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11,
}


def chord_root_pc(chord: str) -> Optional[int]:
    c = chord.strip()
    if not c or c.upper() in {"N", "NOCHORD"}:
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


def to_ascii_for_plot(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


# ── feature-id file ──────────────────────────────────────────────────────────

def parse_feature_id_groups_file(path: Path) -> List[List[int]]:
    groups: List[List[int]] = []
    int_pat = re.compile(r"-?\d+")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                line = line.split(":", 1)[1]
            nums = [int(x) for x in int_pat.findall(line)]
            if not nums:
                continue
            if len(nums) >= 2:
                maybe_len, rest = nums[0], nums[1:]
                nums_use = rest if maybe_len == len(rest) else nums
            else:
                nums_use = nums
            seen: Set[int] = set()
            g: List[int] = []
            for fid in nums_use:
                if fid not in seen:
                    seen.add(fid)
                    g.append(fid)
            if g:
                groups.append(g)
    print(f"[INFO] Parsed {len(groups)} feature-id groups from {path}")
    return groups


# ── model loading / feature standardisation ──────────────────────────────────

def load_sae_and_factory(ckpt_path: Path, device: str = "cpu"):
    ckpt_path = ckpt_path.resolve()
    print(f"[INFO] Loading LitSAE from {ckpt_path}")
    lit: "LitSAE" = LitSAE.load_from_checkpoint(str(ckpt_path), map_location=device, strict=False)
    lit.eval().to(device)
    if hasattr(lit, "model_factory"):
        lit.model_factory.to(device)
    sae = lit.sae
    sae.eval().to(device)
    if not hasattr(sae, "sae"):
        raise RuntimeError("Expected music_sae GroupSAE to have attribute `sae` (MultiSAE).")
    print(f"[INFO] SAE data_tag = {lit.data_tag}")
    return lit, sae, lit.model_factory, lit._normalize_feature_batch


def standardize_to_BTD(feat: torch.Tensor, t_candidates: List[int]) -> torch.Tensor:
    feat = feat.squeeze(1)
    if feat.ndim == 3:
        b, d1, d2 = feat.shape
        if b != 1:
            raise RuntimeError(f"Expect batch size 1, got {feat.shape}")
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


# ── label helpers ────────────────────────────────────────────────────────────

def compress_label_frame(label_frame: np.ndarray) -> Tuple[List[str], List[Tuple[int, int]]]:
    t = label_frame.shape[0]
    labels: List[str] = []
    spans: List[Tuple[int, int]] = []
    if t == 0:
        return labels, spans
    cur = str(label_frame[0])
    s = 0
    for i in range(1, t):
        c = str(label_frame[i])
        if c != cur:
            labels.append(cur)
            spans.append((s, i))
            cur = c
            s = i
    labels.append(cur)
    spans.append((s, t))
    return labels, spans


def _resample_labels_to_T(
    chord_frame: np.ndarray,
    key_frame: Optional[np.ndarray],
    pitch_roll: np.ndarray,
    melody_roll: Optional[np.ndarray],
    target_t: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    if target_t <= 0:
        key0 = None if key_frame is None else key_frame[:0]
        mel0 = None if melody_roll is None else melody_roll[:, :0]
        return chord_frame[:0], key0, pitch_roll[:, :0], mel0
    t_lbl = int(chord_frame.shape[0])
    if t_lbl == target_t:
        return chord_frame, key_frame, pitch_roll, melody_roll
    if t_lbl <= 0:
        chord = np.array(["N"] * target_t, dtype=object)
        key = None if key_frame is None else np.array(["N"] * target_t, dtype=object)
        pitch = np.zeros((128, target_t), dtype=np.uint8)
        mel = None if melody_roll is None else np.zeros((128, target_t), dtype=np.uint8)
        return chord, key, pitch, mel
    src_idx = np.linspace(0, t_lbl - 1, num=target_t)
    src_idx = np.clip(np.rint(src_idx).astype(np.int64), 0, t_lbl - 1)
    chord_rs = chord_frame[src_idx]
    key_rs = None if key_frame is None else key_frame[src_idx]
    pitch_rs = pitch_roll[:, src_idx]
    melody_rs = None if melody_roll is None else melody_roll[:, src_idx]
    return chord_rs, key_rs, pitch_rs, melody_rs


def crop_pitch_to_fixed_span(
    pitch_vals: np.ndarray,
    span: int,
    *,
    default_center: int,
) -> Tuple[np.ndarray, List[int]]:
    """Select an activity-localized contiguous MIDI window of fixed size.

    The returned block always has ``span`` rows (unless span=0). The window
    location adapts to the active pitches, but it is not aligned to octave
    boundaries. If the active range is wider than the requested span, choose
    the window containing the largest amount of pitch activity.
    """
    assert pitch_vals.ndim == 2 and pitch_vals.shape[0] == 128
    span = int(span)
    if span < 0 or span > 128:
        raise ValueError(f"pitch span must be in [0, 128], got {span}")
    if span == 0:
        return pitch_vals[:0, :], []

    activity = (pitch_vals != 0).sum(axis=1).astype(np.float64)
    active = np.flatnonzero(activity > 0)
    max_start = 128 - span

    if active.size == 0:
        center = float(default_center)
        start = int(round(center - (span - 1) / 2.0))
    else:
        lo, hi = int(active[0]), int(active[-1])
        active_width = hi - lo + 1
        weighted_center = float(
            np.dot(np.arange(128, dtype=np.float64), activity)
            / max(activity.sum(), 1.0)
        )

        if active_width <= span:
            # Include the entire active range and distribute the remaining
            # context approximately evenly above and below it.
            center = 0.5 * (lo + hi)
            start = int(round(center - (span - 1) / 2.0))
        else:
            # The active range is wider than the display. Keep the contiguous
            # span with the greatest total activity; break ties by proximity
            # to the activity-weighted pitch centre.
            scores = np.convolve(activity, np.ones(span, dtype=np.float64), mode="valid")
            candidates = np.flatnonzero(np.isclose(scores, scores.max()))
            start = int(
                min(
                    candidates.tolist(),
                    key=lambda s: abs((s + (span - 1) / 2.0) - weighted_center),
                )
            )

    start = int(np.clip(start, 0, max_start))
    keep_idx = list(range(start, start + span))
    return pitch_vals[start:start + span, :], keep_idx


GTZAN_GENRES = {
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
}


def _get_track_and_start(g: h5py.Group, i: int) -> Tuple[str, float]:
    if "trackId" in g:
        track = decode_str_scalar(g["trackId"][i])
    elif "labels" in g and "song_id" in g["labels"]:
        track = decode_str_scalar(g["labels"]["song_id"][i])
    else:
        track = f"seg{i:06d}"
    if "start" in g:
        start = float(g["start"][i])
    elif "labels" in g and "seg_start_sec" in g["labels"]:
        start = float(g["labels"]["seg_start_sec"][i])
    else:
        start = 0.0
    return track, start


def _get_gtzan_genre(track_id: str) -> Optional[str]:
    """Return the GTZAN genre encoded in a track id, if present."""
    track = track_id.replace("\\", "/").strip().lower()
    first = track.split("/", 1)[0]
    if first in GTZAN_GENRES:
        return first
    prefix = Path(track).name.split(".", 1)[0]
    return prefix if prefix in GTZAN_GENRES else None


def _get_labels(
    g: h5py.Group,
    i: int,
    t_hint: int,
    *,
    allow_missing_labels: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    if "labels" in g:
        lbl = g["labels"]
        has_chord = "chord_frame" in lbl
        has_pitch = "pitch_roll" in lbl
        if has_chord or has_pitch:
            chord = decode_str_array(lbl["chord_frame"][i]) if has_chord else np.array(["N"] * t_hint, dtype=object)
            key = decode_str_array(lbl["key_frame"][i]) if "key_frame" in lbl else np.array(["N"] * t_hint, dtype=object)
            pitch = np.array(lbl["pitch_roll"][i], copy=True) if has_pitch else np.zeros((128, t_hint), dtype=np.uint8)
            melody = np.array(lbl["melody_roll"][i], copy=True) if "melody_roll" in lbl else None
            return chord, key, pitch, melody
    if not allow_missing_labels:
        raise RuntimeError(
            "Labels are missing in H5. Expected '/<split>/labels/chord_frame' or "
            "'/<split>/labels/pitch_roll'. Pass '--allow-missing-labels' to accept fallback."
        )
    chord = np.array(["N"] * t_hint, dtype=object)
    key = np.array(["N"] * t_hint, dtype=object)
    pitch = np.zeros((128, t_hint), dtype=np.uint8)
    return chord, key, pitch, None


def _get_segment_duration_sec(g: h5py.Group, i: int, fallback: float = 0.0) -> float:
    if "start" in g and "end" in g:
        dur = float(g["end"][i]) - float(g["start"][i])
        if dur > 0:
            return dur
    if "labels" in g and "segment_sec" in g["labels"].attrs:
        dur = float(g["labels"].attrs["segment_sec"])
        if dur > 0:
            return dur
    return float(fallback)


# ── plotting (layout identical to infer_vis.make_heatmap_time_x) ──────────────

def make_heatmap_time_x(
    out_path: Path,
    mat_feat: np.ndarray,
    feat_labels: List[str],
    feat_group_sizes: List[int],
    pitch_vals: np.ndarray,
    melody_vals: Optional[np.ndarray],
    x_label_frame: List[str],
    chord_spans: List[Tuple[int, int]],
    seg_start_sec: float,
    seg_duration_sec: float,
    title: str,
    all_midi_span: int,
    melody_span: int,
    feature_cell_height: float,
    pianoroll_cell_height: float,
):
    n_feat_total, t = mat_feat.shape
    assert pitch_vals.shape == (128, t)
    assert len(x_label_frame) == t

    # Keep a fixed display height only when the corresponding MIDI
    # reference contains activity. Missing or all-zero references should not
    # create a large empty block in the heatmap.
    if all_midi_span > 0 and np.any(pitch_vals != 0):
        pitch_vals_kept, kept_midi = crop_pitch_to_fixed_span(
            pitch_vals, all_midi_span, default_center=60
        )
    else:
        pitch_vals_kept = np.zeros((0, t), dtype=np.float32)
        kept_midi = []
    n_pitch = int(pitch_vals_kept.shape[0])

    if melody_span > 0 and melody_vals is not None and np.any(melody_vals != 0):
        melody_vals_kept, kept_midi_mel = crop_pitch_to_fixed_span(
            melody_vals, melody_span, default_center=72
        )
    else:
        melody_vals_kept = np.zeros((0, t), dtype=np.float32)
        kept_midi_mel = []
    n_melody = int(melody_vals_kept.shape[0])

    mat = np.concatenate([pitch_vals_kept, melody_vals_kept, mat_feat], axis=0)
    h = n_pitch + n_melody + n_feat_total

    # imshow enforces equal row heights. Use non-uniform y coordinates so
    # piano-roll and SAE-feature cells can have independent relative heights.
    row_heights = np.concatenate([
        np.full(n_pitch + n_melody, pianoroll_cell_height, dtype=np.float64),
        np.full(n_feat_total, feature_cell_height, dtype=np.float64),
    ])
    y_edges = np.concatenate([[0.0], np.cumsum(row_heights)])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    total_display_height = float(y_edges[-1])

    fig_h = max(6.0, 0.055 * total_display_height)
    fig_w = max(10.0, 0.012 * t)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    x_edges = np.arange(t + 1, dtype=np.float64) - 0.5
    im = ax.pcolormesh(
        x_edges, y_edges, mat, shading="flat",
        cmap="YlGnBu_r", vmin=0.0, vmax=1.0,
    )
    ax.set_xlim(-0.5, t - 0.5)
    ax.set_ylim(0.0, total_display_height)
    # ax.set_title(title, fontsize=10)

    seg_centers: List[int] = []
    seg_labels: List[str] = []
    for s, e in chord_spans:
        if e <= s:
            continue
        seg_centers.append((s + e - 1) // 2)
        seg_labels.append(str(x_label_frame[s]))
    # ax.set_xlabel("Label per segment center")
    ax.set_xticks(seg_centers)
    ax.set_xticklabels(seg_labels, rotation=90, fontsize=12)

    if seg_duration_sec <= 0:
        seg_duration_sec = float(max(1, t - 1))
    sec_marks = np.arange(0, int(np.floor(seg_duration_sec)) + 1, 1, dtype=np.int64)
    if sec_marks.size == 0:
        sec_marks = np.array([0], dtype=np.int64)
    frame_pos = np.clip(
        np.rint((sec_marks.astype(np.float64) / max(seg_duration_sec, 1e-8)) * (t - 1)).astype(np.int64),
        0, max(0, t - 1),
    )
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_xticks(frame_pos.tolist())
    ax_top.set_xticklabels([f"{seg_start_sec + float(s):.0f}s" for s in sec_marks], fontsize=7)

    pitch_labels = [midi_to_note_name(i) for i in kept_midi]
    melody_labels = [f"mel:{midi_to_note_name(i)}" for i in kept_midi_mel]
    y_labels = pitch_labels + melody_labels + feat_labels
    ax.set_yticks(y_centers)
    ax.set_yticklabels(y_labels)
    for yi, lab in enumerate(ax.get_yticklabels()):
        lab.set_fontsize(3 if yi < (n_pitch + n_melody) else 5)

    ax.set_yticks(y_edges, minor=True)
    ax.grid(which="minor", axis="y", linewidth=0.25)
    ax.tick_params(which="minor", left=False)

    if n_pitch > 0:
        ax.hlines(y_edges[n_pitch], -0.5, t - 0.5, linewidth=2, color="r")
    if n_melody > 0:
        ax.hlines(y_edges[n_pitch + n_melody], -0.5, t - 0.5, linewidth=2, color="r")

    base = n_pitch + n_melody
    cum = 0
    for gsz in feat_group_sizes[:-1]:
        cum += int(gsz)
        ax.hlines(y_edges[base + cum], -0.5, t - 0.5, linewidth=2, color="r")

    for s, _e in chord_spans[1:]:
        ax.vlines(s - 0.5, 0.0, total_display_height, linewidth=1.2, color="r")

    # cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    # cbar.ax.tick_params(labelsize=7)
    # cbar.set_label("Activation", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser("Visualise SAE feature activations as time-frame heatmaps.")
    ap.add_argument("--ckpt-path", type=str, required=True)
    ap.add_argument("--h5-path", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--feature-id-file", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-tag", type=str, default=None)
    ap.add_argument("--use-tag", type=str, default=None)
    ap.add_argument("--feature-ds", type=str, default="mel")
    ap.add_argument("--sae-idx", type=int, default=0)
    ap.add_argument("--max-plots", type=int, default=30,
                    help="Maximum total number of plots to write.")
    ap.add_argument("--max-per-genre", type=int, default=5,
                    help="For GTZAN, maximum number of plotted songs per genre. Default: 5")
    ap.add_argument("--all-midi-span", "--midi-span", dest="all_midi_span",
                    type=int, default=72,
                    help="Fixed height of the all-MIDI block in semitone rows; 0 hides it. Default: 72 (6 octaves).")
    ap.add_argument("--melody-span", type=int, default=36,
                    help="Fixed height of the melody block in semitone rows; 0 hides it. Default: 36 (3 octaves).")
    ap.add_argument("--hide-midi-reference", "--no-midi-reference",
                    dest="hide_midi_reference", action="store_true",
                    help="Hide both the all-MIDI and melody blocks.")
    ap.add_argument("--feature-cell-height", type=float, default=1.0,
                    help="Relative height of each SAE-feature row. Default: 1.0")
    ap.add_argument("--pianoroll-cell-height", "--midi-cell-height",
                    dest="pianoroll_cell_height", type=float, default=1.0,
                    help="Relative height of each all-MIDI or melody row. Default: 1.0")
    ap.add_argument("--allow-missing-labels", action="store_true",
                    help="Fall back to all-'N' chord and zero pitch-roll when labels are absent.")
    ap.add_argument("--only-labeled", action="store_true", help="Skip segments with no pitch activity.")
    ap.add_argument("--clamp-positive", action="store_true")
    ap.add_argument("--x-label-source", type=str, default="chord+key",
                    choices=["chord", "key", "chord+key"])
    return ap.parse_args()


def _build_x_labels(source: str, chord_frame: np.ndarray, key_frame: Optional[np.ndarray]) -> np.ndarray:
    key_ref = key_frame if key_frame is not None else np.array(["N"] * chord_frame.shape[0], dtype=object)
    if source == "chord":
        return chord_frame
    if source == "key":
        return key_ref
    return np.array([f"{str(c)}|{str(k)}" for c, k in zip(chord_frame.tolist(), key_ref.tolist())], dtype=object)


def main():
    args = parse_args()
    print(f"[INFO] vis_activation.py version {SCRIPT_VERSION}")
    if args.hide_midi_reference:
        args.all_midi_span = 0
        args.melody_span = 0
        print("[INFO] MIDI-reference visualization disabled.")
    for name, value in (("all-midi-span", args.all_midi_span), ("melody-span", args.melody_span)):
        if not (0 <= value <= 128):
            raise ValueError(f"--{name} must be in [0, 128], got {value}")
    for name, value in (
        ("feature-cell-height", args.feature_cell_height),
        ("pianoroll-cell-height", args.pianoroll_cell_height),
    ):
        if value <= 0:
            raise ValueError(f"--{name} must be > 0, got {value}")
    print(
        f"[INFO] fixed pitch spans: all MIDI={args.all_midi_span} rows, "
        f"melody={args.melody_span} rows"
    )
    print(
        f"[INFO] relative cell heights: feature={args.feature_cell_height}, "
        f"pianoroll={args.pianoroll_cell_height}"
    )

    h5_path = Path(args.h5_path).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    out_dir = Path(args.output_dir).resolve()
    feat_id_file = Path(args.feature_id_file).resolve()

    feat_groups = parse_feature_id_groups_file(feat_id_file)
    if not feat_groups:
        raise RuntimeError("No feature ids parsed from feature-id-file.")

    lit, sae, model_factory, norm_fn = load_sae_and_factory(ckpt_path, device=args.device)
    module = sae.sae  # GroupSAE -> MultiSAE

    use_tag = args.data_tag or args.use_tag or str(lit.data_tag)
    if not use_tag:
        raise RuntimeError("Cannot resolve data tag. Pass --data-tag explicitly.")

    n_sub_saes = int(getattr(module, "n_saes", 0))
    if n_sub_saes <= 0:
        raise RuntimeError("Invalid MultiSAE state: n_saes must be > 0.")
    use_idx = int(args.sae_idx)
    if not (0 <= use_idx < n_sub_saes):
        raise ValueError(f"`sae-idx` out of range: {use_idx}, expected [0, {n_sub_saes - 1}]")

    print(f"[INFO] use_tag={use_tag}, sae_idx={use_idx}, split={args.split}")

    dumped = 0
    is_gtzan = "gtzan" in str(h5_path).lower()
    genre_dumped: Dict[str, int] = {}
    if is_gtzan:
        print(f"[INFO] GTZAN detected: at most {args.max_per_genre} plots per genre.")

    with h5py.File(h5_path, "r") as f:
        if args.split not in f:
            raise KeyError(f"Split '{args.split}' not found. Available: {list(f.keys())}")
        g = f[args.split]
        if args.feature_ds not in g:
            raise KeyError(f"Feature dataset '{args.feature_ds}' not in split '{args.split}'.")
        ds_feat = g[args.feature_ds]
        n_total = int(ds_feat.shape[0])
        t_feat = int(ds_feat.shape[-1])
        print(f"[INFO] {n_total} segments, feature T={t_feat}; plotting up to {args.max_plots}.")

        for seg_idx in range(n_total):
            if dumped >= args.max_plots:
                break

            track_id, seg_start = _get_track_and_start(g, seg_idx)
            genre = _get_gtzan_genre(track_id) if is_gtzan else None
            if genre is not None and genre_dumped.get(genre, 0) >= args.max_per_genre:
                continue

            x_np = np.array(ds_feat[seg_idx], copy=True)
            x = torch.from_numpy(x_np).float().to(args.device).unsqueeze(0)
            with torch.no_grad():
                feature_batch: Dict[str, torch.Tensor] = model_factory({use_tag: x}, t=0)
                feature_batch = norm_fn(feature_batch)
            if use_tag not in feature_batch:
                print(f"[WARN] seg={seg_idx}: use_tag not in feature_batch, skip")
                continue

            feat = feature_batch[use_tag]
            x_for_sae = standardize_to_BTD(feat, t_candidates=[t_feat, x_np.shape[-1]])
            with torch.no_grad():
                activation_z = module.inference(x_for_sae, idx=use_idx, topk_wide=0)
            z = activation_z.squeeze(0)
            if args.clamp_positive:
                z = torch.clamp(z, min=0.0)

            chord_frame, key_frame, pitch_roll, melody_roll = _get_labels(
                g, seg_idx, t_hint=t_feat, allow_missing_labels=args.allow_missing_labels,
            )
            if args.only_labeled and pitch_roll.max() == 0:
                continue

            t_z, d_z = z.shape
            target_t = min(t_feat, t_z)
            if target_t <= 0:
                continue
            if chord_frame.shape[0] != target_t:
                chord_frame, key_frame, pitch_roll, melody_roll = _resample_labels_to_T(
                    chord_frame, key_frame, pitch_roll, melody_roll, target_t
                )

            x_label_frame = _build_x_labels(args.x_label_source, chord_frame, key_frame)
            _comp, spans = compress_label_frame(x_label_frame)
            if chord_frame.shape[0] == 0 or not spans:
                continue

            t_common = target_t
            z = z[:t_common]
            pitch_roll = pitch_roll[:, :t_common]
            if melody_roll is not None:
                melody_roll = melody_roll[:, :t_common]
            chord_frame_use = chord_frame[:t_common]
            x_label_frame_use = x_label_frame[:t_common]
            key_frame_use = None if key_frame is None else key_frame[:t_common]

            spans = [(max(0, s), min(t_common, e)) for s, e in spans]
            spans = [(s, e) for s, e in spans if e > s]
            if not spans:
                continue

            # Group feature columns by the feature_ids.txt groups.
            feat_cols_grouped: List[List[int]] = []
            group_sizes: List[int] = []
            for grp_ids in feat_groups:
                cols = [fid for fid in grp_ids if 0 <= fid < d_z]
                if not cols:
                    continue
                feat_cols_grouped.append(cols)
                group_sizes.append(len(cols))
            if not feat_cols_grouped:
                continue

            all_cols = [fid for grp in feat_cols_grouped for fid in grp]
            all_labels = [str(fid) for fid in all_cols]
            z_sel = z[:, all_cols]
            mat_feat = z_sel.transpose(0, 1).detach().cpu().numpy().astype(np.float32)

            # Block-wise normalisation with a global floor.  Each block uses
            # its own peak unless that peak is below 10% of the global peak,
            # in which case 0.1 * global_peak is used as the denominator.
            global_max = float(mat_feat.max()) if mat_feat.size else 0.0
            min_block_scale = 0.1 * global_max
            offset = 0
            for gsz in group_sizes:
                block = mat_feat[offset:offset + gsz, :]
                block_max = float(block.max()) if block.size else 0.0
                scale = max(block_max, min_block_scale)
                if scale > 0:
                    block /= scale
                offset += gsz

            pitch_active = (pitch_roll > 0).astype(np.uint8)
            maxval = 1.0
            val_root, val_other = 0.8 * maxval, 0.5 * maxval
            pitch_vals = np.zeros((128, t_common), dtype=np.float32)
            for ti in range(t_common):
                root_pc = chord_root_pc(str(chord_frame_use[ti]))
                if root_pc is None and key_frame_use is not None:
                    root_pc = chord_root_pc(str(key_frame_use[ti]))
                act_pitches = np.where(pitch_active[:, ti] > 0)[0]
                if act_pitches.size == 0:
                    continue
                if root_pc is None:
                    pitch_vals[act_pitches, ti] = val_other
                else:
                    for p in act_pitches:
                        pitch_vals[p, ti] = val_root if (p % 12) == root_pc else val_other
            melody_vals = (melody_roll > 0).astype(np.float32) * (0.95 * maxval) if melody_roll is not None else None

            seg_dur = _get_segment_duration_sec(g, seg_idx, fallback=float(t_common))
            title = to_ascii_for_plot(
                f"{track_id} | seg_start={seg_start:.2f}s | tag={use_tag} | idx={use_idx}"
            )
            out_path = out_dir / f"{seg_idx:06d}_{to_ascii_for_plot(track_id)}.png"
            make_heatmap_time_x(
                out_path=out_path,
                mat_feat=mat_feat,
                feat_labels=all_labels,
                feat_group_sizes=group_sizes,
                pitch_vals=pitch_vals,
                melody_vals=melody_vals,
                x_label_frame=list(x_label_frame_use),
                chord_spans=spans,
                seg_start_sec=seg_start,
                seg_duration_sec=seg_dur,
                title=title,
                all_midi_span=args.all_midi_span,
                melody_span=args.melody_span,
                feature_cell_height=args.feature_cell_height,
                pianoroll_cell_height=args.pianoroll_cell_height,
            )
            dumped += 1
            if genre is not None:
                genre_dumped[genre] = genre_dumped.get(genre, 0) + 1
                print(
                    f"[{dumped}/{args.max_plots}] seg={seg_idx} "
                    f"genre={genre} ({genre_dumped[genre]}/{args.max_per_genre}) → {out_path}"
                )
            else:
                print(f"[{dumped}/{args.max_plots}] seg={seg_idx} → {out_path}")

    print(f"[DONE] wrote {dumped} plots to {out_dir}")


if __name__ == "__main__":
    main()
