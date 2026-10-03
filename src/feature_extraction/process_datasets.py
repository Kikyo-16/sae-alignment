#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pack audio datasets -> HDF5 with feature extraction + pitch-shift augmentation.

Supported reps:
- muq_layer      : MuQ hidden_states at specified layer (stored as (C, T))
- musicfm_layer  : MusicFM hidden_states at specified layer (stored as (C, T))

Notes:
- Training mode keeps silence checks + 5x pitch shift aug: orig, +2, +1, -1, -2 semitones.
- Inference mode (--inference): no overlap, no aug, no silence filter.
"""

from __future__ import annotations
import os
import argparse
import re
from importlib import import_module
from typing import List, Dict, Any, Optional

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from .audio_utils import (
    is_all_silent,
    has_long_silence,
    pitch_shift_pv_1d,
)
from .h5_utils import ensure_group, create_resizable_string_ds, append_resize
from .registry import get_dataset_handler, get_rep_handler
from .label_providers import get_label_provider

# ensure built-in datasets, reps, and label providers are registered
from . import datasets  # noqa: F401
from . import reps  # noqa: F401
from . import label_providers  # noqa: F401


_DECODE_WARN_PATTERNS = [
    re.compile(r"dequantization failed", re.IGNORECASE),
    re.compile(r"libmpg123", re.IGNORECASE),
    re.compile(r"mpg123", re.IGNORECASE),
]


def _contains_decode_warning(stderr_text: str) -> bool:
    t = str(stderr_text or "")
    return any(p.search(t) is not None for p in _DECODE_WARN_PATTERNS)


def _call_with_captured_stderr(fn, *args, **kwargs):
    """
    Capture C-level and Python stderr during fn(*args, **kwargs).
    Returns (result, stderr_text). Re-raises fn exceptions.
    """
    rfd, wfd = os.pipe()
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(wfd, 2)  # redirect process stderr to pipe
        os.close(wfd)
        result = fn(*args, **kwargs)
    finally:
        try:
            os.dup2(saved_stderr_fd, 2)
        finally:
            os.close(saved_stderr_fd)
    with os.fdopen(rfd, "r", encoding="utf-8", errors="ignore") as rf:
        stderr_text = rf.read()
    return result, stderr_text


def _load_extra_plugins() -> None:
    """Import extra dataset/rep modules listed in feature_plugins.py."""
    try:
        from . import feature_plugins as _fp
    except Exception:
        return
    for mod in getattr(_fp, "EXTRA_DATASET_MODULES", []):
        try:
            import_module(mod)
        except Exception as exc:
            print(f"[Plugins] Failed to import dataset plugin '{mod}': {exc}")
    for mod in getattr(_fp, "EXTRA_REP_MODULES", []):
        try:
            import_module(mod)
        except Exception as exc:
            print(f"[Plugins] Failed to import rep plugin '{mod}': {exc}")


# ---------------------------
# optional DatasetHandler policy hooks (all have safe defaults so a new dataset
# only needs to implement them when it wants non-default behavior)
# ---------------------------
def _handler_wants_long_silence(ds_handler, mode: str) -> bool:
    """Whether to apply the long-silence filter for this dataset/mode.

    Handlers may expose either a bool attr or a callable `wants_long_silence_filter`.
    Defaults to False.
    """
    hook = getattr(ds_handler, "wants_long_silence_filter", False)
    if callable(hook):
        return bool(hook(mode))
    return bool(hook)


def _handler_item_unit(ds_handler, mode: str) -> str:
    """tqdm unit label for one input item. Defaults to 'file'."""
    hook = getattr(ds_handler, "item_unit", "file")
    if callable(hook):
        return str(hook(mode))
    return str(hook)


def _handler_progress_name(ds_handler, mode: str) -> str:
    """Display name for the progress bar. Defaults to the handler name."""
    hook = getattr(ds_handler, "display_name", None)
    if callable(hook):
        return str(hook(mode))
    if hook:
        return str(hook)
    return str(getattr(ds_handler, "name", "dataset"))


# ---------------------------
# core
# ---------------------------
def pack_split_dataset(
    h5: h5py.File,
    root: str,
    split: str,
    target_sr: int = 16000,
    duration_sec: float = 10.0,
    hop_sec: float = 2.0,
    out_dtype: str = "float16",
    compression: str | None = "lzf",
    mel_batch_size: int = 16,
    mode: str = "stem",  # Slakh: "stem" or "mix"
    # silence opts
    min_silence_sec: float = 2.0,
    silence_db: float = -45.0,
    silence_frame_ms: float = 20.0,
    silence_hop_ms: float = 10.0,
    all_silent_eps: float = 1e-5,
    # pitch shift params
    pv_nfft: int = 1024,
    pv_hop: int = 256,
    # reps
    rep: str = "muq_layer",
    muq_model_id: str = "OpenMuQ/MuQ-large-msd-iter",
    muq_layer: int = -1,
    musicfm_model_id: str = "tky823/MusicFM",
    musicfm_layer: int = 7,
    # dataset
    dataset: str = "slakh",
    do_aug: bool = True,
    aug_semitones: Optional[List[int]] = None,
    # inference mode
    inference: bool = False,
    label_root: Optional[str] = None,
    label_frames: int = -1,
    require_labels: bool = False,
    skip_bad_audio: bool = True,
    skip_decode_warnings: bool = True,
    track_id_filter: Optional[set] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")

    # Resolve dataset handler (can be extended via register_dataset_handler).
    ds_handler = get_dataset_handler(dataset)
    files = list(ds_handler.find_files(root, split, mode=mode))

    if not files:
        raise FileNotFoundError(f"No audio found under {root}/{split} for dataset={dataset}")

    # Track-id allowlist: keep only tracks whose id appears in the given lists.
    # When the handler can derive the id from the path we pre-filter here (avoids
    # decoding excluded audio); otherwise the loop below filters after load_track.
    if track_id_filter is not None:
        n_before = len(files)
        if hasattr(ds_handler, "track_id_for_path"):
            files = [
                f for f in files
                if ds_handler.track_id_for_path(f, mode=mode) in track_id_filter
            ]
            print(f"[Filter] track-id allowlist ({len(track_id_filter)} ids): "
                  f"{len(files)}/{n_before} files kept (path pre-filter)")
            if not files:
                raise FileNotFoundError(
                    "track-id allowlist matched no files for "
                    f"dataset={dataset} split={split}."
                )
        else:
            print(f"[Filter] track-id allowlist ({len(track_id_filter)} ids): "
                  f"filtering after audio load (handler lacks track_id_for_path)")

    rep_handler = get_rep_handler(rep)
    win = int(round(duration_sec * target_sr))
    hop = int(round(hop_sec * target_sr))

    # Inference mode: no overlap, no aug, no silence filter
    if inference:
        hop = win
        do_aug = False
        print("[Inference] hop=duration (no overlap), aug disabled, silence filter disabled")

    rep_config = {
        "target_sr": target_sr,
        "muq_model_id": muq_model_id,
        "muq_layer": muq_layer,
        "musicfm_model_id": musicfm_model_id,
        "musicfm_layer": musicfm_layer,
    }

    rep_state = rep_handler.init_models(device, config=rep_config)
    rep_shape, rep_meta = rep_handler.probe_shape(
        win_samples=win,
        device=device,
        state=rep_state,
        config=rep_config,
    )
    print(f"[Probe] rep '{rep}' per segment: {rep_shape}")

    grp = ensure_group(h5, split)
    comp = None if (compression in (None, "none")) else compression
    np_dtype = np.float16 if out_dtype == "float16" else np.float32

    rep_dsets = rep_handler.create_datasets(
        grp,
        rep_shape,
        rep_meta,
        np_dtype=np_dtype,
        compression=comp,
        config=rep_config,
    )
    if "main" not in rep_dsets:
        raise RuntimeError("Rep handler must create a 'main' dataset in create_datasets()")

    # meta datasets. `aug` (string) is not stored: it is fully derivable from the
    # numeric `semitone` field ("orig" if 0 else f"ps{semitone:+d}"). `sr` is a
    # constant for the whole split and is stored as a group attribute below.
    uid_ds      = create_resizable_string_ds(grp, "uid")
    base_uid_ds = create_resizable_string_ds(grp, "base_uid")
    track_ds    = create_resizable_string_ds(grp, "trackId")
    stem_ds     = create_resizable_string_ds(grp, "stemId")

    semitone_ds = grp.create_dataset("semitone", shape=(0,), maxshape=(None,), dtype=np.int16, chunks=True)
    start_ds    = grp.create_dataset("start",    shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)
    end_ds      = grp.create_dataset("end",      shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)

    grp.attrs["sr"] = int(target_sr)

    if not do_aug:
        aug_plan = [("orig", 0)]
    else:
        semitones = aug_semitones if aug_semitones is not None else [2, 1, -1, -2]
        aug_plan = [("orig", 0)] + [(f"ps{s:+d}", s) for s in semitones]

    label_provider = get_label_provider(dataset) if label_root else None
    label_dsets: Dict[str, h5py.Dataset] = {}
    if label_provider and label_root:
        # If label_frames <= 0, auto-align label timeline to rep time length when possible.
        label_frames_eff = int(label_frames)
        if label_frames_eff <= 0:
            if isinstance(rep_shape, tuple) and len(rep_shape) >= 1:
                label_frames_eff = int(rep_shape[-1])
                print(f"[Labels] auto label_frames={label_frames_eff} from rep shape")
            else:
                raise RuntimeError(
                    "Cannot auto-infer label_frames from rep shape; please pass --label-frames explicitly."
                )

        shapes_dtypes = label_provider.get_label_shapes_and_dtypes(duration_sec, label_frames_eff)
        lbl_grp = grp.create_group("labels")
        T_lbl = label_frames_eff
        fps = T_lbl / duration_sec
        for key, (shape_per_seg, dtype) in shapes_dtypes.items():
            if dtype == "str":
                dt = h5py.string_dtype(encoding="utf-8")
            else:
                dt = dtype
            full_shape = (0,) + tuple(shape_per_seg)
            max_shape = (None,) + tuple(shape_per_seg)
            label_dsets[key] = lbl_grp.create_dataset(
                key, shape=full_shape, maxshape=max_shape, dtype=dt, chunks=True
            )
        lbl_grp.attrs["fps"] = float(fps)
        lbl_grp.attrs["T"] = int(T_lbl)
        lbl_grp.attrs["segment_sec"] = float(duration_sec)
        print(f"[Labels] {list(label_dsets.keys())} (fps={fps:.1f}, T={T_lbl})")
    else:
        fps = 0.0
        T_lbl = 0

    total_written = 0
    bad_audio_count = 0
    bad_audio_decode_warn = 0
    label_track_total = 0
    label_track_missing = 0
    label_track_ok = 0
    label_track_skipped = 0
    # Dataset-specific presentation/behavior is read from optional handler hooks
    # (with safe defaults) so new datasets never need to touch this core loop.
    ds_name = _handler_progress_name(ds_handler, mode)
    bar_unit = _handler_item_unit(ds_handler, mode)
    bar_name = f"[{split}] {ds_name}"
    wants_long_silence = _handler_wants_long_silence(ds_handler, mode)

    for p in tqdm(files, desc=bar_name, unit=bar_unit):
        # ---------------------------
        # load audio, derive track/stem ids (via DatasetHandler)
        # ---------------------------
        try:
            if skip_decode_warnings:
                (wav, track_id, stem_id), stderr_text = _call_with_captured_stderr(
                    ds_handler.load_track,
                    p,
                    target_sr=target_sr,
                    mode=mode,
                )
                if _contains_decode_warning(stderr_text):
                    bad_audio_count += 1
                    bad_audio_decode_warn += 1
                    print(f"[Audio][WARN] skip decode-warning file: {p}")
                    continue
            else:
                wav, track_id, stem_id = ds_handler.load_track(
                    p,
                    target_sr=target_sr,
                    mode=mode,
                )
        except Exception as exc:
            bad_audio_count += 1
            if skip_bad_audio:
                print(f"[Audio][WARN] skip unreadable file: {p} ({exc})")
                continue
            raise

        # Fallback allowlist filter (when no path-based pre-filter was possible).
        if track_id_filter is not None and track_id not in track_id_filter:
            continue

        n = wav.shape[-1]
        if n < win:
            continue

        starts = list(range(0, n - win + 1, hop))
        seg_count = len(starts)
        if seg_count == 0:
            continue

        seg_pbar = tqdm(total=seg_count, desc=f"{track_id}/{stem_id}", unit="seg", leave=False)

        # Load label data for this track (when label provider exists)
        track_label_data = None
        if label_provider and label_root:
            label_track_total += 1
            track_label_data = label_provider.load_for_track(track_id, label_root, split)
            if (
                isinstance(track_label_data, dict)
                and bool(track_label_data.get("_skip_track", False))
            ):
                label_track_skipped += 1
                reason = str(track_label_data.get("_skip_reason", "unknown"))
                print(
                    f"[Labels][INFO] skip track_id='{track_id}' by label provider "
                    f"(reason={reason})"
                )
                seg_pbar.close()
                continue
            if track_label_data is None:
                label_track_missing += 1
                print(
                    f"[Labels][WARN] missing labels for track_id='{track_id}' "
                    f"(label_root={label_root})"
                )
                if require_labels:
                    raise RuntimeError(
                        f"Label required but missing for track_id='{track_id}'. "
                        "Check --label-root and dataset track id alignment."
                    )
            else:
                label_track_ok += 1

        # buffers
        rep_buf: List[torch.Tensor] = []
        extra_bufs: Dict[str, List[Any]] = {}
        label_bufs: Dict[str, List] = {k: [] for k in label_dsets} if label_dsets else {}

        uid_buf: List[str] = []
        base_uid_buf: List[str] = []
        semi_buf: List[int] = []
        start_buf: List[float] = []
        end_buf: List[float] = []
        track_buf: List[str] = []
        stem_buf: List[str] = []

        def flush():
            nonlocal total_written
            if not rep_buf:
                return

            acc = len(rep_buf)
            main_ds = rep_dsets["main"]
            rep_np = torch.stack(rep_buf, dim=0).cpu().numpy()
            sl = append_resize(main_ds, acc)
            main_ds[sl, ...] = rep_np.astype(main_ds.dtype)

            for key, ds in rep_dsets.items():
                if key == "main":
                    continue
                vals = extra_bufs.get(key)
                if not vals:
                    continue
                sl_ex = append_resize(ds, acc)
                ds[sl_ex] = np.asarray(vals, dtype=ds.dtype)

            sl_st = append_resize(start_ds, acc); start_ds[sl_st] = np.asarray(start_buf, dtype=np.float32)
            sl_en = append_resize(end_ds,   acc); end_ds[sl_en]   = np.asarray(end_buf,   dtype=np.float32)

            sl_u   = append_resize(uid_ds,      acc); uid_ds[sl_u]       = np.asarray(uid_buf, dtype=object)
            sl_bu  = append_resize(base_uid_ds, acc); base_uid_ds[sl_bu] = np.asarray(base_uid_buf, dtype=object)
            sl_se  = append_resize(semitone_ds, acc); semitone_ds[sl_se] = np.asarray(semi_buf, dtype=np.int16)

            sl_t = append_resize(track_ds, acc); track_ds[sl_t] = np.asarray(track_buf, dtype=object)
            sl_s = append_resize(stem_ds,  acc); stem_ds[sl_s]  = np.asarray(stem_buf,  dtype=object)

            # Write labels when present
            if label_dsets and label_bufs:
                for key, ds in label_dsets.items():
                    vals = label_bufs.get(key)
                    if vals:
                        sl_lbl = append_resize(ds, acc)
                        arr = np.stack(vals, axis=0)
                        if h5py.check_string_dtype(ds.dtype) is not None:
                            ds[sl_lbl] = arr.astype(object)
                        else:
                            ds[sl_lbl] = arr.astype(ds.dtype)

            total_written += acc

            rep_buf.clear()
            extra_bufs.clear()
            for lb in label_bufs.values():
                lb.clear()
            uid_buf.clear()
            base_uid_buf.clear()
            semi_buf.clear()
            start_buf.clear()
            end_buf.clear()
            track_buf.clear()
            stem_buf.clear()

        with torch.no_grad():
            for seg_idx, s in enumerate(starts):
                seg = wav[..., s:s+win].squeeze(0)  # (win,)

                if not inference and is_all_silent(seg, eps=all_silent_eps):
                    seg_pbar.update(1)
                    continue

                # long-silence filter is opt-in per dataset handler (skip in inference)
                if not inference and wants_long_silence and has_long_silence(
                    seg, target_sr,
                    min_silence_sec=min_silence_sec,
                    silence_db=silence_db,
                    frame_ms=silence_frame_ms,
                    hop_ms=silence_hop_ms,
                    all_silent_eps=all_silent_eps,
                ):
                    seg_pbar.update(1)
                    continue

                base_uid = f"{track_id}_{stem_id}_{seg_idx:06d}"
                s_t = float(s / target_sr)
                e_t = float(s_t + float(duration_sec))

                # Compute segment labels once per segment (same for all augs)
                segment_labels: Dict[str, np.ndarray] = {}
                if label_provider and T_lbl > 0:
                    segment_labels = label_provider.compute_segment_labels(
                        s_t, duration_sec, fps, T_lbl, track_label_data
                    )

                for aug_tag, semi in aug_plan:
                    if semi == 0:
                        seg_aug = seg
                    else:
                        seg_aug = pitch_shift_pv_1d(seg, target_sr, semi, n_fft=pv_nfft, hop_length=pv_hop)

                    # -------------------
                    # Feature extraction (delegated to rep handler)
                    # -------------------
                    feat, extra_meta = rep_handler.extract(
                        seg_aug,
                        device=device,
                        state=rep_state,
                        config=rep_config,
                    )
                    if feat is None:
                        continue
                    feat = feat.detach().cpu()

                    # append shared metadata only when we actually kept this sample
                    rep_buf.append(feat)
                    for k, v in extra_meta.items():
                        extra_bufs.setdefault(k, []).append(v)

                    base_uid_buf.append(base_uid)
                    uid_buf.append(f"{base_uid}_{aug_tag}")
                    semi_buf.append(int(semi))
                    start_buf.append(s_t)
                    end_buf.append(e_t)
                    track_buf.append(track_id)
                    stem_buf.append(stem_id)
                    # Labels are stored un-transposed (orig values) for every aug
                    # variant; pitch-shift transposition is applied at training time
                    # using the per-sample `semitone` field.
                    for lk, lv in segment_labels.items():
                        label_bufs.setdefault(lk, []).append(lv)

                if len(rep_buf) >= mel_batch_size * len(aug_plan):
                    flush()

                seg_pbar.update(1)

        flush()
        seg_pbar.close()
        
        

    print(f"[{split}] done. wrote {total_written} entries (includes {len(aug_plan)}x aug). rep={rep}")
    if bad_audio_count > 0:
        print(f"[Audio] skipped unreadable files: {bad_audio_count}")
    if bad_audio_decode_warn > 0:
        print(f"[Audio] skipped due to decode warnings: {bad_audio_decode_warn}")
    if label_provider and label_root:
        print(
            f"[Labels] tracks total={label_track_total}, ok={label_track_ok}, "
            f"missing={label_track_missing}, skipped={label_track_skipped}"
        )



# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Pack datasets to HDF5 with feature extraction + pitch-shift aug")
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-aug", action="store_true", help="Disable pitch-shift augmentation")
    ap.add_argument(
        "--aug-semitones",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Semitone offsets for pitch-shift aug (excluding 0/orig). "
             "Default: 2 1 -1 -2. Example: --aug-semitones 14 7 -7 -14",
    )
    ap.add_argument(
        "--dataset",
        default="slakh",
        help="Dataset name. Built-ins are auto-discovered from feature_extraction/datasets/*.py "
             "(slakh, pop909, rwc, gtzan, giantsteps_key, giantsteps_key_plus, fmakv2, sae_diagnostic). "
             "You can also register custom datasets via feature_plugins.py."
    )
    ap.add_argument("--split", default="train",
                    help="Subfolder under --root (Slakh: train/validation/test; Pop909: train/valid/test)")
    ap.add_argument("--out-h5", required=True)

    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=2.0)

    ap.add_argument("--mel-batch", type=int, default=16)

    ap.add_argument(
        "--rep",
        required=True,
        help=(
            "Feature representation to store: muq_layer | musicfm_layer. "
            "You can also register custom reps via feature_plugins.py."
        ),
    )
    ap.add_argument("--muq-model-id", default="OpenMuQ/MuQ-large-msd-iter")
    ap.add_argument("--muq-layer", type=int, default=-1)
    ap.add_argument("--musicfm-model-id", default="tky823/MusicFM")
    ap.add_argument("--musicfm-layer", type=int, default=7)

    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--compression", default="lzf", choices=["gzip", "lzf", "none"])

    ap.add_argument("--mode", default="stem", choices=["stem", "mix"],
                    help="Feature source for Slakh: stems (default) or mix.*")

    # silence options
    ap.add_argument("--min-silence-sec", type=float, default=2.0)
    ap.add_argument("--silence-db", type=float, default=-45.0)
    ap.add_argument("--silence-frame-ms", type=float, default=20.0)
    ap.add_argument("--silence-hop-ms",   type=float, default=10.0)
    ap.add_argument("--all-silent-eps", type=float, default=1e-5)

    # pitch shift pv options
    ap.add_argument("--pv-nfft", type=int, default=1024)
    ap.add_argument("--pv-hop",  type=int, default=256)

    # inference mode (no overlap, no aug, optional labels)
    ap.add_argument("--inference", action="store_true",
                    help="Inference mode: hop=duration (no overlap), no aug, no silence filter. Use with --label-root for labels.")
    ap.add_argument("--label-root", type=str, default=None,
                    help="Root for labels (dataset-specific, e.g. Pop909 dir with 001/, 002/...). Requires dataset with LabelProvider.")
    ap.add_argument("--label-frames", type=int, default=-1,
                    help="Frames per segment for labels. <=0 means auto-match rep timeline length.")
    ap.add_argument(
        "--require-labels",
        action="store_true",
        help="Fail immediately if any track has missing labels in inference mode.",
    )
    ap.add_argument(
        "--strict-audio",
        action="store_true",
        help="Fail on unreadable/corrupted audio instead of skipping.",
    )
    ap.add_argument(
        "--no-skip-decode-warnings",
        action="store_true",
        help="Do not skip files that emit decode warnings (e.g., mpg123 dequantization warnings).",
    )
    ap.add_argument(
        "--track-id-files",
        type=str,
        nargs="+",
        default=None,
        metavar="FILE",
        help="One or more text files (one trackId per line). Only tracks whose id "
             "appears in the union of these files are extracted.",
    )

    args = ap.parse_args()

    # Load any user‑defined plugin modules before running.
    _load_extra_plugins()

    out_dir = os.path.dirname(args.out_h5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    track_id_filter = None
    if args.track_id_files:
        track_id_filter = set()
        for fpath in args.track_id_files:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    tid = line.strip()
                    if tid:
                        track_id_filter.add(tid)
        if not track_id_filter:
            raise ValueError(f"--track-id-files {args.track_id_files} contained no track ids.")
        print(f"[Filter] loaded {len(track_id_filter)} unique track ids from "
              f"{len(args.track_id_files)} file(s)")

    with h5py.File(args.out_h5, "w") as h5:
        pack_split_dataset(
            h5=h5,
            root=args.root,
            split=args.split,
            target_sr=args.target_sr,
            duration_sec=args.duration,
            hop_sec=args.hop,
            out_dtype=args.dtype,
            compression=None if args.compression == "none" else args.compression,
            mel_batch_size=args.mel_batch,
            mode=args.mode,
            min_silence_sec=args.min_silence_sec,
            silence_db=args.silence_db,
            silence_frame_ms=args.silence_frame_ms,
            silence_hop_ms=args.silence_hop_ms,
            all_silent_eps=args.all_silent_eps,
            pv_nfft=args.pv_nfft,
            pv_hop=args.pv_hop,
            rep=args.rep,
            muq_model_id=args.muq_model_id,
            muq_layer=args.muq_layer,
            musicfm_model_id=args.musicfm_model_id,
            musicfm_layer=args.musicfm_layer,
            dataset=args.dataset,
            do_aug=(not args.no_aug),
            aug_semitones=args.aug_semitones,
            inference=args.inference,
            label_root=args.label_root,
            label_frames=args.label_frames,
            require_labels=args.require_labels,
            skip_bad_audio=(not args.strict_audio),
            skip_decode_warnings=(not args.no_skip_decode_warnings),
            track_id_filter=track_id_filter,
        )

    print("Wrote H5:", args.out_h5)


if __name__ == "__main__":
    main()