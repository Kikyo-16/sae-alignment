from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from ..audio_utils import load_mono, resample_to
from ..label_providers import (
    _DEGREE_N,
    _parse_time_seg_file,
    _segs_to_frames,
    compute_degree_frames,
    register_label_provider,
)
from ..registry import DatasetHandler, register_dataset_handler


def _find_mix_files(root: str, split: str) -> List[str]:
    exts = ("flac", "wav", "mp3", "ogg", "m4a")
    files: List[str] = []
    for e in exts:
        files.extend(glob(os.path.join(root, split, "**", f"mix.{e}"), recursive=True))
    files.sort()
    return files


def _find_stem_files(root: str, split: str) -> List[str]:
    exts = ("flac", "wav", "mp3", "ogg", "m4a")
    files: List[str] = []
    for e in exts:
        files.extend(glob(os.path.join(root, split, "**", "stems", f"S*.{e}"), recursive=True))
    files.sort()
    return files


class SlakhDatasetHandler(DatasetHandler):
    """
    Slakh2100 dataset (flac) with optional stem / mix mode.
    """

    name = "slakh"

    # --- optional policy hooks consulted by process_datasets core ---
    def wants_long_silence_filter(self, mode: str) -> bool:
        return mode == "stem"

    def item_unit(self, mode: str) -> str:
        return "stem" if mode == "stem" else "file"

    def display_name(self, mode: str) -> str:
        return "stems" if mode == "stem" else "slakh"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        if mode == "stem":
            return _find_stem_files(root, split)
        elif mode == "mix":
            return _find_mix_files(root, split)
        else:
            raise ValueError(f"Unsupported mode '{mode}' for Slakh. Use 'stem' or 'mix'.")

    def track_id_for_path(self, path: str, *, mode: str = "stem") -> str:
        """Derive the track_id from a path without decoding audio."""
        if mode == "stem":
            return os.path.basename(os.path.dirname(os.path.dirname(path)))
        return os.path.basename(os.path.dirname(path))

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        # musdb_target is ignored for Slakh.
        if mode not in ("stem", "mix"):
            raise ValueError(f"Unsupported mode '{mode}' for Slakh. Use 'stem' or 'mix'.")

        if mode == "stem":
            track_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
            stem_id = os.path.splitext(os.path.basename(path))[0]
        else:
            track_id = os.path.basename(os.path.dirname(path))
            stem_id = "mix"

        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(SlakhDatasetHandler())


class SlakhLabelProvider:
    """
    Slakh2100 chord/key labels from per-track .lab files.

    Label files live at {label_root}/{split}/{track_id}/:
      all_src.chord.lab  — "start end chord_label" rows (e.g. "Bb:min", "N")
      all_src.key.lab    — "start end key_label"   rows (e.g. "A:minor")

    Works for both mix and stem modes; both share the same track-level labels.
    label_root: dataset root (parent of split dirs, e.g. dataset/slakh2100_flac_redux).
    """

    dataset_name = "slakh"

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Dict[str, Any]]:
        track_dir = Path(label_root) / split / track_id
        if not track_dir.exists():
            return None

        chord_path = track_dir / "all_src.chord.lab"
        key_path = track_dir / "all_src.key.lab"

        chord_segs = _parse_time_seg_file(chord_path) if chord_path.exists() else []
        key_segs = _parse_time_seg_file(key_path) if key_path.exists() else []

        if not chord_segs and not key_segs:
            return None

        return {"chord_segs": chord_segs, "key_segs": key_segs}

    def get_label_keys(self) -> List[str]:
        return ["chord_frame", "key_frame", "tonic_frame", "forth_frame", "fifth_frame"]

    def get_label_shapes_and_dtypes(
        self,
        seg_dur_sec: float,
        label_frames: int,
    ) -> Dict[str, Tuple[Tuple[int, ...], Any]]:
        del seg_dur_sec
        return {
            "chord_frame": ((label_frames,), "str"),
            "key_frame":   ((label_frames,), "str"),
            "tonic_frame": ((label_frames,), np.int8),
            "forth_frame": ((label_frames,), np.int8),
            "fifth_frame": ((label_frames,), np.int8),
        }

    def compute_segment_labels(
        self,
        seg_start_sec: float,
        seg_dur_sec: float,
        fps: float,
        T: int,
        label_data: Any,
    ) -> Dict[str, np.ndarray]:
        if label_data is None:
            return {
                "chord_frame": np.array(["N"] * T, dtype=object),
                "key_frame":   np.array(["N"] * T, dtype=object),
                "tonic_frame": np.full(T, _DEGREE_N, dtype=np.int8),
                "forth_frame": np.full(T, _DEGREE_N, dtype=np.int8),
                "fifth_frame": np.full(T, _DEGREE_N, dtype=np.int8),
            }
        chord_frame = _segs_to_frames(
            label_data.get("chord_segs", []), seg_start_sec, seg_dur_sec, fps, T
        )
        key_frame = _segs_to_frames(
            label_data.get("key_segs", []), seg_start_sec, seg_dur_sec, fps, T
        )
        degree_frames = compute_degree_frames(chord_frame, key_frame)
        return {"chord_frame": chord_frame, "key_frame": key_frame, **degree_frames}


register_label_provider(SlakhLabelProvider())

