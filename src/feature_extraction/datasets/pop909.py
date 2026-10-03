from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from ..audio_utils import load_mono, resample_to
from ..label_providers import (
    _DEGREE_N,
    _melody_roll_from_midi,
    _parse_time_seg_file,
    _pitch_roll_from_midi,
    _segs_to_frames,
    compute_degree_frames,
    register_label_provider,
)
from ..registry import DatasetHandler, register_dataset_handler


def _dedup_sorted(paths: List[str]) -> List[str]:
    return sorted(set(paths))


def _split_aliases(split: str) -> List[str]:
    s = str(split).strip()
    aliases = [s]
    if s == "validation":
        aliases.append("valid")
    if s == "valid":
        aliases.append("validation")
    return aliases


def _find_pop909_original_files(root: str, split: str) -> List[str]:
    """
    Supported layouts:
      1) root/{train,valid,test}/*/original.mp3
      2) root/{train,valid,test}/*/versions/original.{mp3,wav,flac,m4a}
      3) root/*/original.mp3                      (no split folders)
      4) root/*/versions/original.{mp3,wav,flac,m4a}
    """
    exts = ("mp3", "wav", "flac", "m4a")
    candidates: List[str] = []

    # split-aware search
    for sp in _split_aliases(split):
        candidates.extend(glob(os.path.join(root, sp, "*", "original.mp3")))
        for ext in exts:
            candidates.extend(glob(os.path.join(root, sp, "*", "versions", f"original.{ext}")))

    # no-split fallback
    if not candidates:
        candidates.extend(glob(os.path.join(root, "*", "original.mp3")))
        for ext in exts:
            candidates.extend(glob(os.path.join(root, "*", "versions", f"original.{ext}")))

    return _dedup_sorted(candidates)


def _infer_track_id_from_path(path: str) -> str:
    p = os.path.normpath(path)
    parent = os.path.basename(os.path.dirname(p))
    if parent == "versions":
        return os.path.basename(os.path.dirname(os.path.dirname(p)))
    return parent


class Pop909DatasetHandler(DatasetHandler):
    """
    Pop909 original mp3s (mono mix).
    """

    name = "pop909"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        # mode / musdb_target are irrelevant for Pop909.
        return _find_pop909_original_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        track_id = _infer_track_id_from_path(path)
        stem_id = "original"
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(Pop909DatasetHandler())


class Pop909LabelProvider:
    """
    Pop909 labels:
      - chord_frame (from chord_midi.txt/chord_audio.txt)
      - key_frame (from key_audio.txt)
      - pitch_roll/melody_roll (from .mid)
    label_root: Pop909 root with dirs 001/, 002/... each containing
        {sid}.mid, chord_midi.txt/chord_audio.txt, key_audio.txt
    """

    dataset_name = "pop909"

    @staticmethod
    def _candidate_label_roots(label_root: str) -> List[Path]:
        """
        Accept both:
          - <...>/POP909/001/...
          - <...>/001/...   (already inside POP909 root)
        """
        root = Path(label_root)
        cands = [root]
        if (root / "POP909").exists():
            cands.append(root / "POP909")
        if (root / "pop909").exists():
            cands.append(root / "pop909")
        # dedup while preserving order
        out: List[Path] = []
        seen = set()
        for p in cands:
            s = str(p.resolve())
            if s not in seen:
                seen.add(s)
                out.append(p)
        return out

    @staticmethod
    def _candidate_track_ids(track_id: str) -> List[str]:
        tid = str(track_id).strip()
        cands = [tid]

        # Common pop909 audio folder naming: "001-<song title>" -> label folder "001".
        m = re.match(r"^(\d+)", tid)
        if m:
            lead = m.group(1)
            cands.append(lead)
            cands.append(str(int(lead)))         # e.g. 001 -> 1
            cands.append(f"{int(lead):03d}")     # e.g. 1 -> 001

        if tid.isdigit():
            cands.append(str(int(tid)))          # e.g. 001 -> 1
            cands.append(f"{int(tid):03d}")      # e.g. 1 -> 001
        # keep order while removing duplicates
        out: List[str] = []
        seen = set()
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Dict[str, Any]]:
        del split  # pop909 labels are track-folder based; no split-level dependency
        for root in self._candidate_label_roots(label_root):
            for tid in self._candidate_track_ids(track_id):
                song_dir = root / tid
                if not song_dir.exists():
                    continue

                chord_path = song_dir / "chord_midi.txt"
                if not chord_path.exists():
                    chord_path = song_dir / "chord_audio.txt"
                key_path = song_dir / "key_audio.txt"
                primary_mid = song_dir / f"{tid}.mid"
                chord_segs = _parse_time_seg_file(chord_path) if chord_path.exists() else []
                key_segs = _parse_time_seg_file(key_path) if key_path.exists() else []
                if primary_mid.exists():
                    return {
                        "chord_segs": chord_segs,
                        "key_segs": key_segs,
                        "mid_path": primary_mid,
                    }

                mids = sorted(song_dir.glob("*.mid"))
                if mids:
                    return {
                        "chord_segs": chord_segs,
                        "key_segs": key_segs,
                        "mid_path": mids[0],
                    }
                if chord_segs or key_segs:
                    return {
                        "chord_segs": chord_segs,
                        "key_segs": key_segs,
                        "mid_path": None,
                    }
        return None

    def get_label_keys(self) -> List[str]:
        return ["chord_frame", "key_frame", "pitch_roll", "melody_roll",
                "tonic_frame", "forth_frame", "fifth_frame"]

    def get_label_shapes_and_dtypes(
        self,
        seg_dur_sec: float,
        label_frames: int,
    ) -> Dict[str, Tuple[Tuple[int, ...], Any]]:
        # Use "str" sentinel; caller will map to h5py.string_dtype(encoding="utf-8")
        return {
            "chord_frame":  ((label_frames,), "str"),
            "key_frame":    ((label_frames,), "str"),
            "pitch_roll":   ((128, label_frames), np.uint8),
            "melody_roll":  ((128, label_frames), np.uint8),
            "tonic_frame":  ((label_frames,), np.int8),
            "forth_frame":  ((label_frames,), np.int8),
            "fifth_frame":  ((label_frames,), np.int8),
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
                "pitch_roll":  np.zeros((128, T), dtype=np.uint8),
                "melody_roll": np.zeros((128, T), dtype=np.uint8),
                "tonic_frame": np.full(T, _DEGREE_N, dtype=np.int8),
                "forth_frame": np.full(T, _DEGREE_N, dtype=np.int8),
                "fifth_frame": np.full(T, _DEGREE_N, dtype=np.int8),
            }
        chord_segs = label_data.get("chord_segs", [])
        key_segs = label_data.get("key_segs", [])
        mid_path = label_data.get("mid_path")
        chord_frame = _segs_to_frames(chord_segs, seg_start_sec, seg_dur_sec, fps, T)
        key_frame = _segs_to_frames(key_segs, seg_start_sec, seg_dur_sec, fps, T)
        pitch_roll = (
            _pitch_roll_from_midi(mid_path, seg_start_sec, seg_dur_sec, fps, T)
            if mid_path and Path(mid_path).exists()
            else np.zeros((128, T), dtype=np.uint8)
        )
        melody_roll = (
            _melody_roll_from_midi(mid_path, seg_start_sec, seg_dur_sec, fps, T)
            if mid_path and Path(mid_path).exists()
            else np.zeros((128, T), dtype=np.uint8)
        )
        degree_frames = compute_degree_frames(chord_frame, key_frame)
        return {
            "chord_frame": chord_frame,
            "key_frame":   key_frame,
            "pitch_roll":  pitch_roll,
            "melody_roll": melody_roll,
            **degree_frames,
        }


register_label_provider(Pop909LabelProvider())

