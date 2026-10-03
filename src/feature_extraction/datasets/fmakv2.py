from __future__ import annotations

import csv
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..label_providers import (
    _KeyLabelProviderBase,
    _parse_giantsteps_single_key,
    register_label_provider,
)
from ..registry import DatasetHandler, register_dataset_handler


def _find_fmakv2_files(root: str, split: str) -> List[str]:
    """
    FMA-Key v2 layout (test split only):
      root/XXX/YYYYYY.mp3
    where XXX is a 3-digit bucket folder (000, 001, ...) sitting directly
    under root, and YYYYYY is the track id. Key labels live in fmakv2.csv.

    Tries root/<split>/ first; if that doesn't exist falls back to root/.
    """
    split_dir = os.path.join(root, split)
    search_root = split_dir if os.path.isdir(split_dir) else root
    files = glob(os.path.join(search_root, "**", "*.mp3"), recursive=True)
    files.sort()
    return files


class FmaKv2DatasetHandler(DatasetHandler):
    """
    FMA-Key v2 full tracks (mp3 files in bucket subfolders). Test split only.
    """

    name = "fmakv2"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        # mode/musdb_target not used; always single-track audio.
        return _find_fmakv2_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        # Use filename (without extension) as stable track id, e.g. 014000.
        track_id = os.path.splitext(os.path.basename(path))[0]
        stem_id = "mix"
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(FmaKv2DatasetHandler())


class FmaKv2LabelProvider(_KeyLabelProviderBase):
    """
    FMA-Key v2 single-key-per-track labels from fmakv2.csv.

    CSV columns: ,key_and_mode,track_id,spotify_uri
      - key_and_mode: e.g. "F# Major", "A minor", "Bb Major"
      - track_id:     unpadded integer, e.g. 10 (matches "000010.mp3" filename)

    --label-root may point at the dataset folder (containing fmakv2.csv) or
    directly at the csv file. The csv is parsed once and cached per path.
    """

    dataset_name = "fmakv2"

    def __init__(self) -> None:
        # csv_path -> {int track_id: key_label_or_None}
        self._cache: Dict[str, Dict[int, Optional[str]]] = {}

    @staticmethod
    def _resolve_csv_path(label_root: str) -> Optional[Path]:
        root = Path(label_root)
        if root.is_file() and root.suffix.lower() == ".csv":
            return root
        cand = root / "fmakv2.csv"
        if cand.exists():
            return cand
        matches = sorted(root.glob("**/fmakv2.csv"))
        return matches[0] if matches else None

    @staticmethod
    def _normalize_track_id(track_id: str) -> Optional[int]:
        stem = os.path.splitext(os.path.basename(str(track_id)))[0].strip()
        try:
            return int(stem)
        except ValueError:
            return None

    def _load_csv(self, csv_path: Path) -> Dict[int, Optional[str]]:
        key = str(csv_path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        table: Dict[int, Optional[str]] = {}
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = self._normalize_track_id(row.get("track_id", ""))
                if tid is None:
                    continue
                key_label, is_multi = _parse_giantsteps_single_key(
                    row.get("key_and_mode", "")
                )
                # is_multi/unparseable -> store None, skipped per-track at lookup
                table[tid] = None if is_multi else key_label
        self._cache[key] = table
        return table

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Dict[str, Any]]:
        del split
        csv_path = self._resolve_csv_path(label_root)
        if csv_path is None:
            return None
        table = self._load_csv(csv_path)

        tid = self._normalize_track_id(track_id)
        if tid is None or tid not in table:
            return None  # missing label
        key_label = table[tid]
        if key_label is None:
            return {"_skip_track": True, "_skip_reason": "invalid_or_multi_key_label"}
        return {"key_label": key_label}


register_label_provider(FmaKv2LabelProvider())
