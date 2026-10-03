from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..label_providers import _KeyLabelProviderBase, register_label_provider
from ..registry import DatasetHandler, register_dataset_handler


def _dedup_sorted(paths: List[str]) -> List[str]:
    return sorted(set(paths))


def _find_gtzan_wavs(root: str, split: str) -> List[str]:
    """
    GTZAN audio layout (e.g. Marsyas GTZAN):
      - root/genres/<genre>/*.wav
      - root/<split>/genres/<genre>/*.wav   (optional split parent)

    Skips macOS AppleDouble files (names starting with ._).
    """
    bases: List[str] = []
    split_genres = os.path.join(root, split, "genres")
    root_genres = os.path.join(root, "genres")
    if os.path.isdir(split_genres):
        bases.append(split_genres)
    elif os.path.isdir(root_genres):
        bases.append(root_genres)

    files: List[str] = []
    for base in bases:
        files.extend(glob(os.path.join(base, "**", "*.wav"), recursive=True))
        files.extend(glob(os.path.join(base, "**", "*.WAV"), recursive=True))
    out = [f for f in files if not os.path.basename(f).startswith("._")]
    return _dedup_sorted(out)


class GtzanDatasetHandler(DatasetHandler):
    """
    GTZAN genre collection (wav per clip).

    track_id is ``<genre>/<stem>`` (e.g. ``pop/pop.00000``), aligned with
    [dataset-gtzan-key](https://github.com/audiocontentanalysis/dataset-gtzan-key)
    labels under ``<label_root>/genres/<genre>/<stem>.lerch.txt``.
    """

    name = "gtzan"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        del mode
        return _find_gtzan_wavs(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        genre = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        track_id = f"{genre}/{stem}"
        stem_id = genre
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(GtzanDatasetHandler())


def _lerch_gtzan_index_to_key(idx: int) -> Optional[str]:
    """
    Map Lerch GTZAN ``*.lerch.txt`` integer 0..23 to ``<pitch>:maj|min``.

    Chromatic order **from A** (major block then minor block):
      - 0..11: A, A#, B, C, C#, D, D#, E, F, F#, G, G# major
      - 12..23: same roots in minor

    ``-1`` means modulation / unknown; handle in ``load_for_track`` (skip), not here.
    """
    if idx < 0 or idx > 23:
        return None
    roots = (
        "A",
        "A#",
        "B",
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
    )
    pc = idx % 12
    mode = "maj" if idx < 12 else "min"
    return f"{roots[pc]}:{mode}"


class GtzanKeyLabelProvider(_KeyLabelProviderBase):
    dataset_name = "gtzan"

    @staticmethod
    def _resolve_lerch_path(label_root: str, track_id: str) -> Optional[Path]:
        tid = str(track_id).strip()
        if "/" in tid:
            genre, stem = tid.split("/", 1)
            p = Path(label_root) / "genres" / genre / f"{stem}.lerch.txt"
            if p.exists():
                return p
        stem_only = os.path.splitext(os.path.basename(tid))[0]
        root = Path(label_root)
        matches = sorted(root.glob(f"**/genres/*/{stem_only}.lerch.txt"))
        if len(matches) == 1:
            return matches[0]
        return None

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Dict[str, Any]]:
        del split
        path = self._resolve_lerch_path(label_root, track_id)
        if path is None:
            return None
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return {"_skip_track": True, "_skip_reason": "empty_key_label"}
        try:
            idx = int(raw.split()[0])
        except (ValueError, IndexError):
            return {"_skip_track": True, "_skip_reason": "invalid_key_label"}
        if idx == -1:
            return {
                "_skip_track": True,
                "_skip_reason": "gtzan_key_modulation_or_unknown",
            }
        key_label = _lerch_gtzan_index_to_key(idx)
        if key_label is None:
            return {"_skip_track": True, "_skip_reason": "invalid_key_label"}
        return {"key_label": key_label}


register_label_provider(GtzanKeyLabelProvider())
