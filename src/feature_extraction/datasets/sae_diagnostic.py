from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..registry import DatasetHandler, register_dataset_handler


_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _resolve_audio_root(root: str) -> Path:
    """Resolve either the dataset root or its audio/ subdirectory."""
    root_path = Path(root).expanduser().resolve()
    audio_dir = root_path / "audio"
    if audio_dir.is_dir():
        return audio_dir
    if root_path.is_dir():
        return root_path
    raise FileNotFoundError(f"SAE diagnostic dataset root not found: {root_path}")


def _find_audio_files(root: str) -> List[str]:
    audio_root = _resolve_audio_root(root)
    files = sorted(
        str(path)
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTS
    )
    return files


def _sample_id(path: str) -> str:
    """Return the metadata.csv-compatible sample id: family_root_mode."""
    p = Path(path)
    family = p.parent.name
    return f"{family}_{p.stem}"


class SaeDiagnosticDatasetHandler(DatasetHandler):
    """Controlled 30-second SAE diagnostic dataset.

    Expected default layout::

        <root>/
          metadata.csv
          segments.csv
          audio/
            piano/C_major.wav
            strings/C_major.wav
            woodwinds/C_major.wav
            brass/C_major.wav
            ...

    ``--root`` may point to either ``<root>`` or ``<root>/audio``.  The CLI
    split name is used only as the output HDF5 group name; this dataset has no
    physical train/validation/test split.
    """

    name = "sae_diagnostic"

    def wants_long_silence_filter(self, mode: str) -> bool:
        del mode
        return False

    def item_unit(self, mode: str) -> str:
        del mode
        return "sample"

    def display_name(self, mode: str) -> str:
        del mode
        return "SAE diagnostic"

    def find_files(self, root: str, split: str, *, mode: str = "mix") -> Iterable[str]:
        del split, mode
        return _find_audio_files(root)

    def track_id_for_path(self, path: str, *, mode: str = "mix") -> str:
        del mode
        return _sample_id(path)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "mix",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, _sample_id(path), "mix"


register_dataset_handler(SaeDiagnosticDatasetHandler())
