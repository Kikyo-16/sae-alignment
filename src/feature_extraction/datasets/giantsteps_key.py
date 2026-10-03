from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ..audio_utils import load_mono, resample_to
from ..label_providers import (
    _KeyLabelProviderBase,
    _normalize_key_pitch,
    _parse_giantsteps_single_key,
    register_label_provider,
)
from ..registry import DatasetHandler, register_dataset_handler


def _dedup_sorted(paths: List[str]) -> List[str]:
    return sorted(set(paths))


def _find_giantsteps_audio_files(root: str, split: str) -> List[str]:
    """
    Supports:
      - root/<split>/**.<ext>   (if split folder exists)
      - root/**.<ext>           (no split layout; common for giantsteps_key/audio)
    """
    exts = ("mp3", "wav", "flac", "m4a", "aif", "aiff")
    search_roots: List[str] = []
    split_root = os.path.join(root, split)
    if os.path.isdir(split_root):
        search_roots.append(split_root)
    else:
        search_roots.append(root)

    files: List[str] = []
    for sr in search_roots:
        for ext in exts:
            files.extend(glob(os.path.join(sr, f"*.{ext}")))
            files.extend(glob(os.path.join(sr, "**", f"*.{ext}"), recursive=True))
    return _dedup_sorted(files)


def _find_giantsteps_key_audio_files(root: str, split: str) -> List[str]:
    """
    Original GiantSteps key dataset style:
      - root/audio/*.<ext>
      - root/<split>/audio/*.<ext> (optional split parent)
    """
    exts = ("mp3", "wav", "flac", "m4a", "aif", "aiff")
    roots: List[str] = []
    split_root = os.path.join(root, split)
    if os.path.isdir(split_root):
        roots.append(split_root)
    roots.append(root)

    files: List[str] = []
    for r in roots:
        audio_dir = os.path.join(r, "audio")
        if not os.path.isdir(audio_dir):
            continue
        for ext in exts:
            files.extend(glob(os.path.join(audio_dir, f"*.{ext}")))
            files.extend(glob(os.path.join(audio_dir, "**", f"*.{ext}"), recursive=True))
    return _dedup_sorted(files)


class GiantStepsKeyPlusDatasetHandler(DatasetHandler):
    """
    GiantSteps+ key dataset audio handler.
    track_id is file stem, matched to label txt with same stem.
    """

    name = "giantsteps_key_plus"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        del mode  # irrelevant for this dataset
        return _find_giantsteps_audio_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        track_id = os.path.splitext(os.path.basename(path))[0]
        stem_id = "original"
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


class GiantStepsKeyDatasetHandler(DatasetHandler):
    """
    Original GiantSteps key dataset audio handler.
    Expected layout contains 'audio/' folder.
    """

    name = "giantsteps_key"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        del mode
        return _find_giantsteps_key_audio_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        track_id = os.path.splitext(os.path.basename(path))[0]
        stem_id = "original"
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        return wav, track_id, stem_id


register_dataset_handler(GiantStepsKeyPlusDatasetHandler())
register_dataset_handler(GiantStepsKeyDatasetHandler())


# ---------------------------
# Label providers (key-only)
# ---------------------------


class GiantStepsKeyPlusLabelProvider(_KeyLabelProviderBase):
    dataset_name = "giantsteps_key_plus"

    @staticmethod
    def _candidate_label_roots(label_root: str) -> List[Path]:
        root = Path(label_root)
        cands = [root]
        if (root / "keys_gs+").exists():
            cands.append(root / "keys_gs+")
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
        # common filename normalization variants
        cands.append(re.sub(r"\s+", " ", tid))
        cands.append(tid.replace("–", "-"))
        cands.append(tid.replace("—", "-"))
        cands.append(tid.replace(" - ", "-"))
        cands.append(tid.replace("-", " - "))
        m = re.match(r"^(\d+)", tid)
        if m:
            cands.append(m.group(1))
        out: List[str] = []
        seen = set()
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    @staticmethod
    def _resolve_label_file(root: Path, track_id: str) -> Optional[Path]:
        # 1) exact/variant full-name matches
        for tid in GiantStepsKeyPlusLabelProvider._candidate_track_ids(track_id):
            txt = root / f"{tid}.txt"
            if txt.exists():
                return txt
            txt_upper = root / f"{tid}.TXT"
            if txt_upper.exists():
                return txt_upper

        # 2) numeric-prefix fallback: "<id>*.txt"
        m = re.match(r"^(\d+)", str(track_id).strip())
        if m:
            prefix = m.group(1)
            matches = sorted(root.glob(f"{prefix}*.txt")) + sorted(root.glob(f"{prefix}*.TXT"))
            if matches:
                return matches[0]
        return None

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Dict[str, Any]]:
        del split
        for root in self._candidate_label_roots(label_root):
            txt = self._resolve_label_file(root, track_id)
            if txt is None:
                continue
            text = txt.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return {
                    "_skip_track": True,
                    "_skip_reason": "empty_key_label",
                }
            key_label, is_multi = _parse_giantsteps_single_key(text)
            if is_multi:
                return {
                    "_skip_track": True,
                    "_skip_reason": "multi_key_label",
                }
            if key_label is None:
                return {
                    "_skip_track": True,
                    "_skip_reason": "invalid_key_label",
                }
            return {
                "key_label": key_label,
            }
        return None


def _parse_giantsteps_key_value(text: str) -> Tuple[Optional[str], bool]:
    """
    Parse original GiantSteps key labels from plain key files.
    Accepts examples like:
      - "C major"
      - "Eb minor"
      - "C:maj"
      - "A:min"
    Returns (normalized_key, is_multi).
    """
    raw = str(text).strip()
    if not raw:
        return None, False
    if "|" in raw:
        return None, True

    # Fast path: already "<pitch>:<mode>".
    m_colon = re.match(r"^\s*([A-Ga-g][#b]?)\s*:\s*(maj|min)\w*\s*$", raw, re.IGNORECASE)
    if m_colon:
        pitch = _normalize_key_pitch(m_colon.group(1))
        if pitch is None:
            return None, False
        mode = "maj" if m_colon.group(2).lower().startswith("maj") else "min"
        return f"{pitch}:{mode}", False

    # Generic "Pitch mode (...)" form.
    return _parse_giantsteps_single_key(raw)


class GiantStepsKeyLabelProvider(_KeyLabelProviderBase):
    dataset_name = "giantsteps_key"

    @staticmethod
    def _candidate_label_roots(label_root: str) -> List[Path]:
        root = Path(label_root)
        cands = [root]
        if (root / "annotations" / "key").exists():
            cands.append(root / "annotations" / "key")
        if (root / "key").exists():
            cands.append(root / "key")
        if (root / "labels").exists():
            cands.append(root / "labels")
        out: List[Path] = []
        seen = set()
        for p in cands:
            s = str(p.resolve())
            if s not in seen:
                seen.add(s)
                out.append(p)
        return out

    @staticmethod
    def _resolve_label_file(root: Path, track_id: str) -> Optional[Path]:
        tid = str(track_id).strip()
        exts = ("txt", "TXT", "key", "KEY", "lab", "LAB")
        for ext in exts:
            p = root / f"{tid}.{ext}"
            if p.exists():
                return p
        # numeric prefix fallback
        m = re.match(r"^(\d+)", tid)
        if m:
            prefix = m.group(1)
            for pat in (f"{prefix}*.txt", f"{prefix}*.TXT", f"{prefix}*.key", f"{prefix}*.KEY"):
                ms = sorted(root.glob(pat))
                if ms:
                    return ms[0]
        return None

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Dict[str, Any]]:
        del split
        for root in self._candidate_label_roots(label_root):
            lf = self._resolve_label_file(root, track_id)
            if lf is None:
                continue
            text = lf.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return {"_skip_track": True, "_skip_reason": "empty_key_label"}
            key_label, is_multi = _parse_giantsteps_key_value(text)
            if is_multi:
                return {"_skip_track": True, "_skip_reason": "multi_key_label"}
            if key_label is None:
                return {"_skip_track": True, "_skip_reason": "invalid_key_label"}
            return {"key_label": key_label}
        return None


register_label_provider(GiantStepsKeyPlusLabelProvider())
register_label_provider(GiantStepsKeyLabelProvider())

