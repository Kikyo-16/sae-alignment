from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from ..audio_utils import load_mono, resample_to
from ..label_providers import compute_degree_frames, register_label_provider
from ..registry import DatasetHandler, register_dataset_handler


def _dedup_sorted(paths: List[str]) -> List[str]:
    return sorted(set(paths))


def _find_rwc_audio_files(root: str, split: str) -> List[str]:
    """
    Supports:
      - root/AUDIO/*.wav
      - root/<split>/AUDIO/*.wav
      - recursive variants under AUDIO
    """
    roots: List[str] = []
    split_root = os.path.join(root, split)
    if os.path.isdir(split_root):
        roots.append(split_root)
    roots.append(root)

    files: List[str] = []
    for r in roots:
        audio_dir = os.path.join(r, "AUDIO")
        if not os.path.isdir(audio_dir):
            continue
        files.extend(glob(os.path.join(audio_dir, "*.wav")))
        files.extend(glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    return _dedup_sorted(files)


_NOTE_NAME_MAP = {
    "C": "C", "C#": "C#", "Db": "C#", "D": "D", "D#": "D#", "Eb": "D#",
    "E": "E", "Fb": "E", "F": "F", "F#": "F#", "Gb": "F#", "G": "G",
    "G#": "G#", "Ab": "G#", "A": "A", "A#": "A#", "Bb": "A#", "B": "B", "Cb": "B",
}


def _normalize_rwc_key(label: str) -> Optional[str]:
    """
    Convert RWC key label to '<Pitch>:maj|min'.
    Examples: 'C major' -> 'C:maj', 'Ab minor' -> 'G#:min', 'C:maj' -> 'C:maj'.
    Returns None if the label cannot be parsed.
    """
    import re as _re
    s = str(label).strip()
    if not s or s.upper() in {"N", "N/A", "UNKNOWN"}:
        return None
    # Already normalized.
    m = _re.match(r"^([A-Ga-g][#b]?)\s*:\s*(maj|min)\w*$", s, _re.IGNORECASE)
    if m:
        pitch = _NOTE_NAME_MAP.get(m.group(1).capitalize().replace("B", "b").replace("#", "#"), None)
        # rebuild canonical pitch
        raw_pitch = s.split(":")[0].strip()
        canon = _NOTE_NAME_MAP.get(raw_pitch, None)
        if canon is None:
            return None
        mode = "maj" if m.group(2).lower().startswith("maj") else "min"
        return f"{canon}:{mode}"
    # 'C major' / 'Ab minor' style.
    m2 = _re.match(r"^([A-Ga-g][#b]?)\s+(major|minor|maj|min)\b", s, _re.IGNORECASE)
    if m2:
        raw_pitch = m2.group(1)
        # Capitalize first letter, keep accidental.
        canon = _NOTE_NAME_MAP.get(raw_pitch[0].upper() + raw_pitch[1:], None)
        if canon is None:
            return None
        mode = "maj" if m2.group(2).lower().startswith("maj") else "min"
        return f"{canon}:{mode}"
    return None


def _parse_chord_lab(path: Path) -> List[Tuple[float, float, str]]:
    segs: List[Tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                s = float(parts[0])
                e = float(parts[1])
            except Exception:
                continue
            c = " ".join(parts[2:])
            if e > s:
                segs.append((s, e, c))
    segs.sort(key=lambda x: x[0])
    return segs


def _parse_key_lab(path: Path) -> List[Tuple[float, float, str]]:
    """Parse RWC key annotation .lab file (same format as chord lab)."""
    segs: List[Tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                s = float(parts[0])
                e = float(parts[1])
            except Exception:
                continue
            raw_label = " ".join(parts[2:])
            key = _normalize_rwc_key(raw_label)
            if key is not None and e > s:
                segs.append((s, e, key))
    segs.sort(key=lambda x: x[0])
    return segs


def _chord_frame_labels(
    chord_segs: List[Tuple[float, float, str]],
    seg_start: float,
    seg_dur: float,
    fps: float,
    t_frames: int,
    default: str = "N",
) -> np.ndarray:
    out = np.array([default] * t_frames, dtype=object)
    seg_end = seg_start + seg_dur
    for s, e, c in chord_segs:
        if e <= seg_start or s >= seg_end:
            continue
        s2 = max(s, seg_start)
        e2 = min(e, seg_end)
        i0 = int(np.floor((s2 - seg_start) * fps))
        i1 = int(np.ceil((e2 - seg_start) * fps))
        i0 = max(0, min(t_frames, i0))
        i1 = max(0, min(t_frames, i1))
        if i1 > i0:
            out[i0:i1] = c
    return out


class RWCDatasetHandler(DatasetHandler):
    """
    RWC audio handler.
    Track id is wav filename stem, e.g. AUDIO/000.wav -> track_id='000'.
    """

    name = "rwc"

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        del mode
        return _find_rwc_audio_files(root, split)

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        del mode, musdb_target
        wav, sr0 = load_mono(path)
        wav = resample_to(wav, sr0, target_sr).to(torch.float32)
        track_id = os.path.splitext(os.path.basename(path))[0]
        stem_id = "mix"
        return wav, track_id, stem_id


class RWCChordLabelProvider:
    """
    RWC chord labels from .lab files only (ignore .svl).
    Mapping: audio 000 -> chord index 001 (offset +1).
    """

    dataset_name = "rwc"

    @staticmethod
    def _candidate_chord_roots(label_root: str) -> List[Path]:
        root = Path(label_root)
        cands = [root]
        for subdir in ("chords", "RWC_Pop_Chords", "chord"):
            if (root / subdir).exists():
                cands.append(root / subdir)
        out: List[Path] = []
        seen = set()
        for p in cands:
            s = str(p.resolve())
            if s not in seen:
                seen.add(s)
                out.append(p)
        return out

    @staticmethod
    def _candidate_key_roots(label_root: str) -> List[Path]:
        root = Path(label_root)
        cands: List[Path] = []
        for subdir in ("rwc_keys", "keys", "key", "RWC_Pop_Keys"):
            if (root / subdir).exists():
                cands.append(root / subdir)
        out: List[Path] = []
        seen = set()
        for p in cands:
            s = str(p.resolve())
            if s not in seen:
                seen.add(s)
                out.append(p)
        return out

    @staticmethod
    def _resolve_key_lab_file(root: Path, track_id: str) -> Optional[Path]:
        """
        Resolve key .lab file for the given track_id.
        Key files follow the RWC naming convention: RM-P001.lab, RM-P002.lab, ...

        Tries (in order):
          1. Direct match:              {track_id}.lab          (e.g. RM-P001.lab)
          2. RM-P with +1 offset:       RM-P{int(tid)+1:03d}.lab  (audio 000 -> RM-P001)
          3. RM-P with no offset:       RM-P{int(tid):03d}.lab    (audio 001 -> RM-P001)
        """
        tid = str(track_id).strip()

        # 1. Direct match (e.g. track_id is already "RM-P001")
        direct = root / f"{tid}.lab"
        if direct.exists():
            return direct

        # 2 & 3. Numeric track_id -> RM-P{N:03d}.lab
        try:
            n = int(tid)
            for idx in (n + 1, n):  # +1 offset first (matches chord convention)
                p = root / f"RM-P{idx:03d}.lab"
                if p.exists():
                    return p
        except ValueError:
            pass

        # 4. Glob fallback for any RM-P*.lab if only one file present
        matches = sorted(root.glob("RM-P*.lab"))
        if len(matches) == 1:
            return matches[0]

        return None

    @staticmethod
    def _resolve_lab_file(root: Path, track_id: str) -> Optional[Path]:
        tid = str(track_id).strip()
        # Main mapping required by user: audio 000 -> chord 001.
        try:
            idx = int(tid) + 1
            pat = f"N{idx:03d}-*.lab"
            ms = sorted(root.glob(pat))
            if ms:
                return ms[0]
        except Exception:
            pass

        # Fallback: if track id already equals chord index.
        try:
            idx2 = int(tid)
            pat2 = f"N{idx2:03d}-*.lab"
            ms2 = sorted(root.glob(pat2))
            if ms2:
                return ms2[0]
        except Exception:
            pass
        return None

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Dict[str, Any]]:
        del split
        chord_segs: List[Tuple[float, float, str]] = []
        for root in self._candidate_chord_roots(label_root):
            lab = self._resolve_lab_file(root, track_id)
            if lab is None:
                continue
            segs = _parse_chord_lab(lab)
            if not segs:
                return {"_skip_track": True, "_skip_reason": "empty_chord_lab"}
            chord_segs = segs
            break
        if not chord_segs:
            return None

        key_segs: List[Tuple[float, float, str]] = []
        for root in self._candidate_key_roots(label_root):
            lab = self._resolve_key_lab_file(root, track_id)
            if lab is None:
                continue
            key_segs = _parse_key_lab(lab)
            break  # first match wins; empty key_segs is fine (fallback to "N")

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
        _n = np.int8(12)
        if label_data is None:
            return {
                "chord_frame": np.array(["N"] * T, dtype=object),
                "key_frame":   np.array(["N"] * T, dtype=object),
                "tonic_frame": np.full(T, _n, dtype=np.int8),
                "forth_frame": np.full(T, _n, dtype=np.int8),
                "fifth_frame": np.full(T, _n, dtype=np.int8),
            }
        chord_segs = label_data.get("chord_segs", [])
        key_segs = label_data.get("key_segs", [])
        chord_frame = _chord_frame_labels(
            chord_segs=chord_segs,
            seg_start=seg_start_sec,
            seg_dur=seg_dur_sec,
            fps=fps,
            t_frames=T,
            default="N",
        )
        key_frame = _chord_frame_labels(
            chord_segs=key_segs,
            seg_start=seg_start_sec,
            seg_dur=seg_dur_sec,
            fps=fps,
            t_frames=T,
            default="N",
        )
        return {
            "chord_frame": chord_frame,
            "key_frame":   key_frame,
            **compute_degree_frames(chord_frame, key_frame),
        }


register_dataset_handler(RWCDatasetHandler())
register_label_provider(RWCChordLabelProvider())

