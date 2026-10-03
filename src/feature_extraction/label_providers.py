"""
Label-provider framework for inference-mode labels (chord, key, pitch, etc.).

When --inference and --label-root are set, the pipeline loads per-track labels
and computes segment-level labels (chord_frame, pitch_roll, ...) aligned to each
feature segment.

This module holds only the *framework*: the LabelProvider protocol, the
registry, and shared parsing/encoding helpers. Concrete providers live in the
matching dataset module under datasets/ (e.g. datasets/slakh.py registers both
the Slakh DatasetHandler and its SlakhLabelProvider), so adding a new dataset
with labels only touches one file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

_LABEL_REGISTRY: Dict[str, "LabelProvider"] = {}

# ---------------------------
# Pitch class helpers for scale-degree labels
# ---------------------------

_NOTE_TO_PC: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
_DEGREE_N = np.int8(12)  # sentinel: "not this degree" or no valid chord/key


def _parse_root_pc(label: str) -> int:
    """Return pitch class 0-11 from a chord/key label like 'C:maj7' or 'F#:min'.
    Returns -1 if the label is 'N', empty, or unparseable."""
    s = str(label).strip()
    if not s or s in ("N", "X", "n"):
        return -1
    token = s.split(":")[0]
    pc = _NOTE_TO_PC.get(token)
    if pc is None and len(token) >= 1:
        pc = _NOTE_TO_PC.get(token[0].upper() + token[1:])
    return pc if pc is not None else -1


def _parse_key_pc_major(label: str) -> int:
    """Return the relative-major root pitch class from a key label.
    Minor keys (e.g. 'A:min') are converted to their relative major by +3 semitones.
    Major keys are returned as-is. Returns -1 if unparseable."""
    s = str(label).strip()
    if not s or s in ("N", "X", "n"):
        return -1
    parts = s.split(":")
    token = parts[0]
    pc = _NOTE_TO_PC.get(token)
    if pc is None and len(token) >= 1:
        pc = _NOTE_TO_PC.get(token[0].upper() + token[1:])
    if pc is None:
        return -1
    if len(parts) > 1 and parts[1].lower().startswith("min"):
        pc = (pc + 3) % 12
    return pc


def compute_degree_frames(
    chord_frame: np.ndarray,
    key_frame: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Derive tonic/forth/fifth scale-degree labels from per-frame chord and key arrays.

    For each frame t:
      - Parse chord root pitch class c and key root pitch class k.
      - tonic_frame[t] = c  if c == k            else 12 (N)
      - forth_frame[t] = c  if c == (k+5) % 12   else 12 (N)
      - fifth_frame[t] = c  if c == (k+7) % 12   else 12 (N)

    Labels 0-11 are pitch classes (C=0 … B=11); 12 means "not this degree".
    Returns dict of int8 arrays shaped (T,).
    """
    T = len(chord_frame)
    c_pcs = np.array([_parse_root_pc(str(x)) for x in chord_frame], dtype=np.int16)
    k_pcs = np.array([_parse_key_pc_major(str(x)) for x in key_frame], dtype=np.int16)

    valid = (c_pcs >= 0) & (k_pcs >= 0)

    tonic_arr = np.full(T, _DEGREE_N, dtype=np.int8)
    forth_arr  = np.full(T, _DEGREE_N, dtype=np.int8)
    fifth_arr  = np.full(T, _DEGREE_N, dtype=np.int8)

    tonic_arr[valid & (c_pcs == k_pcs)]              = c_pcs[valid & (c_pcs == k_pcs)].astype(np.int8)
    forth_arr[valid & (c_pcs == (k_pcs + 5) % 12)]   = c_pcs[valid & (c_pcs == (k_pcs + 5) % 12)].astype(np.int8)
    fifth_arr[valid & (c_pcs == (k_pcs + 7) % 12)]   = c_pcs[valid & (c_pcs == (k_pcs + 7) % 12)].astype(np.int8)

    return {"tonic_frame": tonic_arr, "forth_frame": forth_arr, "fifth_frame": fifth_arr}


class LabelProvider(Protocol):
    """Interface for dataset-specific label loading and segment computation."""

    dataset_name: str

    def load_for_track(
        self,
        track_id: str,
        label_root: str,
        split: str,
    ) -> Optional[Any]:
        """
        Load label data for a track. Returns opaque object or None if no labels.
        label_root: dataset-specific root (e.g. Pop909 dir with 001/, 002/...)
        """
        ...

    def get_label_keys(self) -> List[str]:
        """Return keys for HDF5 datasets, e.g. ['chord_frame', 'pitch_roll']."""
        ...

    def get_label_shapes_and_dtypes(
        self,
        seg_dur_sec: float,
        label_frames: int,
    ) -> Dict[str, Tuple[Tuple[int, ...], Any]]:
        """Return {key: (shape_per_seg, dtype)} for create_dataset."""
        ...

    def compute_segment_labels(
        self,
        seg_start_sec: float,
        seg_dur_sec: float,
        fps: float,
        T: int,
        label_data: Any,
    ) -> Dict[str, np.ndarray]:
        """Compute per-segment labels. Returns {key: array}."""
        ...


def register_label_provider(provider: LabelProvider) -> None:
    if provider.dataset_name in _LABEL_REGISTRY:
        raise ValueError(f"Label provider for '{provider.dataset_name}' already registered")
    _LABEL_REGISTRY[provider.dataset_name] = provider


def get_label_provider(dataset_name: str) -> Optional[LabelProvider]:
    return _LABEL_REGISTRY.get(dataset_name)


# ---------------------------
# Shared parsing / framing helpers (used by concrete providers in datasets/*)
# ---------------------------


def _parse_time_seg_file(path: Path) -> List[Tuple[float, float, str]]:
    """Parse a "start end label" annotation file into a sorted list of (start, end, label) tuples."""
    segs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            s, e = float(parts[0]), float(parts[1])
            if e > s:
                segs.append((s, e, " ".join(parts[2:])))
    segs.sort(key=lambda x: x[0])
    return segs


def _segs_to_frames(
    segs: List[Tuple[float, float, str]],
    seg_start: float,
    seg_dur: float,
    fps: float,
    T: int,
    default: str = "N",
) -> np.ndarray:
    out = np.array([default] * T, dtype=object)
    seg_end = seg_start + seg_dur
    for s, e, label in segs:
        if e <= seg_start or s >= seg_end:
            continue
        s2, e2 = max(s, seg_start), min(e, seg_end)
        i0 = max(0, min(T, int(np.floor((s2 - seg_start) * fps))))
        i1 = max(0, min(T, int(np.ceil((e2 - seg_start) * fps))))
        if i1 > i0:
            out[i0:i1] = label
    return out


def _pitch_roll_from_midi(
    mid_path: Path,
    seg_start: float,
    seg_dur: float,
    fps: float,
    T: int,
) -> np.ndarray:
    try:
        import pretty_midi
    except ImportError:
        raise RuntimeError("pitch_roll requires pretty_midi: pip install pretty_midi")
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    roll = np.zeros((128, T), dtype=np.uint8)
    seg_end = seg_start + seg_dur
    for inst in pm.instruments:
        for note in inst.notes:
            p = int(note.pitch)
            if p < 0 or p >= 128:
                continue
            ns, ne = float(note.start), float(note.end)
            if ne <= seg_start or ns >= seg_end:
                continue
            ns2 = max(ns, seg_start)
            ne2 = min(ne, seg_end)
            i0 = int(np.floor((ns2 - seg_start) * fps))
            i1 = int(np.ceil((ne2 - seg_start) * fps))
            i0 = max(0, min(T, i0))
            i1 = max(0, min(T, i1))
            if i1 > i0:
                roll[p, i0:i1] = 1
    return roll


def _melody_like_track(name: str) -> bool:
    s = (name or "").strip().lower()
    if not s:
        return False
    keys = ("melody", "vocal", "vocals", "vox", "lead")
    return any(k in s for k in keys)


def _melody_roll_from_midi(
    mid_path: Path,
    seg_start: float,
    seg_dur: float,
    fps: float,
    T: int,
) -> np.ndarray:
    try:
        import pretty_midi
    except ImportError:
        raise RuntimeError("melody_roll requires pretty_midi: pip install pretty_midi")

    pm = pretty_midi.PrettyMIDI(str(mid_path))
    roll = np.zeros((128, T), dtype=np.uint8)
    seg_end = seg_start + seg_dur
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if not _melody_like_track(getattr(inst, "name", "")):
            continue
        for note in inst.notes:
            p = int(note.pitch)
            if p < 0 or p >= 128:
                continue
            ns, ne = float(note.start), float(note.end)
            if ne <= seg_start or ns >= seg_end:
                continue
            ns2 = max(ns, seg_start)
            ne2 = min(ne, seg_end)
            i0 = int(np.floor((ns2 - seg_start) * fps))
            i1 = int(np.ceil((ne2 - seg_start) * fps))
            i0 = max(0, min(T, i0))
            i1 = max(0, min(T, i1))
            if i1 > i0:
                roll[p, i0:i1] = 1
    return roll


# ---------------------------
# Shared base + parsers for single-key-per-track datasets
# ---------------------------


class _KeyLabelProviderBase:
    """Shared implementation for single-key-per-track label providers."""

    def get_label_keys(self) -> List[str]:
        return ["key_frame"]

    def get_label_shapes_and_dtypes(
        self,
        seg_dur_sec: float,
        label_frames: int,
    ) -> Dict[str, Tuple[Tuple[int, ...], Any]]:
        del seg_dur_sec
        return {"key_frame": ((label_frames,), "str")}

    def compute_segment_labels(
        self,
        seg_start_sec: float,
        seg_dur_sec: float,
        fps: float,
        T: int,
        label_data: Any,
    ) -> Dict[str, np.ndarray]:
        del seg_start_sec, seg_dur_sec, fps
        if label_data is None:
            return {"key_frame": np.array(["N"] * T, dtype=object)}
        key_label = str(label_data.get("key_label", "N"))
        return {"key_frame": np.array([key_label] * T, dtype=object)}


def _normalize_key_pitch(token: str) -> Optional[str]:
    t = str(token).strip()
    m = re.match(r"^([A-Ga-g])([#b]?)$", t)
    if not m:
        return None
    letter = m.group(1).upper()
    acc = m.group(2)
    return letter + acc


def _parse_giantsteps_single_key(text: str) -> Tuple[Optional[str], bool]:
    """
    Parse GiantSteps key label:
      - single: "Eb major (desc)" / "C minor (desc)"
      - multi:  "X mode (...) | Y mode (...)"  -> marked as multi
    Returns (normalized_key, is_multi).
    normalized_key format: "<Pitch>:maj|min"
    """
    raw = str(text).strip()
    if not raw:
        return None, False
    if "|" in raw:
        return None, True

    cleaned = re.sub(r"\([^)]*\)", "", raw).strip()
    parts = cleaned.split()
    if len(parts) < 2:
        return None, False
    pitch = _normalize_key_pitch(parts[0])
    if pitch is None:
        return None, False

    mode_raw = parts[1].strip().lower()
    if mode_raw.startswith("maj"):
        mode = "maj"
    elif mode_raw.startswith("min"):
        mode = "min"
    else:
        return None, False
    return f"{pitch}:{mode}", False
