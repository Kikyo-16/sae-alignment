#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H5 data loader for linear-probe training (frame-level, per-segment).

The feature-extraction pipeline stores every label **un-transposed** (the
original-track value) for all pitch-shift aug copies of a segment, together with
a per-segment ``semitone`` field recording the shift.  This loader therefore
applies the pitch transposition **online** at load time, uniformly across all
label types (they are all pitch-class based).

H5 layout (per split, produced by process_datasets.py):
    /<split>/<feature_tag>      (N, D, T)   float16/32  – features (D, T) per segment
    /<split>/trackId            (N,)         str         – song-level split key
    /<split>/semitone           (N,)         int16       – pitch shift of this segment
    /<split>/labels/<tag>       (N, T)       int8 | str  – un-transposed labels

Tasks (see ``TASK_*``):
    key_relative : frame degree labels (tonic/forth/fifth_frame), 12 pc + 12=N
    chord        : chord_frame → int25 (0-11 maj, 12-23 min, 24=N, -100=ignore)
    key_strict   : key_frame   → one global 0-23 label per segment (12 maj + 12 min)

Each item: {feature_tag: Tensor(T, D), "label": Tensor}
  - key_relative / chord : label is (T,) int64 frame labels
  - key_strict           : label is () int64 scalar (segment key class)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

# ── Task identifiers ────────────────────────────────────────────────────────
TASK_KEY_RELATIVE = "key_relative"
TASK_CHORD = "chord"
TASK_KEY_STRICT = "key_strict"
ALL_TASKS = (TASK_KEY_RELATIVE, TASK_CHORD, TASK_KEY_STRICT)

DEGREE_LABEL_TAGS = ("tonic_frame", "forth_frame", "fifth_frame")

# ── Label encoding constants ────────────────────────────────────────────────
DEGREE_N = 12          # scale-degree sentinel: "not this degree" / no key
CHORD_N = 24           # chord int25 class for explicit no-chord (N / X)
CHORD_IGNORE = -100    # chord int25 sentinel for frames excluded from loss
KEY_INVALID = -100     # key_strict sentinel for unusable segments


def _decode_str_array(arr: np.ndarray) -> np.ndarray:
    if len(arr) > 0 and isinstance(arr[0], (bytes, np.bytes_)):
        return np.array([x.decode("utf-8") for x in arr], dtype=object)
    return arr


_NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Majmin folding rules mirrored from evaluate_chord.py (mir_eval.chord.majmin):
# only chords whose triad core (pitch classes within semitones 0-7) is exactly a
# major or minor triad are assigned a maj/min class.  6th/7th extensions fold to
# maj/min; dim, aug, sus, power and added-lower-tone extensions (9/11/13/add) are
# NOT maj/min triads and are EXCLUDED from the loss (CHORD_IGNORE).
_MAJ_LIKE = frozenset({"maj", "7", "maj7", "6", "maj6"})
_MIN_LIKE = frozenset({"min", "m", "min7", "m7", "min6", "m6", "minmaj7", "mmaj7"})


def _root_pc(token: str) -> Optional[int]:
    pc = _NOTE_TO_PC.get(token)
    if pc is None and len(token) >= 1:
        pc = _NOTE_TO_PC.get(token[0].upper() + token[1:])
    return pc


def _chord_str_to_int25(labels: np.ndarray) -> np.ndarray:
    """Convert string chord-frame labels to the int25 chord encoding.

    0–11  = C_maj … B_maj  (maj-like qualities, after majmin folding)
    12–23 = C_min … B_min  (min-like qualities)
    24    = N / X          (explicit no-chord, a trained class)
    -100  = non-maj/min triad (dim/aug/sus/9/11/13/add…) → excluded from loss
    """
    out = np.full(len(labels), CHORD_IGNORE, dtype=np.int64)
    for i, lbl in enumerate(labels):
        s = lbl.decode("utf-8") if isinstance(lbl, (bytes, np.bytes_)) else str(lbl)
        s = s.strip()
        if not s or s in ("N", "X", "n", "x"):
            out[i] = CHORD_N
            continue
        parts = s.split(":")
        if len(parts) < 2:
            continue
        root_pc = _root_pc(parts[0])
        if root_pc is None:
            continue
        quality = parts[1].split("/")[0].lower()  # strip inversion (e.g. maj/3 → maj)
        if quality in _MAJ_LIKE:
            out[i] = root_pc
        elif quality in _MIN_LIKE:
            out[i] = root_pc + 12
        # else: non-maj/min triad → keep CHORD_IGNORE
    return out


def _key_str_to_int24_global(key_frame: np.ndarray) -> int:
    """Collapse a segment's per-frame key labels to one 0-23 class.

    0–11 = major (root pc), 12–23 = minor (root pc).  Returns KEY_INVALID when the
    segment has no valid key, or when the key changes within the segment (those
    segments are dropped from the strict-key dataset).
    """
    seen: set[int] = set()
    for lbl in key_frame:
        s = lbl.decode("utf-8") if isinstance(lbl, (bytes, np.bytes_)) else str(lbl)
        s = s.strip()
        if not s or s in ("N", "X", "n", "x"):
            continue
        parts = s.split(":")
        root_pc = _root_pc(parts[0])
        if root_pc is None:
            continue
        is_min = len(parts) > 1 and parts[1].strip().lower().startswith("min")
        seen.add(root_pc + (12 if is_min else 0))
    if len(seen) != 1:
        return KEY_INVALID
    return next(iter(seen))


def _transpose_degree(lbl: np.ndarray, semi: int) -> np.ndarray:
    """Transpose frame degree labels (int, 0-11 pc, 12=N) by ``semi`` semitones."""
    if semi % 12 == 0:
        return lbl
    out = lbl.copy()
    m = out < DEGREE_N
    out[m] = ((out[m].astype(np.int64) + semi) % 12)
    return out


def _transpose_chord25(lbl: np.ndarray, semi: int) -> np.ndarray:
    """Transpose int25 chord labels by ``semi`` semitones (N / ignore unchanged)."""
    if semi % 12 == 0:
        return lbl
    out = lbl.copy()
    maj = (out >= 0) & (out < 12)
    mn = (out >= 12) & (out < 24)
    out[maj] = (out[maj] + semi) % 12
    out[mn] = 12 + ((out[mn] - 12 + semi) % 12)
    return out


def _transpose_key24(label: int, semi: int) -> int:
    """Transpose a 0-23 key class by ``semi`` semitones (KEY_INVALID unchanged)."""
    if label == KEY_INVALID or semi % 12 == 0:
        return label
    root = label % 12
    mode = label - root  # 0 or 12
    return ((root + semi) % 12) + mode


def _read_track_id_set(path: str) -> set:
    ids: set = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            tid = line.strip()
            if tid:
                ids.add(tid)
    return ids


class ProbeH5Dataset(Dataset):
    """Flat per-segment dataset with online pitch transposition.

    For ``key_strict`` the segment-level label is precomputed at build time so
    that segments with no key / a within-segment key change are dropped.
    """

    def __init__(
        self,
        h5_paths: Sequence[str],
        split: str,
        feature_tag: str,
        task: str,
        label_tag: str = "tonic_frame",
    ) -> None:
        if task not in ALL_TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {ALL_TASKS}")
        self.h5_paths = [str(p) for p in h5_paths]
        self.split = split
        self.feature_tag = feature_tag
        self.task = task
        self.label_tag = label_tag
        self._h5s: List[Optional[h5py.File]] = [None] * len(self.h5_paths)

        # Resolve which H5 dataset holds the label for this task.
        if task == TASK_CHORD:
            h5_label_tag = "chord_frame"
        elif task == TASK_KEY_STRICT:
            h5_label_tag = "key_frame"
        else:
            h5_label_tag = label_tag
        self._h5_label_tag = h5_label_tag

        all_indices: List[Tuple[int, int]] = []
        all_song_ids: List[str] = []
        # Precomputed scalar key labels for key_strict (None for other tasks).
        all_key_labels: List[int] = []
        self._label_paths: List[Tuple[str, ...]] = []
        self._has_semitone: List[bool] = []

        for file_idx, hp in enumerate(self.h5_paths):
            with h5py.File(hp, "r") as f:
                grp = f[split]
                if feature_tag not in grp:
                    raise KeyError(f"'{hp}'['{split}'] missing feature '{feature_tag}'")

                if "labels" in grp and h5_label_tag in grp["labels"]:
                    label_path = ("labels", h5_label_tag)
                elif h5_label_tag in grp:
                    label_path = (h5_label_tag,)
                else:
                    raise KeyError(
                        f"'{hp}'['{split}'] missing label '{h5_label_tag}'. "
                        f"Expected at 'labels/{h5_label_tag}' or '{h5_label_tag}'. "
                        f"Re-run feature extraction with --inference --label-root."
                    )
                self._label_paths.append(label_path)
                self._has_semitone.append("semitone" in grp)

                N = grp[feature_tag].shape[0]
                track_ids = (
                    _decode_str_array(grp["trackId"][:]) if "trackId" in grp
                    else np.array([f"track_{i}" for i in range(N)], dtype=object)
                )

                # For key_strict, collapse each segment's key_frame to one class
                # and drop unusable segments up front.
                key_labels_file = None
                if task == TASK_KEY_STRICT:
                    lbl_ds = grp
                    for key in label_path:
                        lbl_ds = lbl_ds[key]
                    key_labels_file = [
                        _key_str_to_int24_global(np.array(lbl_ds[i])) for i in range(N)
                    ]

                for i in range(N):
                    if task == TASK_KEY_STRICT and key_labels_file[i] == KEY_INVALID:
                        continue
                    all_indices.append((file_idx, i))
                    all_song_ids.append(str(track_ids[i]))
                    if task == TASK_KEY_STRICT:
                        all_key_labels.append(int(key_labels_file[i]))

            print(f"[ProbeH5Dataset] {hp} n_segments={N} kept={len(all_indices)}")

        self.indices = all_indices
        self.song_ids = all_song_ids
        self.key_labels = all_key_labels if task == TASK_KEY_STRICT else []

        fi, li = self.indices[0]
        with h5py.File(self.h5_paths[fi], "r") as f:
            seg_shape = f[self.split][self.feature_tag][li].shape
        self.d_model: int = int(seg_shape[0])
        self.T: int = int(seg_shape[1])

    def _ensure_open(self, file_idx: int) -> h5py.File:
        if self._h5s[file_idx] is None:
            self._h5s[file_idx] = h5py.File(self.h5_paths[file_idx], "r")
        return self._h5s[file_idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        file_idx, local_idx = self.indices[idx]
        f = self._ensure_open(file_idx)
        g = f[self.split]

        feat = np.array(g[self.feature_tag][local_idx], copy=True)   # (D, T)
        feat_t = torch.from_numpy(feat).float().T                    # (T, D)

        semi = int(g["semitone"][local_idx]) if self._has_semitone[file_idx] else 0

        if self.task == TASK_KEY_STRICT:
            label = _transpose_key24(self.key_labels[idx], semi)
            label_t = torch.tensor(label, dtype=torch.int64)
            return {self.feature_tag: feat_t, "label": label_t}

        lbl_ds = g[self._label_paths[file_idx][0]]
        for key in self._label_paths[file_idx][1:]:
            lbl_ds = lbl_ds[key]
        raw = np.array(lbl_ds[local_idx])  # (T,)

        if self.task == TASK_CHORD:
            label = _transpose_chord25(_chord_str_to_int25(raw), semi)
        else:  # key_relative
            label = _transpose_degree(raw.astype(np.int64), semi)
        return {self.feature_tag: feat_t, "label": torch.from_numpy(label)}

    def __del__(self) -> None:
        for fh in self._h5s:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass


class ProbeH5DataModule(pl.LightningDataModule):
    """Song-level train/val split for linear-probe training.

    When both ``train_track_id_file`` and ``val_track_id_file`` are given the
    split follows those lists (matching the music_sae data loader); otherwise a
    random song-level split with ``val_ratio`` is used.  The validation subset is
    only used for lightweight frame-loss logging — the checkpoint-selection
    metric is computed by a ValMetric callback that reads the eval H5 per-song.
    """

    def __init__(
        self,
        h5_paths: Sequence[str],
        split: str,
        feature_tag: str,
        task: str,
        label_tag: str = "tonic_frame",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_ratio: float = 0.1,
        seed: int = 42,
        persistent_workers: Optional[bool] = None,
        train_track_id_file: Optional[str] = None,
        val_track_id_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.d_model: Optional[int] = None

    def setup(self, stage: Optional[str] = None) -> None:
        full_ds = ProbeH5Dataset(
            h5_paths=self.hparams.h5_paths,
            split=self.hparams.split,
            feature_tag=self.hparams.feature_tag,
            task=self.hparams.task,
            label_tag=self.hparams.label_tag,
        )
        self.d_model = full_ds.d_model

        hp = self.hparams
        if hp.train_track_id_file is not None and hp.val_track_id_file is not None:
            train_songs = _read_track_id_set(hp.train_track_id_file)
            val_songs = _read_track_id_set(hp.val_track_id_file)
            train_idxs = [i for i, s in enumerate(full_ds.song_ids) if s in train_songs]
            val_idxs = [i for i, s in enumerate(full_ds.song_ids) if s in val_songs]
            if not train_idxs:
                raise RuntimeError(f"train_track_id_file '{hp.train_track_id_file}' matched no segments.")
            if not val_idxs:
                raise RuntimeError(f"val_track_id_file '{hp.val_track_id_file}' matched no segments.")
            print(
                f"[ProbeH5DataModule] explicit lists | songs train={len(train_songs)} "
                f"val={len(val_songs)} | segments train={len(train_idxs)} val={len(val_idxs)}"
            )
        else:
            seen: dict[str, None] = {}
            for s in full_ds.song_ids:
                seen[s] = None
            unique_songs = list(seen.keys())
            if len(unique_songs) < 2:
                raise ValueError(f"Need ≥2 songs for train/val split, got {len(unique_songs)}.")
            rng = np.random.default_rng(hp.seed)
            unique_songs = [unique_songs[i] for i in rng.permutation(len(unique_songs))]
            n_val = max(1, int(len(unique_songs) * hp.val_ratio))
            val_songs = set(unique_songs[:n_val])
            train_songs = set(unique_songs[n_val:])
            train_idxs = [i for i, s in enumerate(full_ds.song_ids) if s in train_songs]
            val_idxs = [i for i, s in enumerate(full_ds.song_ids) if s in val_songs]
            print(
                f"[ProbeH5DataModule] songs train={len(train_songs)} val={len(val_songs)} | "
                f"segments train={len(train_idxs)} val={len(val_idxs)}"
            )

        self.ds_train = torch.utils.data.Subset(full_ds, train_idxs)
        self.ds_val = torch.utils.data.Subset(full_ds, val_idxs)

    def _persistent_workers_flag(self) -> bool:
        if self.hparams.persistent_workers is not None:
            return self.hparams.persistent_workers
        return self.hparams.num_workers > 0

    def _loader_kwargs(self) -> dict:
        return dict(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self._persistent_workers_flag(),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(self.ds_train, shuffle=True, drop_last=True, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return DataLoader(self.ds_val, shuffle=False, drop_last=False, **self._loader_kwargs())
