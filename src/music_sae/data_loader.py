#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H5 loader for grouped multi-augmentation training.

Assumes H5 layout (per split):
    /<split>/base_uid    (N,)
    /<split>/semitone    (N,)  int  (aug tag derived as orig/ps±N; legacy `aug`
                                     string column is still accepted as fallback)
    /<split>/trackId     (N,)
    /<split>/<feature_tag>  (N, ...)

Returns:
    {data_tag: Tensor(n_saes, T, D)}
"""

from __future__ import annotations

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from pathlib import Path
from typing import List, Sequence, Tuple


def _decode_str_array(arr: np.ndarray) -> np.ndarray:
    if len(arr) > 0 and isinstance(arr[0], (bytes, np.bytes_)):
        return np.array([x.decode("utf-8") for x in arr], dtype=object)
    return arr


def _semitone_to_aug(semi: int) -> str:
    """Reconstruct the aug tag from the numeric semitone field."""
    return "orig" if int(semi) == 0 else f"ps{int(semi):+d}"


def _read_aug_tags(grp: h5py.Group) -> np.ndarray:
    """Per-row aug tags. Prefer numeric `semitone` (current layout); fall back to
    a stored `aug` string column (legacy layout)."""
    if "semitone" in grp:
        return np.array([_semitone_to_aug(s) for s in grp["semitone"][:]], dtype=object)
    if "aug" in grp:
        return _decode_str_array(grp["aug"][:])
    raise KeyError("group has neither 'semitone' nor 'aug' to derive augmentation tags.")


def _read_track_id_set(path: str) -> set:
    ids: set = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            tid = line.strip()
            if tid:
                ids.add(tid)
    return ids


class H5MelDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Sequence[str],
        split: str,
        data_tag: str,
        aug: bool = False,
        feature_tag: str = "feature",
        aug_order: List[str] | None = None,
        n_saes: int | None = None,
        max_songs: int | None = None,
        song_seed: int = 0,
        track_id_file: str | None = None,
    ) -> None:
        if isinstance(h5_path, (list, tuple)):
            self.h5_paths = [str(p) for p in h5_path]
        else:
            self.h5_paths = [str(h5_path)]
        if len(self.h5_paths) == 0:
            raise ValueError("h5_path is empty.")
        self.split = split
        self.feature_tag = feature_tag
        self._h5s: List[h5py.File | None] = [None for _ in self.h5_paths]

        self.data_tag = data_tag
        self.group_keys: List[str] = []
        self.group_indices: List[Tuple[int, List[int]]] = []
        self.group_song_ids: List[str] = []

        if aug:
            _aug_order = aug_order if aug_order is not None else ["ps-2", "ps-1", "orig", "ps+1", "ps+2"]
        else:
            _aug_order = ["orig"]

        if not aug:
            self._window = 1
        elif n_saes is not None:
            self._window = n_saes
        else:
            self._window = len(_aug_order) - 1

        aug_to_pos = {a: i for i, a in enumerate(_aug_order)}

        for file_idx, hp in enumerate(self.h5_paths):
            with h5py.File(hp, "r") as f:
                grp = f[split]
                for required in ("base_uid", self.feature_tag):
                    if required not in grp:
                        raise KeyError(f"File '{hp}' group '{split}' missing '{required}'.")

                base_uids = _decode_str_array(grp["base_uid"][:])
                augs = _read_aug_tags(grp)

                if "trackId" in grp:
                    track_ids = _decode_str_array(grp["trackId"][:])
                else:
                    track_ids = np.array(
                        ["_".join(str(b).split("_")[:2]) for b in base_uids], dtype=object
                    )

                buid_to_track: dict[str, str] = {}
                groups: dict[str, list[int]] = {}
                for i, (buid, aug_tag, tid) in enumerate(zip(base_uids, augs, track_ids)):
                    if aug_tag not in aug_to_pos:
                        continue
                    slot = aug_to_pos[aug_tag]
                    if buid not in groups:
                        groups[buid] = [-1] * len(aug_to_pos)
                        buid_to_track[buid] = str(tid)
                    groups[buid][slot] = i

                added = 0
                file_tag = Path(hp).name
                for buid, idxs in groups.items():
                    if all(j >= 0 for j in idxs):
                        self.group_keys.append(f"{file_tag}:{buid}")
                        self.group_indices.append((file_idx, idxs))
                        self.group_song_ids.append(buid_to_track[buid])
                        added += 1
                print(f"[H5MelDataset] {hp} complete_groups={added}")

        if len(self.group_indices) == 0:
            raise RuntimeError("No complete groups found across all H5 files.")

        if track_id_file is not None:
            allowlist = set()
            with open(track_id_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    tid = line.strip()
                    if tid:
                        allowlist.add(tid)
            kept = [
                (gi, gk, gs)
                for gi, gk, gs in zip(self.group_indices, self.group_keys, self.group_song_ids)
                if gs in allowlist
            ]
            if not kept:
                raise RuntimeError(f"track_id_file '{track_id_file}' matched no groups.")
            self.group_indices, self.group_keys, self.group_song_ids = (list(t) for t in zip(*kept))
            matched = len({gs for gs in self.group_song_ids})
            print(f"[H5MelDataset] track_id_file='{track_id_file}' → {matched} songs, {len(self.group_indices)} groups retained")

        if max_songs is not None:
            seen: dict[str, None] = {}
            for s in self.group_song_ids:
                seen[s] = None
            unique_songs = list(seen.keys())
            rng = np.random.default_rng(song_seed)
            rng.shuffle(unique_songs)
            selected = set(unique_songs[:max_songs])
            kept = [
                (gi, gk, gs)
                for gi, gk, gs in zip(self.group_indices, self.group_keys, self.group_song_ids)
                if gs in selected
            ]
            if not kept:
                raise RuntimeError("max_songs filter removed all groups.")
            self.group_indices, self.group_keys, self.group_song_ids = (list(t) for t in zip(*kept))
            print(f"[H5MelDataset] max_songs={max_songs} → {len(selected)} songs, {len(self.group_indices)} groups retained")

        file_idx0, local_idxs0 = self.group_indices[0]
        f0 = h5py.File(self.h5_paths[file_idx0], "r")
        try:
            self.data_shape = f0[self.split][self.feature_tag][int(local_idxs0[0])].shape
        finally:
            f0.close()

    def _ensure_open(self, file_idx: int) -> h5py.File:
        if self._h5s[file_idx] is None:
            self._h5s[file_idx] = h5py.File(self.h5_paths[file_idx], "r")
        return self._h5s[file_idx]

    def __len__(self) -> int:
        return len(self.group_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, idxs = self.group_indices[idx]
        f = self._ensure_open(file_idx)
        g = f[self.split]

        if len(idxs) == 1:
            sid, eid = 0, 1
        else:
            max_sid = max(0, len(idxs) - self._window)
            sid = np.random.randint(0, max_sid + 1)
            eid = sid + self._window

        xs = [np.array(g[self.feature_tag][j], copy=True) for j in idxs[sid:eid]]
        return {self.data_tag: torch.from_numpy(np.stack(xs, axis=0))}

    def __del__(self):
        for fh in self._h5s:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass


class MelH5DataModule(pl.LightningDataModule):
    """DataModule with song-level train/val split."""

    def __init__(
        self,
        h5_path: str | Sequence[str],
        data_tag: str,
        split: str = "train",
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        feature_tag: str = "feature",
        aug: bool = False,
        aug_order: List[str] | None = None,
        n_saes: int | None = None,
        persistent_workers: bool | None = None,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_songs: int | None = None,
        song_seed: int = 0,
        track_id_file: str | None = None,
        train_track_id_file: str | None = None,
        val_track_id_file: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def setup(self, stage: str | None = None) -> None:
        full_ds = H5MelDataset(
            h5_path=self.hparams.h5_path,
            split=self.hparams.split,
            data_tag=self.hparams.data_tag,
            feature_tag=self.hparams.feature_tag,
            aug=self.hparams.aug,
            aug_order=self.hparams.aug_order,
            n_saes=self.hparams.n_saes,
            max_songs=self.hparams.max_songs,
            song_seed=self.hparams.song_seed,
            track_id_file=self.hparams.track_id_file,
        )

        # ── Explicit train/val track-id lists (no random split) ──────────────
        if self.hparams.train_track_id_file is not None and self.hparams.val_track_id_file is not None:
            train_songs = _read_track_id_set(self.hparams.train_track_id_file)
            val_songs = _read_track_id_set(self.hparams.val_track_id_file)
            train_idxs = [i for i, s in enumerate(full_ds.group_song_ids) if s in train_songs]
            val_idxs = [i for i, s in enumerate(full_ds.group_song_ids) if s in val_songs]
            if not train_idxs:
                raise RuntimeError(
                    f"train_track_id_file '{self.hparams.train_track_id_file}' matched no groups."
                )
            if not val_idxs:
                raise RuntimeError(
                    f"val_track_id_file '{self.hparams.val_track_id_file}' matched no groups."
                )
            print(
                f"[MelH5DataModule] explicit lists | songs train={len(train_songs)} "
                f"val={len(val_songs)} | groups train={len(train_idxs)} val={len(val_idxs)}"
            )
            self.ds_train = torch.utils.data.Subset(full_ds, train_idxs)
            self.ds_val = torch.utils.data.Subset(full_ds, val_idxs)
            self.ds_test = self.ds_val
            return

        seen: dict[str, None] = {}
        for s in full_ds.group_song_ids:
            seen[s] = None
        unique_songs = list(seen.keys())

        if len(unique_songs) < 2:
            raise ValueError(f"Need at least 2 songs to split train/val, got {len(unique_songs)}.")

        rng = np.random.default_rng(self.hparams.seed)
        perm = rng.permutation(len(unique_songs))
        unique_songs = [unique_songs[i] for i in perm]

        n_val_songs = max(1, int(len(unique_songs) * self.hparams.val_ratio))
        val_songs = set(unique_songs[:n_val_songs])
        train_songs = set(unique_songs[n_val_songs:])

        train_idxs = [i for i, s in enumerate(full_ds.group_song_ids) if s in train_songs]
        val_idxs = [i for i, s in enumerate(full_ds.group_song_ids) if s in val_songs]

        print(
            f"[MelH5DataModule] songs train={len(train_songs)} val={len(val_songs)} | "
            f"groups train={len(train_idxs)} val={len(val_idxs)}"
        )

        self.ds_train = torch.utils.data.Subset(full_ds, train_idxs)
        self.ds_val = torch.utils.data.Subset(full_ds, val_idxs)
        self.ds_test = self.ds_val

    def _persistent_workers_flag(self) -> bool:
        return (
            self.hparams.persistent_workers
            if self.hparams.persistent_workers is not None
            else (self.hparams.num_workers > 0)
        )

    def _common_loader_kwargs(self) -> dict:
        return dict(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self._persistent_workers_flag(),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(self.ds_train, shuffle=self.hparams.shuffle, drop_last=True, **self._common_loader_kwargs())

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return DataLoader(self.ds_val, shuffle=False, drop_last=False, **self._common_loader_kwargs())

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None
        return DataLoader(self.ds_test, shuffle=False, drop_last=False, **self._common_loader_kwargs())
