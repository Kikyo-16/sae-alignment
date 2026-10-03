from __future__ import annotations

"""
Small helpers for working with HDF5 datasets in a streaming/append-only way.
"""

from typing import Any

import h5py
import numpy as np


def ensure_group(h5: h5py.File, split: str) -> h5py.Group:
    return h5.require_group(split)


def create_resizable_string_ds(grp: h5py.Group, name: str) -> h5py.Dataset:
    dt = h5py.string_dtype(encoding="utf-8")
    return grp.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=True)


def append_resize(ds: h5py.Dataset, add_n: int) -> slice:
    """
    Increase first dimension of dataset by add_n and return a slice for writing.
    Works for datasets with shape (N, ...) or (N,).
    """
    n0 = ds.shape[0]
    ds.resize((n0 + add_n, *ds.shape[1:]))
    return slice(n0, n0 + add_n)

