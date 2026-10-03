"""
Central registry for DatasetHandler and RepHandler plugins.

Import this module (not process_datasets) when defining new datasets/reps
to avoid circular imports.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, Tuple

import h5py
import torch


class DatasetHandler(Protocol):
    """Interface for dataset plugins."""

    name: str

    def find_files(self, root: str, split: str, *, mode: str = "stem") -> Iterable[str]:
        ...

    def load_track(
        self,
        path: str,
        target_sr: int,
        *,
        mode: str = "stem",
        musdb_target: str = "mixture",
    ) -> Tuple[torch.Tensor, str, str]:
        """Return (wav, track_id, stem_id). wav: mono (1, N) float32 at target_sr."""
        ...

    # --- Optional hooks (process_datasets reads these via getattr with safe
    #     defaults, so a handler only implements them when it needs non-default
    #     behavior). Each may be a plain attribute or a callable taking `mode`:
    #
    #   wants_long_silence_filter(mode) -> bool   # default False
    #   item_unit(mode)                 -> str    # default "file"  (tqdm unit)
    #   display_name(mode)              -> str    # default == name (progress bar)
    #   track_id_for_path(path, *, mode) -> str   # enables path pre-filtering
    #
    # Datasets with labels should register a LabelProvider in the same module
    # (see datasets/rwc.py) instead of editing label_providers.py.


class RepHandler(Protocol):
    """Interface for feature representation plugins."""

    name: str

    def init_models(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> Any:
        """Load models. config: target_sr, <model>_model_id, <model>_layer."""
        ...

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        """Return (shape, meta)."""
        ...

    def create_datasets(
        self,
        grp: h5py.Group,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        *,
        np_dtype,
        compression: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        """Return dict with 'main' and optionally aux keys (e.g. 'max_value')."""
        ...

    def extract(
        self,
        seg: torch.Tensor,
        *,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return (feat, extra_meta). feat may be None to skip sample."""
        ...


_DATASET_REGISTRY: Dict[str, DatasetHandler] = {}
_REP_REGISTRY: Dict[str, RepHandler] = {}


def register_dataset_handler(handler: DatasetHandler) -> None:
    if handler.name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset handler '{handler.name}' already registered")
    _DATASET_REGISTRY[handler.name] = handler


def get_dataset_handler(name: str) -> DatasetHandler:
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[name]


def register_rep_handler(handler: RepHandler) -> None:
    if handler.name in _REP_REGISTRY:
        raise ValueError(f"Rep handler '{handler.name}' already registered")
    _REP_REGISTRY[handler.name] = handler


def get_rep_handler(name: str) -> RepHandler:
    if name not in _REP_REGISTRY:
        raise ValueError(
            f"Unknown rep '{name}'. Available: {sorted(_REP_REGISTRY.keys())}"
        )
    return _REP_REGISTRY[name]
