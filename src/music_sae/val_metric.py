from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytorch_lightning as pl


@runtime_checkable
class ValMetric(Protocol):
    """Protocol for pluggable validation metrics consumed by TopKEvalCallback."""

    @property
    def name(self) -> str:
        """Metric key used for logging, e.g. 'val/probe_key_forth_frame_acc'."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """True → save when score is larger; False → save when score is smaller."""
        ...

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        """Return the scalar metric for the current model state."""
        ...
