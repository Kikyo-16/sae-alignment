from __future__ import annotations

"""Feature-space utilities shared across extractors."""

import torch


def canonicalize_feat_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to (C, T).
    Accepts:
      (C,T) | (1,C,T) | (B,C,T) -> pick first
      (T,C) | (1,T,C) | (B,T,C) -> transpose to (C,T)
    """
    if x.dim() < 2:
        raise RuntimeError(f"Unexpected feat dim={x.dim()}, need at least 2.")

    if x.dim() == 3:
        x = x[0]

    a, b = x.shape
    if b in (64, 128, 256, 512, 768, 1024, 1280) and a not in (64, 128, 256, 512, 768, 1024, 1280):
        x = x.transpose(0, 1)
    return x.contiguous()
