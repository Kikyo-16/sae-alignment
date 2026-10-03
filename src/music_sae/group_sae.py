from __future__ import annotations

import torch.nn as nn

from .sae import MultiSAE


class GroupSAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: int = 4,
        topk: int = 32,
        n_saes: int = 1,
        l1_coeff: float = 1e-4,
    ):
        super().__init__()
        self.sae = MultiSAE(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            topk=topk,
            n_sp=n_saes,
            l1_coeff=l1_coeff,
        )

    def forward(self, x: dict, synchronize: bool, topk: int | None = None, **kwargs) -> dict:
        del synchronize  # kept for API compatibility
        x_tensor = next(iter(x.values())).detach()
        return self.sae(x=x_tensor, topk=topk, **kwargs)
