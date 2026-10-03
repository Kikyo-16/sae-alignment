from __future__ import annotations

import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Ring detection (self-graph algorithm)
# ---------------------------------------------------------------------------

def _get_dec_vec(sae_module, device: str, eps: float = 1e-7) -> torch.Tensor:
    dec = sae_module.fc2.weight.detach().to(device).t()
    return dec / (dec.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def _top1_map(src: torch.Tensor, tgt: torch.Tensor, block: int, prob: float
              ) -> Tuple[np.ndarray, np.ndarray]:
    K = src.shape[0]
    src_n = F.normalize(src.float(), dim=1)
    tgt_n = F.normalize(tgt.float(), dim=1)
    tgtT = tgt_n.t().contiguous()
    idx_out = torch.empty((K,), dtype=torch.long, device=src.device)
    sc_out = torch.empty((K,), dtype=torch.float32, device=src.device)
    for s in range(0, K, block):
        e = min(K, s + block)
        S = src_n[s:e] @ tgtT
        sc, idx = torch.max(S, dim=1)
        idx_out[s:e] = idx
        sc_out[s:e] = sc
    idx_out[sc_out < prob] = -1
    return idx_out.cpu().numpy().astype(np.int64), sc_out.cpu().numpy().astype(np.float32)


def _find_id(x: int, gBA: np.ndarray, gCA: np.ndarray) -> List[int]:
    idx = np.nonzero(gBA == x)[0]
    return [int(gCA[i]) for i in idx if gCA[i] >= 0]


def count_size12_structures(multi_sae, device: str, block: int = 256, prob: float = 0.7) -> int:
    """Count size-12 structures (rings or sequences) in the MultiSAE self-graph."""
    decs = list(multi_sae.sae_modules)
    n = len(decs)
    if n < 3:
        raise RuntimeError(f"Need at least 3 sub-SAEs, got {n}.")

    ws = [_get_dec_vec(d, device) for d in decs]
    maps: List[np.ndarray] = []
    for k in range(1, n):
        g, _ = _top1_map(ws[k], ws[0], block, prob)
        maps.append(g)

    K = len(maps[0])
    res: Dict[int, int] = {}
    rev_res: Dict[int, List[int]] = {}
    timbre: Set[int] = set()
    invalid: Set[int] = set()

    for i in range(K):
        a2_a1 = int(maps[0][i])
        if a2_a1 == -1:
            invalid.add(i)
            continue
        check_maps = maps[:3]
        valid = True
        for k in range(1, len(check_maps)):
            highs = _find_id(i, check_maps[k - 1], check_maps[k])
            if a2_a1 not in highs:
                valid = False
                break
        if valid:
            if i == a2_a1:
                timbre.add(i)
            else:
                res[i] = a2_a1
                rev_res.setdefault(a2_a1, []).append(i)
        else:
            invalid.add(i)

    visited: Set[int] = set()
    n_size12 = 0

    def trace_ring(start: int) -> Tuple[List[int], bool]:
        j = start
        trace = [j]
        while j in res:
            j = res[j]
            if j in trace:
                return trace, True
            trace.append(j)
        return trace, False

    for root in [i for i in res if i not in rev_res]:
        trace, _ = trace_ring(root)
        for v in trace:
            visited.add(v)
        if len(trace) == 12:
            n_size12 += 1

    for i in res:
        if i in visited:
            continue
        trace, _ = trace_ring(i)
        for v in trace:
            visited.add(v)
        if len(trace) == 12:
            n_size12 += 1

    return n_size12


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class RingCheckpointCallback(pl.Callback):
    """
    After each validation epoch:
      1. Count size-12 rings in the current model (ring count = n).
      2. If this checkpoint has the lowest val_mse_loss among all saved ckpt-n,
         save it and delete the previous best ckpt-n.
      3. For each ring count n, at most one checkpoint is kept.
    """

    def __init__(self, ckpt_dir: str, block: int = 256, prob: float = 0.7):
        self.ckpt_dir = ckpt_dir
        self.block = block
        self.prob = prob
        self.metadata_path = os.path.join(ckpt_dir, "ring_ckpts.json")
        # ring_count_str -> {"path": str, "val_mse_loss": float}
        self._best: dict[str, dict] = {}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path) as f:
                self._best = json.load(f)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        val_mse_loss = trainer.callback_metrics.get("val_mse_loss")
        if val_mse_loss is None:
            return
        val_mse_loss = float(val_mse_loss)

        device = str(pl_module.device)
        multi_sae = pl_module.sae.sae
        try:
            n_size12 = count_size12_structures(multi_sae, device, self.block, self.prob)
        except Exception as e:
            print(f"[RingCheckpoint] ring detection failed: {e}")
            return

        key = str(n_size12)
        current_best = self._best.get(key)
        if current_best is not None and val_mse_loss >= current_best["val_mse_loss"]:
            return

        epoch = trainer.current_epoch
        step = trainer.global_step
        filename = f"rings{n_size12:03d}-epoch{epoch:03d}-step{step}-val_mse{val_mse_loss:.4f}.ckpt"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        trainer.save_checkpoint(ckpt_path)

        if current_best is not None:
            old_path = current_best["path"]
            if os.path.exists(old_path):
                os.remove(old_path)

        self._best[key] = {"path": ckpt_path, "val_mse_loss": val_mse_loss}
        with open(self.metadata_path, "w") as f:
            json.dump(self._best, f, indent=2)

        print(f"[RingCheckpoint] rings={n_size12} new best: {filename} (val_mse_loss={val_mse_loss:.6f})")
