from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Hashable, Tuple, Optional


def apply_topk(out: torch.Tensor, k: int, return_indices: bool = False):
    """Keep top-k entries (by absolute value) along the last dim; zero-out others."""
    E = out.size(-1)
    if k <= 0:
        sparse = out.new_zeros(out.shape)
        idx = out.new_empty((*out.shape[:-1], 0), dtype=torch.long)
        return sparse, idx

    k_eff = min(k, E)
    flat = out.reshape(-1, E)
    topk_idx = torch.topk(flat.abs(), k_eff, dim=-1, sorted=True).indices  # [M, k_eff]

    if return_indices:
        return topk_idx.reshape(*out.shape[:-1], k_eff)

    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_idx, value=True)
    return mask.reshape_as(out)


def per_frame_shared_counts(topk_idx_all: torch.Tensor, max_n: int = 4):
    """
    topk_idx_all: [B,T,M,K]
    Returns dict of shared_eq_n: [B,T] — number of indices selected by exactly n SAEs.
    """
    B, T, M, K = topk_idx_all.shape
    L = M * K
    x = topk_idx_all.reshape(B, T, L)
    x = torch.sort(x, dim=-1).values

    start = torch.ones((B, T, L), dtype=torch.bool, device=x.device)
    start[..., 1:] = x[..., 1:] != x[..., :-1]
    gid = start.long().cumsum(dim=-1) - 1

    counts = torch.zeros((B, T, L), dtype=torch.int32, device=x.device)
    ones = torch.ones((B, T, L), dtype=torch.int32, device=x.device)
    counts.scatter_add_(dim=-1, index=gid, src=ones)

    out = {}
    for n in range(1, max_n + 1):
        out[f"shared_eq_{n}"] = (counts == n).sum(dim=-1)
    return out


@torch.no_grad()
def fast_one_to_one_match(
    A: torch.Tensor,
    B: torch.Tensor,
    topk: int = 32,
    block: int = 256,
    use_abs_cos: bool = False,
    normalize: bool = True,
    return_score_and_sign: bool = True,
    idx_masks=None,
) -> Tuple[torch.LongTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Approximate 1-1 matching between rows of A and B using cosine similarity."""
    assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors"
    assert A.ndim == 2 and B.ndim == 2 and A.shape == B.shape
    device = A.device
    N, D = A.shape
    K = min(int(topk), N)

    A_n = F.normalize(A, dim=1) if normalize else A
    B_n = F.normalize(B, dim=1) if normalize else B

    cand_idx = torch.empty((N, K), device=device, dtype=torch.int64)
    cand_val = torch.empty((N, K), device=device, dtype=torch.float32)
    A_t = A_n.transpose(0, 1).contiguous()

    for s in range(0, N, block):
        e = min(N, s + block)
        raw = B_n[s:e] @ A_t
        if idx_masks is not None:
            bsz = e - s
            rows = torch.arange(bsz, device=device)
            cols = rows + s
            penalty = torch.abs(raw).max() / 2
            for idx_mask in idx_masks:
                raw[rows, idx_mask[cols]] -= penalty
        vals, idx = torch.topk(raw, k=K, dim=1, largest=True, sorted=False)
        cand_idx[s:e] = idx
        cand_val[s:e] = vals.float()

    cand_idx_cpu = cand_idx.cpu()
    cand_val_cpu = cand_val.cpu()
    b_ids = torch.arange(N).unsqueeze(1).expand(N, K).reshape(-1).numpy()
    a_ids = cand_idx_cpu.reshape(-1).numpy()
    scores = cand_val_cpu.reshape(-1).numpy()
    order = scores.argsort()[::-1]

    a_used = [False] * N
    out = [-1] * N
    out_score = [0.0] * N

    for t in order:
        b = int(b_ids[t])
        if out[b] != -1:
            continue
        a = int(a_ids[t])
        if a_used[a]:
            continue
        out[b] = a
        a_used[a] = True
        out_score[b] = float(scores[t])

    if -1 in out:
        for b in range(N):
            if out[b] != -1:
                continue
            local = sorted(zip(cand_idx_cpu[b].tolist(), cand_val_cpu[b].tolist()),
                           key=lambda x: x[1], reverse=True)
            for a, sc in local:
                if not a_used[a]:
                    out[b] = a
                    a_used[a] = True
                    out_score[b] = sc
                    break

        if -1 in out:
            remaining_a = iter(i for i in range(N) if not a_used[i])
            for b in range(N):
                if out[b] == -1:
                    out[b] = next(remaining_a)
                    out_score[b] = 0.0

    indices = torch.tensor(out, device=device, dtype=torch.long)
    if not return_score_and_sign:
        return indices, None, None
    score_t = torch.tensor(out_score, device=device, dtype=torch.float32)
    sign_t = torch.ones(N, device=device, dtype=torch.int8)
    return indices, score_t, sign_t
