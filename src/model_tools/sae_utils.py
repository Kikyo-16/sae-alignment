import torch


class ActivationStats:
    """
    Track feature firing rate with EMA.
    For Top-k SAE, "firing" should mean: selected by top-k mask.
    firing_rate[i] ~ P(feature i is selected in top-k for a token)
    """
    def __init__(self, n_feats: int, ema_beta: float = 0.99, eps: float = 1e-6, device="cpu"):
        self.n_feats = n_feats
        self.ema_beta = ema_beta
        self.eps = eps
        self.firing_rate = None

    @torch.no_grad()
    def _init_if_needed(self, device):
        if self.firing_rate is None:
            self.firing_rate = torch.zeros(self.n_feats, device=device)

    @torch.no_grad()
    def update_from_topk(self, topk_idx: torch.Tensor):
        """
        topk_idx: [B,T,K] or [BT,K]
        Updates firing rate by counting selected indices.
        """
        device = topk_idx.device
        self._init_if_needed(device)

        idx = topk_idx.reshape(-1)  # [B*T*K]
        counts = torch.bincount(idx, minlength=self.n_feats).float()  # [E]

        if topk_idx.dim() == 3:
            BT = topk_idx.shape[0] * topk_idx.shape[1]
        else:
            BT = topk_idx.shape[0]

        fired_rate_batch = counts / max(BT, 1)
        self.firing_rate.mul_(self.ema_beta).add_(fired_rate_batch * (1 - self.ema_beta))

    @torch.no_grad()
    def dead_mask(self, device, threshold: float = 1e-4):
        self._init_if_needed(device)
        return self.firing_rate < threshold

    @torch.no_grad()
    def alive_count(self, threshold: float = 1e-4):
        if self.firing_rate is None:
            return 0
        return int((self.firing_rate >= threshold).sum().item())
