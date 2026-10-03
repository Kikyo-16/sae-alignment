from __future__ import annotations

import json
import os

import torch
import pytorch_lightning as pl

from .val_metric import ValMetric


class TopKEvalCallback(pl.Callback):
    """
    After each validation epoch:
      1. Compute metric via ValMetric.compute().
      2. Log the result to the trainer logger and callback_metrics.
      3. Save checkpoint if this beats the worst in the top-k; evict the displaced one.

    Metadata is persisted to eval_ckpts.json for resume support.
    """

    def __init__(self, ckpt_dir: str, metric: ValMetric, top_k: int = 6) -> None:
        self.ckpt_dir = ckpt_dir
        self.metric = metric
        self.top_k = top_k
        self.metadata_path = os.path.join(ckpt_dir, "eval_ckpts.json")
        # List of {"path": str, "score": float}, sorted so index 0 is the best.
        self._saved: list[dict] = []

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path) as f:
                self._saved = json.load(f)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        score = self.metric.compute(pl_module, trainer)

        # Log so the value appears in TensorBoard / WandB and callback_metrics.
        if trainer.logger is not None:
            trainer.logger.log_metrics({self.metric.name: score}, step=trainer.global_step)
        trainer.callback_metrics[self.metric.name] = torch.tensor(float(score))

        sign = 1.0 if self.metric.higher_is_better else -1.0

        if len(self._saved) >= self.top_k:
            worst_signed = sign * self._saved[-1]["score"]
            if sign * score <= worst_signed:
                return

        epoch = trainer.current_epoch
        step = trainer.global_step
        safe_tag = self.metric.name.replace("/", "_")
        filename = f"{safe_tag}-epoch{epoch:03d}-step{step}-score{score:.4f}.ckpt"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        trainer.save_checkpoint(ckpt_path)

        self._saved.append({"path": ckpt_path, "score": score})
        self._saved.sort(key=lambda x: sign * x["score"], reverse=True)

        while len(self._saved) > self.top_k:
            evicted = self._saved.pop()
            if os.path.exists(evicted["path"]):
                os.remove(evicted["path"])

        with open(self.metadata_path, "w") as f:
            json.dump(self._saved, f, indent=2)

        saved_scores = [f"{x['score']:.4f}" for x in self._saved]
        print(
            f"[TopKEval] epoch={epoch}  {self.metric.name}={score:.4f}  "
            f"top-{self.top_k}: {saved_scores}"
        )
