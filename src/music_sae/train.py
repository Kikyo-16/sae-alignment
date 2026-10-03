from __future__ import annotations

import argparse
import json
import os
import pytorch_lightning as pl
import torch


from .sae_lit import LitSAE
from .data_loader import MelH5DataModule
from .config import exp_config
from .ring_checkpoint import RingCheckpointCallback
from .probe_key_metric import MultiProbeKeyMetric
from .chord_ring_metric import ChordRingMetric
from .probe_chord_metric import ProbeChordMetric
from .top_k_eval_callback import TopKEvalCallback
from .score import (
    compute_mid_sae_alignment_scores,
    min_alignment_score_for_feature_ids,
)


class ChordFeatureAlignmentLogger(pl.Callback):
    """Log alignment quality of the maj/min chord features selected this eval.

    The chord metric may select live feature ids every validation evaluation
    (for example ProbeChordMetric maps chord-probe rows to SAE encoder NNs).
    This callback reads those selected ids, computes the current SAE-1 decoder
    alignment scores, and logs the minimum score in the selected major and minor
    rings. Lower is better.
    """

    def __init__(
        self,
        metric,
        target_idx: int = 1,
        mid_neighbor_topk: int = 8,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.target_idx = int(target_idx)
        self.mid_neighbor_topk = int(mid_neighbor_topk)

    @staticmethod
    def _as_int_list(x) -> list[int] | None:
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.detach().cpu().flatten().tolist()
        try:
            out = [int(v) for v in list(x)]
        except Exception:
            return None
        return out if out else None

    def _selected_major_minor_ids(self, pl_module, trainer) -> tuple[list[int], list[int]] | None:
        # Prefer ids cached by the metric during its own compute() call.
        major_names = (
            "last_major_ids", "last_major_feature_ids",
            "selected_major_ids", "selected_major_feature_ids",
            "major_ids", "major_feature_ids",
        )
        minor_names = (
            "last_minor_ids", "last_minor_feature_ids",
            "selected_minor_ids", "selected_minor_feature_ids",
            "minor_ids", "minor_feature_ids",
        )
        major = next((self._as_int_list(getattr(self.metric, name, None)) for name in major_names
                      if self._as_int_list(getattr(self.metric, name, None)) is not None), None)
        minor = next((self._as_int_list(getattr(self.metric, name, None)) for name in minor_names
                      if self._as_int_list(getattr(self.metric, name, None)) is not None), None)
        if major is not None and minor is not None:
            return major, minor

        # ProbeChordMetric exposes this method; calling it reproduces the same
        # live chord-feature selection used by compute().
        selector = getattr(self.metric, "_select_chord_feature_ids", None)
        if callable(selector):
            device = str(pl_module.device)
            major, minor = selector(pl_module, device)
            return [int(x) for x in major], [int(x) for x in minor]
        return None

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        module = getattr(getattr(pl_module, "sae", None), "sae", None)
        sae_modules = getattr(module, "sae_modules", None)
        n_saes = int(getattr(module, "n_saes", 0)) if module is not None else 0
        if sae_modules is None or n_saes < 3:
            return
        if self.target_idx <= 0 or self.target_idx >= n_saes - 1:
            return

        ids = self._selected_major_minor_ids(pl_module, trainer)
        if ids is None:
            return
        major_ids, minor_ids = ids

        scores = compute_mid_sae_alignment_scores(
            sae_modules,
            target_idx=self.target_idx,
            mid_neighbor_topk=self.mid_neighbor_topk,
        )
        major_min = min_alignment_score_for_feature_ids(scores, major_ids)
        minor_min = min_alignment_score_for_feature_ids(scores, minor_ids)

        metrics = {
            f"val/sae_{self.target_idx}_major_feature_min_alignment": float(major_min),
            f"val/sae_{self.target_idx}_minor_feature_min_alignment": float(minor_min),
        }
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
        for k, v in metrics.items():
            trainer.callback_metrics[k] = torch.tensor(v, device=pl_module.device)


def _load_config_file(argv=None) -> dict | None:
    """Return the JSON config passed via --config (or None).

    The file holds ``{"exp_config": {...}, "train": {...}}``: ``exp_config`` is
    used instead of ``config.exp_config[--mode]`` and the ``train`` entries
    become argparse defaults (explicit command-line flags still win).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config is None:
        return None
    with open(known.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if "exp_config" not in cfg:
        raise ValueError(f"{known.config}: missing 'exp_config' section.")
    return cfg


def parse_args(file_cfg: dict | None = None) -> argparse.Namespace:
    has_cfg = file_cfg is not None
    p = argparse.ArgumentParser("Train Music SAE")
    p.add_argument("--config", type=str, default=None,
                   help="JSON config written by scripts/train_sae.py. Provides the "
                        "exp_config (replaces --mode) and default training args.")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed (torch/numpy/python + DataLoader workers).")
    p.add_argument("--h5", required=not has_cfg, nargs="+", help="Path(s) to feature H5 file(s).")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--logDir", type=str, default="logs")
    p.add_argument("--runName", type=str, default="music_sae_run")
    p.add_argument("--batchSize", type=int, default=32)
    p.add_argument("--numWorkers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-warmup-frac", type=float, default=0.,
                   help="Fraction of total steps for linear LR warmup from 0 "
                        "(0 disables warmup).")
    p.add_argument("--lr-decay-frac", type=float, default=0.,
                   help="Fraction of total steps over which LR linearly decays "
                        "to 0 at the end (0 disables decay).")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--featureTag", type=str, default="feat")
    p.add_argument("--maxEpochs", type=int, default=200)
    p.add_argument("--mode", type=str, required=not has_cfg)
    p.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint path.")
    p.add_argument("--precision", type=str, default="32")
    # Explicit train/val track-id lists (one trackId per line). When both are
    # given the data loader builds the train/val sets from these lists instead
    # of a random song-level split.
    p.add_argument("--train-track-ids", type=str, default=None,
                   help="Text file of trackIds for the training set.")
    p.add_argument("--val-track-ids", type=str, default=None,
                   help="Text file of trackIds for the validation set. Enables the "
                        "chord-ring validation metric (root + root/minmaj WCSR).")
    p.add_argument("--chord-probe-ckpt", type=str, default=None,
                   help="LitLinearProbe 'chord' checkpoint. When given (with "
                        "--val-track-ids), the major/minor feature ids are picked "
                        "from the probe weights by cosine NN against the live SAE "
                        "encoder instead of enumerating self-graph rings.")
    # Chord-ring validation metric settings.
    p.add_argument("--val-eval-h5", type=str, default=None,
                   help="H5 used for chord-ring validation (default: first --h5). "
                        "Must contain labels/chord_frame for the validation tracks.")
    p.add_argument("--val-smooth-win", type=int, default=9,
                   help="Moving-average window applied to the chord ring "
                        "activations before mode/root decision (<=1 disables).")
    p.add_argument("--val-majmin-mode", default="seg-mean-max",
                   choices=["sum-peak", "seg-mean-max", "seg-mean-mean", "raw-tmpl"],
                   help="Chord decision for chord-ring validation. sum-peak "
                        "reproduces the original metric; raw-tmpl uses the triad-"
                        "template fallback (default conf_thr/alpha).")
    p.add_argument("--val-ring-prob", type=float, default=0.7,
                   help="Self-graph top-1 cosine threshold for ring enumeration.")
    p.add_argument("--val-semitone-step", type=int, default=1,
                   help="Semitone interval between adjacent sub-SAEs (1=chromatic).")
    # Probe-key eval metric (optional; falls back to ring-checkpoint when omitted).
    # Supply at least one --probe-ckpt-* to enable; each tag uses its own probe ckpt.
    p.add_argument("--probe-ckpt-tonic-frame", type=str, default=None,
                   help="LitLinearProbe checkpoint trained on tonic_frame.")
    p.add_argument("--probe-ckpt-forth-frame", type=str, default=None,
                   help="LitLinearProbe checkpoint trained on forth_frame.")
    p.add_argument("--probe-ckpt-fifth-frame", type=str, default=None,
                   help="LitLinearProbe checkpoint trained on fifth_frame.")
    p.add_argument("--probe-h5", type=str, default=None,
                   help="H5 file to evaluate the probe-key metric on.")
    p.add_argument("--probe-feature-ds", type=str, default=None,
                   help="Feature dataset key in the H5, e.g. 'muq_layer'.")
    p.add_argument("--probe-split", type=str, default="validation",
                   help="H5 split used for probe-key evaluation.")
    p.add_argument("--probe-sae-idx", type=int, default=1,
                   help="Which sub-SAE index to use for encoder NN replacement.")
    p.add_argument("--probe-max-segs", type=int, default=-1,
                   help="Cap on segments evaluated per epoch (-1 = all).")
    p.add_argument("--top-k", type=int, default=6,
                   help="Number of top checkpoints to keep when using probe-key metric.")
    if has_cfg:
        train_defaults = dict(file_cfg.get("train", {}))
        known_dests = {a.dest for a in p._actions}
        unknown = sorted(set(train_defaults) - known_dests)
        if unknown:
            raise ValueError(f"Unknown keys in config 'train' section: {unknown}")
        p.set_defaults(**train_defaults)
    return p.parse_args()


def _build_callbacks(args, ckpt_dir: str, config: dict) -> list:
    # Chord-ring validation metric (no probe weights): find major/minor rings on
    # the validation set, run GT-boundary chord recognition, score = mean of
    # root and root+minmaj WCSR. Best ckpt is selected on this score.
    if args.val_track_ids is not None:
        if args.chord_probe_ckpt is not None:
            metric = ProbeChordMetric(
                probe_ckpt=args.chord_probe_ckpt,
                h5_path=(args.val_eval_h5 or args.h5[0]),
                feature_ds=args.featureTag,
                data_tag=config["model_config"]["data_tag"],
                val_track_id_file=args.val_track_ids,
                split=args.split,
                sae_idx=args.probe_sae_idx,
                batch_size=args.batchSize,
                smooth_win=args.val_smooth_win,
                qual_mode=args.val_majmin_mode.replace("-", "_"),
            )
        else:
            metric = ChordRingMetric(
                h5_path=(args.val_eval_h5 or args.h5[0]),
                feature_ds=args.featureTag,
                data_tag=config["model_config"]["data_tag"],
                val_track_id_file=args.val_track_ids,
                split=args.split,
                sae_idx=args.probe_sae_idx,
                batch_size=args.batchSize,
                smooth_win=args.val_smooth_win,
                qual_mode=args.val_majmin_mode.replace("-", "_"),
                prob=args.val_ring_prob,
                semitone_step=args.val_semitone_step,
            )
        ckpt_cb = TopKEvalCallback(ckpt_dir=ckpt_dir, metric=metric, top_k=args.top_k)
        align_cb = ChordFeatureAlignmentLogger(
            metric=metric,
            target_idx=int(config.get("alignment_target_idx", 1)),
            mid_neighbor_topk=int(config.get("alignment_mid_neighbor_topk", 8)),
        )
        return [
            ckpt_cb,
            align_cb,
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

    probe_ckpts = {
        tag: ckpt
        for tag, ckpt in {
            "tonic_frame": args.probe_ckpt_tonic_frame,
            "forth_frame":  args.probe_ckpt_forth_frame,
            "fifth_frame":  args.probe_ckpt_fifth_frame,
        }.items()
        if ckpt is not None
    }
    use_probe = bool(probe_ckpts)

    if use_probe:
        if args.probe_h5 is None or args.probe_feature_ds is None:
            raise ValueError(
                "--probe-h5 and --probe-feature-ds are required when any --probe-ckpt-* is given."
            )
        metric = MultiProbeKeyMetric(
            probe_ckpts=probe_ckpts,
            h5_path=args.probe_h5,
            feature_ds=args.probe_feature_ds,
            split=args.probe_split,
            sae_idx=args.probe_sae_idx,
            max_segs=args.probe_max_segs,
        )
        ckpt_cb = TopKEvalCallback(ckpt_dir=ckpt_dir, metric=metric, top_k=args.top_k)
    else:
        ckpt_cb = RingCheckpointCallback(ckpt_dir=ckpt_dir)

    return [
        ckpt_cb,
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]


def main() -> None:
    file_cfg = _load_config_file()
    args = parse_args(file_cfg)

    # Fix all RNGs (torch/numpy/python + DataLoader workers) for reproducibility.
    pl.seed_everything(args.seed, workers=True)

    exp_folder = os.path.join(args.logDir, args.runName)
    os.makedirs(exp_folder, exist_ok=True)

    if file_cfg is not None:
        config = dict(file_cfg["exp_config"])
    else:
        config = dict(exp_config[args.mode])

    dm = MelH5DataModule(
        h5_path=args.h5,
        data_tag=config["model_config"]["data_tag"],
        batch_size=args.batchSize,
        num_workers=args.numWorkers,
        pin_memory=True,
        shuffle=True,
        split=args.split,
        aug=config["aug"],
        aug_order=config["aug_order"],
        n_saes=config["n_saes"],
        feature_tag=args.featureTag,
        train_track_id_file=args.train_track_ids,
        val_track_id_file=args.val_track_ids,
    )

    model = LitSAE(
        lr=args.lr,
        lr_warmup_frac=args.lr_warmup_frac,
        lr_decay_frac=args.lr_decay_frac,
        topk=args.topk,
        exp_config=config,
    )

    ckpt_dir = os.path.join(exp_folder, "ckpts")

    trainer = pl.Trainer(
        max_epochs=args.maxEpochs,
        precision=args.precision,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        detect_anomaly=False,
        enable_checkpointing=False,
        default_root_dir=exp_folder,
        callbacks=_build_callbacks(args, ckpt_dir, config),
        check_val_every_n_epoch=10,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
