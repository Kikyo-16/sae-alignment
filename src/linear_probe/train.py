from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl

from .config import probe_config
from .data_loader import (
    ProbeH5DataModule, TASK_KEY_RELATIVE, TASK_CHORD, TASK_KEY_STRICT, ALL_TASKS,
    DEGREE_LABEL_TAGS,
)
from .probe_lit import LitLinearProbe
from .metrics import KeyDegreeMetric, ChordProbeMetric, KeyStrictMetric
from ..music_sae.sae_lit import LitSAE
from ..music_sae.top_k_eval_callback import TopKEvalCallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train linear probe (key_relative / chord / key_strict)")
    p.add_argument("--h5", required=True, nargs="+", help="Path(s) to training H5 file(s).")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--mode", type=str, required=True, choices=list(probe_config),
                   help=f"Feature config. Available: {list(probe_config)}")
    p.add_argument("--task", type=str, default=TASK_KEY_RELATIVE, choices=list(ALL_TASKS),
                   help="Probe task.")
    p.add_argument("--labelTag", type=str, default="forth_frame", choices=list(DEGREE_LABEL_TAGS),
                   help="Scale-degree label for task=key_relative.")
    p.add_argument("--logDir", type=str, default="logs")
    p.add_argument("--runName", type=str, default="linear_probe_run")
    p.add_argument("--batchSize", type=int, default=32)
    p.add_argument("--numWorkers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--maxEpochs", type=int, default=200)
    p.add_argument("--valRatio", type=float, default=0.1,
                   help="Random song-level val fraction when no track-id lists are given.")
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint path.")
    p.add_argument("--saeCkpt", type=str, default=None,
                   help="SAE checkpoint; features are encoded through the SAE before the linear layer.")
    p.add_argument("--threshold", type=float, default=0.7, help="Sigmoid threshold for N-rejection.")
    # Explicit train/val song lists (align with music_sae); both enable the
    # per-song ValMetric used for checkpoint selection.
    p.add_argument("--train-track-ids", type=str, default=None,
                   help="Text file of trackIds for the training set.")
    p.add_argument("--val-track-ids", type=str, default=None,
                   help="Text file of trackIds for validation. Enables the per-song ValMetric.")
    p.add_argument("--val-eval-h5", type=str, default=None,
                   help="H5 used for the per-song validation metric (default: first --h5). "
                        "May be a no-aug labelled H5.")
    p.add_argument("--val-smooth-win", type=int, default=9,
                   help="Moving-average window on chord activations (task=chord).")
    p.add_argument("--val-majmin-mode", default="seg_mean_max",
                   choices=["sum_peak", "seg_mean_max", "seg_mean_mean", "raw_tmpl"],
                   help="Chord decision mode (task=chord).")
    p.add_argument("--val-max-songs", type=int, default=None,
                   help="Cap on validation songs per epoch.")
    p.add_argument("--top-k", type=int, default=3, help="Number of top checkpoints to keep.")
    return p.parse_args()


def _build_metric(args, feature_tag: str):
    h5 = args.val_eval_h5 or args.h5[0]
    common = dict(h5_path=h5, feature_ds=feature_tag, val_track_id_file=args.val_track_ids,
                  split=args.split, batch_size=args.batchSize, max_songs=args.val_max_songs)
    if args.task == TASK_CHORD:
        return ChordProbeMetric(smooth_win=args.val_smooth_win,
                                qual_mode=args.val_majmin_mode, **common)
    if args.task == TASK_KEY_STRICT:
        return KeyStrictMetric(**common)
    return KeyDegreeMetric(label_tag=args.labelTag, **common)


def _build_callbacks(args, ckpt_dir: str, feature_tag: str) -> list:
    if args.val_track_ids is None:
        raise ValueError(
            "--val-track-ids is required: checkpoints are selected by the per-song "
            "ValMetric. Provide a trackId list to enable it."
        )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    metric = _build_metric(args, feature_tag)
    return [TopKEvalCallback(ckpt_dir=ckpt_dir, metric=metric, top_k=args.top_k), lr_cb]


def main() -> None:
    args = parse_args()

    # Fix all RNGs (torch/numpy/python + DataLoader workers) for reproducibility.
    pl.seed_everything(42, workers=True)

    config = probe_config[args.mode]
    feature_tag, d_model = config["feature_tag"], config["d_model"]

    sae_module = None
    if args.saeCkpt is not None:
        lit_sae = LitSAE.load_from_checkpoint(args.saeCkpt, map_location="cpu")
        lit_sae.eval()
        sae_module = lit_sae.sae.sae.sae_modules[0]

    exp_folder = os.path.join(args.logDir, args.runName)
    os.makedirs(exp_folder, exist_ok=True)

    dm = ProbeH5DataModule(
        h5_paths=args.h5,
        split=args.split,
        feature_tag=feature_tag,
        task=args.task,
        label_tag=args.labelTag,
        batch_size=args.batchSize,
        num_workers=args.numWorkers,
        pin_memory=True,
        val_ratio=args.valRatio,
        train_track_id_file=args.train_track_ids,
        val_track_id_file=args.val_track_ids,
    )

    model = LitLinearProbe(
        d_model=d_model,
        feature_tag=feature_tag,
        task=args.task,
        lr=args.lr,
        threshold=args.threshold,
        sae=sae_module,
        sae_ckpt_path=args.saeCkpt,
    )

    ckpt_dir = os.path.join(exp_folder, "ckpts")
    trainer = pl.Trainer(
        max_epochs=args.maxEpochs,
        precision=args.precision,
        log_every_n_steps=10,
        enable_checkpointing=False,
        default_root_dir=exp_folder,
        callbacks=_build_callbacks(args, ckpt_dir, feature_tag),
        check_val_every_n_epoch=2,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
