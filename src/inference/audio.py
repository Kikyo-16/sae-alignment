#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw audio -> 30-second foundation-model feature segments.

This reproduces the evaluation-set feature extraction of
``src.feature_extraction.process_datasets --inference --duration 30``:

  * audio is decoded to mono and resampled to 16 kHz (the ``--target-sr``
    default used for every H5 file in the paper);
  * the waveform is cut into non-overlapping 30 s windows (hop = window);
  * each window goes through the same rep handler (``muq_layer`` /
    ``musicfm_layer``), which resamples to the model rate (24 kHz) internally;
  * features are cast to float16, the precision stored in the H5 files.

The trailing remainder shorter than 30 s is handled by ``last_segment``:

  * ``"drop"``    (default) discard it, exactly like the H5 pipeline;
  * ``"overlap"`` add one more window ending at the last sample. It overlaps
    the previous window; only its non-overlapping frames are kept in the
    results (``AudioSegment.keep_from_sec``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..feature_extraction.audio_utils import load_mono, resample_to
from ..feature_extraction.registry import get_rep_handler
from ..feature_extraction import reps  # noqa: F401  (registers rep handlers)

SEGMENT_SEC = 30.0
TARGET_SR = 16000
LAST_SEGMENT_CHOICES = ("drop", "overlap")


@dataclass
class AudioSegment:
    start_sec: float
    end_sec: float
    keep_from_sec: float   # > 0 only for an overlapped tail window
    feat: np.ndarray       # [D, T] float16, same layout as the H5 datasets

    @property
    def n_frames(self) -> int:
        return int(self.feat.shape[-1])

    @property
    def sec_per_frame(self) -> float:
        return SEGMENT_SEC / float(self.n_frames)

    def keep_from_frame(self) -> int:
        """First frame index whose result is kept (0 for regular windows)."""
        if self.keep_from_sec <= 0.0:
            return 0
        k = int(round(self.keep_from_sec / self.sec_per_frame))
        return min(max(k, 0), self.n_frames - 1)

    def frame_time(self, t: int) -> float:
        return self.start_sec + t * self.sec_per_frame


def plan_segments(
    n_samples: int,
    sr: int = TARGET_SR,
    segment_sec: float = SEGMENT_SEC,
    last_segment: str = "drop",
) -> List[Tuple[int, float]]:
    """Return ``[(start_sample, keep_from_sec), ...]`` for one waveform."""
    if last_segment not in LAST_SEGMENT_CHOICES:
        raise ValueError(f"last_segment must be one of {LAST_SEGMENT_CHOICES}, got {last_segment!r}")
    win = int(round(segment_sec * sr))
    if n_samples < win:
        return []
    # Inference mode of process_datasets: hop == window, remainder dropped.
    plan = [(s, 0.0) for s in range(0, n_samples - win + 1, win)]
    covered = plan[-1][0] + win
    if last_segment == "overlap" and covered < n_samples:
        start = n_samples - win
        plan.append((start, (covered - start) / float(sr)))
    return plan


def load_audio(path: Path) -> torch.Tensor:
    """Mono float32 waveform at 16 kHz, shape (N,)."""
    wav, sr = load_mono(str(path))
    wav = resample_to(wav, sr, TARGET_SR).to(torch.float32)
    return wav.reshape(-1)


class FeatureExtractor:
    """Frozen foundation-model layer features through the registered rep handler."""

    def __init__(self, model: str, layer: int, device: str, model_id: Optional[str] = None) -> None:
        self.model = str(model)
        self.layer = int(layer)
        self.device = torch.device(device)
        self.handler = get_rep_handler(f"{self.model}_layer")
        self.config = {"target_sr": TARGET_SR, f"{self.model}_layer": self.layer}
        if model_id:
            self.config[f"{self.model}_model_id"] = model_id
        self.state = self.handler.init_models(self.device, config=self.config)

    @torch.no_grad()
    def __call__(self, seg: torch.Tensor) -> np.ndarray:
        feat, _ = self.handler.extract(seg, device=self.device, state=self.state, config=self.config)
        return feat.detach().cpu().numpy().astype(np.float16)


def extract_segments(
    path: Path,
    extractor: FeatureExtractor,
    last_segment: str = "drop",
) -> List[AudioSegment]:
    wav = load_audio(path)
    win = int(round(SEGMENT_SEC * TARGET_SR))
    out: List[AudioSegment] = []
    for start, keep_from_sec in plan_segments(int(wav.shape[0]), TARGET_SR, SEGMENT_SEC, last_segment):
        seg = wav[start:start + win]
        start_sec = start / float(TARGET_SR)
        out.append(AudioSegment(
            start_sec=start_sec,
            end_sec=start_sec + SEGMENT_SEC,
            keep_from_sec=keep_from_sec,
            feat=extractor(seg),
        ))
    return out


def stack_features(segments: List[AudioSegment]) -> np.ndarray:
    """[N, D, T] float16 batch (same array layout as an H5 feature dataset)."""
    return np.stack([s.feat for s in segments], axis=0)
