#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared boundary-feature peak utilities for Step 3 selection and Step 4 inference.

Keep all logic that converts a boundary feature activation into final boundary
peaks here, so Step 3 scoring and Step 4 estimated-boundary inference cannot
drift apart.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .infer_key import (
    _segments_from_boundaries,
    detect_peaks_with_drop,
    moving_average_1d,
    thin_times,
)


def detect_boundary_peaks_1d(
    raw: np.ndarray,
    smooth_win: int,
    peak_tol: float,
    peak_drop: float,
    min_gap: int,
    score_floor: float,
    max_labels: int,
) -> Tuple[List[int], np.ndarray]:
    """Detect final estimated chord-boundary peaks from one feature activation.

    This is the single source of truth shared by Step 3 and Step 4:

        raw activation
        -> moving_average_1d(smooth_win)
        -> detect_peaks_with_drop(tol=peak_tol, min_rise/drop=peak_drop)
        -> thin_times(max_labels)
        -> remove endpoints (0 and T)
        -> min_gap thinning (skipped when min_gap <= 1; pass 0 to disable)
        -> score_floor filtering on the smoothed activation

    Returns:
        (peaks, smoothed_activation), where peaks are frame indices suitable for
        _segments_from_boundaries / chord segmentation.
    """
    z1 = np.asarray(raw, dtype=np.float32)
    t_len = int(z1.shape[0])
    if t_len <= 0:
        return [], z1[:0]

    sm = moving_average_1d(z1, int(smooth_win))
    peaks = detect_peaks_with_drop(
        sm,
        tol=float(peak_tol),
        min_drop=float(peak_drop),
        min_rise=float(peak_drop),
    )
    peaks = thin_times(peaks, t_len, max_labels=int(max_labels))
    out = [int(x) for x in peaks if 0 < int(x) < t_len]

    # min_gap <= 1 means no min-gap thinning; this lets step3/step4 both
    # accept --boundary-min-gap 0 as an explicit off switch.
    if int(min_gap) > 1 and out:
        thinned = [out[0]]
        for x in out[1:]:
            x = int(x)
            if x - thinned[-1] >= int(min_gap):
                thinned.append(x)
        out = thinned

    if float(score_floor) > 0.0:
        out = [x for x in out if float(sm[x]) >= float(score_floor)]

    return out, sm


def boundary_peak_mask(
    z: np.ndarray,
    smooth_win: int,
    peak_tol: float,
    peak_drop: float,
    min_gap: int,
    score_floor: float,
    max_labels: int,
) -> np.ndarray:
    """Return boolean [T, D] final boundary-peak mask using detect_boundary_peaks_1d."""
    if z.ndim != 2:
        raise ValueError(f"z must be 2D [T, D], got shape={z.shape}")
    mask = np.zeros(z.shape, dtype=bool)
    for d in range(z.shape[1]):
        idx, _sm = detect_boundary_peaks_1d(
            z[:, d], smooth_win, peak_tol, peak_drop, min_gap, score_floor, max_labels
        )
        if idx:
            mask[idx, d] = True
    return mask


def mutual_boundary_star_mask(
    pm: np.ndarray,
    centers: List[int],
    t_len: int,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (star_mask, missing_count) for final peak mask pm [T, D].

    For every GT boundary and every feature, first find that boundary's nearest
    retained peak. The peak is accepted as a star only if:

      1. the feature has at least one retained peak;
      2. that peak's own nearest GT boundary is the same boundary (mutual match);
      3. when max_dist is set, abs(peak_t - boundary_t) <= max_dist.

    Otherwise that GT boundary is missing for this feature. Peaks accepted by
    other boundaries are not counted as false alarms; only pm & ~star_mask are
    downstream "other" peaks.
    """
    if pm.ndim != 2:
        raise ValueError(f"pm must be 2D [T, D], got shape={pm.shape}")

    T, D = pm.shape
    if T != int(t_len):
        raise ValueError(f"pm length {T} != t_len {t_len}")
    if max_dist is not None and max_dist < 0:
        raise ValueError(f"max_dist must be >= 0 or None, got {max_dist}")

    star = np.zeros_like(pm, dtype=bool)
    miss = np.zeros(D, dtype=np.float64)
    if T <= 0 or D <= 0 or not centers:
        return star, miss

    frames = np.arange(T, dtype=np.float32)
    cps_arr = np.asarray(centers, dtype=np.float32)
    nb_frame = np.abs(frames[:, None] - cps_arr[None, :]).argmin(axis=1)

    valid = pm.any(axis=0)
    far = np.float32(T + 1)
    for k, c in enumerate(centers):
        md = np.where(pm, np.abs(frames - float(c))[:, None], far)
        nearest_t = md.argmin(axis=0)
        nearest_dist = md[nearest_t, np.arange(D)]
        consistent = valid & (nb_frame[nearest_t] == k)
        if max_dist is not None:
            consistent = consistent & (nearest_dist <= float(max_dist))
        accepted_cols = np.where(consistent)[0]
        star[nearest_t[accepted_cols], accepted_cols] = True
        miss[~consistent] += 1.0

    return star, miss


def boundary_segments_from_feature(
    z: np.ndarray,
    feature_id: Optional[int],
    t_z: Optional[int] = None,
    smooth_win: int = 9,
    peak_tol: float = 0.2,
    peak_drop: float = 0.0,
    min_gap: int = 2,
    score_floor: float = 0.0,
    max_labels: int = 120,
) -> List[Tuple[int, int]]:
    """Convert one boundary feature into chord segments using the shared peak path."""
    if z.ndim != 2:
        raise ValueError(f"z must be 2D [T, D], got shape={z.shape}")
    T, D = int(z.shape[0]), int(z.shape[1])
    t_use = T if t_z is None else int(t_z)
    if t_use <= 0:
        return []
    if feature_id is None or int(feature_id) < 0 or int(feature_id) >= D:
        return [(0, t_use)]

    peaks, _sm = detect_boundary_peaks_1d(
        z[:t_use, int(feature_id)].astype(np.float32, copy=False),
        smooth_win=smooth_win,
        peak_tol=peak_tol,
        peak_drop=peak_drop,
        min_gap=min_gap,
        score_floor=score_floor,
        max_labels=max_labels,
    )
    return _segments_from_boundaries(peaks, t_use) or [(0, t_use)]
