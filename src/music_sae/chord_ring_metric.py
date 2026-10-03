from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from tqdm.auto import tqdm

from ..analysis.find_chord_rings import (
    accumulate_global_marks,
    compute_size12_rings,
    iter_songs,
    pooled_ring_cols,
    precompute_chord_pc_qual,
    select_maj_min_rings,
    standardize_to_BTD_batch,
)
from ..analysis.step4_infer_gt_boundary import (
    _gt_chord_boundaries,
    _gt_silence_score,
)
from ..analysis.infer_key import (
    _segments_from_boundaries,
    predict_chord_segments,
)
from ..evaluation.evaluate_chord import ChordRow, evaluate


def _read_track_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            tid = line.strip()
            if tid:
                ids.add(tid)
    return ids


def _chord_rows_for_segment(
    track: str,
    z: np.ndarray,
    chord_frame_seg: Optional[np.ndarray],
    major_cols: List[int],
    minor_cols: List[int],
    smooth_win: int = 1,
    qual_mode: str = "sum_peak",
) -> List[ChordRow]:
    """GT-boundary chord prediction for one segment.

    Boundaries / silence come from the GT chord labels; the actual chord
    inference reuses the shared :func:`predict_chord_segments` interface (same
    code path as step4_infer_gt_boundary), so this metric cannot drift from the
    offline drivers.
    """
    rows: List[ChordRow] = []
    if z.ndim != 2 or z.shape[0] <= 1:
        return rows
    t_z, d_z = int(z.shape[0]), int(z.shape[1])

    if chord_frame_seg is not None:
        chord_frame_seg = chord_frame_seg[:t_z]

    major_cols = [i for i in major_cols if 0 <= i < d_z]
    minor_cols = [i for i in minor_cols if 0 <= i < d_z]
    if len(major_cols) < 12 or len(minor_cols) < 12:
        return rows

    chord_boundaries = _gt_chord_boundaries(chord_frame_seg, t_z)
    chord_segs = _segments_from_boundaries(chord_boundaries, t_z) or [(0, t_z)]
    silence_sc_t = _gt_silence_score(chord_frame_seg, t_z)

    for (cs, ce, ref_chord_str, est_chord) in predict_chord_segments(
        z, major_cols, minor_cols, chord_segs,
        silence_sc_t=silence_sc_t, chord_frame_seg=chord_frame_seg,
        track=track, t_z=t_z, smooth_win=smooth_win, qual_mode=qual_mode,
    ):
        rows.append(ChordRow(
            track_id=track, cs=cs, ce=ce,
            ref_chord=ref_chord_str, est_chord=est_chord,
            block_consistency=-1.0,
        ))
    return rows


class ChordRingMetric:
    """
    Validation metric for the music SAE.

    Each validation epoch this metric, evaluated on the validation track set:
      1. Enumerates size-12 chromatic rings from the live SAE self-graph.
      2. Runs SAE inference over the validation songs.
      3. Identifies the major / minor chord rings via chord-label alignment
         (no linear probe weights involved).
      4. Performs ground-truth-boundary chord recognition (same logic as
         step4_infer_gt_boundary / run_chord_gt_boundary.sh).
      5. Returns the mean of the segment-level WCSR for the 'root' and
         'majmin' (root + min/maj) vocabularies.

    The returned scalar drives top-k checkpoint selection in TopKEvalCallback.
    """

    def __init__(
        self,
        h5_path: str,
        feature_ds: str,
        data_tag: str,
        val_track_id_file: str,
        split: str = "train",
        sae_idx: int = 0,
        batch_size: int = 64,
        smooth_win: int = 9,
        qual_mode: str = "sum_peak",
        block: int = 256,
        prob: float = 0.7,
        semitone_step: int = 1,
        max_songs: Optional[int] = None,
        ring_id_max_batches: int = 8,
    ) -> None:
        self.h5_path = h5_path
        self.feature_ds = feature_ds
        self.data_tag = data_tag
        self.val_track_ids = _read_track_ids(val_track_id_file)
        self.split = split
        self.sae_idx = sae_idx
        self.batch_size = batch_size
        self.smooth_win = smooth_win
        self.qual_mode = qual_mode
        self.block = block
        self.prob = prob
        self.semitone_step = semitone_step
        self.max_songs = max_songs
        self.ring_id_max_batches = ring_id_max_batches
        if not self.val_track_ids:
            raise ValueError(f"val_track_id_file '{val_track_id_file}' is empty.")

    @property
    def name(self) -> str:
        return "val/chord_ring_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _infer_song_z(
        self, module, model_factory, feat_np: np.ndarray,
        use_tag: str, use_idx: int, device: str, keep_cols: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """SAE inference for one song. Returns reduced z [n_segs, T, K] or None.

        Only the ring columns (``keep_cols``) are kept; the song's activations are
        the only thing held in memory at a time (a few MB), so nothing is cached
        across the validation set.
        """
        n_segs = int(feat_np.shape[0])
        t_feat = int(feat_np.shape[-1])
        bs = max(1, int(self.batch_size))
        outs: List[np.ndarray] = []
        for s0 in range(0, n_segs, bs):
            s1 = min(n_segs, s0 + bs)
            x = torch.from_numpy(feat_np[s0:s1]).float().to(device)
            feat_batch = model_factory({use_tag: x}, t=0)
            if use_tag not in feat_batch:
                return None
            feat = feat_batch[use_tag]
            x_sae = standardize_to_BTD_batch(feat, t_candidates=[t_feat, int(feat_np.shape[-1])])
            sparse_z = module.inference(x_sae, idx=use_idx, topk_wide=0)
            if sparse_z.ndim != 3:
                return None
            outs.append(sparse_z.index_select(-1, keep_cols).detach().float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def _iter_val_songs(self):
        return iter_songs(
            Path(self.h5_path), self.split, self.feature_ds,
            max_songs=self.max_songs, track_ids=self.val_track_ids, orig_only=True,
        )

    @torch.no_grad()
    def _accumulate_ring_marks(
        self, module, model_factory, use_tag, use_idx, device, keep_cols_t, pooled,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Pass 1 (streaming, capped): global-argmax per-feature (root, quality) marks.

        All ring columns form one global candidate set; for every valid-root
        frame the single highest-activation feature wins and is tagged with the
        GT (root, quality). Counts accumulate per (feature, root) into ``tot``
        (any quality), ``maj`` (quality 0) and ``min_`` (quality 1).

        Accumulation stops once ``ring_id_max_batches`` forward batches have been
        consumed — ring identification only needs a stable 12x12 histogram, so
        this bounds the cost independent of validation-set size. Each song is
        inferred, accumulated, and discarded (no caching).
        """
        K = len(pooled)
        tot = np.zeros((K, 12), dtype=np.int64)
        maj = np.zeros((K, 12), dtype=np.int64)
        min_ = np.zeros((K, 12), dtype=np.int64)
        bs = max(1, int(self.batch_size))
        cap = max(1, int(self.ring_id_max_batches))
        n_songs = 0
        n_batches = 0
        for _tid, feat_np, chord_2d, _key in tqdm(
            self._iter_val_songs(), total=len(self.val_track_ids),
            desc="ChordRing pass1 (rings)", unit="song", leave=False,
        ):
            z = self._infer_song_z(module, model_factory, feat_np, use_tag, use_idx, device, keep_cols_t)
            if z is None:
                continue
            n_songs += 1
            rp, q = precompute_chord_pc_qual(chord_2d.reshape(-1))
            rp = rp.reshape(chord_2d.shape)   # [n_segs, T]
            q = q.reshape(chord_2d.shape)
            tcut = min(z.shape[1], rp.shape[1])
            zf = z[:, :tcut].reshape(-1, z.shape[-1])
            rpf = rp[:, :tcut].reshape(-1)
            qf = q[:, :tcut].reshape(-1)
            accumulate_global_marks(zf, rpf, qf, pooled, tot, maj, min_)
            n_batches += (int(feat_np.shape[0]) + bs - 1) // bs
            if n_batches >= cap:
                break
        return tot, maj, min_, n_songs

    @torch.no_grad()
    def _chord_pred_pass(
        self, module, model_factory, use_tag, use_idx, device, keep_cols_t,
        major_ring, minor_ring,
    ) -> Tuple[Dict[str, float], Dict[str, float], int]:
        """Pass 2 (streaming): GT-boundary chord recognition; accumulate WCSR counts.

        evaluate() is additive over rows, so summing per-song correct/total frame
        counts equals scoring all rows at once. Each song is scored and discarded.
        """
        correct: Dict[str, float] = defaultdict(float)
        total: Dict[str, float] = defaultdict(float)
        n_rows = 0
        for tid, feat_np, chord_2d, _key in tqdm(
            self._iter_val_songs(), total=len(self.val_track_ids),
            desc="ChordRing pass2 (chord)", unit="song", leave=False,
        ):
            z = self._infer_song_z(module, model_factory, feat_np, use_tag, use_idx, device, keep_cols_t)
            if z is None:
                continue
            rows: List[ChordRow] = []
            for si in range(z.shape[0]):
                rows.extend(_chord_rows_for_segment(
                    tid, z[si], chord_2d[si], major_ring, minor_ring,
                    self.smooth_win, self.qual_mode,
                ))
            if not rows:
                continue
            n_rows += len(rows)
            c, t, _s, _tc, _tt = evaluate(rows)
            for v in ("root", "majmin"):
                correct[v] += c[v]
                total[v] += t[v]
        return correct, total, n_rows

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        device = str(pl_module.device)
        module = pl_module.sae.sae
        model_factory = pl_module.model_factory
        use_tag = self.data_tag or str(pl_module.data_tag)
        n_sub = int(getattr(module, "n_saes", 0))
        use_idx = max(0, min(int(self.sae_idx), n_sub - 1))

        # ── (1/4) Ring enumeration — pure self-graph, no data ────────────────
        print("[ChordRingMetric] (1/4) enumerating size-12 rings (no data) ...", flush=True)
        try:
            rings = compute_size12_rings(module, device, self.block, self.prob)
        except Exception as e:
            print(f"[ChordRingMetric] ring enumeration failed: {e}")
            return 0.0
        print(f"[ChordRingMetric] found {len(rings)} size-12 ring(s).", flush=True)
        if len(rings) < 2:
            print("[ChordRingMetric] fewer than 2 rings; no major/minor target → acc=0.")
            return 0.0

        # Reduce to the union of ring columns and remap ids into that space.
        keep_cols = sorted({int(fid) for ring in rings for fid in ring})
        remap = {fid: k for k, fid in enumerate(keep_cols)}
        rings_reduced = [[remap[int(fid)] for fid in ring] for ring in rings]
        keep_cols_t = torch.tensor(keep_cols, dtype=torch.long, device=device)

        # ── (2/4) Identify major/minor — global-argmax voting, capped pass ───
        pooled, feat_to_k = pooled_ring_cols(rings_reduced)
        print(f"[ChordRingMetric] (2/4) identifying major/minor rings via global-argmax voting "
              f"(<= {self.ring_id_max_batches} batches, {len(keep_cols)} ring cols) ...", flush=True)
        tot, maj, min_, n_songs = self._accumulate_ring_marks(
            module, model_factory, use_tag, use_idx, device, keep_cols_t, pooled,
        )
        if n_songs == 0:
            print("[ChordRingMetric] no validation songs inferred; acc=0.")
            return 0.0
        sel = select_maj_min_rings(rings_reduced, tot, maj, min_, feat_to_k, self.semitone_step)
        major_ring, minor_ring = sel["major_ring"], sel["minor_ring"]
        if not minor_ring:
            print("[ChordRingMetric] only one ring identifiable; no minor target → acc=0.")
            return 0.0
        print(f"[ChordRingMetric] major ring idx={sel['major_ridx']} "
              f"score(maj-min)={sel['major_score']:.0f} off={sel['major_off']}; "
              f"minor ring idx={sel['minor_ridx']} score(min-maj)={sel['minor_score']:.0f} "
              f"off={sel['minor_off']}  (rings identified on {n_songs} song(s))", flush=True)

        # ── (3/4) Chord recognition — full-val streaming pass, released per song ─
        print("[ChordRingMetric] (3/4) GT-boundary chord recognition over the validation set ...",
              flush=True)
        correct, total, n_rows = self._chord_pred_pass(
            module, model_factory, use_tag, use_idx, device, keep_cols_t, major_ring, minor_ring,
        )
        if n_rows == 0:
            print("[ChordRingMetric] no chord rows produced; acc=0.")
            return 0.0

        # ── (4/4) Score = mean of root and root+min/maj WCSR ─────────────────
        root = (100.0 * correct["root"] / total["root"]) if total["root"] > 0 else 0.0
        majmin = (100.0 * correct["majmin"] / total["majmin"]) if total["majmin"] > 0 else 0.0
        score = 0.5 * (root + majmin)
        print(f"[ChordRingMetric] (4/4) root={root:.2f}%  majmin={majmin:.2f}%  "
              f"avg={score:.2f}%  (ring-id songs={n_songs}, rows={n_rows})", flush=True)

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"val/chord_root": root, "val/chord_majmin": majmin},
                step=trainer.global_step,
            )
        trainer.callback_metrics["val/chord_root"] = torch.tensor(root)
        trainer.callback_metrics["val/chord_majmin"] = torch.tensor(majmin)
        return score
