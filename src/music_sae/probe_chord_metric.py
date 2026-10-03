from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ..linear_probe.probe_lit import LitLinearProbe
from .chord_ring_metric import ChordRingMetric


class ProbeChordMetric(ChordRingMetric):
    """
    Validation metric for the music SAE that selects the major/minor chord
    feature ids from a trained linear-probe ``chord`` checkpoint instead of
    enumerating self-graph rings.

    Each validation epoch, on the validation track set:
      1. Loads the chord probe (25-class: 12 maj + 12 min + N) once and caches it.
      2. For each of the 24 maj/min probe output weight vectors, finds the nearest
         SAE encoder feature (cosine similarity, exactly as ProbeKeyMetric) in the
         *live* sub-SAE — so the matched feature ids track the SAE as it trains.
         Rows 0-11 → major feature ids (root pc 0..11), rows 12-23 → minor.
      3. Performs ground-truth-boundary chord recognition with those feature ids
         (the shared ``predict_chord_segments`` path inherited from ChordRingMetric).
      4. Returns the mean of the segment-level WCSR for the 'root' and 'majmin'
         (root + min/maj) vocabularies.

    The N (no-chord) probe row is unused — silence is decided from the GT chord
    labels by the inherited chord-prediction pass.
    """

    def __init__(
        self,
        probe_ckpt: str,
        h5_path: str,
        feature_ds: str,
        data_tag: str,
        val_track_id_file: str,
        split: str = "train",
        sae_idx: int = 0,
        batch_size: int = 64,
        smooth_win: int = 9,
        qual_mode: str = "sum_peak",
        max_songs: Optional[int] = None,
    ) -> None:
        super().__init__(
            h5_path=h5_path,
            feature_ds=feature_ds,
            data_tag=data_tag,
            val_track_id_file=val_track_id_file,
            split=split,
            sae_idx=sae_idx,
            batch_size=batch_size,
            smooth_win=smooth_win,
            qual_mode=qual_mode,
            max_songs=max_songs,
        )
        self.probe_ckpt = probe_ckpt
        self._probe_base: Optional[LitLinearProbe] = None

    @property
    def name(self) -> str:
        return "val/probe_chord_acc"

    def _base_probe(self) -> LitLinearProbe:
        if self._probe_base is None:
            print(f"[ProbeChordMetric] Loading chord probe from {self.probe_ckpt}")
            self._probe_base = LitLinearProbe.load_from_checkpoint(
                self.probe_ckpt, map_location="cpu", strict=False
            )
            self._probe_base.eval()
        return self._probe_base

    @torch.no_grad()
    def _select_chord_feature_ids(
        self, pl_module: pl.LightningModule, device: str
    ) -> Tuple[List[int], List[int]]:
        """Cosine-NN maj/min feature ids from the chord probe in the live SAE.

        Returns ``(major_ids, minor_ids)`` — each a length-12 list of SAE hidden
        feature ids in pitch-class order (position p ↦ root pc p).
        """
        probe = self._base_probe()
        probe_w = probe.linear.weight.data.to(device)  # (25, d_model)

        sae_module = pl_module.sae.sae.sae_modules[self.sae_idx]
        enc_w = sae_module.fc1.weight.data.to(device)  # (hidden, d_model)

        # Rows 0-11 = C_maj..B_maj, 12-23 = C_min..B_min (row 24 = N, unused).
        chord_w = probe_w[:24]
        cos_sim = F.normalize(chord_w, dim=1) @ F.normalize(enc_w, dim=1).T  # (24, hidden)
        best_idx = cos_sim.argmax(dim=1)
        best_scores = cos_sim[torch.arange(24, device=device), best_idx]

        major_ids = best_idx[:12].tolist()
        minor_ids = best_idx[12:24].tolist()
        maj_sc = [round(float(v), 4) for v in best_scores[:12].tolist()]
        min_sc = [round(float(v), 4) for v in best_scores[12:24].tolist()]
        print(f"[ProbeChordMetric] major feature ids: {major_ids}")
        print(f"[ProbeChordMetric] major cosine sims: {maj_sc}")
        print(f"[ProbeChordMetric] minor feature ids: {minor_ids}")
        print(f"[ProbeChordMetric] minor cosine sims: {min_sc}")
        return major_ids, minor_ids

    def compute(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        device = str(pl_module.device)
        module = pl_module.sae.sae
        model_factory = pl_module.model_factory
        use_tag = self.data_tag or str(pl_module.data_tag)
        n_sub = int(getattr(module, "n_saes", 0))
        use_idx = max(0, min(int(self.sae_idx), n_sub - 1))

        # ── (1/3) Pick major/minor feature ids from the chord probe ──────────
        print("[ProbeChordMetric] (1/3) selecting major/minor feature ids from chord probe ...",
              flush=True)
        major_ids, minor_ids = self._select_chord_feature_ids(pl_module, device)

        # Reduce inference to the union of selected columns and remap maj/min
        # lists into that reduced space (preserving pitch-class order).
        keep_cols = sorted({int(c) for c in (major_ids + minor_ids)})
        remap = {fid: k for k, fid in enumerate(keep_cols)}
        major_ring = [remap[int(c)] for c in major_ids]
        minor_ring = [remap[int(c)] for c in minor_ids]
        keep_cols_t = torch.tensor(keep_cols, dtype=torch.long, device=device)

        # ── (2/3) Chord recognition — full-val streaming pass ────────────────
        print("[ProbeChordMetric] (2/3) GT-boundary chord recognition over the validation set ...",
              flush=True)
        correct, total, n_rows = self._chord_pred_pass(
            module, model_factory, use_tag, use_idx, device, keep_cols_t, major_ring, minor_ring,
        )
        if n_rows == 0:
            print("[ProbeChordMetric] no chord rows produced; acc=0.")
            return 0.0

        # ── (3/3) Score = mean of root and root+min/maj WCSR ─────────────────
        root = (100.0 * correct["root"] / total["root"]) if total["root"] > 0 else 0.0
        majmin = (100.0 * correct["majmin"] / total["majmin"]) if total["majmin"] > 0 else 0.0
        score = 0.5 * (root + majmin)
        print(f"[ProbeChordMetric] (3/3) root={root:.2f}%  majmin={majmin:.2f}%  "
              f"avg={score:.2f}%  (rows={n_rows})", flush=True)

        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {"val/chord_root": root, "val/chord_majmin": majmin},
                step=trainer.global_step,
            )
        trainer.callback_metrics["val/chord_root"] = torch.tensor(root)
        trainer.callback_metrics["val/chord_majmin"] = torch.tensor(majmin)
        return score
