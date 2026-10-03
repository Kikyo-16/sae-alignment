"""MuQ reps: muq_layer (hidden states at a specified layer)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import h5py
import torch
import torchaudio as ta

from ..feature_utils import canonicalize_feat_2d
from ..registry import register_rep_handler


def _muq_state(device: torch.device, config: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
    cfg = config or {}
    model_id = cfg.get("muq_model_id", "OpenMuQ/MuQ-large-msd-iter")
    muq_sr = int(cfg.get("muq_sr", 24000))
    try:
        from muq import MuQ  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "MuQ rep requires the official `muq` package. Install it via: pip install muq"
        ) from exc

    model = MuQ.from_pretrained(model_id).to(device)
    model.eval()
    return model, muq_sr


def _prepare_wavs(
    seg: torch.Tensor,
    *,
    target_sr: int,
    muq_sr: int,
    device: torch.device,
) -> torch.Tensor:
    if target_sr != muq_sr:
        seg_rs = ta.functional.resample(seg.unsqueeze(0), target_sr, muq_sr).squeeze(0)
    else:
        seg_rs = seg
    return seg_rs.unsqueeze(0).to(device)  # [1, T]


def _extract_hidden_states(model_out: Any):
    hs = getattr(model_out, "hidden_states", None)
    if hs is not None:
        return hs
    if isinstance(model_out, dict) and "hidden_states" in model_out:
        return model_out["hidden_states"]
    raise RuntimeError("MuQ model output does not contain hidden_states.")


class MuqLayerRepHandler:
    name = "muq_layer"

    def init_models(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> Any:
        return _muq_state(device, config)

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        muq_layer = int(cfg.get("muq_layer", -1))
        duration_sec = win_samples / target_sr
        model, muq_sr = state
        win_muq = int(round(duration_sec * muq_sr))
        probe = torch.zeros(win_muq, device=device)
        wavs = _prepare_wavs(
            probe,
            target_sr=muq_sr,
            muq_sr=muq_sr,
            device=device,
        )
        with torch.no_grad():
            out = model(wavs, output_hidden_states=True)
        hs = _extract_hidden_states(out)
        L = muq_layer if muq_layer >= 0 else len(hs) + muq_layer
        if not (0 <= L < len(hs)):
            raise ValueError(f"Invalid muq_layer={muq_layer}. hidden_states has len={len(hs)}.")
        feat_2d = canonicalize_feat_2d(hs[L][0]).detach().cpu()
        return tuple(feat_2d.shape), {
            "muq_model_id": cfg.get("muq_model_id", "OpenMuQ/MuQ-large-msd-iter"),
            "muq_layer": L,
        }

    def create_datasets(
        self,
        grp: h5py.Group,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        *,
        np_dtype,
        compression: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        comp = None if (compression in (None, "none")) else compression
        c, t = int(shape[0]), int(shape[1])
        main = grp.create_dataset(
            "muq_layer",
            shape=(0, c, t),
            maxshape=(None, c, t),
            dtype=np_dtype,
            chunks=True,
            compression=comp,
        )
        main.attrs["muq_model_id"] = meta.get("muq_model_id", "OpenMuQ/MuQ-large-msd-iter")
        main.attrs["muq_rep"] = self.name
        main.attrs["muq_layer"] = int(meta.get("muq_layer", -1))
        return {"main": main}

    def extract(
        self,
        seg: torch.Tensor,
        *,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        muq_layer = int(cfg.get("muq_layer", -1))
        model, muq_sr = state
        wavs = _prepare_wavs(
            seg,
            target_sr=target_sr,
            muq_sr=muq_sr,
            device=device,
        )
        with torch.no_grad():
            out = model(wavs, output_hidden_states=True)
        hs = _extract_hidden_states(out)
        L = muq_layer if muq_layer >= 0 else len(hs) + muq_layer
        if not (0 <= L < len(hs)):
            raise ValueError(f"Invalid muq_layer={muq_layer}. hidden_states has len={len(hs)}.")
        feat = canonicalize_feat_2d(hs[L][0]).detach().cpu()
        return feat, {}


register_rep_handler(MuqLayerRepHandler())
