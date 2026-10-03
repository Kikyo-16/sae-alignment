"""MusicFM reps: musicfm_layer (hidden states at a specified transformer layer)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import h5py
import torch
import torchaudio as ta

from ..feature_utils import canonicalize_feat_2d
from ..registry import register_rep_handler

_MUSICFM_SR = 24000
_MUSICFM_DEFAULT_MODEL_ID = "tky823/MusicFM"


def _musicfm_state(
    device: torch.device, config: Optional[Dict[str, Any]] = None
) -> Tuple[Any, int]:
    cfg = config or {}
    model_id = cfg.get("musicfm_model_id", _MUSICFM_DEFAULT_MODEL_ID)
    try:
        from transformers import AutoModel  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "MusicFM rep requires `transformers`. Install via: pip install transformers"
        ) from exc

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    return model, _MUSICFM_SR


def _prepare_wav(
    seg: torch.Tensor,
    *,
    target_sr: int,
    musicfm_sr: int,
    device: torch.device,
) -> torch.Tensor:
    """Resample to 24 kHz if needed and return [1, T]."""
    if target_sr != musicfm_sr:
        seg = ta.functional.resample(seg.unsqueeze(0), target_sr, musicfm_sr).squeeze(0)
    return seg.unsqueeze(0).to(device)  # [1, T]


class MusicFMLayerRepHandler:
    name = "musicfm_layer"

    def init_models(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> Any:
        return _musicfm_state(device, config)

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        layer = int(cfg.get("musicfm_layer", 7))
        model, musicfm_sr = state
        duration_sec = win_samples / target_sr
        win_musicfm = int(round(duration_sec * musicfm_sr))
        probe = torch.zeros(1, win_musicfm, device=device)
        with torch.no_grad():
            out = model.get_latent(probe, layer_ix=layer)  # [1, T, C]
        feat_2d = canonicalize_feat_2d(out[0]).detach().cpu()  # [C, T]
        return tuple(feat_2d.shape), {
            "musicfm_model_id": cfg.get("musicfm_model_id", _MUSICFM_DEFAULT_MODEL_ID),
            "musicfm_layer": layer,
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
            "musicfm_layer",
            shape=(0, c, t),
            maxshape=(None, c, t),
            dtype=np_dtype,
            chunks=True,
            compression=comp,
        )
        main.attrs["musicfm_model_id"] = meta.get("musicfm_model_id", _MUSICFM_DEFAULT_MODEL_ID)
        main.attrs["musicfm_rep"] = self.name
        main.attrs["musicfm_layer"] = int(meta.get("musicfm_layer", 7))
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
        layer = int(cfg.get("musicfm_layer", 7))
        model, musicfm_sr = state
        wav = _prepare_wav(seg, target_sr=target_sr, musicfm_sr=musicfm_sr, device=device)
        with torch.no_grad():
            out = model.get_latent(wav, layer_ix=layer)  # [1, T, C]
        feat = canonicalize_feat_2d(out[0]).detach().cpu()  # [C, T]
        return feat, {}


register_rep_handler(MusicFMLayerRepHandler())
