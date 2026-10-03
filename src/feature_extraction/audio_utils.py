from __future__ import annotations

"""
Common audio utilities shared by feature extraction scripts.

Contains:
- Optional TorchCodec-based loader with torchaudio fallback
- Mono loading and resampling
- Silence detection
- Simple phase-vocoder pitch shift that preserves segment length
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio as ta

# ---------------------------
# Optional: TorchCodec for faster decode
# ---------------------------
try:
    from torchcodec.decoders import AudioDecoder as _TC_AudioDecoder

    _HAVE_TC = True
except Exception:
    _HAVE_TC = False


# ---------------------------
# audio I/O
# ---------------------------

def load_mono(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio from disk and return mono waveform (1, N) and sample rate.
    Prefers TorchCodec when available, otherwise falls back to torchaudio.load.
    """
    if _HAVE_TC:
        dec = _TC_AudioDecoder(path)
        samples = dec.get_all_samples()
        wav = samples.data  # (C, N) or (N,)
        sr = int(samples.sample_rate)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sr
    else:
        wav, sr = ta.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav, sr


def resample_to(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform to target_sr if needed."""
    return ta.functional.resample(wav, sr, target_sr) if sr != target_sr else wav


# ---------------------------
# silence detection
# ---------------------------

def is_all_silent(seg: torch.Tensor, eps: float = 1e-5) -> bool:
    x = seg.detach()
    if x.ndim != 1:
        x = x.view(-1)
    if x.numel() == 0:
        return True
    return bool(x.abs().max().item() < eps)


def has_long_silence(
    seg: torch.Tensor,
    sr: int,
    min_silence_sec: float = 2.0,
    silence_db: float = -45.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    all_silent_eps: float = 1e-5,
) -> bool:
    """
    True if continuous silence >= min_silence_sec.
    Silence: frame RMS(dB) < silence_db.
    Duration fixed: (frame_len + (run_frames-1)*hop)/sr
    """
    x = seg.detach()
    if x.ndim != 1:
        x = x.view(-1)

    if x.numel() == 0:
        return True
    if is_all_silent(x, eps=all_silent_eps):
        return True

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))

    if x.numel() < frame_len:
        return True

    frames = x.unfold(0, frame_len, hop)
    if frames.numel() == 0:
        return True

    rms = torch.sqrt((frames.float() ** 2).mean(dim=1) + 1e-12)
    rms_db = 20.0 * torch.log10(rms + 1e-12)

    below = (rms_db < silence_db).cpu().numpy().astype(np.int8)
    if below.size == 0:
        return False

    diffs = np.diff(np.pad(below, (1, 1), mode="constant"))
    run_st = np.where(diffs == 1)[0]
    run_en = np.where(diffs == -1)[0]
    if run_st.size == 0:
        return False

    run_frames = int(np.max(run_en - run_st))
    if run_frames <= 0:
        return False

    longest_sec = (frame_len + (run_frames - 1) * hop) / float(sr)
    return longest_sec >= float(min_silence_sec)


# ---------------------------
# pitch shift (no sox) — phase vocoder + interpolate back to original length
# ---------------------------

_PV_CACHE = {}


def _pv_get_window_and_phase(n_fft: int, hop_length: int, device: torch.device, dtype: torch.dtype):
    key = (n_fft, hop_length, str(device), str(dtype))
    if key in _PV_CACHE:
        return _PV_CACHE[key]
    window = torch.hann_window(n_fft, device=device, dtype=dtype)
    phase_advance = torch.linspace(
        0,
        torch.pi * hop_length,
        n_fft // 2 + 1,
        device=device,
        dtype=dtype,
    ).unsqueeze(-1)
    _PV_CACHE[key] = (window, phase_advance)
    return window, phase_advance


def pitch_shift_pv_1d(
    x: torch.Tensor,
    sr: int,
    semitone: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """1D pitch shift, keeps output length == input length."""
    if semitone == 0:
        return x

    if x.ndim != 1:
        x = x.view(-1)
    N = x.numel()
    if N == 0:
        return x

    device = x.device
    dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    x = x.to(dtype)

    factor = float(2.0 ** (semitone / 12.0))
    rate = 1.0 / factor

    window, phase_advance = _pv_get_window_and_phase(n_fft, hop_length, device, dtype)
    phase_advance_b = phase_advance.unsqueeze(0)

    xb = x.unsqueeze(0)
    spec = torch.stft(
        xb,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )

    spec_stretch = ta.functional.phase_vocoder(spec, rate=rate, phase_advance=phase_advance_b)
    target_len = int(round(N / rate))

    y = torch.istft(
        spec_stretch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=target_len,
    ).squeeze(0)

    y = F.interpolate(y.unsqueeze(0).unsqueeze(0), size=N, mode="linear", align_corners=False).squeeze()
    return y

