"""Small helpers shared by the command-line scripts in this folder."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".au")


def run_module(module: str, args: Sequence[object]) -> None:
    """Run ``python -m <module> <args>`` with the repository on PYTHONPATH."""
    cmd = [sys.executable, "-m", module] + [str(a) for a in args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("[RUN] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def read_sae_hparams(ckpt_path: Path) -> dict:
    """Return the ``exp_config`` stored in a LitSAE checkpoint."""
    import torch
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    if "exp_config" not in hp:
        raise RuntimeError(f"{ckpt_path} does not look like a LitSAE checkpoint (no exp_config).")
    return hp["exp_config"]


def collect_audio(paths: Iterable[str]) -> List[Path]:
    """Expand files and directories (non-recursive) into a sorted audio file list."""
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"Audio path not found: {p}")
    if not out:
        raise FileNotFoundError("No audio files found.")
    return out
