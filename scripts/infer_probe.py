#!/usr/bin/env python3
"""
Music understanding with a trained linear probe (baseline of the paper).

Tasks (--task):
  key-signature  needs a key_relative probe; --degree selects the scale degree
                 it was trained on (forth = subdominant, used in Table 2)
  key-chord      needs the 25-class chord probe (Table 3)
  chord          needs the 25-class chord probe and reference boundaries
                 (--chord-lab), as the probe baseline is defined with Oracle Bd
                 only (Table 1)

Audio segmentation, --level, --last-segment and the outputs are the same as in
scripts/infer_sae.py. Probes do not store the foundation-model layer, so pass
--model / --layer matching the probe.

Example:
    python scripts/infer_probe.py --probe-ckpt ckpts/probes/key_forth_muq-5.ckpt \\
        --model muq --layer 5 --task key-signature --degree forth --audio song.mp3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import collect_audio, default_device
from infer_sae import add_chord_args, chord_opts, resolve_lab

from src.inference.audio import LAST_SEGMENT_CHOICES, FeatureExtractor, extract_segments
from src.inference.decode import LEVELS, TASK_CHORD, TASKS, write_results
from src.inference.runners import ProbeRunner


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-ckpt", required=True, help="LitLinearProbe checkpoint (.ckpt).")
    ap.add_argument("--model", required=True, choices=["muq", "musicfm"], help="Foundation model of the probe.")
    ap.add_argument("--layer", required=True, type=int, help="Foundation-model layer of the probe.")
    ap.add_argument("--audio", required=True, nargs="+", help="Audio files and/or directories.")
    ap.add_argument("--out-dir", default="outputs", help="Output directory. Default: outputs.")
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--degree", default="forth", choices=["forth", "fifth", "tonic"],
                    help="key-signature: scale degree the key_relative probe was trained on.")
    ap.add_argument("--level", default="song", choices=LEVELS, help="Key tasks only. Default: song.")
    ap.add_argument("--last-segment", default="drop", choices=LAST_SEGMENT_CHOICES,
                    help="Trailing audio shorter than 30 s: drop it (default) or overlap.")
    ap.add_argument("--chord-lab", default=None,
                    help="chord: reference boundaries from a .lab file, or a directory of "
                         "<audio stem>.lab files (required for --task chord).")
    add_chord_args(ap)
    ap.add_argument("--model-id", default=None, help="Override the Hugging Face id of the foundation model.")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--batch-size", type=int, default=64)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == TASK_CHORD and args.chord_lab is None:
        raise SystemExit("--task chord with a probe needs --chord-lab (reference boundaries).")
    audios = collect_audio(args.audio)
    runner = ProbeRunner(Path(args.probe_ckpt), args.device, args.batch_size)
    extractor = FeatureExtractor(args.model, args.layer, args.device, args.model_id)

    for audio in audios:
        lab = resolve_lab(args.chord_lab, audio, len(audios)) if args.task == TASK_CHORD else None
        segments = extract_segments(audio, extractor, args.last_segment)
        if not segments:
            print(f"[SKIP] {audio}: shorter than one 30 s segment.")
            continue
        res = runner.run(segments, args.task, degree=args.degree, chord_lab=lab, chord_opts=chord_opts(args))
        write_results(Path(args.out_dir), audio.stem, args.level,
                      key_signature=res.get("key-signature"), key_chord=res.get("key-chord"),
                      chord=res.get("chord"), audio_end_sec=segments[-1].end_sec)


if __name__ == "__main__":
    main()
