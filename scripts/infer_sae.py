#!/usr/bin/env python3
"""
Music understanding with a trained SAE and its labelled orbits (feature_ids.txt).

Tasks (--task, one or more):
  key-signature  key detection ignoring the relative major/minor difference,
                 from the subdominant ([Forth]) ring (Table 2, App. D.2)
  key-chord      24-class major/minor key detection from the major- and
                 minor-chord rings (Table 3, App. D.3)
  chord          chord recognition (12 major + 12 minor + N) from the chord
                 rings (Table 1, App. D.1). Boundaries are estimated from the
                 [Chord Boundary] feature (Est. Bd) unless --chord-lab gives
                 reference boundaries (Oracle Bd).

Audio is cut into 30 s segments as in the paper. A trailing remainder shorter
than 30 s is dropped by default; --last-segment overlap instead infers one more
30 s window ending at the last sample and keeps only its non-overlapping part.

For the key tasks, --level segment writes one key per 30 s segment and
--level song writes one key per file (segment scores aggregated as in the
paper's song-level evaluation).

Outputs per audio file in --out-dir:
  <stem>.key-signature.<level>.tsv   start_sec, end_sec, key signature (e.g. G:maj/E:min)
  <stem>.key-chord.<level>.tsv       start_sec, end_sec, key (e.g. E:min)
  <stem>.chord.lab                   start_sec, end_sec, chord (e.g. C:maj, A:min, N)

Example:
    python scripts/infer_sae.py --ckpt ckpts/muq-2_k48/o-sae.ckpt \\
        --feature-ids results/muq-2_k48/feature_ids.txt \\
        --audio song.mp3 --task key-signature key-chord chord --level song
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import collect_audio, default_device

from src.inference.audio import LAST_SEGMENT_CHOICES, FeatureExtractor, extract_segments
from src.inference.decode import CHORD_DEFAULTS, LEVELS, TASK_CHORD, TASKS, load_lab, write_results
from src.inference.runners import SAERunner


def add_chord_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("chord decoding (defaults = paper settings)")
    g.add_argument("--smooth-win", type=int, default=CHORD_DEFAULTS["smooth_win"])
    g.add_argument("--majmin-mode", default=CHORD_DEFAULTS["majmin_mode"],
                   choices=["sum-peak", "seg-mean-max", "seg-mean-mean", "raw-tmpl"])
    g.add_argument("--majmin-conf-thr", type=float, default=CHORD_DEFAULTS["majmin_conf_thr"])
    g.add_argument("--majmin-alpha", type=float, default=CHORD_DEFAULTS["majmin_alpha"])


def chord_opts(args: argparse.Namespace) -> dict:
    opts = {k: getattr(args, k) for k in CHORD_DEFAULTS if hasattr(args, k)}
    return opts


def resolve_lab(chord_lab: str, audio: Path, n_audio: int):
    """--chord-lab is a .lab file (single audio) or a directory with <stem>.lab files."""
    if chord_lab is None:
        return None
    p = Path(chord_lab)
    if p.is_dir():
        lab = p / f"{audio.stem}.lab"
        if not lab.is_file():
            raise FileNotFoundError(f"No reference chord file {lab} for {audio}")
        return load_lab(lab)
    if n_audio != 1:
        raise ValueError("With several audio files, --chord-lab must be a directory of <stem>.lab files.")
    return load_lab(p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="LitSAE checkpoint (.ckpt).")
    ap.add_argument("--feature-ids", required=True, help="feature_ids.txt written by scripts/find_orbits.py.")
    ap.add_argument("--audio", required=True, nargs="+", help="Audio files and/or directories.")
    ap.add_argument("--out-dir", default="outputs", help="Output directory. Default: outputs.")
    ap.add_argument("--task", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--level", default="song", choices=LEVELS, help="Key tasks only. Default: song.")
    ap.add_argument("--last-segment", default="drop", choices=LAST_SEGMENT_CHOICES,
                    help="Trailing audio shorter than 30 s: drop it (default) or overlap.")
    ap.add_argument("--key-ring", default="forth", choices=["forth", "fifth", "tonic"],
                    help="key-signature: ring to use. The paper's o-SAE result uses forth (subdominant).")
    ap.add_argument("--chord-lab", default=None,
                    help="chord: reference boundaries (Oracle Bd) from a .lab file, or a directory "
                         "of <audio stem>.lab files. Default: estimated boundaries.")
    ap.add_argument("--boundary-rank", type=int, default=1,
                    help="chord (Est. Bd): use the k-th ranked [Chord Boundary] feature. Default: 1.")
    add_chord_args(ap)
    g = ap.add_argument_group("estimated chord boundaries (defaults = paper settings)")
    g.add_argument("--boundary-peak-tol", type=float, default=CHORD_DEFAULTS["boundary_peak_tol"])
    g.add_argument("--boundary-peak-drop", type=float, default=CHORD_DEFAULTS["boundary_peak_drop"])
    g.add_argument("--boundary-min-gap", type=int, default=CHORD_DEFAULTS["boundary_min_gap"])
    g.add_argument("--boundary-score-floor", type=float, default=CHORD_DEFAULTS["boundary_score_floor"])
    g.add_argument("--boundary-max-labels", type=int, default=CHORD_DEFAULTS["boundary_max_labels"])
    ap.add_argument("--model-id", default=None,
                    help="Override the Hugging Face id of the foundation model "
                         "(defaults: OpenMuQ/MuQ-large-msd-iter, tky823/MusicFM).")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--batch-size", type=int, default=32)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    audios = collect_audio(args.audio)
    runner = SAERunner(Path(args.ckpt), Path(args.feature_ids), args.device, args.batch_size)
    extractor = FeatureExtractor(runner.foundation_model, runner.layer, args.device, args.model_id)

    for audio in audios:
        lab = resolve_lab(args.chord_lab, audio, len(audios)) if TASK_CHORD in args.task else None
        segments = extract_segments(audio, extractor, args.last_segment)
        if not segments:
            print(f"[SKIP] {audio}: shorter than one 30 s segment.")
            continue
        res = runner.run(segments, args.task, key_ring=args.key_ring, chord_lab=lab,
                         boundary_rank=args.boundary_rank, chord_opts=chord_opts(args))
        write_results(Path(args.out_dir), audio.stem, args.level,
                      key_signature=res.get("key-signature"), key_chord=res.get("key-chord"),
                      chord=res.get("chord"), audio_end_sec=segments[-1].end_sec)


if __name__ == "__main__":
    main()
