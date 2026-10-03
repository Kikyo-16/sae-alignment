#!/usr/bin/env python3
"""Generate a controlled MIDI/SoundFont diagnostic dataset for pretrained music SAEs.

Dataset design (default):
  4 families x 12 roots x {major, natural minor} = 96 samples.

Each sample is exactly 30 seconds at 120 BPM:
  00.0-11.0  ascending scale across three octave regions
  11.0-22.0  descending scale across the same three octave regions
  22.0-24.0  I / i chord
  24.0-26.0  IV / iv chord
  26.0-28.0  V / v chord
  28.0-30.0  I / i chord

For strings, woodwinds, and brass, octave regions and chord voices are assigned
across different instruments within the family. MIDI files are rendered with
FluidSynth and a user-provided General MIDI SoundFont (.sf2).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import pretty_midi
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "Missing dependency: pretty_midi. Install it with: pip install pretty_midi"
    ) from exc


ROOTS: list[tuple[str, int]] = [
    ("C", 0),
    ("Db", 1),
    ("D", 2),
    ("Eb", 3),
    ("E", 4),
    ("F", 5),
    ("Gb", 6),
    ("G", 7),
    ("Ab", 8),
    ("A", 9),
    ("Bb", 10),
    ("B", 11),
]

SCALE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),  # natural minor
}


@dataclass(frozen=True)
class Voice:
    name: str
    gm_program: int  # zero-based General MIDI program number


@dataclass(frozen=True)
class FamilySpec:
    name: str
    voices: tuple[Voice, ...]
    scale_voice_indices: tuple[int, int, int]  # low, middle, high regions
    chord_voice_indices: tuple[int, int, int, int]  # low to high chord voices


FAMILIES: dict[str, FamilySpec] = {
    "piano": FamilySpec(
        name="piano",
        voices=(Voice("Acoustic Grand Piano", 0),),
        scale_voice_indices=(0, 0, 0),
        chord_voice_indices=(0, 0, 0, 0),
    ),
    "strings": FamilySpec(
        name="strings",
        voices=(
            Voice("Cello", 42),
            Voice("Viola", 41),
            Voice("Violin II", 40),
            Voice("Violin I", 40),
        ),
        scale_voice_indices=(0, 1, 3),
        chord_voice_indices=(0, 1, 2, 3),
    ),
    "woodwinds": FamilySpec(
        name="woodwinds",
        voices=(
            Voice("Bassoon", 70),
            Voice("Clarinet", 71),
            Voice("Oboe", 68),
            Voice("Flute", 73),
        ),
        scale_voice_indices=(0, 1, 3),
        chord_voice_indices=(0, 1, 2, 3),
    ),
    "brass": FamilySpec(
        name="brass",
        voices=(
            Voice("Trombone", 57),
            Voice("French Horn", 60),
            Voice("Trumpet II", 56),
            Voice("Trumpet I", 56),
        ),
        scale_voice_indices=(0, 1, 3),
        chord_voice_indices=(0, 1, 2, 3),
    ),
}


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float
    role: str
    instrument_names: tuple[str, ...]
    pitches: tuple[int, ...]
    harmony_label: str = ""


@dataclass(frozen=True)
class GeneratedSample:
    family: str
    root_name: str
    root_pc: int
    mode: str
    midi_path: Path
    wav_path: Path
    segments: tuple[Segment, ...]


NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


def midi_note_name(note: int) -> str:
    octave = note // 12 - 1
    return f"{NOTE_NAMES[note % 12]}{octave}"


def build_three_octave_scale(root_midi: int, mode: str) -> list[int]:
    intervals = SCALE_INTERVALS[mode]
    notes = [root_midi + 12 * octave + interval for octave in range(3) for interval in intervals]
    notes.append(root_midi + 36)
    if len(notes) != 22:
        raise AssertionError(f"Expected 22 scale notes, got {len(notes)}")
    return notes


def region_index(note: int, root_midi: int) -> int:
    """Return octave-region index 0/1/2 relative to the sample root."""
    relative = note - root_midi
    if relative < 12:
        return 0
    if relative < 24:
        return 1
    return 2


def chord_progression(root_midi: int, mode: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Return voice-led four-note I-IV-V-I or i-iv-v-i voicings.

    The minor version stays inside natural minor, so the dominant is minor v.
    """
    r = root_midi
    if mode == "major":
        return [
            ("I", (r + 0, r + 7, r + 16, r + 24)),
            ("IV", (r + 5, r + 12, r + 21, r + 24)),
            ("V", (r + 7, r + 14, r + 19, r + 23)),
            ("I_return", (r + 0, r + 7, r + 16, r + 24)),
        ]
    return [
        ("i", (r + 0, r + 7, r + 15, r + 24)),
        ("iv", (r + 5, r + 12, r + 20, r + 24)),
        ("v", (r + 7, r + 14, r + 19, r + 22)),
        ("i_return", (r + 0, r + 7, r + 15, r + 24)),
    ]


def add_note(
    instrument: "pretty_midi.Instrument",
    pitch: int,
    start: float,
    end: float,
    velocity: int,
) -> None:
    if not 0 <= pitch <= 127:
        raise ValueError(f"MIDI pitch out of range: {pitch}")
    instrument.notes.append(
        pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
    )


def generate_one_midi(
    output_root: Path,
    family: FamilySpec,
    root_name: str,
    root_pc: int,
    mode: str,
    bpm: float,
    velocity: int,
    note_gate: float,
) -> GeneratedSample:
    seconds_per_beat = 60.0 / bpm
    if abs(seconds_per_beat - 0.5) > 1e-9:
        raise ValueError(
            "This dataset layout assumes 120 BPM so 22 notes occupy 11 seconds. "
            "Use --bpm 120 unless you also modify the timing design."
        )

    # C2-B2 across the 12 roots; the top scale note ends at C5-B5.
    root_midi = 36 + root_pc
    scale_up = build_three_octave_scale(root_midi, mode)
    scale_down = list(reversed(scale_up))

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    tracks = [
        pretty_midi.Instrument(program=v.gm_program, name=v.name)
        for v in family.voices
    ]

    segments: list[Segment] = []

    # Ascending scale: 22 onsets from 0.0 to 10.5 seconds.
    time_sec = 0.0
    region_note_lists: list[list[int]] = [[], [], []]
    region_starts: list[float | None] = [None, None, None]
    region_ends: list[float | None] = [None, None, None]

    for note in scale_up:
        region = region_index(note, root_midi)
        track_index = family.scale_voice_indices[region]
        if region_starts[region] is None:
            region_starts[region] = time_sec
        end = time_sec + seconds_per_beat * note_gate
        add_note(tracks[track_index], note, time_sec, end, velocity)
        region_note_lists[region].append(note)
        region_ends[region] = time_sec + seconds_per_beat
        time_sec += seconds_per_beat

    for region, role in enumerate(("scale_up_low", "scale_up_mid", "scale_up_high")):
        track_index = family.scale_voice_indices[region]
        segments.append(
            Segment(
                start_sec=float(region_starts[region]),
                end_sec=float(region_ends[region]),
                role=role,
                instrument_names=(family.voices[track_index].name,),
                pitches=tuple(region_note_lists[region]),
            )
        )

    # Descending scale: another 22 notes, from 11.0 to 22.0 seconds.
    down_region_note_lists: list[list[int]] = [[], [], []]
    down_region_starts: list[float | None] = [None, None, None]
    down_region_ends: list[float | None] = [None, None, None]

    for note in scale_down:
        region = region_index(note, root_midi)
        track_index = family.scale_voice_indices[region]
        if down_region_starts[region] is None:
            down_region_starts[region] = time_sec
        end = time_sec + seconds_per_beat * note_gate
        add_note(tracks[track_index], note, time_sec, end, velocity)
        down_region_note_lists[region].append(note)
        down_region_ends[region] = time_sec + seconds_per_beat
        time_sec += seconds_per_beat

    # Preserve temporal order: high -> middle -> low while descending.
    for region, role in ((2, "scale_down_high"), (1, "scale_down_mid"), (0, "scale_down_low")):
        track_index = family.scale_voice_indices[region]
        segments.append(
            Segment(
                start_sec=float(down_region_starts[region]),
                end_sec=float(down_region_ends[region]),
                role=role,
                instrument_names=(family.voices[track_index].name,),
                pitches=tuple(down_region_note_lists[region]),
            )
        )

    if abs(time_sec - 22.0) > 1e-9:
        raise AssertionError(f"Scale section should end at 22.0 seconds, got {time_sec}")

    # Four two-second block chords from 22.0 to 30.0 seconds.
    chord_duration = 2.0
    for harmony_label, chord_notes in chord_progression(root_midi, mode):
        chord_start = time_sec
        chord_end = chord_start + chord_duration
        active_names: list[str] = []
        for pitch, track_index in zip(chord_notes, family.chord_voice_indices):
            add_note(tracks[track_index], pitch, chord_start, chord_end, velocity)
            active_names.append(family.voices[track_index].name)
        segments.append(
            Segment(
                start_sec=chord_start,
                end_sec=chord_end,
                role="block_chord",
                instrument_names=tuple(active_names),
                pitches=tuple(chord_notes),
                harmony_label=harmony_label,
            )
        )
        time_sec = chord_end

    if abs(time_sec - 30.0) > 1e-9:
        raise AssertionError(f"Sample should end at 30.0 seconds, got {time_sec}")

    for track in tracks:
        if track.notes:
            track.notes.sort(key=lambda n: (n.start, n.pitch))
            midi.instruments.append(track)

    stem = f"{root_name}_{mode}"
    midi_path = output_root / "midi" / family.name / f"{stem}.mid"
    wav_path = output_root / "audio" / family.name / f"{stem}.wav"
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(midi_path))

    return GeneratedSample(
        family=family.name,
        root_name=root_name,
        root_pc=root_pc,
        mode=mode,
        midi_path=midi_path,
        wav_path=wav_path,
        segments=tuple(segments),
    )


def force_wav_duration(wav_path: Path, duration_sec: float, expected_sr: int) -> None:
    """Trim or zero-pad an uncompressed WAV to an exact frame count."""
    with wave.open(str(wav_path), "rb") as reader:
        params = reader.getparams()
        if params.framerate != expected_sr:
            raise RuntimeError(
                f"Unexpected sample rate for {wav_path}: {params.framerate}, expected {expected_sr}"
            )
        if params.comptype != "NONE":
            raise RuntimeError(f"Compressed WAV is not supported: {wav_path}")
        raw = reader.readframes(params.nframes)

    bytes_per_frame = params.nchannels * params.sampwidth
    target_frames = int(round(duration_sec * expected_sr))
    target_bytes = target_frames * bytes_per_frame
    if len(raw) >= target_bytes:
        fixed = raw[:target_bytes]
    else:
        fixed = raw + (b"\x00" * (target_bytes - len(raw)))

    tmp_path = wav_path.with_suffix(".tmp.wav")
    with wave.open(str(tmp_path), "wb") as writer:
        writer.setparams(
            (
                params.nchannels,
                params.sampwidth,
                params.framerate,
                target_frames,
                params.comptype,
                params.compname,
            )
        )
        writer.writeframes(fixed)
    tmp_path.replace(wav_path)


def render_one(
    sample: GeneratedSample,
    fluidsynth_exe: str,
    soundfont: Path,
    sample_rate: int,
    gain: float,
) -> None:
    sample.wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        fluidsynth_exe,
        "-ni",
        "-R",
        "0",
        "-C",
        "0",
        "-g",
        str(gain),
        "-r",
        str(sample_rate),
        "-F",
        str(sample.wav_path),
        str(soundfont),
        str(sample.midi_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FluidSynth failed for {sample.midi_path}\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    force_wav_duration(sample.wav_path, duration_sec=30.0, expected_sr=sample_rate)


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_metadata(output_root: Path, samples: Sequence[GeneratedSample], bpm: float) -> None:
    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "family",
                "root",
                "root_pc",
                "mode",
                "scale_type",
                "progression",
                "bpm",
                "duration_sec",
                "midi_path",
                "wav_path",
                "family_instruments",
            ],
        )
        writer.writeheader()
        for sample in samples:
            family = FAMILIES[sample.family]
            progression = "I-IV-V-I" if sample.mode == "major" else "i-iv-v-i"
            writer.writerow(
                {
                    "sample_id": f"{sample.family}_{sample.root_name}_{sample.mode}",
                    "family": sample.family,
                    "root": sample.root_name,
                    "root_pc": sample.root_pc,
                    "mode": sample.mode,
                    "scale_type": "major" if sample.mode == "major" else "natural_minor",
                    "progression": progression,
                    "bpm": bpm,
                    "duration_sec": 30.0,
                    "midi_path": relative_to(sample.midi_path, output_root),
                    "wav_path": relative_to(sample.wav_path, output_root),
                    "family_instruments": json.dumps(
                        [voice.name for voice in family.voices], ensure_ascii=False
                    ),
                }
            )

    segments_path = output_root / "segments.csv"
    with segments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "start_sec",
                "end_sec",
                "role",
                "harmony_label",
                "instruments",
                "midi_pitches",
                "pitch_names",
            ],
        )
        writer.writeheader()
        for sample in samples:
            sample_id = f"{sample.family}_{sample.root_name}_{sample.mode}"
            for segment in sample.segments:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "start_sec": f"{segment.start_sec:.3f}",
                        "end_sec": f"{segment.end_sec:.3f}",
                        "role": segment.role,
                        "harmony_label": segment.harmony_label,
                        "instruments": json.dumps(segment.instrument_names, ensure_ascii=False),
                        "midi_pitches": json.dumps(segment.pitches),
                        "pitch_names": json.dumps(
                            [midi_note_name(p) for p in segment.pitches], ensure_ascii=False
                        ),
                    }
                )


def parse_names(value: str, allowed: Iterable[str], argument_name: str) -> list[str]:
    allowed_set = set(allowed)
    if value.strip().lower() == "all":
        return sorted(allowed_set)
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [name for name in names if name not in allowed_set]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid {argument_name}: {invalid}. Allowed: {sorted(allowed_set)} or 'all'."
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optionally SoundFont-render the SAE diagnostic dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sae_diagnostic_dataset"),
        help="Dataset output directory (default: sae_diagnostic_dataset).",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        help="Path to a General MIDI .sf2 SoundFont. Required unless --midi-only is used.",
    )
    parser.add_argument(
        "--fluidsynth",
        default="fluidsynth",
        help="FluidSynth executable name or path (default: fluidsynth).",
    )
    parser.add_argument(
        "--midi-only",
        action="store_true",
        help="Generate MIDI and metadata without rendering WAV files.",
    )
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated subset of piano,strings,woodwinds,brass, or all.",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help="Comma-separated subset of major,minor, or all.",
    )
    parser.add_argument(
        "--roots",
        default="all",
        help="Comma-separated root names such as C,Db,Gb,Bb, or all.",
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--velocity", type=int, default=90)
    parser.add_argument(
        "--note-gate",
        type=float,
        default=0.9,
        help="Scale-note duration as a fraction of one beat (default: 0.9).",
    )
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of concurrent FluidSynth rendering processes (default: 1).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not 1 <= args.velocity <= 127:
        parser.error("--velocity must be in [1, 127].")
    if not 0.0 < args.note_gate <= 1.0:
        parser.error("--note-gate must be in (0, 1].")
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive.")
    if args.jobs <= 0:
        parser.error("--jobs must be positive.")

    try:
        selected_families = parse_names(args.families, FAMILIES.keys(), "families")
        selected_modes = parse_names(args.modes, SCALE_INTERVALS.keys(), "modes")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    root_lookup = {name.lower(): (name, pc) for name, pc in ROOTS}
    if args.roots.strip().lower() == "all":
        selected_roots = ROOTS
    else:
        requested_roots = [item.strip().lower() for item in args.roots.split(",") if item.strip()]
        invalid_roots = [name for name in requested_roots if name not in root_lookup]
        if invalid_roots:
            parser.error(
                f"Invalid roots: {invalid_roots}. Allowed: {[name for name, _ in ROOTS]} or 'all'."
            )
        selected_roots = [root_lookup[name] for name in requested_roots]

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples: list[GeneratedSample] = []
    for family_name in selected_families:
        family = FAMILIES[family_name]
        for root_name, root_pc in selected_roots:
            for mode in selected_modes:
                sample = generate_one_midi(
                    output_root=output_root,
                    family=family,
                    root_name=root_name,
                    root_pc=root_pc,
                    mode=mode,
                    bpm=args.bpm,
                    velocity=args.velocity,
                    note_gate=args.note_gate,
                )
                samples.append(sample)

    samples.sort(key=lambda s: (s.family, s.root_pc, s.mode))
    write_metadata(output_root, samples, bpm=args.bpm)
    print(f"Generated {len(samples)} MIDI files in: {output_root / 'midi'}")

    if args.midi_only:
        print("Skipped WAV rendering (--midi-only).")
        print(f"Metadata: {output_root / 'metadata.csv'}")
        print(f"Segments: {output_root / 'segments.csv'}")
        return 0

    if args.soundfont is None:
        parser.error("--soundfont is required unless --midi-only is used.")
    soundfont = args.soundfont.expanduser().resolve()
    if not soundfont.is_file():
        parser.error(f"SoundFont not found: {soundfont}")

    fluidsynth_exe = args.fluidsynth
    if Path(fluidsynth_exe).expanduser().is_file():
        fluidsynth_exe = str(Path(fluidsynth_exe).expanduser().resolve())
    elif shutil.which(fluidsynth_exe) is None:
        parser.error(
            f"FluidSynth executable not found: {fluidsynth_exe}. "
            "Install FluidSynth or pass --fluidsynth /path/to/fluidsynth."
        )

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_sample = {
            executor.submit(
                render_one,
                sample,
                fluidsynth_exe,
                soundfont,
                args.sample_rate,
                args.gain,
            ): sample
            for sample in samples
        }
        completed_count = 0
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                future.result()
            except Exception as exc:  # continue rendering other files
                failures.append(f"{sample.midi_path}: {exc}")
            completed_count += 1
            print(f"Rendered {completed_count}/{len(samples)}", end="\r", flush=True)
    print()

    if failures:
        print("Rendering completed with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Rendered {len(samples)} WAV files in: {output_root / 'audio'}")
    print(f"Metadata: {output_root / 'metadata.csv'}")
    print(f"Segments: {output_root / 'segments.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
