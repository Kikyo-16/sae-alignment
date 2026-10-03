#!/usr/bin/env python3
"""Evaluate cached LVCR chord predictions with the shared HDF5 evaluator.

By default this script does *not* run LVCR inference.  It reads existing native
large-vocabulary Harte-style ``.lab`` predictions from:

* ``../LVCR/results/baseline_pop909_test``
* ``../LVCR/results/baseline_rwc``
* ``../LVCR/results/baseline_slakh2100_test``

The predictions are sampled onto the exact frame timelines stored in the same
HDF5 files used by the SAE experiments.  A step-4-compatible cache is produced
and evaluated with:

    python -m src.analysis.eval_metrics chord CACHE \
        --h5-path DATA.h5 --split SPLIT

Consequently, LVCR and the SAE/probe systems share:

* the same HDF5 samples and frame-level references;
* the same Root, MajMin, and MIREX comparators;
* the same N/X handling;
* the same song-level macro and corpus-level micro aggregation.

Large-vocabulary predictions are intentionally preserved in the cache.  The
shared ``mir_eval.chord.majmin`` comparator performs the MajMin reduction:
triad-preserving major/minor extensions are folded to major/minor, while an
estimate that cannot be reduced to MajMin is counted as incorrect whenever the
reference frame is valid under MajMin.

Example: evaluate existing predictions
--------------------------------------
python scripts/analyze/run_baseline_chord.py \
    --pop909-h5 sae-data/pop909/muq/pop909_muq_30s_layer_6.h5 \
    --pop909-split train \
    --rwc-h5 sae-data/rwc/muq/rwc_muq_30s_layer_6.h5 \
    --rwc-split test \
    --slakh-h5 sae-data/slakh/muq/slakh_muq_30s_layer_6.h5 \
    --slakh-split test

Inference is opt-in through ``--run-inference``.  In that mode, provide the
corresponding audio roots and ``--chord-script``.  The expected command is:

    python chord_recognition.py INPUT_AUDIO OUTPUT_LAB

Additional model arguments may be supplied repeatedly with ``--chord-arg``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np


NO_CHORD_ESTIMATES = frozenset({"N", "X", "NOCHORD", "N/A", "", "UNKNOWN"})
NO_CHORD_REFERENCES = frozenset({"N", "NOCHORD"})
REQUIRED_METRIC_KEYS = (
    "root",
    "majmin",
    "mirex",
    "micro_root",
    "micro_majmin",
    "micro_mirex",
    "n_songs",
    "n_cache_samples",
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    h5_path: Path
    split: str
    result_dir: Path
    audio_root: Optional[Path] = None


@dataclass(frozen=True)
class AudioTrack:
    dataset: str
    track_id: str
    audio_path: Path


@dataclass(frozen=True)
class LabSegment:
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class H5Sample:
    index: int
    track_id: str
    start_sec: float
    end_sec: float
    ref_labels: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.ref_labels.shape[0])


@dataclass(frozen=True)
class RunResult:
    track: AudioTrack
    lab_path: Path
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "track"


def _track_aliases(track_id: str) -> List[str]:
    """Return conservative aliases used only for audio/HDF5 track matching."""
    raw = str(track_id).strip().replace("\\", "/")
    values = [raw, Path(raw).name, Path(raw).stem]

    aliases: List[str] = []
    for value in values:
        value = value.strip().lower()
        if not value:
            continue
        aliases.append(value)
        if value.isdigit():
            number = int(value)
            aliases.extend((str(number), f"{number:03d}", f"{number:04d}"))

    # A few datasets wrap a numeric identifier in a filename-like string.
    # Numeric aliases are accepted only when they identify one unique audio
    # record in the final index, so adding them here cannot silently choose
    # between multiple tracks.
    match = re.fullmatch(r"[^0-9]*([0-9]+)[^0-9]*", Path(raw).stem)
    if match:
        number = int(match.group(1))
        aliases.extend((str(number), f"{number:03d}", f"{number:04d}"))

    return list(dict.fromkeys(aliases))


def _canonical_estimate(label: str) -> str:
    """Normalize only no-chord aliases; keep the full chord vocabulary."""
    text = str(label).strip()
    return "N" if text.upper() in NO_CHORD_ESTIMATES else text


def parse_lab_file(path: Path) -> List[LabSegment]:
    """Parse a whitespace-delimited ``start end label`` chord file.

    Significant overlaps are rejected because they make the prediction at a
    frame ambiguous.  Gaps are allowed and are later filled with ``N``.
    """
    segments: List[LabSegment] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid start/end") from exc
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                raise ValueError(
                    f"{path}:{line_no}: invalid interval [{start}, {end})"
                )
            label = _canonical_estimate(" ".join(parts[2:]))
            segments.append(LabSegment(start, end, label))

    segments.sort(key=lambda item: (item.start, item.end))
    tolerance = 1e-6
    for previous, current in zip(segments, segments[1:]):
        if current.start < previous.end - tolerance:
            raise ValueError(
                f"{path}: overlapping prediction intervals "
                f"[{previous.start}, {previous.end}) and "
                f"[{current.start}, {current.end})"
            )
    return segments


# ---------------------------------------------------------------------------
# Dataset audio discovery
# ---------------------------------------------------------------------------


def discover_pop909(audio_root: Path) -> List[AudioTrack]:
    tracks: List[AudioTrack] = []
    for audio_path in sorted(audio_root.glob("**/original.mp3")):
        track_id = audio_path.parent.name
        if track_id.lower() == "versions":
            track_id = audio_path.parent.parent.name
        tracks.append(AudioTrack("pop909", track_id, audio_path))
    return tracks


def discover_rwc(root: Path) -> List[AudioTrack]:
    audio_dir = root / "AUDIO"
    if not audio_dir.exists():
        raise FileNotFoundError(f"RWC AUDIO directory not found: {audio_dir}")
    paths: List[Path] = []
    for extension in ("*.wav", "*.flac", "*.mp3"):
        paths.extend(audio_dir.glob(extension))
    return [AudioTrack("rwc", path.stem, path) for path in sorted(set(paths))]


def discover_slakh(root: Path, split: str) -> List[AudioTrack]:
    split_root = root / split
    tracks: List[AudioTrack] = []
    for extension in ("flac", "wav", "mp3", "ogg", "m4a"):
        paths = sorted(split_root.glob(f"**/mix.{extension}"))
        if paths:
            tracks.extend(
                AudioTrack("slakh", path.parent.name, path) for path in paths
            )
            break
    return tracks


def discover_audio(config: DatasetConfig) -> List[AudioTrack]:
    if config.name == "pop909":
        tracks = discover_pop909(config.audio_root)
    elif config.name == "rwc":
        tracks = discover_rwc(config.audio_root)
    elif config.name == "slakh":
        tracks = discover_slakh(config.audio_root, config.split)
    else:
        raise ValueError(f"Unsupported dataset: {config.name}")
    if not tracks:
        raise RuntimeError(
            f"No audio tracks found for {config.name} under {config.audio_root}"
        )
    return tracks


def build_audio_index(tracks: Sequence[AudioTrack]) -> Dict[str, List[AudioTrack]]:
    index: Dict[str, List[AudioTrack]] = {}
    for track in tracks:
        for alias in _track_aliases(track.track_id):
            bucket = index.setdefault(alias, [])
            if track not in bucket:
                bucket.append(track)
    return index


def match_audio_track(
    h5_track_id: str,
    index: Mapping[str, Sequence[AudioTrack]],
) -> AudioTrack:
    candidates: Dict[Path, AudioTrack] = {}
    for alias in _track_aliases(h5_track_id):
        for track in index.get(alias, ()):
            candidates[track.audio_path.resolve()] = track
    if not candidates:
        raise KeyError(f"No audio file matches HDF5 trackId={h5_track_id!r}")
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise RuntimeError(
            f"Ambiguous audio match for HDF5 trackId={h5_track_id!r}: {paths}"
        )
    return next(iter(candidates.values()))


# ---------------------------------------------------------------------------
# Existing LVCR result discovery
# ---------------------------------------------------------------------------


def _lab_aliases(path: Path, dataset: str) -> List[str]:
    """Aliases for a cached LVCR lab filename.

    Historical runs used names such as ``pop909_001.lab``, ``rwc_12.lab``,
    ``slakh_Track00001.lab``, or simply ``001.lab``.  Prefix stripping is
    conservative, and the final match must still be unique.
    """
    stem = path.stem.strip()
    variants = [stem]
    prefixes = (
        "baseline_",
        f"baseline_{dataset}_",
        f"{dataset}_",
        "slakh2100_",
        "baseline_slakh2100_",
    )

    changed = True
    while changed:
        changed = False
        for value in list(variants):
            lower = value.lower()
            for prefix in prefixes:
                if lower.startswith(prefix.lower()) and len(value) > len(prefix):
                    stripped = value[len(prefix):]
                    if stripped not in variants:
                        variants.append(stripped)
                        changed = True

    aliases: List[str] = []
    for value in variants:
        aliases.extend(_track_aliases(value))
    return list(dict.fromkeys(aliases))


def build_lab_index(result_dir: Path, dataset: str) -> Dict[str, List[Path]]:
    if not result_dir.is_dir():
        raise FileNotFoundError(
            f"Cached LVCR result directory not found: {result_dir}"
        )
    lab_files = sorted(path for path in result_dir.glob("**/*.lab") if path.is_file())
    if not lab_files:
        raise RuntimeError(f"No .lab files found under {result_dir}")

    index: Dict[str, List[Path]] = {}
    for lab_path in lab_files:
        for alias in _lab_aliases(lab_path, dataset):
            bucket = index.setdefault(alias, [])
            resolved = lab_path.resolve()
            if resolved not in bucket:
                bucket.append(resolved)
    return index


def match_lab_file(
    h5_track_id: str,
    index: Mapping[str, Sequence[Path]],
) -> Path:
    candidates: Dict[Path, Path] = {}
    for alias in _track_aliases(h5_track_id):
        for path in index.get(alias, ()):
            candidates[path.resolve()] = path.resolve()
    if not candidates:
        raise KeyError(f"No cached LVCR .lab matches HDF5 trackId={h5_track_id!r}")
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise RuntimeError(
            f"Ambiguous cached LVCR match for HDF5 trackId={h5_track_id!r}: "
            f"{paths}"
        )
    return next(iter(candidates.values()))


# ---------------------------------------------------------------------------
# HDF5 timelines
# ---------------------------------------------------------------------------


def load_h5_samples(
    h5_path: Path,
    split: str,
    default_fps: float,
    orig_only: bool,
) -> List[H5Sample]:
    samples: List[H5Sample] = []
    with h5py.File(h5_path, "r") as handle:
        if split not in handle:
            raise KeyError(f"Split {split!r} not found in {h5_path}")
        group = handle[split]
        if "trackId" not in group:
            raise KeyError(f"{split}/trackId not found in {h5_path}")
        if "labels" not in group or "chord_frame" not in group["labels"]:
            raise KeyError(f"{split}/labels/chord_frame not found in {h5_path}")

        track_ids = group["trackId"]
        chord_frames = group["labels"]["chord_frame"]
        starts = group.get("start")
        ends = group.get("end")
        semitones = group.get("semitone")
        segment_sec = group["labels"].attrs.get("segment_sec")

        if len(track_ids) != len(chord_frames):
            raise ValueError(
                f"trackId count {len(track_ids)} != chord_frame count "
                f"{len(chord_frames)} in {h5_path}"
            )

        for index in range(len(track_ids)):
            if orig_only and semitones is not None and int(semitones[index]) != 0:
                continue

            labels = np.asarray(
                [_decode(value).strip() for value in chord_frames[index]],
                dtype=object,
            )
            n_frames = int(labels.shape[0])
            if n_frames <= 0:
                raise ValueError(f"Empty chord_frame at HDF5 sample {index}")

            start_sec = float(starts[index]) if starts is not None else 0.0
            if ends is not None:
                end_sec = float(ends[index])
            elif segment_sec is not None:
                end_sec = start_sec + float(segment_sec)
            else:
                end_sec = start_sec + n_frames / float(default_fps)

            if not np.isfinite(start_sec) or not np.isfinite(end_sec):
                raise ValueError(f"Non-finite HDF5 timing at sample {index}")
            if end_sec <= start_sec:
                raise ValueError(
                    f"Invalid HDF5 time range [{start_sec}, {end_sec}) "
                    f"at sample {index}"
                )

            samples.append(
                H5Sample(
                    index=index,
                    track_id=_decode(track_ids[index]).strip(),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    ref_labels=labels,
                )
            )

    if not samples:
        suffix = " after semitone==0 filtering" if orig_only else ""
        raise RuntimeError(f"No HDF5 samples loaded from {h5_path}:{split}{suffix}")
    return samples


# ---------------------------------------------------------------------------
# Baseline execution
# ---------------------------------------------------------------------------


def run_baseline_track(
    track: AudioTrack,
    lab_path: Path,
    chord_script: Path,
    chord_args: Sequence[str],
    timeout_sec: int,
    overwrite: bool,
) -> RunResult:
    if lab_path.exists() and lab_path.stat().st_size > 0 and not overwrite:
        return RunResult(track=track, lab_path=lab_path)

    lab_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(chord_script),
        str(track.audio_path),
        str(lab_path),
        *chord_args,
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RunResult(track, lab_path, f"timeout after {timeout_sec}s")
    except Exception as exc:  # pragma: no cover - defensive around subprocess
        return RunResult(track, lab_path, str(exc))

    if process.returncode != 0:
        stderr = process.stderr.strip().replace("\n", " ")[:500]
        return RunResult(
            track,
            lab_path,
            f"exit {process.returncode}: {stderr}",
        )
    if not lab_path.exists() or lab_path.stat().st_size == 0:
        return RunResult(track, lab_path, "no non-empty output .lab produced")

    try:
        parse_lab_file(lab_path)
    except Exception as exc:
        return RunResult(track, lab_path, f"invalid output .lab: {exc}")
    return RunResult(track=track, lab_path=lab_path)


def run_required_tracks(
    tracks: Sequence[AudioTrack],
    lab_dir: Path,
    chord_script: Path,
    chord_args: Sequence[str],
    timeout_sec: int,
    jobs: int,
    overwrite: bool,
) -> Dict[Path, Path]:
    unique: Dict[Path, AudioTrack] = {
        track.audio_path.resolve(): track for track in tracks
    }
    ordered_tracks = sorted(unique.values(), key=lambda item: str(item.audio_path))

    def output_path(track: AudioTrack) -> Path:
        return lab_dir / f"{track.dataset}_{_safe_name(track.track_id)}.lab"

    results: List[RunResult] = []
    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    run_baseline_track,
                    track,
                    output_path(track),
                    chord_script,
                    chord_args,
                    timeout_sec,
                    overwrite,
                ): track
                for track in ordered_tracks
            }
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                status = "OK" if result.error is None else f"ERROR: {result.error}"
                print(
                    f"  [{done:>{len(str(len(futures)))}}/{len(futures)}] "
                    f"{result.track.track_id}: {status}"
                )
    else:
        for done, track in enumerate(ordered_tracks, 1):
            result = run_baseline_track(
                track,
                output_path(track),
                chord_script,
                chord_args,
                timeout_sec,
                overwrite,
            )
            results.append(result)
            status = "OK" if result.error is None else f"ERROR: {result.error}"
            print(
                f"  [{done:>{len(str(len(ordered_tracks)))}}/"
                f"{len(ordered_tracks)}] {track.track_id}: {status}"
            )

    errors = [result for result in results if result.error is not None]
    if errors:
        details = "\n".join(
            f"  {result.track.track_id}: {result.error}" for result in errors
        )
        raise RuntimeError(
            f"Baseline inference failed for {len(errors)} track(s):\n{details}"
        )

    return {
        result.track.audio_path.resolve(): result.lab_path for result in results
    }


# ---------------------------------------------------------------------------
# Projection to HDF5 frames and cache generation
# ---------------------------------------------------------------------------


def sample_lab_at_frame_centers(
    segments: Sequence[LabSegment],
    sample: H5Sample,
) -> np.ndarray:
    """Sample a full-track chord lab at centers of the HDF5 label frames."""
    n_frames = sample.n_frames
    frame_duration = (sample.end_sec - sample.start_sec) / float(n_frames)
    times = sample.start_sec + (np.arange(n_frames) + 0.5) * frame_duration

    labels = np.full(n_frames, "N", dtype=object)
    segment_index = 0
    for frame_index, time_sec in enumerate(times):
        while (
            segment_index < len(segments)
            and segments[segment_index].end <= time_sec
        ):
            segment_index += 1
        if segment_index >= len(segments):
            break
        segment = segments[segment_index]
        if segment.start <= time_sec < segment.end:
            labels[frame_index] = _canonical_estimate(segment.label)
    return labels


def contiguous_runs(labels: Sequence[Any]) -> Iterable[Tuple[int, int, str]]:
    if len(labels) == 0:
        return
    start = 0
    previous = _canonical_estimate(_decode(labels[0]))
    for index in range(1, len(labels)):
        current = _canonical_estimate(_decode(labels[index]))
        if current != previous:
            yield start, index, previous
            start = index
            previous = current
    yield start, len(labels), previous


def legacy_cached_reference(
    ref_labels: Sequence[Any],
    start: int,
    end: int,
) -> str:
    """Exactly reproduce eval_metrics._legacy_cached_ref for safe alignment."""
    counts: Dict[str, int] = {}
    for value in ref_labels[start:end]:
        text = _decode(value).strip()
        if text and text.upper() not in NO_CHORD_REFERENCES:
            counts[text] = counts.get(text, 0) + 1
    if not counts:
        return "N"
    return max(counts.items(), key=lambda item: item[1])[0]


def write_h5_aligned_cache(
    cache_path: Path,
    samples: Sequence[H5Sample],
    sample_labs: Mapping[int, Path],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_labs: Dict[Path, List[LabSegment]] = {}

    with cache_path.open("w", encoding="utf-8") as handle:
        handle.write("# track_id\tseg_range\tref_chord\test_chord\n")
        for sample in samples:
            if sample.index not in sample_labs:
                raise KeyError(
                    f"Missing cached .lab for HDF5 sample {sample.index} "
                    f"(trackId={sample.track_id!r})"
                )
            lab_path = sample_labs[sample.index].resolve()
            if lab_path not in parsed_labs:
                parsed_labs[lab_path] = parse_lab_file(lab_path)

            frame_labels = sample_lab_at_frame_centers(
                parsed_labs[lab_path], sample
            )
            covered = 0
            for start, end, estimate in contiguous_runs(frame_labels):
                if start != covered:
                    raise AssertionError(
                        f"Internal cache gap at sample {sample.index}: "
                        f"expected {covered}, got {start}"
                    )
                cached_ref = legacy_cached_reference(
                    sample.ref_labels, start, end
                )
                handle.write(
                    f"{sample.track_id}\t{start}-{end}\t"
                    f"{cached_ref}\t{estimate}\n"
                )
                covered = end
            if covered != sample.n_frames:
                raise AssertionError(
                    f"Internal cache coverage error at sample {sample.index}: "
                    f"covered {covered}/{sample.n_frames} frames"
                )


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------


def run_shared_evaluator(
    cache_path: Path,
    h5_path: Path,
    split: str,
    eval_module: str,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        eval_module,
        "chord",
        str(cache_path),
        "--h5-path",
        str(h5_path),
        "--split",
        split,
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Shared evaluator failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )

    stdout = process.stdout.strip()
    try:
        metrics = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Evaluator did not return valid JSON:\n{stdout}\n"
            f"stderr:\n{process.stderr}"
        ) from exc

    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing:
        raise KeyError(f"Evaluator JSON is missing keys: {missing}")
    return metrics


def append_summary(
    path: Path,
    dataset: str,
    split: str,
    metrics: Mapping[str, Any],
) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(
                "model\tdataset\tsplit\tboundary\troot\tmajmin\tmirex\t"
                "micro_root\tmicro_majmin\tmicro_mirex\t"
                "n_songs\tn_cache_samples\n"
            )
        handle.write(
            "lvcr\t{dataset}\t{split}\testimated\t{root}\t{majmin}\t"
            "{mirex}\t{micro_root}\t{micro_majmin}\t{micro_mirex}\t"
            "{n_songs}\t{n_cache_samples}\n".format(
                dataset=dataset,
                split=split,
                **{key: metrics[key] for key in REQUIRED_METRIC_KEYS},
            )
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cached LVCR predictions on the same HDF5 frame timelines "
            "and with the same eval_metrics.py used by SAE/probe results. "
            "LVCR inference is disabled by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pop909-h5", type=Path)
    parser.add_argument("--pop909-split", default="train")
    parser.add_argument(
        "--pop909-results",
        type=Path,
        default=Path("../LVCR/results/baseline_pop909_test"),
        help="Directory containing existing POP909 LVCR .lab predictions",
    )
    parser.add_argument(
        "--pop909-audio",
        type=Path,
        help="POP909 audio root; required only with --run-inference",
    )

    parser.add_argument("--rwc-h5", type=Path)
    parser.add_argument("--rwc-split", default="test")
    parser.add_argument(
        "--rwc-results",
        type=Path,
        default=Path("../LVCR/results/baseline_rwc"),
        help="Directory containing existing RWC LVCR .lab predictions",
    )
    parser.add_argument(
        "--rwc",
        type=Path,
        help="RWC audio root; required only with --run-inference",
    )

    parser.add_argument("--slakh-h5", type=Path)
    parser.add_argument("--slakh-split", default="test")
    parser.add_argument(
        "--slakh-results",
        type=Path,
        default=Path("../LVCR/results/baseline_slakh2100_test"),
        help="Directory containing existing Slakh LVCR .lab predictions",
    )
    parser.add_argument(
        "--slakh",
        type=Path,
        help="Slakh audio root; required only with --run-inference",
    )

    parser.add_argument(
        "--run-inference",
        action="store_true",
        help=(
            "Run LVCR before evaluation. Without this flag, only existing .lab "
            "files in the configured result directories are used."
        ),
    )
    parser.add_argument(
        "--chord-script",
        type=Path,
        default=Path("../LVCR/chord_recognition.py"),
        help="LVCR entry point; used only with --run-inference",
    )
    parser.add_argument(
        "--chord-arg",
        action="append",
        default=[],
        help="Additional LVCR argument; repeatable and used only for inference",
    )
    parser.add_argument(
        "--eval-module",
        default="src.analysis.eval_metrics",
        help="Python module containing the shared HDF5 chord evaluator",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../LVCR/results/unified_eval"),
        help="Directory for HDF5-aligned caches, metric JSON, and summary TSV",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--default-fps",
        type=float,
        default=10.0,
        help="Fallback only when HDF5 start/end and segment_sec are absent",
    )
    parser.add_argument(
        "--include-transposed-h5",
        action="store_true",
        help="Include HDF5 rows whose semitone field is nonzero",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run LVCR over an existing non-empty .lab (inference mode only)",
    )
    return parser.parse_args()


def collect_configs(args: argparse.Namespace) -> List[DatasetConfig]:
    rows = [
        (
            "pop909",
            args.pop909_h5,
            args.pop909_split,
            args.pop909_results,
            args.pop909_audio,
        ),
        ("rwc", args.rwc_h5, args.rwc_split, args.rwc_results, args.rwc),
        (
            "slakh",
            args.slakh_h5,
            args.slakh_split,
            args.slakh_results,
            args.slakh,
        ),
    ]
    configs: List[DatasetConfig] = []
    for name, h5_path, split, result_dir, audio_root in rows:
        if h5_path is None:
            if audio_root is not None:
                raise ValueError(
                    f"{name}: an audio root was supplied without --{name}-h5"
                )
            continue
        configs.append(
            DatasetConfig(
                name=name,
                h5_path=h5_path.resolve(),
                split=split,
                result_dir=result_dir.resolve(),
                audio_root=audio_root.resolve() if audio_root is not None else None,
            )
        )
    if not configs:
        raise ValueError(
            "Provide at least one HDF5 path, e.g. --rwc-h5 FILE. "
            "Cached LVCR result directories already have defaults."
        )
    return configs


def main() -> None:
    args = parse_args()
    configs = collect_configs(args)

    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.default_fps <= 0:
        raise ValueError("--default-fps must be positive")
    if args.run_inference and not args.chord_script.is_file():
        raise FileNotFoundError(f"Chord script not found: {args.chord_script}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "baseline_chord_summary.tsv"
    if summary_path.exists():
        summary_path.unlink()

    all_metrics: Dict[str, Dict[str, Any]] = {}

    for config in configs:
        print("\n" + "=" * 72)
        print(
            f"Dataset: {config.name} | split={config.split} | "
            f"HDF5={config.h5_path}"
        )
        print(f"LVCR results: {config.result_dir}")
        print(f"Inference: {'enabled' if args.run_inference else 'disabled (cached labs only)'}")
        print("=" * 72)

        if not config.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {config.h5_path}")

        h5_samples = load_h5_samples(
            config.h5_path,
            config.split,
            default_fps=args.default_fps,
            orig_only=not args.include_transposed_h5,
        )

        if args.run_inference:
            if config.audio_root is None:
                raise ValueError(
                    f"{config.name}: --run-inference requires its audio-root option"
                )
            if not config.audio_root.exists():
                raise FileNotFoundError(
                    f"Audio root not found: {config.audio_root}"
                )
            config.result_dir.mkdir(parents=True, exist_ok=True)
            discovered = discover_audio(config)
            audio_index = build_audio_index(discovered)
            sample_tracks: Dict[int, AudioTrack] = {
                sample.index: match_audio_track(sample.track_id, audio_index)
                for sample in h5_samples
            }
            lab_paths_by_audio = run_required_tracks(
                list(sample_tracks.values()),
                config.result_dir,
                args.chord_script.resolve(),
                args.chord_arg,
                args.timeout,
                args.jobs,
                args.overwrite,
            )
            sample_labs: Dict[int, Path] = {
                sample_index: lab_paths_by_audio[track.audio_path.resolve()]
                for sample_index, track in sample_tracks.items()
            }
        else:
            lab_index = build_lab_index(config.result_dir, config.name)
            sample_labs = {
                sample.index: match_lab_file(sample.track_id, lab_index)
                for sample in h5_samples
            }

        dataset_dir = out_dir / config.name
        cache_path = dataset_dir / f"{config.name}_{config.split}_lvcr_cache.txt"
        metrics_path = dataset_dir / f"{config.name}_{config.split}_lvcr_metrics.json"

        print(
            f"[INFO] HDF5 samples={len(h5_samples)}, "
            f"songs={len({sample.track_id for sample in h5_samples})}, "
            f"matched labs={len(set(sample_labs.values()))}"
        )

        write_h5_aligned_cache(cache_path, h5_samples, sample_labs)
        print(f"[INFO] HDF5-aligned cache -> {cache_path}")

        metrics = run_shared_evaluator(
            cache_path,
            config.h5_path,
            config.split,
            args.eval_module,
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_summary(summary_path, config.name, config.split, metrics)
        all_metrics[config.name] = metrics

        print(
            "[RESULT] song-macro: "
            f"Root={metrics['root']:.2f}%  "
            f"MajMin={metrics['majmin']:.2f}%  "
            f"MIREX={metrics['mirex']:.2f}%"
        )
        print(
            "[RESULT] micro:      "
            f"Root={metrics['micro_root']:.2f}%  "
            f"MajMin={metrics['micro_majmin']:.2f}%  "
            f"MIREX={metrics['micro_mirex']:.2f}%"
        )
        print(f"[INFO] Metrics JSON -> {metrics_path}")

    print("\n" + "=" * 72)
    print(f"Summary -> {summary_path}")
    print("=" * 72)
    for dataset, metrics in all_metrics.items():
        print(
            f"{dataset:<8} Root={metrics['root']:>6.2f}%  "
            f"MajMin={metrics['majmin']:>6.2f}%  "
            f"MIREX={metrics['mirex']:>6.2f}%"
        )


if __name__ == "__main__":
    main()
