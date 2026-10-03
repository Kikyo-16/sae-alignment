#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import random
import sys


def normalize_key(key: str) -> str:
    """
    Normalize key labels.

    Examples:
        C:maj    -> C:major
        C:major  -> C:major
        A:min    -> A:minor
        A:minor  -> A:minor
    """
    key = key.strip()

    tonic, sep, mode = key.partition(":")
    if sep == "":
        return key

    mode_map = {
        "maj": "major",
        "major": "major",
        "min": "minor",
        "minor": "minor",
    }

    mode = mode_map.get(mode, mode)
    return f"{tonic}:{mode}"


def read_key_from_lab(lab_file: Path) -> str:
    """
    Read key from all_src.key.lab.

    Expected format:
        0.000000    239.196426    B:minor
    """
    text = lab_file.read_text().strip()
    if not text:
        raise ValueError("empty key lab file")

    first_line = text.splitlines()[0]
    parts = first_line.split()

    if len(parts) < 3:
        raise ValueError(f"bad lab format: {first_line}")

    return normalize_key(parts[-1])


def get_track_id(lab_file: Path) -> str:
    """
    For:
        .../validation/Track01875/all_src.key.lab

    Return:
        Track01875
    """
    return lab_file.parent.name


def main():
    parser = argparse.ArgumentParser(
        description="Sample equal number of tracks for each key."
    )

    parser.add_argument(
        "root",
        help="Path to validation directory, e.g. dataset/slakh2100_flac_redux/validation",
    )

    parser.add_argument(
        "-n",
        "--num-per-key",
        type=int,
        required=True,
        help="Number of samples to select for each key",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="selected_track_ids.txt",
        help="Output txt file for selected track ids",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--allow-less",
        action="store_true",
        help="If a key has fewer than N samples, select all available samples instead of exiting",
    )

    parser.add_argument(
        "--output-with-key",
        action="store_true",
        help="Output format: track_id<TAB>key instead of only track_id",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    if args.num_per_key <= 0:
        raise ValueError("--num-per-key must be positive")

    random.seed(args.seed)

    key_to_tracks = defaultdict(list)
    bad_files = []

    lab_files = sorted(root.rglob("all_src.key.lab"))

    for lab_file in lab_files:
        try:
            key = read_key_from_lab(lab_file)
            track_id = get_track_id(lab_file)
            key_to_tracks[key].append(track_id)
        except Exception as e:
            bad_files.append((str(lab_file), str(e)))

    if not key_to_tracks:
        print("No valid key lab files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanned files: {len(lab_files)}")
    print(f"Valid tracks: {sum(len(v) for v in key_to_tracks.values())}")
    print(f"Number of keys: {len(key_to_tracks)}")
    print()

    print("Original key distribution:")
    key_counter = Counter({key: len(tracks) for key, tracks in key_to_tracks.items()})
    for key, count in key_counter.most_common():
        print(f"  {key:12s} {count}")
    print()

    selected = []

    for key in sorted(key_to_tracks.keys()):
        tracks = key_to_tracks[key]
        count = len(tracks)

        if count < args.num_per_key:
            msg = (
                f"Key {key} has only {count} tracks, "
                f"but --num-per-key is {args.num_per_key}"
            )

            if args.allow_less:
                print(f"Warning: {msg}. Selecting all available tracks.")
                chosen = tracks[:]
            else:
                print(f"Error: {msg}.", file=sys.stderr)
                print(
                    "Use --allow-less if you want to keep smaller keys.",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            chosen = random.sample(tracks, args.num_per_key)

        for track_id in sorted(chosen):
            selected.append((track_id, key))

    output_path = Path(args.output)

    with output_path.open("w") as f:
        for track_id, key in selected:
            if args.output_with_key:
                f.write(f"{track_id}\t{key}\n")
            else:
                f.write(f"{track_id}\n")

    print("Selected key distribution:")
    selected_counter = Counter(key for _, key in selected)
    for key, count in sorted(selected_counter.items()):
        print(f"  {key:12s} {count}")

    print()
    print(f"Total selected tracks: {len(selected)}")
    print(f"Saved to: {output_path}")

    if bad_files:
        print()
        print(f"Bad files: {len(bad_files)}")
        for path, err in bad_files:
            print(f"  {path} | {err}")


if __name__ == "__main__":
    main()