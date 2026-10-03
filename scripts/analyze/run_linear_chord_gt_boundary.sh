#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_linear_chord_gt_boundary.sh
#
# Evaluate 25-class linear chord probes under reference chord boundaries.
# Reference annotations provide boundaries only; root, quality, and no-chord
# are predicted by src.linear_probe.infer_chord_gt_boundary.
#
# For each probe run under <probe_group_dir>, this script:
#   1. selects the checkpoint with the highest score encoded in its filename
#      inside that run's ckpts/ directory;
#   2. runs reference-boundary chord inference;
#   3. evaluates predictions against the original HDF5 chord_frame labels with
#      src.analysis.eval_metrics;
#   4. writes corpus-level micro WCSR and song-level macro WCSR summaries.
#
# Usage:
#   bash scripts/analyze/run_linear_chord_gt_boundary.sh \
#       <probe_group_dir> <chord_out_dir>
#
# Example:
#   FORCE_INFER=1 bash scripts/analyze/run_linear_chord_gt_boundary.sh \
#       store/results_latest/models/probing/linear_chord_new \
#       store/results_latest/results/probing/linear_chord_refbnd_predn
#
# FORCE_INFER=1 regenerates prediction caches even when they already exist.
# ---------------------------------------------------------------------------
set -euo pipefail

probe_group_dir="${1:?Usage: $0 <probe_group_dir> <chord_out_dir>}"
chord_out_dir="${2:?Usage: $0 <probe_group_dir> <chord_out_dir>}"

probe_group_dir="${probe_group_dir%/}"
chord_out_dir="${chord_out_dir%/}"
force_infer="${FORCE_INFER:-0}"

# ── Configuration ────────────────────────────────────────────────────────────
sae_data_root="sae-data"

# These splits must match the SAE and baseline evaluations.
chord_datasets=(pop909 rwc slakh)
chord_splits=(train test test)

batch_size=64
device=cuda
smooth_win=9
majmin_mode=seg-mean-max  # sum-peak | seg-mean-max | seg-mean-mean | raw-tmpl
majmin_conf_thr=0.18      # raw-tmpl only
majmin_alpha=0.4          # raw-tmpl only

# ── Output files ─────────────────────────────────────────────────────────────
mkdir -p "$chord_out_dir"

micro_tsv="${chord_out_dir}/chord_micro_refbnd_predn.tsv"
macro_tsv="${chord_out_dir}/chord_song_macro_refbnd_predn.tsv"

header='feature\tlid\tdataset\troot\tmajmin\tmirex\tn_songs\tn_cache_samples'
printf '%b\n' "$header" > "$micro_tsv"
printf '%b\n' "$header" > "$macro_tsv"

LOG_SEP="============================================================"
PASS=0
FAIL=0

# ── Helpers ──────────────────────────────────────────────────────────────────

get_split() {
    local dataset="$1"
    local -n datasets_ref="$2"
    local -n splits_ref="$3"

    local i
    for i in "${!datasets_ref[@]}"; do
        if [[ "${datasets_ref[$i]}" == "$dataset" ]]; then
            printf '%s\n' "${splits_ref[$i]}"
            return 0
        fi
    done

    printf '%s\n' "test"
}

# Select the checkpoint with the highest score encoded in its filename.
# Expected example:
#   val_chord_acc-epoch015-step18560-score83.6735.ckpt
# Only files in the supplied ckpt_dir are considered. JSON metadata is ignored.
best_ckpt() {
    local ckpt_dir="$1"

    python3 - "$ckpt_dir" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
if not ckpt_dir.is_dir():
    raise SystemExit(0)

score_re = re.compile(r"-score(-?\d+(?:\.\d+)?)(?=\.ckpt$)")
epoch_re = re.compile(r"-epoch(\d+)")
step_re = re.compile(r"-step(\d+)")

candidates = []
for path in ckpt_dir.glob("*.ckpt"):
    score_match = score_re.search(path.name)
    if score_match is None:
        continue

    score = float(score_match.group(1))
    epoch_match = epoch_re.search(path.name)
    step_match = step_re.search(path.name)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    step = int(step_match.group(1)) if step_match else -1
    candidates.append((score, epoch, step, path))

if not candidates:
    raise SystemExit(0)

# Score is primary; epoch and step break exact score ties.
best = max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
print(best.resolve())
PY
}

# eval_and_write <chord_out> <h5_path> <split_name>
#
# eval_metrics returns:
#   micro_root / micro_majmin / micro_mirex : corpus-level frame-weighted WCSR
#   root / majmin / mirex                   : song-level macro WCSR
#
# The evaluator reads original frame-level references from the matching HDF5.
eval_and_write() {
    local chord_out="$1"
    local h5_path="$2"
    local split_name="$3"

    if [[ ! -f "$chord_out" ]]; then
        echo "  [ERROR] Prediction output missing: $chord_out" >&2
        return 1
    fi

    local metrics_json
    if ! metrics_json="$(
        python -m src.analysis.eval_metrics \
            --h5-path "$h5_path" \
            --split "$split_name" \
            chord "$chord_out"
    )"; then
        echo "  [ERROR] Chord evaluation failed: $chord_out" >&2
        return 1
    fi

    local parsed
    if ! parsed="$(
        python3 -c '
import json
import sys

d = json.load(sys.stdin)
required = [
    "micro_root",
    "micro_majmin",
    "micro_mirex",
    "root",
    "majmin",
    "mirex",
    "n_songs",
    "n_cache_samples",
]
missing = [k for k in required if k not in d]
if missing:
    raise SystemExit("Missing evaluator fields: " + ", ".join(missing))
print("\t".join(str(d[k]) for k in required))
' <<< "$metrics_json"
    )"; then
        echo "  [ERROR] Invalid evaluator JSON for: $chord_out" >&2
        echo "$metrics_json" >&2
        return 1
    fi

    local micro_root micro_majmin micro_mirex
    local macro_root macro_majmin macro_mirex
    local n_songs n_cache_samples

    IFS=$'\t' read -r \
        micro_root micro_majmin micro_mirex \
        macro_root macro_majmin macro_mirex \
        n_songs n_cache_samples \
        <<< "$parsed"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$feature" "$lid" "$dataset" \
        "$micro_root" "$micro_majmin" "$micro_mirex" \
        "$n_songs" "$n_cache_samples" \
        >> "$micro_tsv"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$feature" "$lid" "$dataset" \
        "$macro_root" "$macro_majmin" "$macro_mirex" \
        "$n_songs" "$n_cache_samples" \
        >> "$macro_tsv"

    echo "  micro: root=${micro_root}% majmin=${micro_majmin}% mirex=${micro_mirex}%"
    echo "  macro: root=${macro_root}% majmin=${macro_majmin}% mirex=${macro_mirex}%"
    echo "  count: songs=${n_songs} cache_samples=${n_cache_samples}"
}

# ── Main loop ────────────────────────────────────────────────────────────────

for run_dir in "${probe_group_dir}"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name="$(basename "$run_dir")"

    # Expected run names:
    #   chord_muq-0
    #   chord_musicfm-6
    #   chord_cqt
    mode="${run_name#chord_}"
    if [[ "$mode" =~ ^(muq|musicfm)-([0-9]+)$ ]]; then
        feature="${BASH_REMATCH[1]}"
        lid="${BASH_REMATCH[2]}"
    elif [[ "$mode" == "cqt" ]]; then
        feature="cqt"
        lid="0"
    else
        echo "[SKIP] Unrecognised run name: $run_name" >&2
        continue
    fi

    echo
    echo "$LOG_SEP"
    echo " PROCESS  feature=$feature  lid=$lid  run=$run_name"
    echo "$LOG_SEP"

    ckpt="$(best_ckpt "${run_dir}ckpts")"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "[ERROR] No scored checkpoint found in ${run_dir}ckpts/" >&2
        FAIL=$((FAIL + 1))
        continue
    fi
    echo "  ckpt: $ckpt"

    run_out_dir="${chord_out_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    run_failed=0

    for dataset in "${chord_datasets[@]}"; do
        split="$(get_split "$dataset" chord_datasets chord_splits)"
        h5="${sae_data_root}/${dataset}/${feature}/${dataset}_${feature}_30s_layer_${lid}.h5"
        chord_out="${run_out_dir}/${dataset}_${split}_${mode}_chord_refbnd_predn.txt"

        echo
        echo "  [$dataset / $split]"

        if [[ ! -f "$h5" ]]; then
            echo "  [ERROR] HDF5 not found: $h5" >&2
            run_failed=1
            continue
        fi

        if [[ "$force_infer" == "1" || ! -f "$chord_out" ]]; then
            if ! python -m src.linear_probe.infer_chord_gt_boundary \
                --h5-path         "$h5" \
                --split           "$split" \
                --ckpt-path       "$ckpt" \
                --chord-out-file  "$chord_out" \
                --smooth-win      "$smooth_win" \
                --majmin-mode     "$majmin_mode" \
                --majmin-conf-thr "$majmin_conf_thr" \
                --majmin-alpha    "$majmin_alpha" \
                --batch-size      "$batch_size" \
                --device          "$device"; then
                echo "  [ERROR] Inference failed: $dataset / $split" >&2
                run_failed=1
                continue
            fi
        else
            echo "  [CACHED] $chord_out"
        fi

        if ! eval_and_write "$chord_out" "$h5" "$split"; then
            run_failed=1
            continue
        fi
    done

    if [[ "$run_failed" == "0" ]]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "$LOG_SEP"
echo " Done: $PASS passed, $FAIL failed"
echo "$LOG_SEP"
echo
echo "── Corpus-level micro WCSR : $micro_tsv"
echo "── Song-level macro WCSR   : $macro_tsv"
echo
echo "  probe_group_dir : $probe_group_dir"
echo "  chord_out_dir   : $chord_out_dir"
echo "  FORCE_INFER     : $force_infer"
