#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_linear_relative_key.sh — Key accuracy from the linear-probe relative-key
# (scale-degree) model.
#
# Relative-key counterpart of run_linear_minor_key.sh.  Where the chord probe
# emits major/minor rings, the relative-key probe emits a single ring of 12
# scale-degree logits, so each run predicts only the key SIGNATURE (relative-
# major space): there is no major/minor cue and hence a single variant per run.
# The scale degree (tonic / forth / fifth = tonic / subdominant / dominant) is a
# property of the run — it is not stored in the checkpoint — so it is taken from
# the run-dir name and passed to inference as --label-tag.
#
# For each probe run under <probe_group_dir> it picks the best checkpoint (by
# the score embedded in the ckpt filename), runs key inference
# (src.linear_probe.infer_key_relative), and writes, per level (segment + song),
# one merged-signature result file.
#
# Run-dir naming (per-run subdirs of <probe_group_dir>):
#   [key_]<degree>_<mode>   where
#     degree ∈ {tonic, forth, fifth}                (sets the signature offset)
#     mode   ∈ {muq-<lid>, musicfm-<lid>, cqt}      (feature + layer)
#   e.g. forth_muq-2, key_forth_muq-2, tonic_musicfm-6, fifth_cqt
#
# Usage:
#   bash scripts/analyze/run_linear_relative_key.sh \
#       <probe_group_dir> <key_out_dir>
#
# Example:
#   bash scripts/analyze/run_linear_relative_key.sh \
#       store/linear_relkey_new \
#       store/linear_relkey_new/results-relkey
# ---------------------------------------------------------------------------
set -euo pipefail

probe_group_dir="${1:?Usage: $0 <probe_group_dir> <key_out_dir>}"
key_out_dir="${2:?Usage: $0 <probe_group_dir> <key_out_dir>}"

probe_group_dir="${probe_group_dir%/}"
key_out_dir="${key_out_dir%/}"

# ── Configuration ────────────────────────────────────────────────────────────
sae_data_root="sae-data"

key_datasets=(rwc gtzan giantsteps_key fmakv2)
key_splits=(test test test test)

batch_size=64
device=cuda
# Set FORCE_INFER=1 to regenerate cached key output files.
force_infer="${FORCE_INFER:-0}"

# degree token → infer_key_relative --label-tag
declare -A DEGREE_TO_TAG=(
    [tonic]=tonic_frame
    [forth]=forth_frame
    [fifth]=fifth_frame
)

# ── Output files ─────────────────────────────────────────────────────────────
mkdir -p "$key_out_dir"
for lvl in segment song; do
    eval "f_${lvl}=\"${key_out_dir}/key_${lvl}_relative.tsv\""
done

h='feature\tlid\tmode\tdegree\tdataset\tmaj_total\tmin_total\ttotal\tacc\tacc_strict\tmirex\n'
for lvl in segment song; do
    eval "printf '$h' > \"\$f_${lvl}\""
done

LOG_SEP="============================================================"
PASS=0; FAIL=0

# ── Helpers ──────────────────────────────────────────────────────────────────

get_split() {
    local dataset="$1"; shift
    local -n _ds_arr="$1"; shift
    local -n _sp_arr="$1"
    for i in "${!_ds_arr[@]}"; do
        [[ "${_ds_arr[$i]}" == "$dataset" ]] && { echo "${_sp_arr[$i]}"; return; }
    done
    echo "test"
}

# Pick the ckpt with the highest score parsed from the filename
# (…-score<val>.ckpt); eval_ckpts.json is ignored since its paths may be stale.
best_ckpt() {
    local ckpt_dir="$1"
    ls "${ckpt_dir}"/*.ckpt 2>/dev/null | \
        awk -F'score' 'NR==1 || $2+0 > best {best=$2+0; line=$0} END{print line}'
}

# ── Main loop ────────────────────────────────────────────────────────────────

for run_dir in "${probe_group_dir}"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name=$(basename "$run_dir")

    # Expect [key_]<degree>_<mode>: degree ∈ {tonic,forth,fifth};
    # mode ∈ {muq-<lid>,musicfm-<lid>,cqt}.  e.g. forth_muq-2, key_forth_muq-2.
    parse_name="${run_name#key_}"
    degree="${parse_name%%_*}"
    mode="${parse_name#*_}"
    label_tag="${DEGREE_TO_TAG[$degree]:-}"
    if [[ -z "$label_tag" ]]; then
        echo "[SKIP] Unrecognised degree in run name: $run_name (want tonic_/forth_/fifth_)" >&2
        continue
    fi
    if [[ "$mode" =~ ^(muq|musicfm)-([0-9]+)$ ]]; then
        feature="${BASH_REMATCH[1]}"
        lid="${BASH_REMATCH[2]}"
    elif [[ "$mode" == "cqt" ]]; then
        feature="cqt"
        lid="0"
    else
        echo "[SKIP] Unrecognised mode in run name: $run_name" >&2
        continue
    fi

    echo ""; echo "$LOG_SEP"
    echo " PROCESS  feature=$feature  lid=$lid  degree=$degree  (run=$run_name)"
    echo "$LOG_SEP"

    ckpt=$(best_ckpt "${run_dir}ckpts")
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "[SKIP] No checkpoint in ${run_dir}ckpts/" >&2; FAIL=$((FAIL+1)); continue
    fi
    echo "  ckpt: $ckpt"

    run_out_dir="${key_out_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    for dataset in "${key_datasets[@]}"; do
        split=$(get_split "$dataset" key_datasets key_splits)
        h5="${sae_data_root}/${dataset}/${feature}/${dataset}_${feature}_30s_layer_${lid}.h5"

        key_out="${run_out_dir}/${dataset}_${split}_${mode}_key_${degree}.txt"

        echo ""; echo "  [$dataset / $split]"
        if [[ ! -f "$h5" ]]; then
            echo "  [SKIP] H5 not found: $h5" >&2; continue
        fi

        if [[ "$force_infer" != "0" || ! -f "$key_out" ]]; then
            python -m src.linear_probe.infer_key_relative \
                --h5-path       "$h5" \
                --split         "$split" \
                --ckpt-path     "$ckpt" \
                --label-tag     "$label_tag" \
                --key-out-file  "$key_out" \
                --batch-size    $batch_size \
                --device        $device \
                --allow-missing-labels
        else
            echo "  [CACHED] $key_out"
        fi
        [[ -f "$key_out" ]] || { echo "  [SKIP] output missing after inference"; continue; }

        # Single-ring (no maj/min block) → eval reads merged-signature accuracy;
        # maj_degree / vote2_decision are unused (the degree is baked into the file).
        m=$(python -m src.analysis.eval_metrics key "$key_out")
        unset M; declare -A M
        while IFS=$'\t' read -r k v; do M["$k"]="$v"; done < <(
            printf '%s' "$m" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('\n'.join('%s\t%s' % (k, v) for k, v in d.items()))"
        )

        echo "  [$dataset] gt (stable-key only): songs maj=${M[song_maj]} min=${M[song_min]} | segs maj=${M[seg_maj]} min=${M[seg_min]}"
        for lvl in segment song; do
            p=seg; [[ "$lvl" == song ]] && p=song
            f="f_${lvl}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$feature" "$lid" "$mode" "$degree" "$dataset" \
                "${M[${p}_maj]}" "${M[${p}_min]}" "${M[${p}_total]}" \
                "${M[${p}_acc]}" "${M[${p}_acc_strict]}" "${M[${p}_mirex]}" >> "${!f}"
        done
        echo "  key[$degree] seg=${M[seg_acc]}%  song=${M[song_acc]}%"
    done

    PASS=$((PASS+1))
done

echo ""; echo "$LOG_SEP"
echo " Done: $PASS passed, $FAIL failed"
echo "$LOG_SEP"
echo ""
for lvl in segment song; do
    f="f_${lvl}"
    echo "── ${lvl} relative : ${!f}"
done
echo ""
echo "  probe_group_dir : $probe_group_dir"
echo "  key_out_dir     : $key_out_dir"
