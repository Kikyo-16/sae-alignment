#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_relative_key.sh — Key-signature accuracy from the SAE [Forth] / [Fifth] /
# [Tonic] rings, scored exactly like run_linear_relative_key.sh.
#
# SAE counterpart of run_linear_relative_key.sh.  Each ring is a plain
# single-ring signature predictor (src.analysis.infer_key.estimate_key_from_ring:
# time-average → argmax → minus the degree offset), with NO 3-SAE averaging —
# unlike the forth variant in run_minor_key.sh.  Metrics come from
# src.analysis.eval_metrics key with no extra flags (merged-signature acc,
# strict acc, mirex; stable-key songs only), identical to the linear-probe
# relative-key evaluation.
#
# Like run_minor_key.sh, this assumes run_dir.sh has already produced the
# per-run feature_ids.txt; 1-1 models have no SAE rings, so they read the
# probe rings from probe_feature_ids.txt instead.  Rings absent from the
# feature-ids file are skipped.
#
# For each level (segment + song) it writes one merged-signature result file
# with the same columns as run_linear_relative_key.sh.
#
# Usage:
#   bash scripts/analyze/run_relative_key.sh \
#       <sae_group_dir> <feat_ids_dir> <key_out_dir>
#
#   sae_group_dir  – root of SAE model directories (contains per-run subdirs
#                    with ckpts/, same structure as run_dir.sh input)
#   feat_ids_dir   – results directory written by run_dir.sh (contains per-run
#                    subdirs with epoch*_feature_ids.txt)
#   key_out_dir    – directory to write key output files and summary TSVs
#
# Example:
#   bash scripts/analyze/run_relative_key.sh \
#       store/important_exps/saes/slakh-64-3-1 \
#       store/important_exps/results/slakh-64-3-1 \
#       store/important_exps/results/slakh-64-3-1-relkey
# ---------------------------------------------------------------------------
set -euo pipefail

sae_group_dir="${1:?Usage: $0 <sae_group_dir> <feat_ids_dir> <key_out_dir>}"
feat_ids_dir="${2:?Usage: $0 <sae_group_dir> <feat_ids_dir> <key_out_dir>}"
key_out_dir="${3:?Usage: $0 <sae_group_dir> <feat_ids_dir> <key_out_dir>}"

sae_group_dir="${sae_group_dir%/}"
feat_ids_dir="${feat_ids_dir%/}"
key_out_dir="${key_out_dir%/}"

# ── Configuration ────────────────────────────────────────────────────────────
sae_data_root="sae-data"

key_datasets=(rwc gtzan giantsteps_key fmakv2)
key_splits=(test test test test)

batch_size=64
device=cuda
# Set FORCE_INFER=1 to regenerate cached key output files.
force_infer="${FORCE_INFER:-0}"

# degree token → step4_infer plain single-ring out-file flag
declare -A DEGREE_TO_FLAG=(
    [forth]=--forth-ring-out-file
    [fifth]=--fifth-out-file
    [tonic]=--tonic-out-file
)
all_degrees=(forth fifth tonic)

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

# ── Main loop ────────────────────────────────────────────────────────────────

for run_dir in "${sae_group_dir}"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name=$(basename "$run_dir")

    # Same regex as run_dir.sh
    if [[ "$run_name" =~ ^[^_]+_([^-]+)_([^-]+)_([^-]+)-([0-9]+)_([0-9]+-[0-9]+)_([^_]+)_([0-9]+)_max$ ]]; then
        feature="${BASH_REMATCH[3]}"
        lid="${BASH_REMATCH[4]}"
        mode="${BASH_REMATCH[5]}"
    else
        echo "[SKIP] Unrecognised name: $run_name" >&2
        continue
    fi

    echo ""; echo "$LOG_SEP"
    echo " PROCESS  feature=$feature  lid=$lid  mode=$mode"
    echo "$LOG_SEP"

    # Checkpoint with the largest epoch (latest), matching run_dir.sh.
    # Names are val_probe_chord_acc-epoch<N>-step<N>-score<V>.ckpt.
    ckpt=$(ls "${run_dir}ckpts"/val_probe_chord_acc-*.ckpt 2>/dev/null | \
           awk -F'epoch' 'NR==1 || $2+0 > best {best=$2+0; line=$0} END{print line}' || true)
    if [[ -z "$ckpt" ]]; then
        echo "[SKIP] No checkpoint in ${run_dir}ckpts/" >&2; FAIL=$((FAIL+1)); continue
    fi
    epoch_str=$(basename "$ckpt" | sed -E 's/.*-epoch([0-9]+)-.*/\1/')

    [[ "$feature" == "cqt" ]] && feature_ds="cqt" || feature_ds="${feature}_layer"

    # 1-1 models have no SAE ring structure: run_dir.sh skips steps 1-3 for them
    # and only writes the probe rings, so predict from probe_feature_ids.txt.
    if [[ "$mode" == "1-1" ]]; then
        feat_ids="${feat_ids_dir}/${run_name}/epoch${epoch_str}_probe_feature_ids.txt"
    else
        feat_ids="${feat_ids_dir}/${run_name}/epoch${epoch_str}_feature_ids.txt"
    fi
    if [[ ! -f "$feat_ids" ]]; then
        echo "[SKIP] $(basename "$feat_ids") not found: $feat_ids" >&2
        echo "       Run run_dir.sh first." >&2
        FAIL=$((FAIL+1)); continue
    fi

    # Only evaluate rings present in feature_ids.txt (step4 skips absent rings,
    # so their output files would never exist and defeat the infer cache).
    degrees=()
    for d in "${all_degrees[@]}"; do
        grep -qiE "^\[${d}\]" "$feat_ids" 2>/dev/null && degrees+=("$d")
    done
    if [[ ${#degrees[@]} -eq 0 ]]; then
        echo "[SKIP] No [Forth] / [Fifth] / [Tonic] ring in $feat_ids" >&2
        FAIL=$((FAIL+1)); continue
    fi
    echo "  rings: ${degrees[*]}"

    run_out_dir="${key_out_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    for dataset in "${key_datasets[@]}"; do
        split=$(get_split "$dataset" key_datasets key_splits)
        h5="${sae_data_root}/${dataset}/${feature}/${dataset}_${feature}_30s_layer_${lid}.h5"

        echo ""; echo "  [$dataset / $split]"
        if [[ ! -f "$h5" ]]; then
            echo "  [SKIP] H5 not found: $h5" >&2; continue
        fi

        # (Re)run inference if any ring output is missing, or FORCE_INFER=1.
        infer_args=()
        need_infer=false
        [[ "$force_infer" != "0" ]] && need_infer=true
        for degree in "${degrees[@]}"; do
            key_out="${run_out_dir}/${dataset}_${split}_${mode}_ep${epoch_str}_key_${degree}.txt"
            infer_args+=("${DEGREE_TO_FLAG[$degree]}" "$key_out")
            [[ ! -f "$key_out" ]] && need_infer=true
        done

        if $need_infer; then
            python -m src.analysis.step4_infer \
                --h5-path          "$h5" \
                --split            "$split" \
                --ckpt-path        "$ckpt" \
                --feature-id-file  "$feat_ids" \
                --data-tag         "${feature}_layer_${lid}" \
                --feature-ds       "$feature_ds" \
                "${infer_args[@]}" \
                --batch-size       $batch_size \
                --device           $device \
                --allow-missing-labels
        else
            echo "  [CACHED] ring outputs present"
        fi

        counts_printed=false
        for degree in "${degrees[@]}"; do
            key_out="${run_out_dir}/${dataset}_${split}_${mode}_ep${epoch_str}_key_${degree}.txt"
            [[ -f "$key_out" ]] || { echo "  [SKIP] $degree output missing"; continue; }

            # Plain single-ring file (degree offset baked in at inference) → eval
            # with no flags, exactly like run_linear_relative_key.sh.
            m=$(python -m src.analysis.eval_metrics key "$key_out")
            unset M; declare -A M
            while IFS=$'\t' read -r k v; do M["$k"]="$v"; done < <(
                printf '%s' "$m" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('\n'.join('%s\t%s' % (k, v) for k, v in d.items()))"
            )

            if ! $counts_printed; then
                echo "  [$dataset] gt (stable-key only): songs maj=${M[song_maj]} min=${M[song_min]} | segs maj=${M[seg_maj]} min=${M[seg_min]}"
                counts_printed=true
            fi
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
echo "  sae_group_dir : $sae_group_dir"
echo "  feat_ids_dir  : $feat_ids_dir"
echo "  key_out_dir   : $key_out_dir"
