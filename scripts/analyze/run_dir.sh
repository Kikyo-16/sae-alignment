#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_dir.sh  —  Ring-finding pipeline for one SAE group directory.
#
# For every model sub-directory under <sae_group_dir> this script:
#   Step 1: Compute SAE self-graph   → orbits_raw.txt, orbits_raw_score.txt,
#                                      timbre.txt, invalid.txt
#   Step 2: Classify size-12 rings   → feature_ids.txt
#   Step 3: Boundary + silence       → append to feature_ids.txt
#   Step P: Chord-probe weights → [Major/Minor Chord] probe_feature_ids.txt
#           Key-probe weights   → [Forth]/[Fifth]/[Tonic] (appended)
#
# Inference and metric computation are intentionally NOT done here — that is
# handled by separate scripts that consume the *_feature_ids.txt files.
#
# All output goes under <results_dir>/{run_name}/.
# The original SAE model directory is never written to.
#
# Usage:
#   bash scripts/analyze/run_dir.sh <sae_group_dir> [<results_dir>]
#
# Example:
#   bash scripts/analyze/run_dir.sh \
#       store/important_exps/saes/slakh-64-3-1 \
#       store/important_exps/results/slakh-64-3-1
# ---------------------------------------------------------------------------
set -euo pipefail

sae_group_dir="${1:?Usage: $0 <sae_group_dir> [<results_dir>]}"
sae_group_dir="${sae_group_dir%/}"
group_name=$(basename "$sae_group_dir")
results_dir="${2:-store/important_exps/results/${group_name}}"

# ── Configuration ────────────────────────────────────────────────────────────
sae_data_root="sae-data"

# H5 dataset used for ring discovery (step 2)
ring_dataset="pop909-test"
ring_split="test"

# H5 dataset used for boundary + silence detection (step 3)
boundary_dataset="pop909-test"
boundary_split="test"

# Shared inference settings
batch_size=64
device=cuda

# Chord linear-probe checkpoints (step P).
# Root directory of chord linear-probe experiments. Sub-directories follow:
#   {probe_base_dir}/chord_{feature}-{lid}/ckpts/<metric>-...-score<V>.ckpt
# Set to "" to skip the chord-probe part of step P.
probe_base_dir="store/results_latest/models/probing/linear_chord_new"

# Key (scale-degree) linear-probe checkpoints (step P). Per degree
# d ∈ {forth, fifth, tonic} the experiments live under:
#   {key_probe_root}/{feature}_{d}/key_{d}_{feature}-{lid}/ckpts/<metric>-...-score<V>.ckpt
# Set to "" to skip the key-probe rings.
key_probe_root="store/results_latest/models/probing"

# ── Output files ─────────────────────────────────────────────────────────────
mkdir -p "$results_dir"
ckpts_tsv="${results_dir}/ckpts_used.tsv"
printf 'feature\tlid\tmode\tckpt\n' > "$ckpts_tsv"

LOG_SEP="============================================================"
PASS=0; FAIL=0

# Pick the ckpt with the highest score parsed from the filename
# (…-score<V>.ckpt), same as run_linear_relative_key.sh.
# Prints nothing (exit 0) when the dir/ckpts are missing — under
# `set -euo pipefail` a failing `ls` inside the $() would otherwise
# silently kill the whole script at the assignment.
best_score_ckpt() {
    ls "$1"/*-score*.ckpt 2>/dev/null | \
        awk -F'score' 'NR==1 || $2+0 > best {best=$2+0; line=$0} END{print line}' || true
}

# ── Main loop ────────────────────────────────────────────────────────────────

for run_dir in "${sae_group_dir}"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name=$(basename "$run_dir")

    # Parse: {prefix}_{feature}-{lid}_{mode}_{l1}_{expansion}_max
    # prefix  = any string without underscores (e.g. slakh2100-val, musicnet)
    # feature = any string without hyphens    (e.g. muq, cqt)
    if [[ "$run_name" =~ ^[^_]+_([^-]+)_([^-]+)_([^-]+)-([0-9]+)_([0-9]+-[0-9]+)_([^_]+)_([0-9]+)_max$ ]]; then
        feature="${BASH_REMATCH[3]}"
        lid="${BASH_REMATCH[4]}"
        mode="${BASH_REMATCH[5]}"
        l1="${BASH_REMATCH[6]}"
        expansion="${BASH_REMATCH[7]}"
    else
        echo "[SKIP] Unrecognised name: $run_name" >&2
        continue
    fi

    echo ""; echo "$LOG_SEP"
    echo " PROCESS  feature=$feature  lid=$lid  mode=$mode"
    echo "$LOG_SEP"

    # ── Find checkpoint with the largest epoch ───────────────────────────────
    # Names are ...-epoch<N>-step<N>-score<V>.ckpt; pick the highest epoch
    # (latest), not the best score.
    ckpt=$(ls "${run_dir}ckpts"/val_probe_chord_acc-*.ckpt 2>/dev/null | \
           awk -F'epoch' 'NR==1 || $2+0 > best {best=$2+0; line=$0} END{print line}' || true)
    if [[ -z "$ckpt" ]]; then
        echo "[SKIP] No checkpoint in ${run_dir}ckpts/" >&2
        FAIL=$((FAIL+1)); continue
    fi
    epoch_str=$(basename "$ckpt" | sed -E 's/.*-epoch([0-9]+)-.*/\1/')
    printf '%s\t%s\t%s\t%s\n' "$feature" "$lid" "$mode" "$ckpt" >> "$ckpts_tsv"

    [[ "$feature" == "cqt" ]] && feature_ds="cqt" || feature_ds="${feature}_layer"

    # ── Per-run output directory (all files go here, not into the model dir) ──
    run_out_dir="${results_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    # ── H5 paths ─────────────────────────────────────────────────────────────
    ring_h5="${sae_data_root}/${ring_dataset}/${feature}/${ring_dataset}_${feature}_30s_layer_${lid}.h5"
    boundary_h5="${sae_data_root}/${boundary_dataset}/${feature}/${boundary_dataset}_${feature}_30s_layer_${lid}.h5"

    # ── Steps 1-3: prepare feature_ids.txt if not already done ───────────────
    # Skipped for 1-1 models (single sub-SAE, no pitch-shift graph).
    feat_ids="${run_out_dir}/epoch${epoch_str}_feature_ids.txt"

    if [[ "$mode" == "1-1" ]]; then
        echo "[INFO] mode=1-1 → skipping steps 1-3 (no ring structure; boundary/silence added to probe_feature_ids in step P)"
    elif [[ -f "$feat_ids" ]]; then
        echo "[CACHED] feature_ids.txt already exists: $feat_ids"
    else
        echo "[STEP 1] Self-graph ..."
        if ! python -m src.analysis.step1_graph \
                --ckpt-path "$ckpt" \
                --out-dir   "$run_out_dir" \
                --device    $device; then
            echo "[FAIL] step1_graph" >&2; FAIL=$((FAIL+1)); continue
        fi

        echo "[STEP 2] Classify rings ..."
        if ! python -m src.analysis.step2_rings \
                --ckpt-path  "$ckpt" \
                --rings-file "${run_out_dir}/orbits_raw.txt" \
                --h5-path    "$ring_h5" \
                --split      "$ring_split" \
                --feature-ds "$feature_ds" \
                --out-file   "$feat_ids" \
                --device     $device \
                --batch-size $batch_size; then
            echo "[WARN] step2_rings: no size-12 rings — continuing to step 3 for boundary/silence" >&2
            touch "$feat_ids"
        fi

        echo "[STEP 3] Boundary + silence ..."
        if ! python -m src.analysis.step3_boundary \
                --ckpt-path        "$ckpt" \
                --h5-path          "$boundary_h5" \
                --split            "$boundary_split" \
                --feature-ds       "$feature_ds" \
                --feature-ids-file "$feat_ids" \
                --device           $device \
                --batch-size       $batch_size; then
            echo "[FAIL] step3_boundary" >&2; FAIL=$((FAIL+1)); continue
        fi
    fi

    # ── Step P: probe-based ring finding (chord + key degrees) ───────────────
    if [[ -n "$probe_base_dir" || -n "$key_probe_root" ]]; then
        probe_args=()

        # Chord probe: {probe_base_dir}/chord_{feature}-{lid}/ckpts/<metric>-...-score<V>.ckpt
        if [[ -n "$probe_base_dir" ]]; then
            chord_probe_dir="${probe_base_dir}/chord_${feature}-${lid}/ckpts"
            chord_ckpt=$(best_score_ckpt "$chord_probe_dir")
            if [[ -n "$chord_ckpt" ]]; then
                probe_args+=(--chord-probe "$chord_ckpt")
            else
                echo "[SKIP] No chord probe ckpt in ${chord_probe_dir}" >&2
            fi
        fi

        # Key probes: {key_probe_root}/{feature}_{d}/key_{d}_{feature}-{lid}/ckpts/
        if [[ -n "$key_probe_root" ]]; then
            for degree in forth fifth tonic; do
                key_probe_dir="${key_probe_root}/${feature}_${degree}/key_${degree}_${feature}-${lid}/ckpts"
                key_ckpt=$(best_score_ckpt "$key_probe_dir")
                if [[ -n "$key_ckpt" ]]; then
                    probe_args+=("--${degree}-probe" "$key_ckpt")
                else
                    echo "[SKIP] No ${degree} probe ckpt in ${key_probe_dir}" >&2
                fi
            done
        fi

        probe_feat_ids="${run_out_dir}/epoch${epoch_str}_probe_feature_ids.txt"

        if [[ ! -f "$probe_feat_ids" ]]; then
            echo "[STEP P] Probe rings ..."
            if [[ ${#probe_args[@]} -eq 0 ]]; then
                echo "[SKIP] No probe ckpts found for ${feature}-${lid}" >&2
            else
                if ! python -m src.analysis.step_probe_rings \
                        --ckpt-path   "$ckpt" \
                        "${probe_args[@]}" \
                        --out-file    "$probe_feat_ids" \
                        --device      $device; then
                    echo "[FAIL] step_probe_rings" >&2
                fi

                # Copy [Chord Boundary] and [Silence] from SAE feature_ids
                # (1-1 has none; handled by the step3 run below)
                if [[ "$mode" != "1-1" && -f "$probe_feat_ids" && -f "$feat_ids" ]]; then
                    grep -E '^\[(Chord Boundary|Silence)\]:' "$feat_ids" >> "$probe_feat_ids" || true
                fi
            fi
        else
            echo "[CACHED] probe_feature_ids.txt already exists: $probe_feat_ids"
        fi

        # 1-1 models have no SAE feature_ids.txt to copy [Chord Boundary]/[Silence]
        # from — detect them directly into probe_feature_ids.txt (step3 scans every
        # feature and does not need the ring structure), so est-boundary chord
        # inference (run_chord_gt_boundary.sh) works for 1-1 runs too.
        if [[ "$mode" == "1-1" && -f "$probe_feat_ids" ]] && \
           ! grep -qE '^\[Chord Boundary\]' "$probe_feat_ids" 2>/dev/null; then
            echo "[STEP 3] Boundary + silence (1-1 → probe_feature_ids) ..."
            if ! python -m src.analysis.step3_boundary \
                    --ckpt-path        "$ckpt" \
                    --h5-path          "$boundary_h5" \
                    --split            "$boundary_split" \
                    --feature-ds       "$feature_ds" \
                    --feature-ids-file "$probe_feat_ids" \
                    --device           $device \
                    --batch-size       $batch_size; then
                echo "[FAIL] step3_boundary (1-1)" >&2
            fi
        fi
    fi

    PASS=$((PASS+1))
done

echo ""; echo "$LOG_SEP"
echo " Done: $PASS passed, $FAIL failed"
echo "$LOG_SEP"
echo ""
echo "── Per-run ring files under : ${results_dir}/{run_name}/"
echo "──   SAE rings              : epoch{N}_feature_ids.txt"
echo "──   Probe rings            : epoch{N}_probe_feature_ids.txt"
echo "── Checkpoints used         : $ckpts_tsv"
