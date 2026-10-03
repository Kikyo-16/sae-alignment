#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_activation_counts.sh  —  Batch per-feature activation-count statistics
# for all model sub-directories under <sae_group_dir>.
#
# For every recognised model sub-directory this script runs
# step_activation_counts.py and writes two files under
# <results_dir>/{run_name}/:
#   activation_counts.npy   int64 array [D] — count per feature
#   activation_counts.txt   TSV sorted by count descending
#
# Usage:
#   bash scripts/analyze/run_activation_counts.sh <sae_group_dir> [<results_dir>]
#
# Example:
#   bash scripts/analyze/run_activation_counts.sh \
#       <models_root>/slakh-muq-64-3-1 \
#       store/results_latest/analysis/activation_counts/slakh-muq-64-3-1
# ---------------------------------------------------------------------------
set -euo pipefail

sae_group_dir="${1:?Usage: $0 <sae_group_dir> [<results_dir>]}"
sae_group_dir="${sae_group_dir%/}"
group_name=$(basename "$sae_group_dir")
results_dir="${2:-store/results_latest/analysis/activation_counts/${group_name}}"

# ── Configuration ────────────────────────────────────────────────────────────
sae_data_root="sae-data"

# H5 dataset used for activation-count statistics
# (sae-data/${count_dataset}/${feature}/${count_dataset}_${feature}_30s_layer_${lid}.h5)
count_dataset="slakh2100_train_16"
count_split="train"

# Shared inference settings
batch_size=64
sae_idx=0
device=cuda

# ── Output directory ──────────────────────────────────────────────────────────
mkdir -p "$results_dir"

LOG_SEP="============================================================"
PASS=0; FAIL=0

# ── Main loop ─────────────────────────────────────────────────────────────────

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

    # ── Find checkpoint with the largest epoch (latest), matching run_dir.sh ──
    # Names are val_probe_chord_acc-epoch<N>-step<N>-score<V>.ckpt.
    ckpt=$(ls "${run_dir}ckpts"/val_probe_chord_acc-*.ckpt 2>/dev/null | \
           awk -F'epoch' 'NR==1 || $2+0 > best {best=$2+0; line=$0} END{print line}' || true)
    if [[ -z "$ckpt" ]]; then
        echo "[SKIP] No checkpoint in ${run_dir}ckpts/" >&2
        FAIL=$((FAIL+1)); continue
    fi

    [[ "$feature" == "cqt" ]] && feature_ds="cqt" || feature_ds="${feature}_layer"

    # ── Per-run output directory ───────────────────────────────────────────────
    run_out_dir="${results_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    # ── H5 path ───────────────────────────────────────────────────────────────
    count_h5="${sae_data_root}/${count_dataset}/${feature}/${count_dataset}_${feature}_30s_layer_${lid}.h5"

    # ── Run activation counts (cached if output already present) ──────────────
    counts_npy="${run_out_dir}/activation_counts.npy"
    if [[ -f "$counts_npy" ]]; then
        echo "[CACHED] activation_counts.npy already exists: $counts_npy"
        PASS=$((PASS+1)); continue
    fi

    echo "[STEP] Activation counts ..."
    if ! python -m src.analysis.step_activation_counts \
            --ckpt-path   "$ckpt" \
            --h5-path     "$count_h5" \
            --split       "$count_split" \
            --feature-ds  "$feature_ds" \
            --out-dir     "$run_out_dir" \
            --sae-idx     $sae_idx \
            --device      $device \
            --batch-size  $batch_size; then
        echo "[FAIL] step_activation_counts" >&2
        FAIL=$((FAIL+1)); continue
    fi

    PASS=$((PASS+1))
done

echo ""; echo "$LOG_SEP"
echo " Done: $PASS passed, $FAIL failed"
echo "$LOG_SEP"
echo ""
echo "── Results under: $results_dir"
