#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_vis_features.sh  —  Batch SAE feature visualisation for all model
# sub-directories under <sae_group_dir>.
#
# For every recognised sub-directory this script runs step_vis_features.py,
# which writes images under <out_dir>/{run_name}/:
#   features_all.png    all SAE features, coloured by semantic category
#   features_forth.png  [Forth] ring only, with pitch-class labels
#   features_major.png  [Major Chord] ring  (mode=3-1 only)
#   features_minor.png  [Minor Chord] ring  (mode=3-1 only)
#
# Checkpoints are found under  <sae_group_dir>/{run_name}/ckpts/.
# For mode=3-1: orbits_raw.txt + epoch*_feature_ids.txt from <feat_ids_dir>/{run_name}/.
# For mode=1-1: epoch*_probe_feature_ids.txt from <feat_ids_dir>/{run_name}/;
#               --orbits is omitted so only Forth vs Others coloring is used.
# activation_counts.npy is read from <alive_counts_dir>/{run_name}/ when
# <alive_counts_dir> is provided (produced by run_activation_counts.sh).
#
# Skips:
#   runs where features_all.png already exists (cached)
#   mode=3-1 runs where orbits_raw.txt or feature_ids.txt are missing
#   mode=1-1 runs where probe_feature_ids.txt is missing
#
# Usage:
#   bash scripts/analyze/run_vis_features.sh \
#       <sae_group_dir> <feat_ids_dir> [<alive_counts_dir> [<out_dir>]]
#
# Example (with activation filtering):
#   bash scripts/analyze/run_vis_features.sh \
#       <models_root>/slakh-muq-64-3-1 \
#       <feat_ids_root>/slakh-muq-64-3-1 \
#       store/results_latest/analysis/activation_counts/slakh-muq-64-3-1
# ---------------------------------------------------------------------------
set -euo pipefail

sae_group_dir="${1:?Usage: $0 <sae_group_dir> <feat_ids_dir> [<alive_counts_dir> [<out_dir>]]}"
feat_ids_dir="${2:?Usage: $0 <sae_group_dir> <feat_ids_dir> [<alive_counts_dir> [<out_dir>]]}"
alive_counts_dir="${3:-}"           # optional: alive_features_analysis/{group_name}
sae_group_dir="${sae_group_dir%/}"
feat_ids_dir="${feat_ids_dir%/}"
[[ -n "$alive_counts_dir" ]] && alive_counts_dir="${alive_counts_dir%/}"
out_dir="${4:-${feat_ids_dir}}"

# ── Configuration ────────────────────────────────────────────────────────────
device=cuda
vec=dec
method=tsne
min_act_count=2000                     # minimum activation count; ignored if no alive_counts_dir

# ── Output directory ──────────────────────────────────────────────────────────
mkdir -p "$out_dir"

LOG_SEP="============================================================"
PASS=0; FAIL=0; SKIP=0

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
        SKIP=$((SKIP+1)); continue
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
    epoch_str=$(basename "$ckpt" | sed -E 's/.*-epoch([0-9]+)-.*/\1/')

    # ── Per-run directories ────────────────────────────────────────────────────
    feat_ids_run_dir="${feat_ids_dir}/${run_name}"
    run_out_dir="${out_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    # ── Input files — mode-dependent ─────────────────────────────────────────
    if [[ "$mode" == "1-1" ]]; then
        # No ring structure; use probe_feature_ids only (Forth/Fifth/Tonic),
        # omit --orbits so the script uses Forth-vs-Others coloring only.
        orbits_args=()
        ids_file="${feat_ids_run_dir}/epoch${epoch_str}_probe_feature_ids.txt"
        if [[ ! -f "$ids_file" ]]; then
            echo "[SKIP] probe_feature_ids.txt not found: $ids_file" >&2
            SKIP=$((SKIP+1)); continue
        fi
    else
        orbits_file="${feat_ids_run_dir}/orbits_raw.txt"
        ids_file="${feat_ids_run_dir}/epoch${epoch_str}_feature_ids.txt"
        if [[ ! -f "$orbits_file" ]]; then
            echo "[SKIP] orbits_raw.txt not found: $orbits_file" >&2
            SKIP=$((SKIP+1)); continue
        fi
        if [[ ! -f "$ids_file" ]]; then
            echo "[SKIP] feature_ids.txt not found: $ids_file" >&2
            SKIP=$((SKIP+1)); continue
        fi
        orbits_args=(--orbits "$orbits_file")
    fi

    # ── Cached check ──────────────────────────────────────────────────────────
    if [[ -f "${run_out_dir}/features_all.png" ]]; then
        echo "[CACHED] features_all.png already exists: ${run_out_dir}/features_all.png"
        PASS=$((PASS+1)); continue
    fi

    # ── Optional activation-count filtering ──────────────────────────────────
    act_count_args=()
    if [[ -n "$alive_counts_dir" ]]; then
        act_counts_file="${alive_counts_dir}/${run_name}/activation_counts.npy"
        if [[ -f "$act_counts_file" ]]; then
            act_count_args=(--act-counts-file "$act_counts_file"
                            --min-act-count   $min_act_count)
            echo "[INFO] Using activation counts: $act_counts_file (min=$min_act_count)"
        else
            echo "[WARN] activation_counts.npy not found, skipping filter: $act_counts_file" >&2
        fi
    fi

    echo "[STEP] Visualise features ..."
    if ! python -m src.analysis.step_vis_features \
            --ckpt-path   "$ckpt" \
            --feature-ids "$ids_file" \
            --out-dir     "$run_out_dir" \
            --vec         $vec \
            --method      $method \
            --device      $device \
            "${orbits_args[@]}" \
            "${act_count_args[@]}"; then
        echo "[FAIL] step_vis_features" >&2
        FAIL=$((FAIL+1)); continue
    fi

    PASS=$((PASS+1))
done

echo ""; echo "$LOG_SEP"
echo " Done: $PASS passed, $FAIL failed, $SKIP skipped"
echo "$LOG_SEP"
echo ""
echo "── Results under: $out_dir"
