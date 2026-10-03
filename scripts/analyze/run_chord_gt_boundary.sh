#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_chord_gt_boundary.sh — Chord accuracy with GT vs estimated boundaries.
#
# Assumes run_dir.sh has already been run.  Reads the existing
# feature_ids.txt from the feature-ids results directory (for Major/Minor
# ring IDs); 1-1 models have no SAE rings, so they read the probe rings from
# probe_feature_ids.txt instead.  Then runs chord inference two ways:
#   - GT boundary  : chord_frame label changes + N/X silence labels
#                    (src.analysis.step4_infer_gt_boundary)
#   - Est boundary : SAE [Chord Boundary] / silence features
#                    (src.analysis.step4_infer; needs a [Chord Boundary] ring)
# and writes four summary TSVs = {micro, song-macro} level x {gt, est} boundary.
# Est-boundary is evaluated three times: [Chord Boundary] top1/top2/top3.
#
# Usage:
#   bash scripts/analyze/run_chord_gt_boundary.sh \
#       <sae_group_dir> <feat_ids_dir> <chord_out_dir>
#
#   sae_group_dir  – root of SAE model directories (contains per-run subdirs
#                    with ckpts/, same structure as run_dir.sh input)
#   feat_ids_dir   – results directory written by run_dir.sh (contains per-run
#                    subdirs with epoch*_feature_ids.txt)
#   chord_out_dir  – directory to write chord output files and summary TSVs
#
# Example:
#   bash scripts/analyze/run_chord_gt_boundary.sh \
#       store/important_exps/saes/slakh-64-3-1 \
#       store/important_exps/results/slakh-64-3-1 \
#       store/important_exps/results/slakh-64-3-1-gtbnd
# ---------------------------------------------------------------------------
set -euo pipefail

sae_group_dir="${1:?Usage: $0 <sae_group_dir> <feat_ids_dir> <chord_out_dir>}"
feat_ids_dir="${2:?Usage: $0 <sae_group_dir> <feat_ids_dir> <chord_out_dir>}"
chord_out_dir="${3:?Usage: $0 <sae_group_dir> <feat_ids_dir> <chord_out_dir>}"

sae_group_dir="${sae_group_dir%/}"
feat_ids_dir="${feat_ids_dir%/}"
chord_out_dir="${chord_out_dir%/}"

# ── Configuration ────────────────────────────────────────────────────────────
sae_data_root="sae-data"

chord_datasets=(pop909 rwc slakh)
chord_splits=(train test test)

batch_size=64
device=cuda
smooth_win=9          # MA window on chord ring acts before mode/root (also SAE boundary/silence for est)
majmin_mode=seg-mean-max  # sum-peak | seg-mean-max | seg-mean-mean | raw-tmpl (both gt & est)
majmin_conf_thr=0.18      # raw-tmpl only: top-2 raw-margin for template fallback
majmin_alpha=0.4          # raw-tmpl only: triad-template blend weight

# Estimated-boundary peak logic. Keep these matched to step3_boundary.
boundary_top_k=3
chord_boundary_idx=0
boundary_peak_tol=0.2
boundary_peak_drop=0.0
boundary_min_gap=0
boundary_score_floor=0.0
boundary_max_labels=120

# ── Output files ─────────────────────────────────────────────────────────────
mkdir -p "$chord_out_dir"
# Four summaries: corpus-level micro WCSR and song-level macro WCSR for
# GT and estimated boundaries.  The legacy *_segment_level_* filenames are
# retained for compatibility, but those files now contain micro frame-level
# WCSR from eval_metrics.py, not a mean over predicted segments.
seg_gt_tsv="${chord_out_dir}/chord_segment_level_gtbnd.tsv"
song_gt_tsv="${chord_out_dir}/chord_song_level_gtbnd.tsv"
seg_est_tsv="${chord_out_dir}/chord_segment_level_estbnd.tsv"
song_est_tsv="${chord_out_dir}/chord_song_level_estbnd.tsv"

for tsv in "$seg_gt_tsv" "$song_gt_tsv" "$seg_est_tsv" "$song_est_tsv"; do
    printf 'feature\tlid\tmode\tdataset\tboundary_feature_rank\troot\tmajmin\tmirex\tn_songs\n' > "$tsv"
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

# eval_and_write <chord_out> <h5_path> <split> <micro_tsv> <song_tsv> <label> <boundary_tag>
# The evaluator always scores against the original frame-level chord_frame in
# the HDF5 file.  micro_* values are corpus-level frame-weighted WCSR; root /
# majmin / mirex are song-level macro WCSR.
eval_and_write() {
    local chord_out="$1"
    local h5_path="$2"
    local split_name="$3"
    local micro_tsv="$4"
    local song_tsv="$5"
    local label="$6"
    local boundary_tag="$7"

    [[ -f "$chord_out" ]] || {
        echo "  [SKIP $label] output missing after inference: $chord_out" >&2
        return 1
    }
    [[ -f "$h5_path" ]] || {
        echo "  [SKIP $label] HDF5 missing: $h5_path" >&2
        return 1
    }

    local metrics_json
    if ! metrics_json="$(
        python -m src.analysis.eval_metrics chord "$chord_out" \
            --h5-path "$h5_path" \
            --split "$split_name"
    )"; then
        echo "  [ERROR $label] eval_metrics failed for $chord_out" >&2
        return 1
    fi

    local parsed
    if ! parsed="$(
        python3 -c '
import json
import sys

required = (
    "micro_root", "micro_majmin", "micro_mirex",
    "root", "majmin", "mirex", "n_songs",
)
data = json.load(sys.stdin)
missing = [key for key in required if key not in data]
if missing:
    raise KeyError(f"missing metric keys: {missing}")
print("\t".join(str(data[key]) for key in required))
' <<< "$metrics_json"
    )"; then
        echo "  [ERROR $label] invalid evaluator JSON:" >&2
        echo "$metrics_json" >&2
        return 1
    fi

    local micro_root micro_majmin micro_mirex
    local song_root song_majmin song_mirex n_songs
    IFS=$'\t' read -r \
        micro_root micro_majmin micro_mirex \
        song_root song_majmin song_mirex n_songs \
        <<< "$parsed"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$feature" "$lid" "$mode" "$dataset" "$boundary_tag" \
        "$micro_root" "$micro_majmin" "$micro_mirex" "$n_songs" \
        >> "$micro_tsv"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$feature" "$lid" "$mode" "$dataset" "$boundary_tag" \
        "$song_root" "$song_majmin" "$song_mirex" "$n_songs" \
        >> "$song_tsv"

    echo "  $label ($boundary_tag)  micro: root=${micro_root}% majmin=${micro_majmin}% mirex=${micro_mirex}%   song-macro: root=${song_root}% majmin=${song_majmin}% mirex=${song_mirex}%"
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

    run_out_dir="${chord_out_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    # Check that major/minor rings are present (needed for chord inference)
    if ! grep -qE '^\[(Major|Minor) Chord\]' "$feat_ids" 2>/dev/null; then
        echo "[SKIP] No [Major Chord] / [Minor Chord] in $feat_ids" >&2
        FAIL=$((FAIL+1)); continue
    fi

    # Est-boundary inference additionally needs a [Chord Boundary] ring.
    has_boundary=false
    grep -qE '^\[Chord Boundary\]' "$feat_ids" 2>/dev/null && has_boundary=true

    for dataset in "${chord_datasets[@]}"; do
        split=$(get_split "$dataset" chord_datasets chord_splits)
        h5="${sae_data_root}/${dataset}/${feature}/${dataset}_${feature}_30s_layer_${lid}.h5"
        out_pfx="${run_out_dir}/${dataset}_${split}_${mode}_ep${epoch_str}_chord"

        echo ""; echo "  [$dataset / $split]"

        # ── GT boundary (chord_frame label changes + N/X silence) ────────────
        chord_out_gt="${out_pfx}_gtbnd.txt"
        if [[ ! -f "$chord_out_gt" ]]; then
            python -m src.analysis.step4_infer_gt_boundary \
                --h5-path            "$h5" \
                --split              "$split" \
                --ckpt-path          "$ckpt" \
                --feature-id-file    "$feat_ids" \
                --data-tag           "${feature}_layer_${lid}" \
                --feature-ds         "$feature_ds" \
                --enable-chord \
                --chord-out-file     "$chord_out_gt" \
                --smooth-win         $smooth_win \
                --majmin-mode        "$majmin_mode" \
                --majmin-conf-thr    "$majmin_conf_thr" \
                --majmin-alpha       "$majmin_alpha" \
                --batch-size         $batch_size \
                --device             $device \
                --allow-missing-labels
        else
            echo "  [CACHED] $chord_out_gt"
        fi
        eval_and_write "$chord_out_gt" "$h5" "$split" "$seg_gt_tsv" "$song_gt_tsv" "gtbnd" "gt" || true

        # ── Estimated boundary (SAE [Chord Boundary] / silence features) ─────
        if $has_boundary; then
            chord_out_est="${out_pfx}_estbnd.txt"
            est_stem="${chord_out_est%.txt}"
            all_est_cached=true
            for r in $(seq 1 "$boundary_top_k"); do
                [[ -f "${est_stem}_bndtop${r}.txt" ]] || all_est_cached=false
            done
            if ! $all_est_cached; then
                python -m src.analysis.step4_infer \
                    --h5-path              "$h5" \
                    --split                "$split" \
                    --ckpt-path            "$ckpt" \
                    --feature-id-file      "$feat_ids" \
                    --data-tag             "${feature}_layer_${lid}" \
                    --feature-ds           "$feature_ds" \
                    --enable-chord \
                    --chord-out-file       "$chord_out_est" \
                    --chord-boundary-idx   $chord_boundary_idx \
                    --boundary-top-k       $boundary_top_k \
                    --boundary-peak-tol    $boundary_peak_tol \
                    --boundary-peak-drop   $boundary_peak_drop \
                    --boundary-min-gap     $boundary_min_gap \
                    --boundary-score-floor $boundary_score_floor \
                    --boundary-max-labels  $boundary_max_labels \
                    --smooth-win           $smooth_win \
                    --majmin-mode          "$majmin_mode" \
                    --majmin-conf-thr      "$majmin_conf_thr" \
                    --majmin-alpha         "$majmin_alpha" \
                    --batch-size           $batch_size \
                    --device               $device \
                    --allow-missing-labels
            else
                echo "  [CACHED] ${est_stem}_bndtop1..${boundary_top_k}.txt"
            fi
            for r in $(seq 1 "$boundary_top_k"); do
                chord_out_rank="${est_stem}_bndtop${r}.txt"
                eval_and_write "$chord_out_rank" "$h5" "$split" "$seg_est_tsv" "$song_est_tsv" "estbnd_top${r}" "top${r}" || true
            done
        else
            echo "  [SKIP estbnd] no [Chord Boundary] ring in feat_ids"
        fi
    done

    PASS=$((PASS+1))
done

echo ""; echo "$LOG_SEP"
echo " Done: $PASS passed, $FAIL failed"
echo "$LOG_SEP"
echo ""
echo "── GT boundary  micro WCSR    : $seg_gt_tsv"
echo "── GT boundary  song macro    : $song_gt_tsv"
echo "── Est boundary micro WCSR    : $seg_est_tsv"
echo "── Est boundary song macro    : $song_est_tsv"
echo ""
echo "  feat_ids_dir : $feat_ids_dir"
echo "  chord_out_dir: $chord_out_dir"
