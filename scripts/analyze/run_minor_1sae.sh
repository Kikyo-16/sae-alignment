#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_minor_key_1sae.sh — run_minor_key.sh counterpart for 1-1 models
# (single sub-SAE): the joint estimator degrades to ONE sub-SAE, no 3-SAE
# averaging.
#
# 1-1 models have no SAE ring structure, so the [Major Chord] / [Minor Chord]
# rings come from probe_feature_ids.txt (run_dir.sh step P, chord probe).
# Two variants, same scoring model as run_minor_key.sh:
#
#   joint  major_minor_joint     joint template scoring (power compress +
#                                per-ring L1 + signature-first decision) on ONE
#                                sub-SAE.  step4_infer writes the vote2 layout
#                                with sub-SAE blocks 1/2 zeroed — the joint
#                                merge L1-normalizes after averaging, so this
#                                is exactly single-SAE scoring and
#                                eval_metrics --vote2-decision joint applies
#                                unchanged.
#          → joint_strict file: overall strict (mode-aware) acc + maj/min judgment acc
#   gt     major_minor_joint_gt  same scoring, argmax within the gt mode.
#          → gt file: overall acc + overall strict acc
#
# All accuracies count stable-key songs only; "merged" = key-signature space,
# "strict" = relative major/minor distinguished (same as run_minor_key.sh).
#
# Usage:
#   bash scripts/analyze/run_minor_key_1sae.sh \
#       <sae_group_dir> <feat_ids_dir> <key_out_dir>
#
#   sae_group_dir  – root of SAE model directories (contains per-run subdirs
#                    with ckpts/); only mode 1-1 runs are processed
#   feat_ids_dir   – results directory written by run_dir.sh (contains per-run
#                    subdirs with epoch*_probe_feature_ids.txt)
#   key_out_dir    – directory to write key output files and summary TSVs
#
# Example:
#   bash scripts/analyze/run_minor_key_1sae.sh \
#       store/important_exps/saes/slakh-1-1 \
#       store/important_exps/results/slakh-1-1 \
#       store/important_exps/results/slakh-1-1-minorkey
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
sae_idx=0
device=cuda
# Set FORCE_INFER=1 to regenerate cached key output files.
force_infer="${FORCE_INFER:-0}"

# ── Prediction variants (single-SAE) ─────────────────────────────────────────
key_variants=(major_minor_joint major_minor_joint_gt)

joint_variants=" major_minor_joint "
gt_variants=" major_minor_joint_gt "

# ── Output files ─────────────────────────────────────────────────────────────
# Per level (segment + song) there are two result files (same columns as
# run_minor_key.sh so 1-1 and 3-1 results can be compared side by side):
#   joint_strict : overall strict (mode-aware) acc of the joint
#   gt           : overall acc + overall strict acc of the gt-mode variant
mkdir -p "$key_out_dir"
for lvl in segment song; do
    eval "f3_${lvl}=\"${key_out_dir}/key_${lvl}_joint_strict.tsv\""
    eval "f4_${lvl}=\"${key_out_dir}/key_${lvl}_gt.tsv\""
done

h3='feature\tlid\tmode\tlabel_tag\tdataset\ttotal\tacc_strict\tmirex\tmajmin_acc\n'
h4='feature\tlid\tmode\tlabel_tag\tdataset\ttotal\tacc\tacc_strict\tmirex\n'
for lvl in segment song; do
    eval "printf '$h3' > \"\$f3_${lvl}\""
    eval "printf '$h4' > \"\$f4_${lvl}\""
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

    # This script is the 1-1 (single sub-SAE) counterpart of run_minor_key.sh;
    # multi-SAE runs belong to run_minor_key.sh.
    [[ "$mode" == "1-1" ]] || continue

    echo ""; echo "$LOG_SEP"
    echo " PROCESS  feature=$feature  lid=$lid  mode=$mode"
    echo "$LOG_SEP"

    # Checkpoint with the largest epoch (latest), matching run_dir.sh.
    ckpt=$(ls "${run_dir}ckpts"/val_probe_chord_acc-*.ckpt 2>/dev/null | \
           awk -F'epoch' 'NR==1 || $2+0 > best {best=$2+0; line=$0} END{print line}' || true)
    if [[ -z "$ckpt" ]]; then
        echo "[SKIP] No checkpoint in ${run_dir}ckpts/" >&2; FAIL=$((FAIL+1)); continue
    fi
    epoch_str=$(basename "$ckpt" | sed -E 's/.*-epoch([0-9]+)-.*/\1/')

    [[ "$feature" == "cqt" ]] && feature_ds="cqt" || feature_ds="${feature}_layer"

    feat_ids="${feat_ids_dir}/${run_name}/epoch${epoch_str}_probe_feature_ids.txt"
    if [[ ! -f "$feat_ids" ]]; then
        echo "[SKIP] probe_feature_ids.txt not found: $feat_ids" >&2
        echo "       Run run_dir.sh first." >&2
        FAIL=$((FAIL+1)); continue
    fi

    # The joint needs both probe chord rings.
    if ! grep -qE '^\[Major Chord\]' "$feat_ids" 2>/dev/null || \
       ! grep -qE '^\[Minor Chord\]' "$feat_ids" 2>/dev/null; then
        echo "[SKIP] Missing [Major Chord] / [Minor Chord] ring in $feat_ids" >&2
        echo "       Re-run run_dir.sh step P with a chord probe." >&2
        FAIL=$((FAIL+1)); continue
    fi

    run_out_dir="${key_out_dir}/${run_name}"
    mkdir -p "$run_out_dir"

    for dataset in "${key_datasets[@]}"; do
        split=$(get_split "$dataset" key_datasets key_splits)
        h5="${sae_data_root}/${dataset}/${feature}/${dataset}_${feature}_30s_layer_${lid}.h5"

        mmj_out="${run_out_dir}/${dataset}_${split}_${mode}_ep${epoch_str}_key_major_minor_joint.txt"
        mmjg_out="${run_out_dir}/${dataset}_${split}_${mode}_ep${epoch_str}_key_major_minor_joint_gt.txt"

        echo ""; echo "  [$dataset / $split]"
        if [[ ! -f "$h5" ]]; then
            echo "  [SKIP] H5 not found: $h5" >&2; continue
        fi
        counts_printed=false

        # (Re)run inference if any variant output is missing, or FORCE_INFER=1.
        need_infer=false
        [[ "$force_infer" != "0" ]] && need_infer=true
        for f in "$mmj_out" "$mmjg_out"; do
            [[ ! -f "$f" ]] && need_infer=true && break
        done

        if $need_infer; then
            python -m src.analysis.step4_infer \
                --h5-path                             "$h5" \
                --split                               "$split" \
                --ckpt-path                           "$ckpt" \
                --feature-id-file                     "$feat_ids" \
                --data-tag                            "${feature}_layer_${lid}" \
                --feature-ds                          "$feature_ds" \
                --sae-idx                             $sae_idx \
                --major-minor-joint-1sae-out-file     "$mmj_out" \
                --major-minor-joint-1sae-gt-out-file  "$mmjg_out" \
                --batch-size                          $batch_size \
                --device                              $device \
                --allow-missing-labels
        else
            echo "  [CACHED] variant outputs present"
        fi

        for label_tag in "${key_variants[@]}"; do
            case "$label_tag" in
                major_minor_joint)    key_out="$mmj_out" ;;
                major_minor_joint_gt) key_out="$mmjg_out" ;;
            esac
            [[ -f "$key_out" ]] || { echo "  [SKIP] $label_tag output missing"; continue; }

            # Same eval as run_minor_key.sh: the files are in the vote2 layout
            # (blocks 1/2 zeroed = exact single-SAE joint scores).
            vote2_decision=joint
            [[ "$label_tag" == major_minor_joint_gt ]] && vote2_decision=joint_gt
            m=$(python -m src.analysis.eval_metrics key "$key_out" \
                    --vote2-decision $vote2_decision)
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
                f3="f3_${lvl}"; f4="f4_${lvl}"
                if [[ "$joint_variants" == *" $label_tag "* ]]; then
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$feature" "$lid" "$mode" "$label_tag" "$dataset" \
                        "${M[${p}_total]}" "${M[${p}_acc_strict]}" "${M[${p}_mirex]}" "${M[${p}_majmin_acc]}" >> "${!f3}"
                elif [[ "$gt_variants" == *" $label_tag "* ]]; then
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$feature" "$lid" "$mode" "$label_tag" "$dataset" \
                        "${M[${p}_total]}" "${M[${p}_acc]}" "${M[${p}_acc_strict]}" "${M[${p}_mirex]}" >> "${!f4}"
                fi
            done
            echo "  key[$label_tag] seg=${M[seg_acc]}%  song=${M[song_acc]}%"
        done
    done

    PASS=$((PASS+1))
done

echo ""; echo "$LOG_SEP"
echo " Done: $PASS passed, $FAIL failed"
echo "$LOG_SEP"
echo ""
for lvl in segment song; do
    f3="f3_${lvl}"; f4="f4_${lvl}"
    echo "── ${lvl} joint strict : ${!f3}"
    echo "── ${lvl} gt           : ${!f4}"
done
echo ""
echo "  feat_ids_dir: $feat_ids_dir"
echo "  key_out_dir : $key_out_dir"
