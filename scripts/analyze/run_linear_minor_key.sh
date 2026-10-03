#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_linear_minor_key.sh — Key accuracy from the linear-probe chord model.
#
# Linear-probe counterpart of run_minor_key.sh.  The chord probe's 24 logits
# (relu'd, time-summed) act as the [Major Chord] / [Minor Chord] rings, so no
# feature_ids / rings are needed.  For each probe run under <probe_group_dir> it
# picks the best checkpoint (from eval_ckpts.json, sorted best-first), runs key
# inference (src.linear_probe.infer_key) for the six major/minor variants, and
# writes, per level (segment + song), four result files.
#
# The probe is a single decoder (no sub-SAE vote), so the forth-ring variants
# (forth / forth_minor_tonic / *_gt) are omitted — there is no probe equivalent.
#
# Usage:
#   bash scripts/analyze/run_linear_minor_key.sh \
#       <probe_group_dir> <key_out_dir>
#
#   probe_group_dir – root containing per-run subdirs named chord_<mode>
#                     (e.g. chord_muq-2), each with a ckpts/ dir.
#   key_out_dir     – directory to write key output files and summary TSVs.
#
# Example:
#   bash scripts/analyze/run_linear_minor_key.sh \
#       store/linear_chord_new \
#       store/linear_chord_new/results-minorkey
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

# ── Prediction variants (major/minor only; no forth ring on the chord probe) ──
key_variants=(minor_tonic major_tonic \
              major_minor_tonic major_minor_joint \
              major_minor_tonic_gt major_minor_joint_gt)

single_variants=" minor_tonic major_tonic "
joint_variants=" major_minor_tonic major_minor_joint "
gt_variants=" major_minor_tonic_gt major_minor_joint_gt "

# ── Output files ─────────────────────────────────────────────────────────────
mkdir -p "$key_out_dir"
for lvl in segment song; do
    eval "f1_${lvl}=\"${key_out_dir}/key_${lvl}_single.tsv\""
    eval "f2_${lvl}=\"${key_out_dir}/key_${lvl}_joint.tsv\""
    eval "f3_${lvl}=\"${key_out_dir}/key_${lvl}_joint_strict.tsv\""
    eval "f4_${lvl}=\"${key_out_dir}/key_${lvl}_gt.tsv\""
done

h1='feature\tlid\tmode\tlabel_tag\tdataset\tmaj_total\tmaj_correct\tmaj_acc\tmin_total\tmin_correct\tmin_acc\ttotal\tacc\n'
h2='feature\tlid\tmode\tlabel_tag\tdataset\tmaj_total\tmaj_pred_maj\tmaj_pred_ratio\tmin_total\tmin_pred_min\tmin_pred_ratio\ttotal\tacc\n'
h3='feature\tlid\tmode\tlabel_tag\tdataset\ttotal\tacc_strict\tmirex\tmajmin_acc\n'
h4='feature\tlid\tmode\tlabel_tag\tdataset\ttotal\tacc\tacc_strict\tmirex\n'
for lvl in segment song; do
    eval "printf '$h1' > \"\$f1_${lvl}\""
    eval "printf '$h2' > \"\$f2_${lvl}\""
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

best_ckpt() {
    local ckpt_dir="$1"
    local meta="${ckpt_dir}/eval_ckpts.json"
    if [[ -f "$meta" ]]; then
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[0]['path'] if d else '')" "$meta"
        return
    fi
    ls "${ckpt_dir}"/*.ckpt 2>/dev/null | \
        awk -F'score' 'NR==1 || $2+0 > best {best=$2+0; line=$0} END{print line}'
}

# ── Main loop ────────────────────────────────────────────────────────────────

for run_dir in "${probe_group_dir}"/*/; do
    [[ -d "$run_dir" ]] || continue
    run_name=$(basename "$run_dir")

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

    echo ""; echo "$LOG_SEP"
    echo " PROCESS  feature=$feature  lid=$lid  (run=$run_name)"
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

        pfx="${run_out_dir}/${dataset}_${split}_${mode}_key"
        maj_tonic_out="${pfx}_major_tonic.txt"
        min_tonic_out="${pfx}_minor_tonic.txt"
        mmt_out="${pfx}_major_minor_tonic.txt"
        mmj_out="${pfx}_major_minor_joint.txt"
        mmg_tonic_out="${pfx}_major_minor_tonic_gt.txt"
        mmjg_out="${pfx}_major_minor_joint_gt.txt"

        echo ""; echo "  [$dataset / $split]"
        if [[ ! -f "$h5" ]]; then
            echo "  [SKIP] H5 not found: $h5" >&2; continue
        fi
        counts_printed=false

        need_infer=false
        [[ "$force_infer" != "0" ]] && need_infer=true
        for f in "$maj_tonic_out" "$min_tonic_out" "$mmt_out" "$mmj_out" \
                 "$mmg_tonic_out" "$mmjg_out"; do
            [[ ! -f "$f" ]] && need_infer=true && break
        done

        if $need_infer; then
            python -m src.linear_probe.infer_key \
                --h5-path                        "$h5" \
                --split                          "$split" \
                --ckpt-path                      "$ckpt" \
                --major-out-file                 "$maj_tonic_out" \
                --minor-out-file                 "$min_tonic_out" \
                --major-minor-tonic-out-file     "$mmt_out" \
                --major-minor-joint-out-file     "$mmj_out" \
                --major-minor-joint-gt-out-file  "$mmjg_out" \
                --major-minor-tonic-gt-out-file  "$mmg_tonic_out" \
                --batch-size                     $batch_size \
                --device                         $device \
                --allow-missing-labels
        else
            echo "  [CACHED] variant outputs present"
        fi

        for label_tag in "${key_variants[@]}"; do
            case "$label_tag" in
                major_tonic)          key_out="$maj_tonic_out" ;;
                minor_tonic)          key_out="$min_tonic_out" ;;
                major_minor_tonic)    key_out="$mmt_out" ;;
                major_minor_joint)    key_out="$mmj_out" ;;
                major_minor_tonic_gt) key_out="$mmg_tonic_out" ;;
                major_minor_joint_gt) key_out="$mmjg_out" ;;
            esac
            [[ -f "$key_out" ]] || { echo "  [SKIP] $label_tag output missing"; continue; }

            # All variants here are non-forth → maj_degree=0.  Decision per eval:
            #   single rings + *_tonic / *_tonic_gt: align-average ('avg')
            #   major_minor_joint: joint template ('joint')
            #   major_minor_joint_gt: joint template, gt mode ('joint_gt')
            vote2_decision=avg
            case "$label_tag" in
                major_minor_joint)    vote2_decision=joint ;;
                major_minor_joint_gt) vote2_decision=joint_gt ;;
            esac
            m=$(python -m src.analysis.eval_metrics key "$key_out" \
                    --maj-degree 0 --vote2-decision $vote2_decision)
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
                f1="f1_${lvl}"; f2="f2_${lvl}"; f3="f3_${lvl}"; f4="f4_${lvl}"
                if [[ "$single_variants" == *" $label_tag "* ]]; then
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$feature" "$lid" "$mode" "$label_tag" "$dataset" \
                        "${M[${p}_maj]}" "${M[${p}_hit_maj]}" "${M[${p}_acc_maj]}" \
                        "${M[${p}_min]}" "${M[${p}_hit_min]}" "${M[${p}_acc_min]}" \
                        "${M[${p}_total]}" "${M[${p}_acc]}" >> "${!f1}"
                elif [[ "$joint_variants" == *" $label_tag "* ]]; then
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$feature" "$lid" "$mode" "$label_tag" "$dataset" \
                        "${M[${p}_maj]}" "${M[${p}_predmaj_in_maj]}" "${M[${p}_predmaj_ratio]}" \
                        "${M[${p}_min]}" "${M[${p}_predmin_in_min]}" "${M[${p}_predmin_ratio]}" \
                        "${M[${p}_total]}" "${M[${p}_acc]}" >> "${!f2}"
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
    f1="f1_${lvl}"; f2="f2_${lvl}"; f3="f3_${lvl}"; f4="f4_${lvl}"
    echo "── ${lvl} single       : ${!f1}"
    echo "── ${lvl} joint        : ${!f2}"
    echo "── ${lvl} joint strict : ${!f3}"
    echo "── ${lvl} gt           : ${!f4}"
done
echo ""
echo "  probe_group_dir : $probe_group_dir"
echo "  key_out_dir     : $key_out_dir"
