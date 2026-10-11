cd /home/coder/wei/music/sae-alignment
source sae-alignment-store/envs/sae-alignment/bin/activate
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1

S=sae-alignment-store/results_latest
CKPT=$S/models/sae/slakh-muq-32-3-1/slakh2100_train_16_muq-2_3-1_1e-3_32_max/ckpts/val_probe_chord_acc-epoch129-step30160-score78.9653.ckpt
FIDS=$S/analysis/feature_ids/slakh-muq-32-3-1/slakh2100_train_16_muq-2_3-1_1e-3_32_max/epoch129_feature_ids.txt

python scripts/infer_sae.py \
    --ckpt $CKPT \
    --feature-ids $FIDS \
    --audio sae-alignment-store/dataset/rwc/AUDIO/004.wav \
    --task key-signature key-chord chord \
    --level song \
    --out-dir sae-alignment-store/outputs/infer_test
