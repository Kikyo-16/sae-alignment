cd /home/coder/wei/music/sae-alignment
source sae-alignment-store/envs/sae-alignment/bin/activate
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1

S=sae-alignment-store/results_latest
CKPT=$S/models/sae/slakh-muq-32-3-1/slakh2100_train_16_muq-2_3-1_1e-3_32_max/ckpts/val_probe_chord_acc-epoch129-step30160-score78.9653.ckpt
ORBITS=sae-alignment-store/outputs/orbits_test/refine/orbits_raw.txt   # from find_orbits.sh
OUT=sae-alignment-store/outputs/plots_test

python scripts/plot_activations.py \
    --ckpt $CKPT \
    --feature-ids $ORBITS --min-orbit-size 10 \
    --audio examples/004.wav \
    --out-dir $OUT
