
python -m src.analysis.vis_activation \
    --feature-cell-height 20 \
    --ckpt-path       store/results_latest/models/sae/slakh-muq-32-3-1/slakh2100_train_16_muq-2_3-1_1e-3_32_max/ckpts/val_probe_chord_acc-epoch129-step30160-score78.9653.ckpt \
    --h5-path         sae-data/pop909-test/muq/pop909-test_muq_30s_layer_2.h5       \
    --split           test \
    --feature-id-file store/results_latest/analysis/activation_vis/feature_ids_clean/muq-32-3-1/2-bd.txt \
    --feature-ds      muq_layer \
    --output-dir      store/results_latest/analysis/activation_vis/figs/muq-32-3-1/2/pop909_bd \
    --sae-idx 0 --max-plots 50 --device cuda --allow-missing-labels


#sae-data/pop909-test/muq/pop909-test_muq_30s_layer_2.h5
#sae-data/rwc/muq/rwc-train_muq_30s_layer_2.h5
#sae-data/slakh/muq/slakh_muq_30s_layer_2.h5
#sae-data/gtzan/muq/gtzan_muq_30s_layer_2.h5 
#sae-data/sae_diagnostic_dataset/muq/sae_diagnostic_dataset_muq_30s_layer_2.h5 