## Repository layout

```text
  music_sae/              multi-SAE model, shared Top-K training, validation metrics
  orbits_recovery/        successor-map orbit recovery (Sec. 3.3), threshold study
  analysis/               refined recovery, orbit labelling, boundary/silence features,
                          step-4 inference, metrics, visualization
  inference/              audio segmentation and task decoders for scripts/infer_*.py
  feature_extraction/     dataset readers, labels, MuQ / MusicFM feature extraction
  linear_probe/           probe baselines
  evaluation/             chord / key metrics, LVCR baseline evaluation
  data_preprocess/        Slakh2100 key-balanced selection, synthetic diagnostic set
```