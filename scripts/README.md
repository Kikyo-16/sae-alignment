## Repository layout

```text
  find_orbits.py          orbit recovery + labelling -> feature_ids.txt
  infer_sae.py            chord recognition / key detection on audio with an SAE
  infer_probe.py          the same tasks with a linear-probe baseline
  extract_features.py     MuQ / MusicFM feature extraction to H5
  train_sae.py            o-SAE / s-SAE training (writes config.json)
  train_probe.py          linear-probe training
  analyze/                batch evaluation on H5 datasets used for the paper tables
```