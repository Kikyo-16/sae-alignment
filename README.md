# From Isolated Feature to Orbits: Discovering Music Concepts via Multi-SAE Alignment

**Liwei Lin** (liwei.lin@nyu.edu) · **Gus Xia** (Gus.Xia@mbzuai.ac.ae)

 [Paper](https://arxiv.org/abs/2610.01864) (NeurIPS 2026 Poster) · [Demo page](https://Kikyo-16.github.io/sae-alignment-demo-page) · [Model weights (Google Drive)](https://drive.google.com/drive/folders/1_oqe5miPLmdfy4nk-_1qnO3LzNlk9LlS?usp=sharing)  

***Under construction*** Liwei authored the initial codebase, which was then refactored and organized using Claude Code.

---

## Contents

- [Installation](#installation)
- [Model weights](#model-weights)
- [Quick start: inference](#quick-start-inference-on-audio)
- [1. Recovering orbits from a checkpoint](#1-recovering-orbits-from-a-checkpoint)
- [2. Chord recognition and key detection](#2-chord-recognition-and-key-detection)
- [Training an SAE](#3-training-an-sae)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/Kikyo-16/sae-alignment.git
cd sae-alignment
pip install -r requirements.txt
```

The code was developed with Python 3.12 and PyTorch with CUDA. MuQ is loaded with the official `muq` package and MusicFM through `transformers` (`trust_remote_code`). Run all commands from the repository root.

## Model weights

Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1_oqe5miPLmdfy4nk-_1qnO3LzNlk9LlS?usp=sharing):

- **o-SAE checkpoints** (`.ckpt`). Example model under a setting of **MuQ layer 2** at **K = 32** is avalable here [ckpt](models/sae/slakh-muq-32-3-1/slakh2100_train_16_muq-2_3-1_1e-3_32_max/ckpts/val_probe_chord_acc-epoch129-step30160-score78.9653.ckpt) and [feature_ids.txt](...).
- [**Linear-probe checkpoints**](...) (optional). They are the paper's baselines and the reference for SAE checkpoint selection (`--chord-probe-ckpt`). They are also used to retrieve s-SAE features.

## Quick start: inference

```bash
python scripts/infer_sae.py \
    --ckpt <o-sae.ckpt> \
    --feature-ids <feature_ids.txt> \
    --audio song.mp3 \
    --task key-signature key-chord chord \
    --level song \
    --out-dir outputs
```

The foundation model (MuQ / MusicFM) and its layer are read from the checkpoint, and the foundation model is downloaded automatically. `feature_ids.txt` lists the labelled orbits of the checkpoint; download it from [google drive](https://drive.google.com/drive/folders/1wyQnRppR8sXX9QkemAoU6LPs17BXhtsy?usp=sharing) or create it with step 1.

### 1. Recovering orbits from a checkpoint

```bash
python scripts/find_orbits.py --ckpt <o-sae.ckpt> --out-dir results/muq-2 [--threshold 0.5] [--refine]
```

Orbit recovery reads only the decoder dictionaries, so it needs no data or labels.

- By default it uses the successor-map recovery of Sec. 3.3: three sub-SAEs, the middle SAE as reference, bidirectional confidence, and threshold `--threshold` (τ, default 0.5).
- `--refine` switches to the sparse directed-graph recovery of App. G.2. It also keeps near-cyclic orbits that the strict one-edge rule misses.

Output files in `--out-dir`:

| File | Content |
|---|---|
| `orbits_raw.txt` | All recovered structures, one per line: `[Ring 00]: 12 a -> b -> ...` for rings, `[Seq 00]: n a -> ...` for open sequences. With `--refine`, the tags are `Ring12` / `Seq12` / `Chain` / `GreedyRing12` and carry edge-score metadata. |
| `orbits_raw_score.txt` | `--refine` only: the edge scores of every structure in `orbits_raw.txt`. |
| `timbre.txt` | Fixed points: features whose successor is themselves, i.e. candidate pitch-invariant concepts. |
| `invalid.txt` | Features without an accepted successor (default recovery only). |

#### Interpreting orbits

Plot the activations of the recovered orbits on a few songs whose chords or key you know:

```bash
python scripts/plot_activations.py \
    --ckpt <o-sae.ckpt> \
    --feature-ids results/muq-2/orbits_raw.txt --min-orbit-size 10 \
    --audio song.mp3 --out-dir plots
```

This writes one heatmap per 30-second segment to `plots/<stem>/`. Time runs along the x-axis, the y-axis lists feature ids, and red lines separate orbits. `--feature-ids` accepts `orbits_raw.txt` or any file with one feature-id group per line.

From the plots, pick the orbits you need and write them into `feature_ids.txt`. Order each 12-feature ring so that position *p* is pitch class *p* (C, C#, …, B). Because a ring is cyclic, this only means rotating it so that it starts at the C feature.

```text
[Major Chord]: 12 f_C -> f_C# -> ... -> f_B      # position p = major chord with root p
[Minor Chord]: 12 ...                             # position p = minor chord with root p
[Forth]: 12 ...                                   # subdominant ring: position p = subdominant pitch class p
[Chord Boundary]: 1 b                             # feature that peaks at chord changes
[Silence]: s1 s2                                  # features active on silence
```

Only the groups that a task uses are required (see the table in step 2).

### 2. Chord recognition and key detection

```bash
python scripts/infer_sae.py --ckpt <ckpt> --feature-ids <feature_ids.txt> \
    --audio <files or folders> --task <tasks> [options]
```

| `--task` | Description | Orbits used |
|---|---|---|
| `key-signature` | Key detection that ignores the relative major/minor difference (Table 2, App. D.2). The output is a key signature such as `G:maj/E:min`. | `[Forth]` (or `--key-ring fifth/tonic`) |
| `key-chord` | Key detection over 24 major/minor keys from chord features (Table 3, App. D.3). | `[Major Chord]`, `[Minor Chord]` |
| `chord` | Chord recognition over 12 major + 12 minor chords + `N` (Table 1, App. D.1). | chord rings, `[Chord Boundary]`, `[Silence]` |

**Segmentation.** Audio is resampled to 16 kHz and cut into non-overlapping 30-second segments, exactly as in the paper. `--last-segment` controls what happens to the remainder shorter than 30 s:

- `drop` (default): discard it, as in the paper.
- `overlap`: run one more 30 s window that ends at the last sample, and keep only the results of its non-overlapping part.

Audio shorter than 30 s is skipped.

**Key level.** `--level segment` writes one key per 30 s segment. `--level song` (default) writes one key per file, aggregated as in the paper's song-level evaluation: the 12-d key-signature scores are summed over segments (`key-signature`), and the 24 template scores are averaged over segments (`key-chord`).

**Chord boundaries.** By default boundaries are estimated from the `[Chord Boundary]` feature (Est. Bd). `--chord-lab` gives reference boundaries instead (Oracle Bd). It takes a `start end label` file for a single audio, or a folder of `<audio stem>.lab` files.


Outputs in `--out-dir`, per audio file:

```text
<stem>.key-signature.<level>.tsv   start_sec  end_sec  key signature (e.g. G:maj/E:min)
<stem>.key-chord.<level>.tsv       start_sec  end_sec  key (e.g. E:min)
<stem>.chord.lab                   start_sec  end_sec  chord (e.g. C:maj, A:min, N)
```

## Training an SAE

### 1. Data

The SAEs are trained on a key-balanced subset of Slakh2100: 326 training and 164 validation tracks (App. B.1). The two subsets are given as track-id lists, with one Slakh2100 track id per line (`splits/slakh_train.txt`, `splits/slakh_val.txt`).

### 2. Feature extraction

```bash
python scripts/extract_features.py --purpose train \
    --model muq --layer 2 \
    --dataset slakh --root dataset/slakh2100_flac_redux --label-root dataset/slakh2100_flac_redux \
    --split train --track-id-files splits/slakh_train.txt splits/slakh_val.txt \
    --out-h5 sae-data/slakh2100_train_16/muq/slakh2100_train_16_muq_30s_layer_2.h5
```

`--purpose train` cuts 30 s segments with a 10 s hop, removes silent segments, and stores five pitch-shifted views at {−2, −1, 0, +1, +2} semitones (phase vocoder, App. B.1). `--purpose eval` stores non-overlapping 30 s segments without augmentation. Use it for evaluation sets (`--dataset pop909 | rwc | slakh | gtzan | giantsteps_key | fmakv2`). Use `--model musicfm` for MusicFM.

### 3. Training

```bash
python scripts/train_sae.py \
    --model muq --layer 2 --topk 32 \
    --h5 sae-data/slakh2100_train_16/muq/slakh2100_train_16_muq_30s_layer_2.h5 \
    --train-track-ids splits/slakh_train.txt --val-track-ids splits/slakh_val.txt \
    [--chord-probe-ckpt <chord probe .ckpt>] [--sae-type s-sae] [--seed 42]
```

The script writes `runs/sae/<run-name>/config.json` and starts `python -m src.music_sae.train --config <config.json>`. The config holds:

- `exp_config`: the model settings. These are 3 sub-SAEs with a shared pre-encoder bias, Top-K support shared across pitch-shifted views, expansion factor 4, and decoder columns renormalized after every step.
- `train`: the trainer settings. The defaults are the paper's: batch size 32, learning rate 1e-3, 400 epochs, seed 42.

To rerun an experiment, run `python -m src.music_sae.train --config <config.json>`. `--sae-type s-sae` trains the standard single-SAE ablation.

Checkpoints are written to `runs/sae/<run-name>/ckpts/` and are ranked by a chord-recognition validation score every 10 epochs (App. F). With `--chord-probe-ckpt`, the SAE features closest to the chord-probe weights are scored. This is the setting used in the paper, and the probe checkpoints are part of the released weights. Without it, chord rings are enumerated from the SAE itself. The probe only ranks checkpoints; it never provides gradients to the SAE. For an s-SAE, pass `--chord-probe-ckpt`.

After training, recover and interpret the orbits of the chosen checkpoint with [step 1](#1-recovering-orbits-from-a-checkpoint) and run inference with [step 2](#2-chord-recognition-and-key-detection).

The linear-probe baselines can be trained and applied with `scripts/train_probe.py` and `scripts/infer_probe.py` (see `--help`).


## Citation

```bibtex
@misc{lin2026isolatedfeatureorbitsdiscovering,
      title={From Isolated Feature to Orbits: Discovering Music Concepts via Multi-SAE Alignment}, 
      author={Liwei Lin and Gus Xia},
      year={2026},
      eprint={2610.01864},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2610.01864}, 
}
```
