# 🫀 MESA ECG SleepNet — 1D ResNet-34 Sleep Stage Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-MESA%20PSG-green" />
  <img src="https://img.shields.io/badge/Task-5--Class%20Sleep%20Staging-blueviolet" />
  <img src="https://img.shields.io/badge/Best%20Acc-84.25%25-brightgreen" />
  <img src="https://img.shields.io/badge/Best%20%CE%BA-0.7671-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  Automatic sleep stage classification from a <strong>single ECG channel</strong> using a custom 1D ResNet-34 built from scratch in PyTorch. Trained and evaluated on the <a href="https://sleepdata.org/datasets/mesa">MESA</a> polysomnography dataset across five experimental variants exploring data augmentation strategies.
</p>

---

## 📑 Table of Contents

- [Overview](#overview)
- [Results Summary](#results-summary)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Experimental Variants](#experimental-variants)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Overview

Traditional polysomnography (PSG) requires simultaneous EEG, EMG, EOG, airflow, and SpO₂ — a clinic-bound, resource-intensive procedure. This project explores whether a **single ECG electrode** — feasible in consumer wearables — is sufficient to automatically classify sleep stages.

Each 30-second ECG window is classified into one of five AASM sleep stages:

| Class ID | Stage | Description |
|:---:|:---:|---|
| 0 | **Wake (W)** | Subject is awake |
| 1 | **N1** | Light sleep, transition from wake |
| 2 | **N2** | Stable light sleep, sleep spindles |
| 3 | **N3** | Deep slow-wave sleep |
| 4 | **REM** | Rapid Eye Movement sleep, dreaming |

**Key design decisions shared across all variants:**
- Single ECG channel only — no EEG, EMG, or other sensors
- 30-second windows at 256 Hz → **7,680 samples** per window
- **1D ResNet-34** adapted for time-series ECG signals
- Class-weighted Cross-Entropy loss for natural class imbalance
- Early stopping (patience = 5)
- **Stratified split BEFORE augmentation** to prevent data leakage

---

## Results Summary

| Version | Strategy | Best Val Acc | Test Acc | Cohen's κ | Verdict |
|:---:|---|:---:|:---:|:---:|:---:|
| **V6** ★ | Manual aug ×10, batch=32 | 83.70% | **84.25%** | **0.7671** | ★ Best |
| V3 | Manual aug ×10, batch=16 | 83.93% | 83.15% | 0.7544 | ✓ Strong |
| V4 | Manual aug ×6.4, batch=16 | 81.35% | 80.41% | 0.7131 | △ Moderate |
| V7 | SMOTE ×10, batch=16 | 81.58% | 78.84% | 0.6896 | ▽ Weak |
| V5 | No augmentation | 48.75%* | 45.92% | 0.2533 | ✗ Baseline |

> \* V5 stopped at epoch 6 via early stopping; severely underfitted due to natural class imbalance.

**Key findings:**
- Larger batch size (V6, batch=32) stabilises gradient estimates and yields the best overall generalisation.
- Aggressive manual augmentation (×10) consistently outperforms reduced augmentation (×6.4) and no augmentation.
- SMOTE-generated ECG segments underperform manual signal-space augmentation — likely because linear feature-space interpolation does not produce physiologically valid ECG waveforms.
- Without augmentation (V5), the model collapses to majority-class prediction and fails to learn N1 and REM.

---

## Architecture

A **1D ResNet-34** adapted from the original image-domain ResNet to process 1D time-series ECG signals.

```
Input: (B, 1, 7680)
  │
  ▼
Stem: Conv1d(1→64, k=15, s=2) + BN + ReLU + MaxPool(s=2)  →  (B, 64, 1920)
  │
  ▼
Layer 1: 3 × BasicBlock1D(64→64,  s=1)  →  (B, 64,  1920)
Layer 2: 4 × BasicBlock1D(64→128, s=2)  →  (B, 128,  960)
Layer 3: 6 × BasicBlock1D(128→256,s=2)  →  (B, 256,  480)
Layer 4: 3 × BasicBlock1D(256→512,s=2)  →  (B, 512,  240)
  │
  ▼
Global Average Pooling  →  (B, 512, 1)
  │
  ▼
Head: Flatten → Linear(512→256) + ReLU + Dropout(0.3) → Linear(256→5)
  │
  ▼
Output: (B, 5) logits
```

**BasicBlock1D — Residual Unit:**
```
out = ReLU(BN(Conv1d(x, k=3))) → Dropout(0.3) → BN(Conv1d(out, k=3))
output = ReLU(out + shortcut(x))
```
Shortcut uses `1×1 Conv1d + BN` when channel dimensions change; otherwise identity.

| Metric | Value |
|---|---|
| Architecture | 1D ResNet-34 |
| Total Parameters | ~21 million |
| Input Shape | (B, 1, 7680) |
| Normalization | BatchNorm1d after every Conv1d |
| Regularization | Dropout(0.3) in classification head |
| Weight Init | Kaiming Normal (He) |

---

## Dataset

This project uses the **[MESA Sleep Study](https://sleepdata.org/datasets/mesa)** (Multi-Ethnic Study of Atherosclerosis) — overnight polysomnography recordings in EDF format with corresponding NSRR XML annotation files.

> **Access required:** MESA is a gated dataset available via [sleepdata.org](https://sleepdata.org) after signing a Data Use Agreement.

**Data loading pipeline (shared across all versions):**

1. **File Discovery** — Scans `EDF_DIR` for `.edf` files; pairs each with its `-nsrr.xml` annotation.
2. **ECG Extraction** — Searches channel labels for `ECG`, `EKG`, or `CARDIAC`; records original sampling frequency.
3. **Bandpass Filter** — 4th-order zero-phase Butterworth filter, `[0.5 Hz – 40.0 Hz]`, using `filtfilt()` to avoid phase distortion.
4. **Resampling** — `scipy.signal.resample()` to 256 Hz for uniform 7,680-sample windows.
5. **Annotation Parsing** — NSRR XML `EventConcept` field mapped to integer labels `0–4`.
6. **Window Extraction & Z-Score Normalization** — Per-window: `(x − μ) / σ` (only when σ > 1e-6).

**Stratified 80 / 10 / 10 split — BEFORE augmentation:**
```
All data  →  StratifiedShuffleSplit(test_size=0.10)  →  90% trainval + 10% test
trainval  →  StratifiedShuffleSplit(test_size=0.111) →  80% train   + 10% val
```

---

## Experimental Variants

Five notebooks explore different data balancing strategies. All share the same architecture, optimizer, and training loop — only Cell 5 (augmentation/balancing) and `BATCH_SIZE` differ.

### Manual Augmentation Primitives (V3, V4, V5, V6)

| Transform | Parameters | Probability |
|---|---|:---:|
| Gaussian Noise | std ∈ [0.005, 0.03] | Always |
| Amplitude Scaling | factor ∈ [0.7, 1.3] | 60% |
| Time Shift (Roll) | shift ∈ [−2×fs, +2×fs] | 60% |
| Baseline Wander | freq ∈ [0.05, 0.5] Hz | 50% |
| Re-normalization | Z-score per window | Always |

### Version Comparison

| Aspect | V3 | V4 | V5 | V6 ★ | V7 |
|---|:---:|:---:|:---:|:---:|:---:|
| Train function | `augment_split()` | `augment_split()` | `split()` | `augment_split()` | `smote_split()` |
| Oversample factor | 10 | 6.4 | N/A | 10 | 10 |
| `BATCH_SIZE` | 16 | 16 | 16 | **32** | 16 |
| Synthetic method | Signal transforms | Signal transforms | None | Signal transforms | KNN interpolation |
| Train class balance | Equal | Equal | Natural | Equal | Equal (majority not reduced) |

### SMOTE (V7) vs Manual Augmentation

| Aspect | Manual Aug (V3/V4/V6) | SMOTE (V7) |
|---|---|---|
| Generation method | Random signal transforms | Linear interpolation in feature space |
| Physiological validity | Domain-inspired (noise, drift) | Interpolated — may not be valid ECG |
| Downsampling | Majority classes capped to target | Only oversamples; majority unchanged |
| Library | `numpy` / `scipy` | `imblearn.over_sampling.SMOTE` |

---

## Pipeline Overview

| Cell | Purpose | Shared? | Key Output |
|:---:|---|:---:|---|
| 1 | Environment verification | All | GPU/CPU status, import checks |
| 2 | Imports & CFG configuration | All (params vary) | `CFG` dict, `LABEL_MAP`, `DEVICE` |
| 3 | MESA data loading & preprocessing | All | `class_samples[0..4]`, 7,680-sample windows |
| 4 | Stratified 80/10/10 split | All | `split_samples{train/val/test}[0..4]` |
| 5 | Augmentation & balancing | **Differs per version** | `X_train`, `y_train`, `X_val`, `X_test` |
| 6 | PyTorch Dataset + DataLoaders | All (batch differs) | `train/val/test_loader` |
| 7 | 1D ResNet-34 definition | All | `model` (~21M parameters) |
| 8 | (Optional) Pretrained weights | All | Transfer learning init |
| 9 | Training loop | All | `best_model.pt` + history |
| 10 | Training curves | All | `training_curves.png` |
| 11 | Test evaluation | All | Report + `confusion_matrix.png` |
| 12 | Model persistence | All | Full checkpoint `.pt` file |

**Training configuration:**

| Component | Setting |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR Schedule | CosineAnnealingLR (T_max=30, η_min=2e-5) |
| Loss | Class-weighted CrossEntropyLoss |
| Gradient Clipping | `clip_grad_norm_(max_norm=1.0)` |
| Early Stopping | patience=5 epochs |
| Max Epochs | 30 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/MESA-ECG-SleepNet.git
cd MESA-ECG-SleepNet

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`:**
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
imbalanced-learn>=0.11.0   # V7 only
pyEDFlib>=0.1.28
lxml>=4.9.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## Usage

1. **Set your dataset path** in `CFG` (Cell 2 of any notebook):
```python
CFG = {
    'EDF_DIR':  '/path/to/mesa/edf/',
    'XML_DIR':  '/path/to/mesa/xml/',
    'BATCH_SIZE': 16,          # 32 for V6
    'LR': 1e-3,
    'EPOCHS': 30,
    'DROPOUT': 0.3,
    'patience': 5,
    'TRAIN_OVERSAMPLE_FACTOR': 10,
    'TARGET_FS': 256,
    'EPOCH_SEC': 30,
}
```

2. **Run the recommended version (V6):**
```bash
jupyter notebook MESA_ECG_SleepNet_LOCAL_\(6\).ipynb
```

3. **Run all cells sequentially** (Cells 1–12). The best model checkpoint is saved to `best_model.pt` automatically.

4. **Evaluate** — Cell 11 loads `best_model.pt` and prints the full classification report, Cohen's κ, and saves the confusion matrix plot.

---

## Project Structure

```
MESA-ECG-SleepNet/
│
├── MESA_ECG_SleepNet_LOCAL_(3).ipynb   # V3 — Manual aug ×10, batch=16  (Strong)
├── MESA_ECG_SleepNet_LOCAL_(4).ipynb   # V4 — Manual aug ×6.4, batch=16 (Moderate)
├── MESA_ECG_SleepNet_LOCAL_(5).ipynb   # V5 — No augmentation            (Baseline)
├── MESA_ECG_SleepNet_LOCAL_(6).ipynb   # V6 — Manual aug ×10, batch=32   ★ Best
├── MESA_ECG_SleepNet_LOCAL_(7).ipynb   # V7 — SMOTE ×10, batch=16        (Weak)
│
├── best_model.pt                        # Best checkpoint (auto-saved during training)
├── training_curves.png                  # Loss & accuracy plots (auto-generated)
├── confusion_matrix.png                 # Normalized 5×5 confusion matrix
│
├── requirements.txt
└── README.md
```

---

## Limitations

- **N1 ambiguity** — ECG changes at Wake/N1/N2 boundaries are physiologically subtle; all ECG-only models struggle with this class.
- **No EEG-specific markers** — Sleep spindles, K-complexes, and delta waves are not visible in a single ECG channel.
- **SMOTE on raw ECG** — Linear interpolation in 7,680-dimensional sample space does not guarantee physiologically valid waveforms.
- **Arrhythmia sensitivity** — Performance may degrade for subjects with cardiac arrhythmias underrepresented in the MESA cohort.
- **No patient-aware split** — Current splitting uses `StratifiedShuffleSplit` on windows; a patient-level `GroupShuffleSplit` would provide a stricter leakage-free evaluation. This is a known limitation and a planned improvement.

---

## Citation

If you use this work, please cite the MESA dataset:

```bibtex
@article{chen2015multi,
  title   = {The Multi-Ethnic Study of Atherosclerosis and its relevance to sleep apnea research},
  author  = {Chen, X and Wang, R and Zee, P and others},
  journal = {Journal of Clinical Sleep Medicine},
  year    = {2015}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for open biomedical AI research · Single-channel ECG · AASM 5-Class Sleep Staging
</p>
