# Dynamic ViT: Levantine Ceramics

Code for *"Evaluation of a Dynamic Vision Transformer for Petrographic Fabric
Classification of Levantine Archaeological Ceramics"*.

Fine-tunes **DynamicViT-LVViT-M/0.7** on a petrographic image
dataset of Levantine ceramics (20 fabric classes), and compares it against
baselines (Swin-Tiny/Small, static LV-ViT-M).

---

## Repository structure

```
dynamic-vit-levantine-ceramics/
├── notebooks/
│   ├── 01_training.ipynb   # Dynamic ViT: training + evaluation (3 experiments)
│   └── 02_xai.ipynb        # Uncertainty, calibration, token-pruning analysis (XAI)
├── src/
│   ├── dynamicvit/          # Dynamic ViT: model, dataset, training, eval, XAI
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── xai.py
│   │   ├── train_dynamicvit_3seeds.py
│   │   └── benchmark_efficiency.py
│   └── baselines/           # baselines: config, dataset, models, training, metrics
│       ├── config.py
│       ├── dataset.py
│       ├── model_baselines.py
│       ├── train_baseline.py
│       ├── evaluate.py
│       ├── eval_metrics_table.py
│       └── run_all_baselines.sh
├── data/                    # dataset goes here (not tracked, see data/README.md)
├── requirements.txt
└── LICENSE
```

The **Dynamic ViT** has two parallel entry points that implement the same model and
training recipe:

- **`src/dynamicvit/`** is the actual development codebase: plain Python modules
  (`config.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `xai.py`), run
  from the command line. This is what produced the 3-seed runs
  (`train_dynamicvit_3seeds.py`) and the efficiency benchmark
  (`benchmark_efficiency.py`, FLOPs/latency/throughput/peak memory vs. the static
  LV-ViT-M counterpart) reported in the paper.
- **`notebooks/`** is a self-contained, Colab-ready reproduction of the same model
  and the *main* experiment and the two generalization tests (seeds 5, 16 and 25),
  with the figures used in the paper. No imports from `src/`.

The **baselines** are a separate small Python package under
`src/baselines/`, also meant to be run from the command line.

---

## Environment setup

### 1. Clone the official DynamicViT repo

Needed for the LV-ViT architecture code, used
both by the notebooks and by the static LV-ViT baseline:

```bash
git clone https://github.com/raoyongming/DynamicViT.git

# compatibility fix for modern PyTorch (torch._six was removed)
sed -i 's/from torch._six import inf/inf = float("inf")/' DynamicViT/utils.py
```

The notebooks do this automatically in their first cell; for the baselines, point
`DYNAMICVIT_REPO` in `src/baselines/config.py` at the clone.

### 2. Install dependencies

```bash
# install PyTorch matching your CUDA version, see https://pytorch.org/get-started
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### 3. Pretrained weights

Backbone: **DynamicViT-LVViT-M/0.7** pretrained on ImageNet (top-1 acc 83.82%).

```bash
mkdir -p pretrained
gdown 1dNab1B5ZOTVNpnpO6H1TsXKFM8BAlA3I -O pretrained/dynamic-vit_lv-m_r0.7.pth
```

### 4. Dataset

See [`data/README.md`](data/README.md) for the expected CSV/image layout and how to
extract and point the code at it. The dataset is not public (available on request).

---

## Usage

### Dynamic ViT: training & evaluation (CLI)

```bash
cd src/dynamicvit
python train.py       # trains the student with KD, saves best checkpoint + metrics CSV
python evaluate.py    # accuracy, confusion matrix, per-class entropy, calibration (ECE)
python xai.py         # token pruning viz, keep-prob heatmap, token survival, influence overlay
```

Paths and hyperparameters are centralised in `config.py` (same defaults as the
baselines: `SEED=5`, 70/15/15 split, 20 classes; plus the Dynamic ViT-specific
pruning/distillation settings, see the [Configuration](#configuration) table below).

To reproduce the paper's mean±std results and the efficiency comparison against
the static LV-ViT-M counterpart:

```bash
python train_dynamicvit_3seeds.py    # 3 runs (seeds 0/1/2), same fixed data split
python benchmark_efficiency.py       # FLOPs, latency, throughput, peak GPU memory
```

### Dynamic ViT: training & evaluation (notebook)

Run **`notebooks/01_training.ipynb`**. It covers three experiments, selected in the
configuration cell via `EXPERIMENT`:

| `EXPERIMENT` | Description | Paper |
|---|---|---|
| `"main"` | 70/15/15 split of the dataset | Table 3, Fig. 2 |
| `"heldout"` | generalisation test, held-out subset | Table 4, Fig. 8 |
| `"sites"` | generalisation test, held-out sites | Table S2, Fig. S21 |

Set `DATA_DIR` to your local (or Google Drive, on Colab) dataset folder before
running. Saves the best/final checkpoint and reports accuracy, macro
precision/recall/F1 and weighted ROC-AUC on the test split.

### Dynamic ViT: XAI / interpretability

Run **`notebooks/02_xai.ipynb`**. Reproduces Figures 3–7 of the paper for the
trained model (main experiment, seed 5), the checkpoint is downloaded
automatically from Google Drive if not already present:

| Section | Figure |
|---|---|
| Predictive entropy vs per-class accuracy | Fig. 3 |
| Calibration (reliability diagram, class-wise ECE) | Fig. 4 |
| Sparsification process for the 20 fabrics | Fig. 5 |
| Token keep-probability heatmaps | Fig. 6 |
| Token exit depth and token influence (Δprob) | Fig. 7 |

### Baselines: training

```bash
cd src/baselines
python train_baseline.py --model swin_small   # or: swin_tiny, lvvit
```

The train/val/test split is fixed by `config.SEED` and identical across models and
runs; only the init/shuffling seed changes between runs of the same model. To train
all static baselines sequentially (3 runs each, for mean/std), at low CPU/IO
priority so it doesn't contend with other jobs on a shared machine:

```bash
./run_all_baselines.sh
```

### Baselines: evaluation

```bash
python eval_metrics_table.py                  # all baselines
python eval_metrics_table.py --models swin_small
```

Reports accuracy / macro precision / macro recall / macro F1 / weighted ROC-AUC,
mean ± sample std over the saved runs, same metric definitions used for the
Dynamic ViT, for the paper's results table (Table 3).

---

## Configuration

Paths and hyperparameters live in `src/dynamicvit/config.py` (Dynamic ViT CLI) and
`src/baselines/config.py` (baselines); the notebooks have the same settings in
their configuration cell. All three share the same dataset split logic
(`SEED = 5`, 70/15/15 train/val/test, 20 classes).

| Parameter | Default | Description |
|---|---|---|
| `SEED` | 5 | reproducibility seed / fixed data split |
| `NUM_CLASSES` | 20 | number of fabric classes |
| `BATCH_SIZE` | 64 | batch size |
| `NUM_EPOCHS` | 25 | maximum epochs |
| `LEARNING_RATE` | 1e-4 | AdamW learning rate |
| `PATIENCE` | 8 | early stopping patience |
| `PRUNING_LOC` | [5, 10, 15] | transformer layers where pruning occurs (Dynamic ViT) |
| `BASE_KEEP_RATE` | 0.7 | base keep-rate (ρ) (Dynamic ViT) |

## Training loss (Dynamic ViT)

Training combines four losses (eq. 16 of [DynamicViT, NeurIPS 2021](https://arxiv.org/abs/2106.02034)):

1. **Classification loss**: CrossEntropy with balanced class weights
2. **KL divergence**: between student and teacher logits (temperature T=4)
3. **Token distillation**: MSE between student and teacher feature tokens
4. **Ratio loss**: MSE between actual and target keep-rate at each pruning stage
