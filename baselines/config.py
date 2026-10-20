import os

# ── Paths (set these to your local folders) ────────────────────────────────────
DATASET_DIR     = "path/to/data"                 # folder with dataset.csv and DATASET/
DYNAMICVIT_REPO = "path/to/DynamicViT"           # clone of https://github.com/raoyongming/DynamicViT
OUTPUT_DIR      = "path/to/output"               # checkpoints and results

CSV_FILE        = os.path.join(DATASET_DIR, "dataset.csv")
IMG_DIR         = os.path.join(DATASET_DIR, "DATASET")
PRETRAINED_PATH = os.path.join(OUTPUT_DIR, "pretrained", "dynamic-vit_lv-m_r0.7.pth")
BASELINES_DIR   = os.path.join(OUTPUT_DIR, "baselines_output")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 5   # fixes the train/val/test split, identical for every run and model

# ── Dataset ────────────────────────────────────────────────────────────────────
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# test_ratio = 1 - TRAIN_RATIO - VAL_RATIO
NUM_CLASSES  = 20
BATCH_SIZE   = 64
NUM_WORKERS  = 2

# Training uses no data augmentation, as in the experiments reported in the paper.
# Set to True to apply random crops, flips, rotation, colour jitter and blur to the training set.
USE_AUGMENTATION = False

# ── ImageNet normalisation ─────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── LV-ViT-M architecture (static baseline) ────────────────────────────────────
PATCH_SIZE   = 16
EMBED_DIM    = 512
DEPTH        = 20
NUM_HEADS    = 8
MLP_RATIO    = 3.0
P_EMB        = "4_2"
SKIP_LAM     = 2.0

# ── Training ───────────────────────────────────────────────────────────────────
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-4
PATIENCE      = 8
