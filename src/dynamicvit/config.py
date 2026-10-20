import os

# ── Paths (set these to your local folders) ────────────────────────────────────
DATA_DIR        = "path/to/data"                 # folder with dataset.csv and DATASET/ (see data/README.md)
REPO_DIR        = "path/to/output"               # checkpoints, metrics and results are written here
DYNAMICVIT_REPO = "path/to/DynamicViT"           # clone of https://github.com/raoyongming/DynamicViT

CSV_FILE        = os.path.join(DATA_DIR, "dataset.csv")
IMG_DIR         = os.path.join(DATA_DIR, "DATASET")
CHECKPOINT_DIR  = os.path.join(REPO_DIR, "checkpoints")
PRETRAINED_PATH = os.path.join(REPO_DIR, "pretrained", "dynamic-vit_lv-m_r0.7.pth")
MODEL_SAVE_PATH = os.path.join(REPO_DIR, "DYNAMIC_VIT_model.pth")
BEST_MODEL_PATH = os.path.join(REPO_DIR, "best_dynamicvit.pth")
METRICS_CSV     = os.path.join(REPO_DIR, "DYNAMICVIT_metrics.csv")

# ── Static baselines (no token pruning) ─────────────────────────────────────────
BASELINES_DIR   = os.path.join(REPO_DIR, "baselines_output")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 5

# ── Dataset ────────────────────────────────────────────────────────────────────
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# test_ratio = 1 - TRAIN_RATIO - VAL_RATIO
NUM_CLASSES  = 20
BATCH_SIZE   = 64
NUM_WORKERS  = 2

# ── ImageNet normalisation ─────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Model architecture ─────────────────────────────────────────────────────────
PATCH_SIZE   = 16
EMBED_DIM    = 512
DEPTH        = 20
NUM_HEADS    = 8
MLP_RATIO    = 3.0
P_EMB        = "4_2"
SKIP_LAM     = 2.0
RETURN_DENSE = True
MIX_TOKEN    = True

BASE_KEEP_RATE = 0.7
PRUNING_LOC    = [5, 10, 15]
TOKEN_RATIO    = [BASE_KEEP_RATE, BASE_KEEP_RATE ** 2, BASE_KEEP_RATE ** 3]

# ── Training ───────────────────────────────────────────────────────────────────
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-4
PATIENCE      = 8

# Knowledge-distillation loss weights
DISTILL_TEMPERATURE = 4
LAMBDA_KL      = 0.5
LAMBDA_DISTILL = 0.5
LAMBDA_RATIO   = 2.0
