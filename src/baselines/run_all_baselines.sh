#!/usr/bin/env bash
# Sequential training of the 2 static baselines (Swin-Small, LV-ViT-M), one run
# per seed (5, 16, 25), at low CPU/IO priority (nice 19 / ionice idle) so it
# does not contend with other jobs on a shared machine.
# Activate your Python environment before running this script.
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for MODEL in swin_small lvvit; do
    echo "===== starting $MODEL $(date) ====="
    nice -n 19 ionice -c 3 python train_baseline.py --model "$MODEL" --seeds 5 16 25 \
        2>&1 | tee "$LOG_DIR/${MODEL}.log"
    echo "===== finished $MODEL $(date) ====="
done

echo "All baselines complete."
