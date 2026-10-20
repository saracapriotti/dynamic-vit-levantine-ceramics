#!/usr/bin/env bash
# Sequential training of the 3 static baselines (3 runs each for mean/std).
# Runs at low CPU/IO priority (nice 19 / ionice idle) so it never contends
# with other users' jobs on the shared GPU box -- it only ever uses cycles
# nobody else needs at that instant.
set -euo pipefail

cd "$(dirname "$0")"
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dynamicvit

LOG_DIR="baselines_output/logs"
mkdir -p "$LOG_DIR"

for MODEL in efficientnet_b0 swin_tiny lvvit; do
    echo "===== starting $MODEL $(date) ====="
    nice -n 19 ionice -c 3 python train_baseline.py --model "$MODEL" --n-runs 3 \
        2>&1 | tee "$LOG_DIR/${MODEL}.log"
    echo "===== finished $MODEL $(date) ====="
done

echo "All baselines complete."
