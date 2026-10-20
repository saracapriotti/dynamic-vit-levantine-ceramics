"""
Runs DynamicViT training 3x with seeds 0/1/2 (same data split every time,
fixed by config.SEED, matching the train_baseline.py protocol used for the
other baselines) so the paper's table can report mean +/- std instead of a
single run. Writes to baselines_output/dynamicvit/run{i}/ -- does NOT touch
the existing best_dynamicvit.pth / checkpoints/ / DYNAMICVIT_metrics.csv at
the repo root, those stay as they are.
"""
import csv
import os

import numpy as np
import torch

import config
from dataset import build_dataloaders, set_seed
from evaluate import compute_metrics
from model import build_model, build_teacher
from train import train, get_class_weights

OUT_DIR = os.path.join(config.BASELINES_DIR, "dynamicvit")
N_RUNS  = 3


def eval_checkpoint(model, test_loader, num_classes, device):
    m = compute_metrics(model, test_loader, num_classes, device)
    return dict(
        accuracy=100 * m["accuracy"],
        precision=100 * m["precision"].mean(),
        recall=100 * m["recall"].mean(),
        f1=100 * m["f1"].mean(),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, dataset, *_ = build_dataloaders()
    num_classes   = len(dataset.labels.unique())
    class_weights = get_class_weights(dataset, device)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["run", "seed", "best_val_acc", "test_acc", "test_precision", "test_recall", "test_f1"]
        )

    rows = []
    for run_idx in range(N_RUNS):
        seed = run_idx
        print(f"\n===== [dynamicvit] run {run_idx + 1}/{N_RUNS}  (seed={seed}) =====")

        set_seed(seed)  # reseeds model/teacher init + dropout; data split stays fixed
        model   = build_model(num_classes=num_classes, device=device)
        teacher = build_teacher(num_classes=num_classes, device=device)

        run_dir     = os.path.join(OUT_DIR, f"run{run_idx}")
        ckpt_dir    = os.path.join(run_dir, "checkpoints")
        best_path   = os.path.join(run_dir, "best_dynamicvit.pth")
        metrics_csv = os.path.join(run_dir, "metrics.csv")
        os.makedirs(run_dir, exist_ok=True)

        model = train(
            model, teacher, train_loader, val_loader, class_weights, device,
            checkpoint_dir=ckpt_dir, best_model_path=best_path, metrics_csv=metrics_csv,
        )

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        row = eval_checkpoint(model, test_loader, num_classes, device)

        with open(metrics_csv) as f:
            recorded = list(csv.DictReader(f))
        best_val_acc = max(float(r["val_acc"]) for r in recorded)

        print(f"[dynamicvit] run {run_idx + 1} -> best_val_acc={best_val_acc:.2f}%  "
              f"test_acc={row['accuracy']:.2f}%  precision={row['precision']:.2f}  "
              f"recall={row['recall']:.2f}  f1={row['f1']:.2f}")

        with open(summary_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [run_idx, seed, best_val_acc, row["accuracy"], row["precision"], row["recall"], row["f1"]]
            )

        rows.append(row)
        del model, teacher
        torch.cuda.empty_cache()

    keys = ["accuracy", "precision", "recall", "f1"]
    mean = {k: np.mean([r[k] for r in rows]) for k in keys}
    std  = {k: np.std([r[k] for r in rows]) for k in keys}

    with open(summary_csv, "a", newline="") as f:
        csv.writer(f).writerow(["mean", "", "", mean["accuracy"], mean["precision"], mean["recall"], mean["f1"]])
        csv.writer(f).writerow(["std",  "", "", std["accuracy"],  std["precision"],  std["recall"],  std["f1"]])

    print(f"\n[dynamicvit] {N_RUNS} runs -> "
          f"acc={mean['accuracy']:.2f}±{std['accuracy']:.2f}  "
          f"prec={mean['precision']:.2f}±{std['precision']:.2f}  "
          f"rec={mean['recall']:.2f}±{std['recall']:.2f}  "
          f"f1={mean['f1']:.2f}±{std['f1']:.2f}")
    print(f"Summary written to {summary_csv}")
