"""
Test-set metrics of the trained baselines for the paper's results table (Table 3):
Accuracy / macro Precision / macro Recall / macro F1 / weighted ROC-AUC,
reported as mean ± sample standard deviation (divide by N-1) over the seeds
(default 5, 16, 25), as for the Dynamic ViT. Each run is evaluated on the test
split of its own seed, i.e. on the same test images as the Dynamic ViT.

Usage:
    python eval_metrics_table.py                       # both baselines (Swin-Small, LV-ViT-M)
    python eval_metrics_table.py --models swin_small
"""
import argparse
import os

import numpy as np
import torch

import config
from dataset import build_dataloaders
from evaluate import compute_metrics
from model_baselines import BUILDERS

KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def eval_checkpoint(model, test_loader, num_classes, device):
    m = compute_metrics(model, test_loader, num_classes, device)
    return dict(
        accuracy=100 * m["accuracy"],
        precision=100 * m["precision"].mean(),
        recall=100 * m["recall"].mean(),
        f1=100 * m["f1"].mean(),
        roc_auc=100 * m["roc_auc"],
    )


def summarize(rows):
    mean = {k: np.mean([r[k] for r in rows]) for k in KEYS}
    std  = {k: np.std([r[k] for r in rows], ddof=1) for k in KEYS}   # sample std (N-1)
    return mean, std


def fmt_latex(name, mean, std):
    cells = " \n".join(f"& {mean[k]:.2f} $\\pm$ {std[k]:.2f}" for k in KEYS)
    return f"\\textbf{{{name}}} \n{cells} \\\\"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(BUILDERS.keys()),
                        choices=list(BUILDERS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--eval-checkpoint", choices=["final", "best"], default=config.EVAL_CHECKPOINT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows_by_model = {name: [] for name in args.models}
    for seed in args.seeds:
        # test split of this seed (same images as the Dynamic ViT for this seed)
        _, _, test_loader, dataset, *_ = build_dataloaders(seed=seed)
        num_classes = dataset.labels.nunique()

        for name in args.models:
            ckpt = os.path.join(config.BASELINES_DIR, name, f"seed{seed}", f"{args.eval_checkpoint}_{name}.pth")
            if not os.path.exists(ckpt):
                print(f"[{name}] seed {seed}: {ckpt} not found, skipped")
                continue
            print(f"\n[{name}] seed {seed}")
            model = BUILDERS[name](num_classes=num_classes, device=device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            rows_by_model[name].append(eval_checkpoint(model, test_loader, num_classes, device))
            del model
            torch.cuda.empty_cache()

    for name, rows in rows_by_model.items():
        if len(rows) < 2:
            print(f"\n{name}: at least 2 runs are needed for mean ± std, found {len(rows)}")
            continue

        mean, std = summarize(rows)
        print(f"\n=== {name}: mean ± std over {len(rows)} runs ({args.eval_checkpoint} checkpoint) ===")
        for k in KEYS:
            print(f"  {k:10s} {mean[k]:.2f} ± {std[k]:.2f}")
        print("\n" + fmt_latex(name, mean, std) + "\n")
