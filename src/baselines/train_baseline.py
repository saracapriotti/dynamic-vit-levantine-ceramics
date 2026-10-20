"""
Train a static baseline (Swin-Small or LV-ViT-M without token pruning) for
several runs. The data split is fixed by config.SEED and identical for every
run; only the initialisation / shuffling seed changes between runs.

Usage:
    python train_baseline.py --model swin_small
    python train_baseline.py --model lvvit
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config
from dataset import build_dataloaders, set_seed
from model_baselines import BUILDERS


def get_class_weights(dataset, device: torch.device) -> torch.Tensor:
    """Balanced class weights (same as for the Dynamic ViT)."""
    labels = dataset.labels.values
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)


@torch.no_grad()
def evaluate_test(model, test_loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in test_loader:
        if images.size(0) == 0:
            continue
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def train_baseline(
    model,
    model_name: str,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    num_epochs: int,
    patience: int,
    lr: float,
    out_dir: str,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.0,
):
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f"best_{model_name}.pth")
    metrics_csv     = os.path.join(out_dir, "metrics.csv")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = GradScaler()

    best_val_acc      = 0.0
    epochs_no_improve = 0

    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(num_epochs):
        model.train()
        total_loss = correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs}"):
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

        train_acc  = 100.0 * correct / total
        train_loss = total_loss / len(train_loader)

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                if images.size(0) == 0:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss_total += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = 100.0 * val_correct / val_total
        val_loss = val_loss_total / len(val_loader)

        print(f"\n[{model_name}] Epoch {epoch+1}  |  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%"
              f"  |  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        scheduler.step()

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model -> {best_model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    print(f"\n[{model_name}] Training complete — best val acc: {best_val_acc:.2f}%")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a static (non-dynamic) baseline on the ceramics dataset")
    parser.add_argument("--model", required=True, choices=list(BUILDERS.keys()))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--n-runs", type=int, default=3, help="number of repeated runs for mean/std")
    parser.add_argument("--seed-base", type=int, default=0, help="run i uses seed seed-base+i for init/shuffle")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data split built ONCE with config.SEED -> identical train/val/test across
    # every run and every model. Only the initialisation / shuffling seed varies per run.
    config.BATCH_SIZE = args.batch_size
    train_loader, val_loader, test_loader, dataset, *_ = build_dataloaders()

    num_classes   = dataset.labels.nunique()
    class_weights = get_class_weights(dataset, device)

    model_out_dir = os.path.join(config.BASELINES_DIR, args.model)
    os.makedirs(model_out_dir, exist_ok=True)
    summary_csv = os.path.join(model_out_dir, "summary.csv")

    val_accs, test_accs = [], []

    with open(summary_csv, "w", newline="") as f:
        csv.writer(f).writerow(["run", "seed", "best_val_acc", "test_acc"])

    for run_idx in range(args.n_runs):
        run_seed = args.seed_base + run_idx
        print(f"\n===== [{args.model}] run {run_idx+1}/{args.n_runs}  (seed={run_seed}) =====")

        set_seed(run_seed)  # reseeds init / dropout / shuffle order only
        model = BUILDERS[args.model](num_classes=num_classes, device=device)

        run_dir = os.path.join(model_out_dir, f"run{run_idx}")
        model = train_baseline(
            model, args.model, train_loader, val_loader, class_weights, device,
            num_epochs=args.epochs, patience=args.patience, lr=args.lr, out_dir=run_dir,
            weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
        )

        best_ckpt = os.path.join(run_dir, f"best_{args.model}.pth")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        test_acc = evaluate_test(model, test_loader, device)

        with open(os.path.join(run_dir, "metrics.csv")) as f:
            rows = list(csv.DictReader(f))
        best_val_acc = max(float(r["val_acc"]) for r in rows)

        print(f"[{args.model}] run {run_idx+1} -> best_val_acc={best_val_acc:.2f}%  test_acc={test_acc:.2f}%")

        val_accs.append(best_val_acc)
        test_accs.append(test_acc)

        with open(summary_csv, "a", newline="") as f:
            csv.writer(f).writerow([run_idx, run_seed, best_val_acc, test_acc])

        del model
        torch.cuda.empty_cache()

    # Sample standard deviation (divide by N-1), as for the Dynamic ViT
    val_mean, val_std   = np.mean(val_accs),  np.std(val_accs, ddof=1)
    test_mean, test_std = np.mean(test_accs), np.std(test_accs, ddof=1)

    with open(summary_csv, "a", newline="") as f:
        csv.writer(f).writerow(["mean", "", val_mean, test_mean])
        csv.writer(f).writerow(["std",  "", val_std,  test_std])

    print(f"\n[{args.model}] {args.n_runs} runs -> "
          f"val_acc = {val_mean:.2f}% ± {val_std:.2f}  |  test_acc = {test_mean:.2f}% ± {test_std:.2f}")
    print(f"Summary written to {summary_csv}")
    print("Full test metrics (Table 3): python eval_metrics_table.py")
