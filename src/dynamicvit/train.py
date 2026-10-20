import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config
from dataset import build_dataloaders
from model import build_model, build_teacher


def compute_ratio_loss(decision_masks, target_ratios: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for mask, target in zip(decision_masks, target_ratios):
        actual_ratio = mask.mean(dim=1)
        loss += F.mse_loss(actual_ratio, torch.full_like(actual_ratio, target))
    return loss / len(decision_masks)


def get_class_weights(dataset, device: torch.device) -> torch.Tensor:
    all_labels = dataset.labels.values
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels,
    )
    return torch.tensor(weights, dtype=torch.float).to(device)


def train(
    model,
    teacher,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    best_model_path: str = config.BEST_MODEL_PATH,
    metrics_csv: str = config.METRICS_CSV,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion_cls    = nn.CrossEntropyLoss(weight=class_weights)
    criterion_kl     = nn.KLDivLoss(reduction="batchmean")
    criterion_mse    = nn.MSELoss()
    target_ratios    = torch.tensor(config.TOKEN_RATIO, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    scaler    = GradScaler()

    T = config.DISTILL_TEMPERATURE

    best_val_acc     = 0.0
    epochs_no_improve = 0

    # initialise metrics CSV
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        model.distill = True

        total_loss = correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                student_cls, student_aux, prev_decision, decision_masks = model(images)

                with torch.no_grad():
                    teacher_cls, teacher_aux = teacher(images)

                loss_cls = criterion_cls(student_cls, labels)

                loss_kl = criterion_kl(
                    F.log_softmax(student_cls / T, dim=1),
                    F.softmax(teacher_cls / T, dim=1),
                ) * (T * T)

                token_mask    = prev_decision.squeeze(-1).bool()
                loss_distill  = criterion_mse(student_aux[token_mask], teacher_aux[token_mask])

                loss_ratio = compute_ratio_loss(decision_masks, target_ratios)

                loss = (
                    loss_cls
                    + config.LAMBDA_KL      * loss_kl
                    + config.LAMBDA_DISTILL * loss_distill
                    + config.LAMBDA_RATIO   * loss_ratio
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = student_cls.max(1)
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
                logits  = outputs[0] if isinstance(outputs, tuple) else outputs
                val_loss_total += criterion_cls(logits, labels).item()
                _, predicted = logits.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = 100.0 * val_correct / val_total
        val_loss = val_loss_total / len(val_loader)

        print(f"\nEpoch {epoch+1}  |  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%"
              f"  |  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        scheduler.step()

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print("  Saved best model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print("Early stopping triggered")
            break

    print(f"\nTraining complete — best val acc: {best_val_acc:.2f}%")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, dataset, *_ = build_dataloaders()

    num_classes   = len(dataset.labels.unique())
    class_weights = get_class_weights(dataset, device)

    model   = build_model(num_classes=num_classes, device=device)
    teacher = build_teacher(num_classes=num_classes, device=device)

    train(model, teacher, train_loader, val_loader, class_weights, device)
