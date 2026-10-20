"""
Evaluation utilities: accuracy, metrics, confusion matrix, entropy, calibration.
"""
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import config


# ── Helpers ────────────────────────────────────────────────────────────────────

def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


# ── Basic accuracy ─────────────────────────────────────────────────────────────

def test_accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits  = outputs[0] if isinstance(outputs, tuple) else outputs
            _, predicted = logits.max(1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    acc = 100.0 * correct / total
    print(f"Test accuracy: {acc:.2f}%")
    return acc


# ── Per-class metrics ──────────────────────────────────────────────────────────

def compute_metrics(model, loader, num_classes: int, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits  = outputs[0] if isinstance(outputs, tuple) else outputs
            _, preds = logits.max(1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).cpu()
    all_labels = torch.cat(all_labels).cpu()

    cm = torch.zeros(num_classes, num_classes)
    for t, p in zip(all_labels, all_preds):
        cm[t.long(), p.long()] += 1

    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    precision = (tp / (tp + fp + 1e-8)).numpy()
    recall    = (tp / (tp + fn + 1e-8)).numpy()
    f1        = (2 * precision * recall / (precision + recall + 1e-8))
    accuracy  = (tp.sum() / cm.sum()).item()

    print(f"Overall accuracy: {100*accuracy:.2f}%")
    print(f"Mean precision:   {precision.mean():.4f}")
    print(f"Mean recall:      {recall.mean():.4f}")
    print(f"Mean F1:          {f1.mean():.4f}")

    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, cm=cm.numpy())


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    loader,
    num_classes: int,
    device: torch.device,
    class_names=None,
    save_path: str | None = None,
):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits  = outputs[0] if isinstance(outputs, tuple) else outputs
            _, preds = logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    if class_names is None:
        class_names = [f"Class {i+1}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return cm


def show_most_confused_images(
    model,
    loader,
    device: torch.device,
    class_names,
    cm: np.ndarray,
    num_images: int = 6,
    true_label: int | None = None,
    pred_label: int | None = None,
):
    model.eval()

    if true_label is None or pred_label is None:
        mask = np.ones_like(cm, dtype=bool)
        np.fill_diagonal(mask, False)
        idx = np.unravel_index(cm[mask].argmax(), cm.shape)
        true_label, pred_label = int(idx[0]), int(idx[1])

    print(f"Showing: true={class_names[true_label]} → pred={class_names[pred_label]}")

    images_shown = []
    with torch.no_grad():
        for images, labels in loader:
            if images.size(0) == 0:
                continue
            images_dev = images.to(device)
            labels_dev = labels.to(device)
            outputs = model(images_dev)
            logits  = outputs[0] if isinstance(outputs, tuple) else outputs
            _, preds = logits.max(1)

            for i in range(len(labels)):
                if (labels[i].item() == true_label and
                        preds[i].item() == pred_label):
                    images_shown.append(images[i])
                    if len(images_shown) >= num_images:
                        break
            if len(images_shown) >= num_images:
                break

    fig, axes = plt.subplots(1, len(images_shown), figsize=(4 * len(images_shown), 4))
    if len(images_shown) == 1:
        axes = [axes]
    for ax, img in zip(axes, images_shown):
        img_np = denormalize(img).permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.axis("off")
    fig.suptitle(f"True: {class_names[true_label]}  →  Pred: {class_names[pred_label]}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── Entropy per class ──────────────────────────────────────────────────────────

def compute_entropy_per_class(
    model,
    loader,
    device: torch.device,
    num_classes: int = config.NUM_CLASSES,
    save_path: str | None = None,
):
    model.eval()
    class_entropy  = defaultdict(list)
    class_counts   = defaultdict(int)

    with torch.no_grad():
        for images, labels in loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits  = outputs[0] if isinstance(outputs, tuple) else outputs

            log_probs = torch.log_softmax(logits, dim=1)
            probs     = log_probs.exp()
            entropy   = -torch.sum(probs * log_probs, dim=1) / math.log(num_classes)

            for e, lab in zip(entropy, labels):
                cls = int(lab.item())
                class_entropy[cls].append(e.item())
                class_counts[cls] += 1

    classes         = np.arange(1, num_classes + 1)
    entropy_values  = [class_entropy[i] for i in range(num_classes)]
    entropy_means   = [np.mean(v) if v else 0 for v in entropy_values]
    entropy_std     = [np.std(v)  if v else 0 for v in entropy_values]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(classes, entropy_means, color="steelblue", alpha=0.75,
                   edgecolor="black", linewidth=0.6)
    plt.errorbar(classes, entropy_means, yerr=entropy_std, fmt="none",
                 ecolor="black", capsize=3)
    plt.xlabel("Class", fontsize=13)
    plt.ylabel("Normalised entropy", fontsize=13)
    plt.title("Entropy per class (test set)", fontsize=14)
    plt.xticks(classes, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return class_entropy, class_counts, entropy_means, entropy_std


# ── Calibration ────────────────────────────────────────────────────────────────

def get_prediction_probs(model, loader, device: torch.device):
    model.eval()
    y_true, y_pred_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            if images.size(0) == 0:
                continue
            images = images.to(device)
            outputs = model(images)
            logits  = outputs[0] if isinstance(outputs, tuple) else outputs
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    return np.array(y_true), np.array(y_pred_probs)


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.mean() > 0:
            ece += np.abs(confidences[in_bin].mean() - correct[in_bin].mean()) * in_bin.mean()
    return ece


def plot_calibration(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 10,
    save_path: str | None = None,
):
    confidences = y_pred_probs.max(axis=1)
    predictions = y_pred_probs.argmax(axis=1)
    correct     = predictions == y_true

    ece = compute_ece(confidences, correct, n_bins)
    print(f"ECE: {ece:.4f}")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_sizes.append(mask.sum())
        else:
            bin_accs.append(0.0)
            bin_confs.append(bin_edges[i] + 0.5 / n_bins)
            bin_sizes.append(0)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = bin_edges[:-1] + 0.5 / n_bins
    ax.bar(x, bin_accs, width=0.9 / n_bins, align="center",
           color="#3A6EA5", alpha=0.7, edgecolor="black", linewidth=0.6, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy",   fontsize=12)
    ax.set_title(f"Reliability diagram  (ECE = {ece:.3f})", fontsize=13)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return ece


if __name__ == "__main__":
    import os
    from dataset import build_dataloaders
    from model import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, dataset, *_ = build_dataloaders()

    num_classes = len(dataset.labels.unique())
    class_names = [f"Class {i+1}" for i in range(num_classes)]

    model = load_model(config.BEST_MODEL_PATH, num_classes=num_classes, device=device)

    out_dir = os.path.join(config.REPO_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving images to {out_dir}")

    test_accuracy(model, test_loader, device)
    compute_metrics(model, test_loader, num_classes, device)
    plot_confusion_matrix(model, test_loader, num_classes, device, class_names,
                          save_path=os.path.join(out_dir, "confusion_matrix.png"))
    plt.close("all")
    compute_entropy_per_class(model, test_loader, device, num_classes,
                              save_path=os.path.join(out_dir, "entropy_per_class.png"))
    plt.close("all")
    y_true, y_pred_probs = get_prediction_probs(model, test_loader, device)
    plot_calibration(y_true, y_pred_probs,
                     save_path=os.path.join(out_dir, "calibration.png"))
    plt.close("all")
    print(f"\nDone — images saved in {out_dir}/")
