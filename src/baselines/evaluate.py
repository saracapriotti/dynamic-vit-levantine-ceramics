"""
Test-set metrics for the baselines: accuracy, macro precision / recall / F1
(same definition used for the Dynamic ViT) and weighted one-vs-rest ROC-AUC.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict(model, loader, device: torch.device):
    """Softmax probabilities and true labels on a loader."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        if images.size(0) == 0:
            continue
        outputs = model(images.to(device))
        logits  = outputs[0] if isinstance(outputs, tuple) else outputs
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(model, loader, num_classes: int, device: torch.device) -> dict:
    probs, labels = predict(model, loader, device)
    preds = probs.argmax(1)

    cm = np.zeros((num_classes, num_classes))
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = tp.sum() / cm.sum()
    roc_auc   = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")

    print(f"Accuracy:       {100 * accuracy:.2f}%")
    print(f"Mean precision: {100 * precision.mean():.2f}%")
    print(f"Mean recall:    {100 * recall.mean():.2f}%")
    print(f"Mean F1:        {100 * f1.mean():.2f}%")
    print(f"ROC-AUC:        {100 * roc_auc:.2f}%")

    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                roc_auc=roc_auc, cm=cm)
