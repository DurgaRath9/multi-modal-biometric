"""Training metrics computation.

Includes standard classification metrics and biometric-specific metrics
such as Equal Error Rate (EER) and ROC-AUC.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def compute_top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy."""
    k = min(k, logits.size(1))
    _, topk_preds = logits.topk(k, dim=1)
    correct = topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
    return correct / labels.size(0)


def compute_eer(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) — the standard biometric metric.

    EER is the point where False Acceptance Rate equals False Rejection Rate.
    Lower is better.
    """
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    # Use one-vs-rest: for each sample, use the probability of the true class
    true_scores = probs[np.arange(len(labels)), labels]

    # Binary: genuine (1) vs impostor (0)
    # Genuine = score assigned to the correct class
    # Impostor = scores assigned to wrong classes
    genuine_scores = true_scores
    impostor_scores = []
    for i in range(len(labels)):
        mask = np.ones(probs.shape[1], dtype=bool)
        mask[labels[i]] = False
        impostor_scores.extend(probs[i, mask].tolist())

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, np.array(impostor_scores)])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Interpolate to find EER (where FPR == FNR)
    eer_func = interp1d(fpr, fnr)
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer


def compute_epoch_metrics(all_logits: torch.Tensor, all_labels: torch.Tensor) -> dict[str, float]:
    """Compute comprehensive metrics for an epoch.

    Args:
        all_logits: Concatenated logits from all batches, shape (N, C).
        all_labels: Concatenated labels from all batches, shape (N,).

    Returns:
        Dict with top1_acc, top5_acc, f1_macro, roc_auc, eer.
    """
    preds = all_logits.argmax(dim=1).cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    probs = torch.softmax(all_logits, dim=1).cpu().numpy()

    top1 = compute_accuracy(all_logits, all_labels)
    top5 = compute_top_k_accuracy(all_logits, all_labels, k=5)

    f1 = float(f1_score(labels_np, preds, average="macro", zero_division=0))

    # ROC-AUC (one-vs-rest, macro)
    try:
        unique_labels = np.unique(labels_np)
        if len(unique_labels) >= 2:
            # Remap labels to contiguous [0, N) and slice + renormalize probs
            label_map = {old: new for new, old in enumerate(unique_labels)}
            remapped = np.array([label_map[l] for l in labels_np])
            probs_subset = probs[:, unique_labels]
            probs_subset = probs_subset / probs_subset.sum(axis=1, keepdims=True)
            auc = float(roc_auc_score(remapped, probs_subset, multi_class="ovr", average="macro"))
        else:
            auc = 0.0
    except (ValueError, IndexError):
        auc = 0.0

    # EER
    try:
        eer = compute_eer(probs, labels_np)
    except Exception:
        eer = 1.0

    return {
        "top1_acc": top1,
        "top5_acc": top5,
        "f1_macro": f1,
        "roc_auc": auc,
        "eer": eer,
    }


def compute_confusion_matrix(all_logits: torch.Tensor, all_labels: torch.Tensor) -> np.ndarray:
    """Return the confusion matrix as a numpy array."""
    preds = all_logits.argmax(dim=1).cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    return confusion_matrix(labels_np, preds)
