"""Primary evaluation metrics for deepfake detection research."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Equal Error Rate via ROC curve interpolation."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # Find point where |fpr - fnr| is minimal
    abs_diff = np.abs(fpr - fnr)
    idx = np.nanargmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def compute_metrics(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute standard binary detection metrics.
    logits: raw model outputs (before sigmoid)
    labels: 0 (real) / 1 (fake)
    """
    logits_np = _to_numpy(logits).ravel()
    labels_np = _to_numpy(labels).ravel().astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
    preds = (probs >= threshold).astype(int)

    out: Dict[str, float] = {}
    try:
        out["auroc"] = float(roc_auc_score(labels_np, probs))
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["ap"] = float(average_precision_score(labels_np, probs))
    except ValueError:
        out["ap"] = float("nan")
    out["accuracy"] = float(accuracy_score(labels_np, preds))
    out["precision"] = float(precision_score(labels_np, preds, zero_division=0))
    out["recall"] = float(recall_score(labels_np, preds, zero_division=0))
    out["f1"] = float(f1_score(labels_np, preds, zero_division=0))
    try:
        out["eer"] = compute_eer(labels_np, probs)
    except Exception:
        out["eer"] = float("nan")

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        out["tn"] = int(tn)
        out["fp"] = int(fp)
        out["fn"] = int(fn)
        out["tp"] = int(tp)
        out["fpr"] = float(fp / (fp + tn + 1e-8))
        out["fnr"] = float(fn / (fn + tp + 1e-8))
    return out


def generalization_gap(source_auroc: float, target_auroc: float) -> float:
    """Positive value = drop when moving to target (worse generalization)."""
    return float(source_auroc - target_auroc)
