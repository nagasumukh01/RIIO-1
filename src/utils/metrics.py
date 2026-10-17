"""
Metrics computation for deepfake detection.
Computes accuracy, precision, recall, F1, AUROC, EER, confusion matrix, etc.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    Returns (eer, threshold_at_eer).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # Find the point where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    return float(eer), float(threshold)


def compute_all_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        labels: Ground truth binary labels (0=real, 1=fake)
        predictions: Binary predictions
        scores: Continuous probability scores for the positive (fake) class
        threshold: Decision threshold used for the predictions

    Returns:
        Dictionary with all metrics
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    scores = np.array(scores)

    # Basic classification metrics
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # AUROC
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.0  # Only one class present

    # EER
    try:
        eer, eer_threshold = compute_eer(labels, scores)
    except Exception:
        eer, eer_threshold = 0.0, 0.5

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # FPR and FNR
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ROC curve data
    try:
        fpr_curve, tpr_curve, roc_thresholds = roc_curve(labels, scores)
    except ValueError:
        fpr_curve, tpr_curve, roc_thresholds = [], [], []

    # Average precision
    try:
        avg_prec = average_precision_score(labels, scores)
    except ValueError:
        avg_prec = 0.0

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'auroc': float(auroc),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'fpr': float(fpr_val),
        'fnr': float(fnr_val),
        'average_precision': float(avg_prec),
        'confusion_matrix': cm.tolist(),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'threshold': float(threshold),
        'total_samples': int(len(labels)),
        'positive_samples': int(np.sum(labels == 1)),
        'negative_samples': int(np.sum(labels == 0)),
        'roc_curve': {
            'fpr': fpr_curve.tolist() if hasattr(fpr_curve, 'tolist') else list(fpr_curve),
            'tpr': tpr_curve.tolist() if hasattr(tpr_curve, 'tolist') else list(tpr_curve),
        },
    }


def compute_cross_dataset_drop(
    in_domain_metrics: Dict[str, float],
    cross_domain_metrics: Dict[str, float],
    metric_name: str = 'auroc'
) -> float:
    """Compute performance drop from in-domain to cross-domain."""
    in_val = in_domain_metrics.get(metric_name, 0)
    cross_val = cross_domain_metrics.get(metric_name, 0)
    if in_val == 0:
        return 0.0
    return float((in_val - cross_val) / in_val * 100)


def compute_robustness_drop(
    clean_metrics: Dict[str, float],
    perturbed_metrics: Dict[str, float],
    metric_name: str = 'auroc'
) -> float:
    """Compute performance drop under perturbation."""
    clean_val = clean_metrics.get(metric_name, 0)
    perturbed_val = perturbed_metrics.get(metric_name, 0)
    if clean_val == 0:
        return 0.0
    return float((clean_val - perturbed_val) / clean_val * 100)
