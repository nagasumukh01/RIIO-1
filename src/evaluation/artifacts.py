"""Research artifact writers for real-data evaluation runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics


def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    video_ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch["video"].to(device)).detach().cpu())
            labels.append(batch["label"].detach().cpu())
            video_ids.extend(batch["video_id"])
    if not logits:
        raise ValueError("Evaluation loader produced no samples")
    return torch.cat(logits), torch.cat(labels), video_ids


def export_evaluation_artifacts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    video_ids: Sequence[str],
    split: str,
    dataset: str,
    out_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write reproducible metrics, predictions, ROC, and confusion artifacts."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores = torch.sigmoid(logits).numpy().ravel()
    y_true = labels.numpy().astype(int).ravel()
    metrics = compute_metrics(logits, labels)
    metrics.update({"dataset": dataset, "split": split, "n_videos": len(video_ids), "is_synthetic": False})
    if extra:
        metrics.update(extra)

    predictions_path = output / ("predictions.csv" if split == "test" else f"predictions_{split}.csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "label", "score", "split"])
        writer.writeheader()
        writer.writerows(
            {"video_id": video_id, "label": int(label), "score": float(score), "split": split}
            for video_id, label, score in zip(video_ids, y_true, scores)
        )

    roc_payload: dict[str, Any] = {"dataset": dataset, "split": split, "metrics": metrics}
    try:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_payload.update({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()})
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUROC={metrics['auroc']:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{dataset} {split} ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output / f"roc_{split}.png", dpi=160)
        plt.close()
    except ValueError:
        roc_payload["note"] = "ROC unavailable because both classes are not present."
    with (output / f"roc_{split}.json").open("w", encoding="utf-8") as handle:
        json.dump(roc_payload, handle, indent=2, allow_nan=True)

    cm = confusion_matrix(y_true, scores >= 0.5, labels=[0, 1]).tolist()
    with (output / f"confusion_matrix_{split}.json").open("w", encoding="utf-8") as handle:
        json.dump({"labels": [0, 1], "matrix": cm, "dataset": dataset, "split": split}, handle, indent=2)
    plt.figure(figsize=(4, 3.5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{dataset} {split} confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, cm[row][col], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output / f"confusion_matrix_{split}.png", dpi=160)
    plt.close()
    return {"metrics": metrics, "predictions": str(predictions_path), "roc": str(output / f"roc_{split}.json")}