#!/usr/bin/env python3
"""Run evaluation-time JPEG and blur stress tests on a real checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real_dataset import RealFaceVideoDataset, collate_real
from src.evaluation.metrics import compute_metrics
from src.models.factory import build_model
from src.robustness.stress import STRESS_FN, apply_stress_to_batch
from src.utils.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["faceforensics", "celebdf", "dfdc"])
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dataset = RealFaceVideoDataset(args.dataset, args.split, frames_per_video=args.frames, max_videos=args.max_videos)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_real)
    model = build_model(args.model, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    clean_logits, labels = [], []
    stressed = {name: [] for name in STRESS_FN}
    with torch.no_grad():
        for batch in loader:
            video = batch["video"]
            labels.append(batch["label"])
            clean_logits.append(model(video))
            for name in STRESS_FN:
                stressed[name].append(model(apply_stress_to_batch(video, name)))
    labels_tensor = torch.cat(labels)
    clean_metrics = compute_metrics(torch.cat(clean_logits), labels_tensor)
    rows = []
    for name, logits in stressed.items():
        metrics = compute_metrics(torch.cat(logits), labels_tensor)
        rows.append({"dataset": args.dataset, "split": args.split, "stress": name, "clean_auroc": clean_metrics["auroc"], "stressed_auroc": metrics["auroc"], "auroc_drop": clean_metrics["auroc"] - metrics["auroc"], "accuracy": metrics["accuracy"], "f1": metrics["f1"], "eer": metrics["eer"], "checkpoint": args.checkpoint, "is_synthetic": False})
    output = PROJECT_ROOT / args.out if args.out else PROJECT_ROOT / "artifacts" / "experiment_logs" / f"robustness_{args.dataset}_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"label": "REAL DATA", "protocol": "evaluation-time compression and blur stress; no retraining", "rows": rows}, indent=2, allow_nan=True), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
