#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a separate user-provided diffusion holdout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real_dataset import RealFaceVideoDataset, collate_real
from src.evaluation.artifacts import export_evaluation_artifacts, predict_loader
from src.models.factory import build_model
from src.utils.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a separate diffusion holdout")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--out", default="artifacts/experiment_logs/diffusion_eval.json")
    args = parser.parse_args()

    report = {
        "label": "REAL DATA - SEPARATE DIFFUSION HOLDOUT",
        "dataset": "diffusion",
        "checkpoint": args.checkpoint,
        "provenance": "user-provided held-out diffusion samples; never used for training or the cross-dataset matrix",
        "is_synthetic": False,
    }
    try:
        dataset = RealFaceVideoDataset("diffusion", args.split, frames_per_video=args.frames, max_videos=args.max_videos, allow_diffusion_holdout=True)
    except FileNotFoundError:
        report.update({"status": "pending - no diffusion samples found", "sample_count": 0})
        output = PROJECT_ROOT / args.out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    model = build_model(args.model, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_real)
    logits, labels, video_ids = predict_loader(model, loader, torch.device("cpu"))
    output = PROJECT_ROOT / args.out
    artifact_dir = output.parent / "diffusion_artifacts"
    artifacts = export_evaluation_artifacts(logits, labels, video_ids, args.split, "diffusion", artifact_dir, {"checkpoint": args.checkpoint})
    report.update({
        "status": "complete",
        "sample_count": len(video_ids),
        "metrics": artifacts["metrics"],
        "class_specific_errors": {"false_positives": artifacts["metrics"].get("fp", 0), "false_negatives": artifacts["metrics"].get("fn", 0)},
        "artifacts": artifacts,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
