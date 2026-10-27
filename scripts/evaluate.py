#!/usr/bin/env python3
"""Evaluate a checkpoint on a real dataset split. No target tuning."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real_dataset import RealFaceVideoDataset, collate_real
from src.evaluation.artifacts import export_evaluation_artifacts, predict_loader
from src.models.factory import build_model
from src.utils.logging_utils import setup_logger

logger = setup_logger("evaluate")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["faceforensics", "celebdf", "dfdc"])
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--model", default="efficientnet_b0")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    ds = RealFaceVideoDataset(
        dataset=args.dataset, split=args.split,
        frames_per_video=args.frames, max_videos=args.max_videos,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_real)

    model = build_model(args.model, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    logits, labels, all_ids = predict_loader(model, loader, device)
    out_dir = Path(args.out).parent if args.out else Path(args.checkpoint).parent / "evaluations" / f"{args.dataset}_{args.split}"
    artifacts = export_evaluation_artifacts(
        logits, labels, all_ids, args.split, args.dataset, out_dir,
        {"checkpoint": args.checkpoint, "model": args.model},
    )
    metrics = artifacts["metrics"]

    print(json.dumps(metrics, indent=2))
    output_json = Path(args.out) if args.out else out_dir / "evaluation.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "video_ids": all_ids,
                   "logits": logits.tolist(), "labels": labels.tolist(),
                   "artifacts": artifacts}, f, indent=2, allow_nan=True)
    logger.info("Saved evaluation artifacts → %s", out_dir)


if __name__ == "__main__":
    main()
