#!/usr/bin/env python3
"""Recover an honest experiment summary from a saved real checkpoint and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["faceforensics", "celebdf", "dfdc"])
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    summary = {
        "experiment_id": checkpoint.get("experiment_id", checkpoint_path.parent.name),
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "artifact_dir": str(checkpoint_path.parent),
        "epochs_ran": int(checkpoint.get("epoch", 0)) + 1,
        "training_complete": False,
        "status": "checkpoint recovered after interrupted training; metrics are real and incomplete-run metadata is explicit",
        "is_synthetic": False,
        "label": "REAL DATA",
        "last_checkpoint_metrics": checkpoint.get("metrics", {}),
        "val_metrics": checkpoint.get("metrics", {}),
        "test_metrics": evaluation["metrics"],
        "evaluation_artifacts": evaluation.get("artifacts", {}),
    }
    summary_path = checkpoint_path.parent / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True, default=str), encoding="utf-8")
    log_path = checkpoint_path.parents[2] / "experiment_logs" / f"{summary['experiment_id']}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, allow_nan=True, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
