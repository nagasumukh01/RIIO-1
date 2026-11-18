#!/usr/bin/env python3
"""Execute the six ordered cross-dataset evaluations without target tuning."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import PROJECT_ROOT, ensure_dir, get_experiment_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("cross_dataset")


def _run(command: list[str]) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the six-pair cross-dataset matrix")
    parser.add_argument("--models", nargs="+", default=["efficientnet_b0"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Train and evaluate every pair")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if not args.dry_run and not args.execute:
        parser.error("choose --dry-run or --execute")

    pairs = get_experiment_config()["cross_dataset"]["pairs"]
    out_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "experiment_logs" / f"cross_{datetime.now():%Y%m%d_%H%M%S}")
    jobs = []
    for model in args.models:
        for pair in pairs:
            exp_id = f"cross_{pair['source']}_to_{pair['target']}_{model}"
            train_cmd = [sys.executable, "scripts/train.py", "--dataset", pair["source"], "--model", model, "--experiment-id", exp_id]
            eval_cmd = [sys.executable, "scripts/evaluate.py", "--dataset", pair["target"], "--split", "test", "--model", model, "--checkpoint", str(PROJECT_ROOT / "artifacts" / "models" / exp_id / "best.pt")]
            for command in (train_cmd, eval_cmd):
                if args.max_videos is not None:
                    command.extend(["--max-videos", str(args.max_videos)])
            if args.epochs is not None:
                train_cmd.extend(["--epochs", str(args.epochs)])
            jobs.append({"source": pair["source"], "target": pair["target"], "model": model, "experiment_id": exp_id, "train": train_cmd, "evaluate": eval_cmd})

    with (out_dir / "plan.json").open("w", encoding="utf-8") as handle:
        json.dump({"pairs": [{k: v for k, v in job.items() if k not in ("train", "evaluate")} for job in jobs], "protocol": "source train only; target test only; no target threshold tuning"}, handle, indent=2)
    if args.dry_run:
        for index, job in enumerate(jobs, 1):
            print(f"{index}. {job['source']} -> {job['target']} | {job['model']}")
            print("   TRAIN:", " ".join(job["train"]))
            print("   EVAL: ", " ".join(job["evaluate"]))
        print(f"Plan written to {out_dir / 'plan.json'}")
        return

    rows = []
    for job in jobs:
        _run(job["train"])
        _run(job["evaluate"])
        checkpoint_dir = PROJECT_ROOT / "artifacts" / "models" / job["experiment_id"]
        source_summary = json.loads((checkpoint_dir / "training_summary.json").read_text(encoding="utf-8"))
        evaluation_path = checkpoint_dir / "evaluations" / f"{job['target']}_test" / "evaluation.json"
        target_metrics = json.loads(evaluation_path.read_text(encoding="utf-8"))["metrics"]
        source_metrics = source_summary["test_metrics"]
        rows.append({"source": job["source"], "target": job["target"], "model": job["model"], "source_auroc": source_metrics["auroc"], "target_auroc": target_metrics["auroc"], "accuracy": target_metrics["accuracy"], "f1": target_metrics["f1"], "eer": target_metrics["eer"], "generalization_gap": source_metrics["auroc"] - target_metrics["auroc"], "checkpoint": source_summary["checkpoint"]})
    fields = list(rows[0]) if rows else ["source", "target", "model"]
    with (out_dir / "cross_dataset_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "cross_dataset_matrix.json").write_text(json.dumps({"definition": "generalization gap = source test AUROC - target test AUROC", "rows": rows}, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Matrix written to {out_dir}")


if __name__ == "__main__":
    main()
