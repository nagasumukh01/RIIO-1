#!/usr/bin/env python3
"""Execute controlled augmentation, loss, and capacity ablations."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["augmentation", "loss", "capacity", "all"], default="all")
    parser.add_argument("--dataset", default="faceforensics")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if not args.dry_run and not args.execute:
        parser.error("choose --dry-run or --execute")
    config = get_experiment_config()["ablations"]
    out_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "experiment_logs" / f"ablations_{datetime.now():%Y%m%d_%H%M%S}")
    jobs = []
    if args.type in ("augmentation", "all"):
        jobs.extend({"type": "augmentation", "variant": value, "augmentation": value, "model": "efficientnet_b0", "loss": "bce"} for value in config["augmentation"]["strategies"])
    if args.type in ("loss", "all"):
        jobs.extend({"type": "loss", "variant": value, "augmentation": "none", "model": "efficientnet_b0", "loss": value} for value in config["loss"]["functions"])
    if args.type in ("capacity", "all"):
        jobs.extend({"type": "capacity", "variant": value, "augmentation": "none", "model": value, "loss": "bce"} for value in config["capacity"]["models"])
    if args.dry_run:
        for job in jobs:
            print(f"[{job['type']}] {job['variant']} | dataset={args.dataset} model={job['model']} loss={job['loss']} augmentation={job['augmentation']}")
        print(f"Plan contains {len(jobs)} jobs; no training was run.")
        return

    rows = []
    for job in jobs:
        exp_id = f"ablation_{job['type']}_{job['variant']}_{args.dataset}"
        command = [sys.executable, "scripts/train.py", "--dataset", args.dataset, "--model", job["model"], "--loss", job["loss"], "--augmentation", job["augmentation"], "--experiment-id", exp_id]
        if args.max_videos is not None:
            command.extend(["--max-videos", str(args.max_videos)])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        subprocess.run(command, cwd=ROOT, check=True)
        summary = json.loads((PROJECT_ROOT / "artifacts" / "models" / exp_id / "training_summary.json").read_text(encoding="utf-8"))
        metrics = summary["test_metrics"]
        rows.append({"ablation": job["type"], "variant": job["variant"], "dataset": args.dataset, "model": job["model"], "loss": job["loss"], "augmentation": job["augmentation"], "auroc": metrics["auroc"], "accuracy": metrics["accuracy"], "f1": metrics["f1"], "eer": metrics["eer"], "experiment_id": exp_id})
    (out_dir / "ablation_results.json").write_text(json.dumps({"rows": rows}, indent=2, allow_nan=True), encoding="utf-8")
    with (out_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["ablation", "variant"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ablation results written to {out_dir}")


if __name__ == "__main__":
    main()
