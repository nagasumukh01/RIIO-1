#!/usr/bin/env python3
"""Compile the matched FF++/Celeb-DF cross-dataset evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    output = Path(args.out) if args.out else ROOT / "artifacts" / "experiment_logs" / f"cross_ff_celeb_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    experiments = [
        ("faceforensics", "celebdf", "efficientnet_b0", "ff_c23_b0_subset50_f4", "ff_c23_b0_subset50_f4"),
        ("faceforensics", "celebdf", "vit_small", "ff_c23_vit_small_subset50_f4", "ff_c23_vit_small_subset50_f4"),
        ("celebdf", "faceforensics", "efficientnet_b0", "celebdf_b0_subset200_f4", "celebdf_b0_subset200_f4"),
        ("celebdf", "faceforensics", "vit_small", "celebdf_vit_small_subset50_f4", "celebdf_vit_small_subset50_f4"),
    ]
    rows = []
    for source, target, model, experiment, checkpoint_dir in experiments:
        base = ROOT / "artifacts" / "models" / checkpoint_dir
        summary = load(base / "training_summary.json")
        target_eval = load(base / "cross_evaluations" / f"{target}_test" / "evaluation.json")["metrics"]
        source_metrics = summary["test_metrics"]
        rows.append({
            "source": source,
            "target": target,
            "model": model,
            "source_test_auroc": source_metrics["auroc"],
            "target_test_auroc": target_eval["auroc"],
            "target_accuracy": target_eval["accuracy"],
            "target_f1": target_eval["f1"],
            "target_eer": target_eval["eer"],
            "generalization_gap": source_metrics["auroc"] - target_eval["auroc"],
            "source_checkpoint": summary["checkpoint"],
            "is_synthetic": False,
        })
    fields = list(rows[0])
    with (output / "cross_dataset_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "cross_dataset_matrix.json").write_text(json.dumps({
        "label": "REAL DATA",
        "datasets": ["faceforensics", "celebdf"],
        "models": ["efficientnet_b0", "vit_small"],
        "protocol": "source train only; target test only; no target tuning",
        "generalization_gap_definition": "source test AUROC - target test AUROC",
        "rows": rows,
    }, indent=2, allow_nan=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
