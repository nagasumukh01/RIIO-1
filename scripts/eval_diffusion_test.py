"""
Evaluate the 5 deepfake models and ensemble on the held-out Diffusion Synthetic Face test split.

This test measures zero-shot generalization of face-swap deepfake models to modern diffusion faces (SDXL).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble.registry import get_registry
from src.ensemble.fusion import analyze_image_faces

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("robustdeep.eval_diffusion")


def main():
    manifest_path = PROJECT_ROOT / "data" / "diffusion_face" / "manifests" / "test_manifest.json"
    if not manifest_path.exists():
        logger.error("Test manifest not found: %s", manifest_path)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    logger.info("Loaded %d test samples from %s", len(samples), manifest_path)

    registry = get_registry()
    logger.info("Loading models into registry...")
    loaded_models = registry.load_all()
    logger.info("Loaded %d/%d models: %s", len(loaded_models), len(registry.model_ids), list(loaded_models.keys()))

    model_ids = registry.model_ids
    per_model_preds: Dict[str, List[float]] = {m: [] for m in model_ids}
    per_model_correct: Dict[str, int] = {m: 0 for m in model_ids}
    ensemble_scores: List[float] = []
    ensemble_correct: int = 0
    ensemble_total: int = 0

    results_table = []

    for s in samples:
        if type(s.get("label")) is not int or s["label"] not in (0, 1):
            logger.warning("Skip sample with missing/unsupported ground-truth label: %s", s.get("path"))
            continue
        img_path = Path(s["path"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        ground_truth = int(s["label"])

        if not img_path.exists():
            logger.warning("Image missing: %s", img_path)
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_arr = np.array(pil_img)
        except Exception as e:
            logger.warning("Could not read image %s: %s", img_path, e)
            continue

        # Run ensemble fusion on face
        verdict_obj = analyze_image_faces([{"crop": img_arr, "box": None}], registry=registry)
        if not verdict_obj.per_face:
            logger.warning("No face result for %s", img_path)
            continue
        face_res = verdict_obj.per_face[0]
        if verdict_obj.fused_score is None or not face_res.model_results:
            logger.warning("No model scores available for %s", img_path)
            continue

        row_record = {
            "image": img_path.name,
            "ground_truth": "FAKE" if ground_truth == 1 else "REAL",
            "ensemble_score": round(verdict_obj.fused_score, 4),
            "ensemble_verdict": verdict_obj.verdict,
            "models": {}
        }

        if verdict_obj.verdict in ("FAKE", "REAL"):
            ensemble_total += 1
            if verdict_obj.verdict == row_record["ground_truth"]:
                ensemble_correct += 1
        ensemble_scores.append(verdict_obj.fused_score)

        for m_res in face_res.model_results:
            m_id = m_res.model_id
            prob = m_res.probability
            thresh = registry.get(m_id).threshold
            is_pred_fake = prob >= thresh

            per_model_preds[m_id].append(prob)
            if (ground_truth == 1 and is_pred_fake) or (ground_truth == 0 and not is_pred_fake):
                per_model_correct[m_id] += 1

            row_record["models"][m_id] = {
                "prob_fake": round(prob, 4),
                "threshold": thresh,
                "pred": "FAKE" if is_pred_fake else "REAL",
                "weight": m_res.weight,
                "included": m_res.included,
            }

        results_table.append(row_record)

    total_samples = len(results_table)

    # Print detailed report
    print("\n" + "=" * 80)
    print("  ZERO-SHOT EVALUATION ON DIFFUSION SYNTHETIC FACES (TEST SPLIT)")
    print("=" * 80)
    print(f"  Test Samples Evaluated: {total_samples}")
    print("  Target Ground Truth   : Labels recorded in the test manifest")
    print("-" * 80)
    print(f"{'Model':<16} | {'Backbone':<14} | {'Dataset':<10} | {'Mean P(Fake)':<12} | {'Fake Recall / Acc':<16}")
    print("-" * 80)

    summary_models = {}
    for m_id in model_ids:
        cfg = registry.get(m_id)
        probs = per_model_preds[m_id]
        mean_p = float(np.mean(probs)) if probs else None
        model_total = len(probs)
        acc = per_model_correct[m_id] / model_total if model_total else None
        summary_models[m_id] = {
            "display_name": cfg.display_name,
            "architecture": cfg.architecture,
            "dataset": cfg.dataset,
            "mean_p_fake": round(mean_p, 4) if mean_p is not None else None,
            "accuracy": round(float(acc), 4) if acc is not None else None,
            "correct": per_model_correct[m_id],
            "total": model_total,
            "status": "available" if model_total else "pending",
            "weight": cfg.weight,
        }
        mean_display = f"{mean_p:.4f}" if mean_p is not None else "N/A"
        acc_display = f"{acc*100:.1f}%" if acc is not None else "N/A"
        print(f"{m_id:<16} | {cfg.architecture:<14} | {cfg.dataset:<10} | {mean_display:>12} | {acc_display:>16} ({per_model_correct[m_id]}/{model_total})")

    ens_acc = ensemble_correct / ensemble_total if ensemble_total else None
    mean_ens_score = float(np.mean(ensemble_scores)) if ensemble_scores else None
    print("-" * 80)
    ens_mean_display = f"{mean_ens_score:.4f}" if mean_ens_score is not None else "N/A"
    ens_acc_display = f"{ens_acc*100:.1f}%" if ens_acc is not None else "N/A"
    print(f"{'ENSEMBLE FUSION':<16} | {'AUC-Weighted':<14} | {'Loaded':<10} | {ens_mean_display:>11} | {ens_acc_display:>13} ({ensemble_correct}/{ensemble_total})")
    print("=" * 80 + "\n")

    # Save artifact
    out_dir = PROJECT_ROOT / "artifacts" / "experiment_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "diffusion_test_eval.json"

    eval_artifact = {
        "status": "available" if results_table else "pending",
        "message": None if results_table else "No supported test samples produced model scores.",
        "timestamp": manifest.get("created"),
        "num_samples": total_samples,
        "models": summary_models,
        "ensemble": {
            "status": "available" if ensemble_total else "pending",
            "accuracy": round(float(ens_acc), 4) if ens_acc is not None else None,
            "mean_score": round(float(mean_ens_score), 4) if mean_ens_score is not None else None,
            "correct": ensemble_correct,
            "total": ensemble_total,
        },
        "per_sample_results": results_table,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_artifact, f, indent=2)
    logger.info("Saved evaluation artifact to %s", report_file)


if __name__ == "__main__":
    main()
