#!/usr/bin/env python3
"""
Calibrate ensemble weights and thresholds from a labelled validation folder.

Usage:
    python scripts/calibrate_ensemble.py --val-dir path/to/val
        where val/ has  real/  and  fake/  subfolders of images.

For each model:
  1. Run inference on every image (face-crop first).
  2. Compute AUC, best-F1 threshold (Youden's J), accuracy.
  3. Set weight = max(0, auc - 0.5) if auc >= 0.6, else 0.

For the ensemble:
  4. Compute weighted-average fused score for each image.
  5. Find optimal ensemble threshold via Youden's J.
  6. Compare ensemble vs. best single model.
  7. Write back to configs/ensemble.yaml.

If the ensemble is worse than the best single model, warn and list which
models are dragging it down.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.ensemble.registry import EnsembleRegistry
from src.api.server import _detect_and_verify_faces
from PIL import Image


def load_val_images(val_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    """Load images from val_dir/real and val_dir/fake."""
    images, labels = [], []
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = val_dir / label_name
        if not folder.exists():
            print(f"WARNING: {folder} does not exist, skipping")
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                images.append(img)
                labels.append(label_val)
            except Exception as exc:
                print(f"  Skip {img_path.name}: {exc}")
    return images, labels


def extract_face(image: np.ndarray) -> np.ndarray | None:
    """Detect and crop the primary face; returns None if no face found."""
    verified = _detect_and_verify_faces(image)
    if verified:
        return verified[0]["crop"]
    return None


def youdens_j_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find threshold maximising Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thresholds[best_idx])


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray,
                       n_steps: int = 200) -> float:
    """Grid-search threshold that maximises F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, n_steps):
        preds = (y_score >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def main():
    parser = argparse.ArgumentParser(description="Calibrate ensemble from validation data")
    parser.add_argument("--val-dir", type=str, required=True,
                        help="Path to validation folder with real/ and fake/ subfolders")
    parser.add_argument("--config", type=str, default="configs/ensemble.yaml",
                        help="Path to ensemble config (will be updated in-place)")
    parser.add_argument("--output-report", type=str, default=None,
                        help="Path to save calibration report JSON")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        print(f"ERROR: Validation directory not found: {val_dir}")
        sys.exit(1)

    print(f"Loading validation images from {val_dir} ...")
    images, labels = load_val_images(val_dir)
    print(f"  {len(images)} images loaded ({sum(1 for l in labels if l == 0)} real, "
          f"{sum(1 for l in labels if l == 1)} fake)")

    if len(images) < 4:
        print("ERROR: Need at least 4 validation images")
        sys.exit(1)

    # Extract face crops
    print("Extracting face crops ...")
    face_crops = []
    valid_labels = []
    skipped = 0
    for img, lbl in zip(images, labels):
        crop = extract_face(img)
        if crop is None:
            skipped += 1
            continue
        face_crops.append(crop)
        valid_labels.append(lbl)
    print(f"  {len(face_crops)} face crops extracted, {skipped} images skipped (no face)")

    if len(face_crops) < 4:
        print("ERROR: Need at least 4 face crops for calibration")
        sys.exit(1)

    y_true = np.array(valid_labels)

    # Load registry and models
    config_path = ROOT / args.config
    registry = EnsembleRegistry(config_path)
    loaded = registry.load_all()
    print(f"\nLoaded {len(loaded)}/{len(registry.enabled_configs())} models")

    if not loaded:
        print("ERROR: No models could be loaded")
        sys.exit(1)

    # Per-model inference
    model_scores: dict[str, np.ndarray] = {}
    model_report: list[dict] = []

    for model_id, model in loaded.items():
        cfg = registry.get(model_id)
        print(f"\n--- {model_id} ({cfg.display_name}) ---")
        probs = []
        for crop in face_crops:
            tensor = registry.normalise(crop, model_id).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
            prob = registry.fake_probability(logits, model_id)
            probs.append(prob)

        y_score = np.array(probs)
        model_scores[model_id] = y_score

        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = 0.5

        threshold_j = youdens_j_threshold(y_true, y_score)
        threshold_f1 = best_f1_threshold(y_true, y_score)
        # Use Youden's J threshold
        chosen_threshold = threshold_j

        preds = (y_score >= chosen_threshold).astype(int)
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        # Compute weight
        if auc >= 0.6:
            weight = max(0.0, auc - 0.5)
        else:
            weight = 0.0

        print(f"  AUC:       {auc:.4f}")
        print(f"  Threshold: {chosen_threshold:.4f} (Youden's J)")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Weight:    {weight:.4f}" + (" (EXCLUDED: AUC < 0.6)" if auc < 0.6 else ""))

        # Update registry config
        cfg.val_auc = auc
        cfg.threshold = chosen_threshold
        cfg.weight = weight

        model_report.append({
            "model_id": model_id,
            "display_name": cfg.display_name,
            "dataset": cfg.dataset,
            "val_auc": round(auc, 4),
            "threshold": round(chosen_threshold, 4),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "weight": round(weight, 4),
            "included": auc >= 0.6,
        })

    # Ensemble fusion
    print("\n=== ENSEMBLE FUSION ===")
    included_models = [r for r in model_report if r["included"]]
    all_model_ids = list(model_scores.keys())

    if included_models:
        # Weighted average
        ensemble_scores = np.zeros(len(y_true))
        total_weight = 0.0
        for r in included_models:
            mid = r["model_id"]
            w = r["weight"]
            ensemble_scores += w * model_scores[mid]
            total_weight += w
        if total_weight > 0:
            ensemble_scores /= total_weight
    else:
        # Uniform average fallback
        ensemble_scores = np.mean(
            [model_scores[mid] for mid in all_model_ids], axis=0
        )

    try:
        ensemble_auc = float(roc_auc_score(y_true, ensemble_scores))
    except ValueError:
        ensemble_auc = 0.5

    ensemble_threshold = youdens_j_threshold(y_true, ensemble_scores)
    ensemble_preds = (ensemble_scores >= ensemble_threshold).astype(int)
    ensemble_acc = float(accuracy_score(y_true, ensemble_preds))
    ensemble_f1 = float(f1_score(y_true, ensemble_preds, zero_division=0))

    print(f"  Ensemble AUC:       {ensemble_auc:.4f}")
    print(f"  Ensemble Threshold: {ensemble_threshold:.4f}")
    print(f"  Ensemble Accuracy:  {ensemble_acc:.4f}")
    print(f"  Ensemble F1:        {ensemble_f1:.4f}")

    # Compare vs best single model
    best_single = max(model_report, key=lambda r: r["val_auc"])
    print(f"\n  Best single model: {best_single['model_id']} (AUC={best_single['val_auc']:.4f})")

    if ensemble_auc < best_single["val_auc"]:
        print("\n  ⚠️  WARNING: Ensemble AUC is WORSE than the best single model!")
        print("  Models potentially dragging down the ensemble:")
        for r in model_report:
            if r["val_auc"] < ensemble_auc and r["included"]:
                print(f"    - {r['model_id']}: AUC={r['val_auc']:.4f} (below ensemble {ensemble_auc:.4f})")
    else:
        print(f"\n  ✅ Ensemble AUC ({ensemble_auc:.4f}) >= best single ({best_single['val_auc']:.4f})")

    # Write back to config
    print(f"\nWriting calibrated config to {config_path} ...")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["ensemble"]["threshold"] = round(ensemble_threshold, 4)

    for r in model_report:
        mid = r["model_id"]
        if mid in raw["models"]:
            raw["models"][mid]["val_auc"] = r["val_auc"]
            raw["models"][mid]["threshold"] = r["threshold"]
            raw["models"][mid]["weight"] = r["weight"]

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print("  ✅ Config updated")

    # Print final summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<35} {'Weight':>8} {'Val AUC':>9} {'Accuracy':>9} {'Included?':>10}")
    print("-" * 90)
    for r in model_report:
        inc = "YES" if r["included"] else "NO"
        print(f"{r['model_id']:<35} {r['weight']:>8.4f} {r['val_auc']:>9.4f} {r['accuracy']:>9.4f} {inc:>10}")
    print("-" * 90)
    print(f"{'ENSEMBLE':<35} {'—':>8} {ensemble_auc:>9.4f} {ensemble_acc:>9.4f} {'—':>10}")
    print(f"{'Best single':<35} {'—':>8} {best_single['val_auc']:>9.4f} {best_single['accuracy']:>9.4f} {'—':>10}")
    print("=" * 90)

    # Save report
    report = {
        "val_dir": str(val_dir),
        "n_images": len(images),
        "n_face_crops": len(face_crops),
        "n_skipped_no_face": skipped,
        "models": model_report,
        "ensemble": {
            "auc": round(ensemble_auc, 4),
            "threshold": round(ensemble_threshold, 4),
            "accuracy": round(ensemble_acc, 4),
            "f1": round(ensemble_f1, 4),
        },
        "best_single": {
            "model_id": best_single["model_id"],
            "auc": best_single["val_auc"],
            "accuracy": best_single["accuracy"],
        },
        "ensemble_better": ensemble_auc >= best_single["val_auc"],
    }

    report_path = args.output_report or str(
        ROOT / "artifacts" / "calibration_report.json"
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nCalibration report saved to {report_path}")


if __name__ == "__main__":
    main()
