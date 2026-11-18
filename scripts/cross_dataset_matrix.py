#!/usr/bin/env python3
"""
Cross-Dataset Generalization Matrix
====================================
Evaluates every trained model checkpoint on every available test split.
Produces a train-dataset × test-dataset table of AUROC, Accuracy, F1.

Usage:
  python scripts/cross_dataset_matrix.py [--max-samples 200] [--output-dir artifacts/experiment_logs]

No test-set contamination: only uses the held-out 'test' split of each dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import timm

from src.utils.config import PROJECT_ROOT, load_yaml
from src.utils.logging_utils import setup_logger
from src.models.factory import VideoDeepfakeClassifier

logger = setup_logger("robustdeep.crossdataset")

# ---------------------------------------------------------------------------
# Preprocessing helpers (mirrors EnsembleRegistry.normalise)
# ---------------------------------------------------------------------------

def build_transform(input_size: int, mean: list, std: list):
    """Build a simple PIL → tensor preprocessing function."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def predict_on_samples(
    model: nn.Module,
    samples: list[tuple[str, int]],
    transform,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference while retaining the label paired with each successful prediction."""
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for path, label in samples:
            try:
                img = Image.open(path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                # VideoDeepfakeClassifier expects (B, T, C, H, W) but supports (B, C, H, W) via squeeze
                # Treat single frame as T=1
                x = x.unsqueeze(1)  # (1, 1, C, H, W)
                logit = model(x)
                p = torch.sigmoid(logit).squeeze().item()
                scores.append(float(p))
                labels.append(int(label))
            except Exception as e:
                logger.warning(f"Skip {path}: {e}")
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# Dataset loader (image-level for efficiency, honours test split only)
# ---------------------------------------------------------------------------

def load_test_samples(dataset_name: str, max_samples: int | None = None):
    """
    Returns list of (image_path, label) pairs from the held-out test split.
    Labels: 0=real, 1=fake.
    Prefers manifest JSON over name heuristics so DFDC labels are correct.
    """
    if dataset_name == "diffusion":
        return _load_diffusion_test(max_samples)

    faces_root = PROJECT_ROOT / "data" / "faces" / dataset_name / "test"
    if not faces_root.exists():
        logger.warning(f"Test split not found: {faces_root}")
        return []

    # Load manifest if present — has authoritative labels (critical for DFDC)
    manifest_path = PROJECT_ROOT / "data" / "processed" / dataset_name / "test_manifest.json"
    manifest_labels: dict = {}
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for s in data.get("samples", []):
                manifest_labels[s["video_id"]] = int(s["label"])
            logger.info(f"  Loaded manifest for {dataset_name}: {len(manifest_labels)} entries")
        except Exception as e:
            logger.warning(f"  Could not read manifest for {dataset_name}: {e}")

    samples = []
    for video_dir in sorted(faces_root.iterdir()):
        if not video_dir.is_dir():
            continue
        vid_name = video_dir.name
        vid = vid_name.lower()

        # Prefer manifest label; fall back to name heuristic
        if vid_name in manifest_labels:
            label = manifest_labels[vid_name]
        elif any(k in vid for k in ("real", "original", "pristine", "authentic")):
            label = 0
        elif any(k in vid for k in ("fake", "deepfake", "manip", "synth", "df_", "fs_", "f2f", "nt_")):
            label = 1
        elif vid.startswith("0_") or vid.startswith("real"):
            label = 0
        else:
            label = 1  # conservative default

        frames = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
        if not frames:
            continue
        # Take up to 4 evenly-spaced frames per video
        chosen = frames[::max(1, len(frames)//4)][:4]
        samples.extend((str(f), label) for f in chosen)

    # Balance real/fake
    real = [(p, l) for p, l in samples if l == 0]
    fake = [(p, l) for p, l in samples if l == 1]
    n = min(len(real), len(fake))
    if max_samples:
        n = min(n, max_samples // 2)
    if n == 0:
        logger.warning(f"  {dataset_name}/test: no balanced samples (real={len(real)}, fake={len(fake)})")
        return []
    rng = np.random.RandomState(42)
    real_idx = rng.choice(len(real), n, replace=False).tolist()
    fake_idx = rng.choice(len(fake), n, replace=False).tolist()
    balanced = [real[i] for i in real_idx] + [fake[i] for i in fake_idx]
    rng.shuffle(balanced)
    logger.info(f"  {dataset_name}/test: {n} real + {n} fake = {len(balanced)} samples")
    return balanced


def _load_diffusion_test(max_samples: int | None):
    """Load diffusion benchmark test manifest."""
    manifest_path = PROJECT_ROOT / "data" / "diffusion_face" / "metadata" / "test_manifest.json"
    if not manifest_path.exists():
        logger.warning(f"Diffusion manifest not found: {manifest_path}")
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = []
    for entry in data:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            samples.append((str(path), int(entry["label"])))
    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]
    logger.info(f"  diffusion/test: {len(samples)} samples loaded")
    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5):
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float("nan")
    return {"auroc": round(auc, 4), "accuracy": round(acc, 4), "f1": round(f1, 4),
            "n_samples": int(len(labels))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = ["faceforensics", "celebdf", "dfdc", "diffusion"]


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset generalization matrix")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max test samples per dataset (balanced real/fake)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
        PROJECT_ROOT / "artifacts" / "experiment_logs" / f"cross_dataset_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ensemble config to get model info
    cfg = load_yaml(PROJECT_ROOT / "configs" / "ensemble.yaml")
    models_cfg = cfg.get("models", {})

    # Build model → train-dataset mapping (for axis labels)
    model_to_train_ds = {
        "ff_b0": "faceforensics",
        "ff_vit": "faceforensics",
        "celebdf_b0": "celebdf",
        "celebdf_vit": "celebdf",
        "dfdc_b0": "dfdc",
        "combined_multidomain_b0": "all_combined",
    }

    # Models with valid checkpoints
    valid_models = {}
    for model_id, mcfg in models_cfg.items():
        ckpt = Path(mcfg["checkpoint"])
        if not ckpt.is_absolute():
            ckpt = PROJECT_ROOT / ckpt
        if ckpt.is_file():
            valid_models[model_id] = {**mcfg, "checkpoint": str(ckpt)}
        else:
            logger.warning(f"Skip {model_id}: checkpoint not found at {ckpt}")

    logger.info(f"Found {len(valid_models)} valid checkpoints: {list(valid_models.keys())}")

    # Pre-load test sets once
    logger.info("Loading test datasets...")
    test_sets = {}
    for ds in DATASETS:
        samples = load_test_samples(ds, args.max_samples)
        if samples:
            test_sets[ds] = samples
        else:
            logger.warning(f"  {ds}: no test samples found, skipping column")

    # Result matrix: rows = train-dataset, cols = test-dataset
    # For models trained on "all_combined", we use "all_combined" as train-dataset label
    results = {}  # model_id -> {test_ds -> metrics}

    for model_id, mcfg in valid_models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model_id}  ({mcfg['display_name']})")
        logger.info(f"  checkpoint: {mcfg['checkpoint']}")

        try:
            backbone = timm.create_model(
                mcfg["timm_name"], pretrained=False, num_classes=int(mcfg["output_units"])
            )
            model = VideoDeepfakeClassifier(
                backbone=backbone,
                backbone_type=mcfg["backbone_type"],
                aggregation="mean",
            )
            ckpt_data = torch.load(mcfg["checkpoint"], map_location="cpu", weights_only=False)
            state = ckpt_data.get("model_state", ckpt_data)
            has_prefix = any(k.startswith("backbone.") for k in state)
            if not has_prefix:
                state = {f"backbone.{k}": v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            model.eval()
            logger.info(f"  Model loaded OK")
        except Exception as e:
            logger.error(f"  Failed to load {model_id}: {e}")
            continue

        transform = build_transform(
            input_size=int(mcfg["input_size"]),
            mean=list(mcfg["mean"]),
            std=list(mcfg["std"]),
        )
        threshold = float(mcfg.get("threshold", 0.5))
        model_results = {}

        for test_ds, samples in test_sets.items():
            logger.info(f"  → Testing on {test_ds} ({len(samples)} samples)...")
            t0 = time.time()
            labels, scores = predict_on_samples(model, samples, transform)
            if len(scores) == 0:
                logger.warning(f"    No predictions for {test_ds}")
                continue
            metrics = compute_metrics(labels, scores, threshold)
            elapsed = time.time() - t0
            logger.info(f"    AUROC={metrics['auroc']:.4f}  ACC={metrics['accuracy']:.4f}  "
                        f"F1={metrics['f1']:.4f}  ({elapsed:.1f}s)")
            model_results[test_ds] = metrics

        results[model_id] = {
            "train_dataset": model_to_train_ds.get(model_id, "unknown"),
            "display_name": mcfg["display_name"],
            "val_auc": float(mcfg.get("val_auc", 0.0)),
            "test_results": model_results,
        }

    # ----- Build and print the generalization matrix -----
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-DATASET GENERALIZATION MATRIX (AUROC)")
    logger.info(f"{'='*70}")

    test_ds_list = sorted(test_sets.keys())
    model_ids = list(results.keys())

    # Header
    header = f"{'Model (trained on)':<40}" + "".join(f"{ds:>15}" for ds in test_ds_list)
    logger.info(header)
    logger.info("-" * len(header))

    rows_for_csv = []
    for model_id in model_ids:
        r = results[model_id]
        row_label = f"{model_id} ({r['train_dataset']})"
        row = f"{row_label:<40}"
        csv_row = {"model": model_id, "train_dataset": r["train_dataset"], "display_name": r["display_name"]}
        for ds in test_ds_list:
            if ds in r["test_results"]:
                auc = r["test_results"][ds]["auroc"]
                row += f"{auc:>15.4f}"
                csv_row[f"test_{ds}_auroc"] = auc
                csv_row[f"test_{ds}_acc"] = r["test_results"][ds]["accuracy"]
                csv_row[f"test_{ds}_f1"] = r["test_results"][ds]["f1"]
                csv_row[f"test_{ds}_n"] = r["test_results"][ds]["n_samples"]
            else:
                row += f"{'N/A':>15}"
                csv_row[f"test_{ds}_auroc"] = None
        logger.info(row)
        rows_for_csv.append(csv_row)

    # Save JSON
    output_json = out_dir / "cross_dataset_matrix.json"
    output_json.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "results": results, "matrix_rows": rows_for_csv},
                   indent=2, allow_nan=True),
        encoding="utf-8",
    )
    logger.info(f"\nFull results saved to: {output_json}")

    # Save CSV
    import csv
    output_csv = out_dir / "cross_dataset_matrix.csv"
    if rows_for_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)
        logger.info(f"CSV saved to: {output_csv}")

    # Save pretty markdown table
    output_md = out_dir / "cross_dataset_matrix.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Cross-Dataset Generalization Matrix\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## AUROC (higher is better; >0.5 = better than chance)\n\n")
        f.write("| Model | Trained On |" + "".join(f" {ds} |" for ds in test_ds_list) + "\n")
        f.write("|" + "---|" * (2 + len(test_ds_list)) + "\n")
        for r_id in model_ids:
            r = results[r_id]
            row = f"| `{r_id}` | {r['train_dataset']} |"
            for ds in test_ds_list:
                if ds in r["test_results"]:
                    auc = r["test_results"][ds]["auroc"]
                    row += f" **{auc:.4f}** |" if auc > 0.7 else f" {auc:.4f} |"
                else:
                    row += " N/A |"
            f.write(row + "\n")
        f.write("\n\n## Accuracy\n\n")
        f.write("| Model | Trained On |" + "".join(f" {ds} |" for ds in test_ds_list) + "\n")
        f.write("|" + "---|" * (2 + len(test_ds_list)) + "\n")
        for r_id in model_ids:
            r = results[r_id]
            row = f"| `{r_id}` | {r['train_dataset']} |"
            for ds in test_ds_list:
                if ds in r["test_results"]:
                    acc = r["test_results"][ds]["accuracy"]
                    row += f" {acc:.4f} |"
                else:
                    row += " N/A |"
            f.write(row + "\n")
    logger.info(f"Markdown table saved to: {output_md}")
    logger.info(f"\nAll outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
