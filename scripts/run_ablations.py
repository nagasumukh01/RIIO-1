"""
Run ablation studies.

Usage:
    python scripts/run_ablations.py --type augmentation --epochs 5
    python scripts/run_ablations.py --type loss --epochs 5
    python scripts/run_ablations.py --type backbone --epochs 5
    python scripts/run_ablations.py --type all --epochs 3

WARNING: Uses synthetic data by default.
Replace with real datasets for research-grade results.
"""

import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger
from src.experiments.runner import ExperimentRunner

logger = setup_logger("run_ablations", log_dir="artifacts/experiment_logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Run RobustDeep ablation studies")
    parser.add_argument(
        "--type", "-t",
        choices=["augmentation", "aug", "loss", "backbone", "bb", "all"],
        default="all",
        help="Type of ablation to run (default: all)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Number of training epochs per ablation variant (default: 3)",
    )
    parser.add_argument(
        "--model", "-m",
        default="efficientnet_b0",
        help="Model key from models.yaml for aug/loss ablations (default: efficientnet_b0)",
    )
    parser.add_argument(
        "--config-dir", "-c",
        default="configs",
        help="Path to config directory (default: configs)",
    )
    return parser.parse_args()


def make_serializable(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info(f"RobustDeep: Ablation Study — type={args.type}, epochs={args.epochs}")
    logger.info("=" * 80)
    logger.info("⚠ WARNING: Synthetic data is used by default.")
    logger.info("  Replace with real datasets (FF++, Celeb-DF, DFDC) for research results.")

    runner = ExperimentRunner(config_dir=args.config_dir)
    ablation_type = args.type
    results = {}

    # ── Augmentation ablation ─────────────────────────────────────────────────
    if ablation_type in ("augmentation", "aug", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("AUGMENTATION ABLATION (Gap C)")
        logger.info("=" * 60)
        try:
            aug_results = runner.run_augmentation_ablation(
                model_key=args.model,
                num_epochs=args.epochs,
            )
            results["augmentation"] = aug_results
            logger.info(f"✓ Augmentation ablation: {len(aug_results)} variants completed")
            for r in aug_results:
                m = r.get("test_metrics", {})
                logger.info(
                    f"  [{r.get('strategy', '?')}] "
                    f"Acc={m.get('accuracy', 0):.4f}  AUROC={m.get('auroc', 0):.4f}"
                )
        except Exception as e:
            logger.error(f"Augmentation ablation failed: {e}", exc_info=True)
            results["augmentation"] = {"error": str(e)}

    # ── Loss function ablation ────────────────────────────────────────────────
    if ablation_type in ("loss", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("LOSS FUNCTION ABLATION (Gap D)")
        logger.info("=" * 60)
        try:
            loss_results = runner.run_loss_ablation(
                model_key=args.model,
                num_epochs=args.epochs,
            )
            results["loss"] = loss_results
            logger.info(f"✓ Loss ablation: {len(loss_results)} variants completed")
            for r in loss_results:
                m = r.get("test_metrics", {})
                logger.info(
                    f"  [{r.get('loss_function', '?')}] "
                    f"Acc={m.get('accuracy', 0):.4f}  AUROC={m.get('auroc', 0):.4f}"
                )
        except Exception as e:
            logger.error(f"Loss ablation failed: {e}", exc_info=True)
            results["loss"] = {"error": str(e)}

    # ── Backbone capacity ablation ────────────────────────────────────────────
    if ablation_type in ("backbone", "bb", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("BACKBONE CAPACITY ABLATION (Gap E)")
        logger.info("=" * 60)
        try:
            bb_results = runner.run_backbone_ablation(num_epochs=args.epochs)
            results["backbone"] = bb_results
            logger.info(f"✓ Backbone ablation: {len(bb_results)} variants completed")
            for r in bb_results:
                m = r.get("test_metrics", {})
                info = r.get("model_info", {})
                size = r.get("model_size", {})
                logger.info(
                    f"  [{r.get('backbone', '?')}] "
                    f"Acc={m.get('accuracy', 0):.4f}  AUROC={m.get('auroc', 0):.4f}  "
                    f"Params={size.get('parameters_formatted', '?')}"
                )
        except Exception as e:
            logger.error(f"Backbone ablation failed: {e}", exc_info=True)
            results["backbone"] = {"error": str(e)}

    # ── Save combined results ─────────────────────────────────────────────────
    out_path = Path("artifacts/experiment_logs/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)

    logger.info(f"\n✓ Ablation results saved to {out_path}")
    logger.info("DONE")

    # Print compact summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    for abl_type, abl_result in results.items():
        if isinstance(abl_result, list):
            logger.info(f"  {abl_type}: {len(abl_result)} variants")
            for r in abl_result:
                if isinstance(r, dict) and "test_metrics" in r:
                    m = r["test_metrics"]
                    key = r.get("strategy") or r.get("loss_function") or r.get("backbone", "?")
                    logger.info(f"    {key}: AUROC={m.get('auroc', 0):.4f}")
        elif isinstance(abl_result, dict) and "error" in abl_result:
            logger.info(f"  {abl_type}: FAILED — {abl_result['error']}")


if __name__ == "__main__":
    main()
