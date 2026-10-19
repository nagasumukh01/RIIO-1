"""
Run diffusion-generated face manipulation evaluation.

Usage:
    # Status check + pipeline validation (no real data needed):
    python scripts/run_diffusion_eval.py

    # With a trained checkpoint:
    python scripts/run_diffusion_eval.py --checkpoint artifacts/checkpoints/best_cnn.pt

    # With a ViT model:
    python scripts/run_diffusion_eval.py --model-type vit --backbone vit_small_patch16_224

    # Disable synthetic fallback (fail clearly when data is missing):
    python scripts/run_diffusion_eval.py --no-synthetic-fallback

Status: PENDING — DATA ACCESS LIMITED
See src/experiments/diffusion_eval.py for data format requirements.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger
from src.utils.config import ModelConfig, get_device
from src.models.factory import create_model, load_checkpoint
from src.experiments.diffusion_eval import DiffusionEvaluator

logger = setup_logger("run_diffusion_eval", log_dir="artifacts/experiment_logs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detector on diffusion-generated face manipulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_diffusion_eval.py
  python scripts/run_diffusion_eval.py --checkpoint artifacts/checkpoints/best.pt
  python scripts/run_diffusion_eval.py --model-type vit --backbone vit_small_patch16_224
        """,
    )
    parser.add_argument(
        "--checkpoint", "-k",
        default=None,
        help="Path to trained model checkpoint (.pt). If None, uses random weights.",
    )
    parser.add_argument(
        "--model-type", "-m",
        choices=["cnn", "vit", "hybrid"],
        default="cnn",
        help="Model architecture type (default: cnn)",
    )
    parser.add_argument(
        "--backbone", "-b",
        default="efficientnet_b0",
        help="Backbone name (default: efficientnet_b0)",
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Compute device: auto, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/diffusion",
        help="Path to diffusion evaluation data (default: data/diffusion)",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Disable synthetic pipeline validation when real data is unavailable",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("RobustDeep: Diffusion-Generated Face Manipulation Evaluation")
    logger.info("STATUS: PENDING — DATA ACCESS LIMITED (see docs for setup)")
    logger.info("=" * 80)

    device = get_device(args.device)
    logger.info(f"Device: {device}")

    # Build model
    config = ModelConfig(
        type=args.model_type,
        backbone=args.backbone,
        pretrained=False,
    )
    model = create_model(config, device)
    logger.info(
        f"Model: {args.model_type} / {args.backbone}  "
        f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)"
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        meta = load_checkpoint(model, args.checkpoint, device)
        logger.info(f"  Epoch: {meta.get('epoch', '?')}  "
                    f"Best metric: {meta.get('best_metric', '?')}")
    else:
        if args.checkpoint:
            logger.warning(f"Checkpoint not found: {args.checkpoint}")
        logger.warning(
            "Using untrained model — predictions will be random. "
            "Provide --checkpoint for meaningful results."
        )

    # Run evaluation
    evaluator = DiffusionEvaluator(
        model=model,
        device=device,
        data_dir=args.data_dir,
    )
    evaluator.generate_report()

    result = evaluator.evaluate(
        use_synthetic_fallback=not args.no_synthetic_fallback,
        batch_size=args.batch_size,
    )

    # Print summary
    logger.info("\n" + "=" * 65)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 65)
    logger.info(f"Status: {result.get('status', 'UNKNOWN')}")

    if result.get("warning"):
        logger.warning(result["warning"])

    if result.get("metrics"):
        m = result["metrics"]
        logger.info("\nResults on DIFFUSION-GENERATED data:")
        logger.info(f"  Accuracy:  {m.get('accuracy', 0):.4f}")
        logger.info(f"  AUROC:     {m.get('auroc', 0):.4f}")
        logger.info(f"  EER:       {m.get('eer', 0):.4f}")
        logger.info(f"  Precision: {m.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {m.get('recall', 0):.4f}")
        logger.info(f"  F1:        {m.get('f1_score', 0):.4f}")
        logger.info(f"  FPR:       {m.get('fpr', 0):.4f}")
        logger.info(f"  FNR:       {m.get('fnr', 0):.4f}")

    if result.get("pipeline_validation"):
        pv = result["pipeline_validation"]
        m = pv.get("metrics", {})
        logger.info(f"\nPipeline validation (SYNTHETIC, NOT research results):")
        logger.info(f"  {pv.get('note', '')}")
        logger.info(f"  Accuracy: {m.get('accuracy', 0):.4f}")
        logger.info(f"  AUROC:    {m.get('auroc', 0):.4f}")

    logger.info("\nWARNING: This is a probabilistic research system.")
    logger.info("  Results must NOT be treated as conclusive forensic evidence.")


if __name__ == "__main__":
    main()
