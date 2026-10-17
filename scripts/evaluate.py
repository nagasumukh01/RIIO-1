"""
Evaluation script.
Usage: python scripts/evaluate.py --checkpoint <path> --dataset <name> --model <type>
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger
from src.utils.config import ModelConfig
from src.utils.reproducibility import set_seed
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.factory import create_model, load_checkpoint
from src.evaluation.evaluator import Evaluator

logger = setup_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       help='Dataset to evaluate on')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Create model
    model_configs = {
        'efficientnet_b0': ModelConfig(type="cnn", backbone="efficientnet_b0", pretrained=False),
        'efficientnet_b4': ModelConfig(type="cnn", backbone="efficientnet_b4", pretrained=False),
        'vit_small': ModelConfig(type="vit", backbone="vit_small_patch16_224", pretrained=False),
        'vit_base': ModelConfig(type="vit", backbone="vit_base_patch16_224", pretrained=False),
    }

    config = model_configs.get(args.model, model_configs['efficientnet_b0'])
    model = create_model(config, "cpu")

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        metadata = load_checkpoint(model, args.checkpoint, "cpu")
        logger.info(f"Loaded checkpoint from epoch {metadata.get('epoch', '?')}")

    # Create dataset
    if args.dataset == "synthetic":
        test_ds = SyntheticDataset(num_samples=100, seed=args.seed + 2, split='test')
    else:
        logger.error(f"Dataset '{args.dataset}' not available. Using synthetic.")
        test_ds = SyntheticDataset(num_samples=100, seed=args.seed + 2, split='test')

    loaders = create_dataloaders(
        test_ds, test_ds, test_ds,
        batch_size=args.batch_size, num_workers=0, balanced=False,
    )

    # Evaluate
    evaluator = Evaluator(model, "cpu")
    metrics = evaluator.evaluate(loaders['test'], dataset_name=args.dataset)

    logger.info(f"\nResults on {args.dataset}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        elif isinstance(v, int):
            logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
