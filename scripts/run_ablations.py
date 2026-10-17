"""
Run ablation studies.
Usage: python scripts/run_ablations.py [--type augmentation|loss|backbone|all]
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger
from src.experiments.runner import ExperimentRunner

logger = setup_logger("ablations")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument('--type', type=str, default='all',
                       choices=['augmentation', 'loss', 'backbone', 'all'],
                       help='Type of ablation to run')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Epochs per ablation run')
    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.type in ('augmentation', 'all'):
        logger.info("\n=== AUGMENTATION ABLATION ===")
        results = runner.run_augmentation_ablation(num_epochs=args.epochs)
        for r in results:
            logger.info(f"  {r['strategy']}: AUROC={r['test_metrics']['auroc']:.4f}")

    if args.type in ('loss', 'all'):
        logger.info("\n=== LOSS FUNCTION ABLATION ===")
        results = runner.run_loss_ablation(num_epochs=args.epochs)
        for r in results:
            logger.info(f"  {r['loss_function']}: AUROC={r['test_metrics']['auroc']:.4f}")

    if args.type in ('backbone', 'all'):
        logger.info("\n=== BACKBONE CAPACITY ABLATION ===")
        results = runner.run_backbone_ablation(num_epochs=args.epochs)
        for r in results:
            logger.info(f"  {r['backbone']}: AUROC={r['test_metrics']['auroc']:.4f}, "
                       f"Params={r['model_size']['parameters_formatted']}")


if __name__ == "__main__":
    main()
