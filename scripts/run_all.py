"""
Run complete experiment pipeline.
Usage: python scripts/run_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger
from src.experiments.runner import ExperimentRunner

logger = setup_logger("run_all", log_dir="artifacts/experiment_logs")


def main():
    logger.info("=" * 80)
    logger.info("RobustDeep: Complete Experimental Pipeline")
    logger.info("=" * 80)

    runner = ExperimentRunner()
    results = runner.run_all_experiments()

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    for key, result in results.items():
        if isinstance(result, dict) and 'error' in result:
            logger.info(f"  {key}: FAILED - {result['error']}")
        elif isinstance(result, dict) and 'test_metrics' in result:
            m = result['test_metrics']
            logger.info(f"  {key}: Acc={m.get('accuracy', 0):.4f}, AUROC={m.get('auroc', 0):.4f}")
        elif isinstance(result, list):
            logger.info(f"  {key}: {len(result)} experiments completed")
        else:
            logger.info(f"  {key}: Completed")


if __name__ == "__main__":
    main()
