"""
Run cross-dataset evaluation matrix.

Usage:
    # Using synthetic data (for pipeline validation only):
    python scripts/run_cross_dataset.py --synthetic

    # Using real datasets (after downloading and preprocessing):
    python scripts/run_cross_dataset.py --datasets faceforensics celebdf dfdc

    # Single pair only:
    python scripts/run_cross_dataset.py --source faceforensics --target celebdf

    # Using a pre-trained checkpoint:
    python scripts/run_cross_dataset.py --datasets faceforensics celebdf \\
        --checkpoint_dir artifacts/checkpoints/

NOTE: Cross-dataset evaluation intentionally uses NO fine-tuning on target data.
      This measures true generalization (Gap A in our research).
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger

logger = setup_logger("cross_dataset", log_dir="artifacts/experiment_logs")


def run_synthetic_demo():
    """Demonstrate cross-dataset infrastructure using synthetic data."""
    logger.info("=" * 70)
    logger.info("CROSS-DATASET DEMO - SYNTHETIC DATA")
    logger.info("(For pipeline validation only — NOT research results)")
    logger.info("=" * 70)

    from src.experiments.runner import ExperimentRunner
    runner = ExperimentRunner()

    # Stage 1: Train on "synthetic source A"
    logger.info("\nStep 1: Train model on synthetic source dataset...")
    result_source = runner.run_smoke_test("efficientnet_b0")

    # Stage 2: Simulate cross-dataset evaluation
    # (In real use, you'd load a different dataset's test set here)
    import torch
    import numpy as np
    from src.data.dataset import SyntheticDataset, create_dataloaders
    from src.evaluation.evaluator import CrossDatasetEvaluator, Evaluator
    from src.utils.config import TrainingConfig
    from src.models.factory import create_model
    from src.utils.config import ModelConfig
    from src.utils.reproducibility import set_seed

    set_seed(99)  # Different seed → different distribution

    # Simulate "target" dataset with different seed
    target_dataset = SyntheticDataset(num_samples=50, seed=9999, split="test")
    from torch.utils.data import DataLoader
    target_loader = DataLoader(target_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Re-create model (in real cross-dataset eval you'd load trained checkpoint)
    model_config = ModelConfig()
    model = create_model(model_config, "cpu")
    model.eval()

    evaluator = CrossDatasetEvaluator()
    pair_result = evaluator.evaluate_pair(
        model=model,
        source_name="synthetic_A",
        target_name="synthetic_B",
        target_loader=target_loader,
        device="cpu",
        in_domain_metrics=result_source.get("test_metrics"),
    )

    logger.info("\n" + "=" * 50)
    logger.info("Cross-Dataset Demo Result (SYNTHETIC):")
    logger.info(f"  Source: synthetic_A → Target: synthetic_B")
    logger.info(f"  AUROC: {pair_result.get('auroc', 0):.4f}")
    logger.info(f"  Accuracy: {pair_result.get('accuracy', 0):.4f}")
    if "generalization_gap_pct" in pair_result:
        logger.info(f"  Generalization Gap: {pair_result['generalization_gap_pct']:.1f}%")
    logger.info("=" * 50)
    logger.info("\n⚠ WARNING: These are SYNTHETIC demo results.")
    logger.info("  For research results, use real datasets via --datasets flag.")

    return {"synthetic_demo": pair_result}


def run_real_cross_dataset(
    datasets: list,
    checkpoint_dir: str = "artifacts/checkpoints",
    source: str = None,
    target: str = None,
):
    """
    Run cross-dataset evaluation on real datasets.

    Expects:
     - Trained model checkpoints in artifacts/checkpoints/{dataset}_best.pt
     - Face crops preprocessed in data/{dataset}/faces/
    """
    import torch
    from src.evaluation.evaluator import CrossDatasetEvaluator
    from src.utils.config import ModelConfig
    from src.models.factory import create_model, load_checkpoint
    from src.data.real_dataset import create_real_dataloaders

    ckpt_dir = Path(checkpoint_dir)

    # Load models trained on each dataset
    models_by_dataset = {}
    loaders_by_dataset = {}
    in_domain_metrics = {}

    eval_datasets = datasets if not source else [source]
    target_datasets = datasets if not target else [target]

    logger.info("Loading models...")
    for ds in eval_datasets:
        ckpt_path = ckpt_dir / f"{ds}_best.pt"
        model_config = ModelConfig()
        model = create_model(model_config, "cpu")

        if ckpt_path.exists():
            meta = load_checkpoint(model, str(ckpt_path), "cpu")
            logger.info(f"  Loaded {ds} model from {ckpt_path} (epoch {meta.get('epoch', '?')})")
        else:
            logger.warning(f"  No checkpoint for {ds} at {ckpt_path}. Using random weights.")

        model.eval()
        models_by_dataset[ds] = model

    logger.info("\nLoading test data loaders...")
    for ds in target_datasets:
        try:
            loaders = create_real_dataloaders(ds, batch_size=16, num_workers=0)
            loaders_by_dataset[ds] = loaders["test"]
            logger.info(f"  Loaded {ds} test split")
        except FileNotFoundError as e:
            logger.warning(f"  Could not load {ds}: {e}")

    if not loaders_by_dataset:
        logger.error(
            "No datasets could be loaded. Download datasets and run preprocessing first."
        )
        return {}

    # Run evaluation
    evaluator = CrossDatasetEvaluator()

    if source and target:
        # Single pair evaluation
        if source not in models_by_dataset:
            logger.error(f"No model for source dataset: {source}")
            return {}
        if target not in loaders_by_dataset:
            logger.error(f"No data loader for target dataset: {target}")
            return {}

        result = evaluator.evaluate_pair(
            model=models_by_dataset[source],
            source_name=source,
            target_name=target,
            target_loader=loaders_by_dataset[target],
            device="cpu",
        )
        logger.info(f"\n{source} → {target}: AUROC={result['auroc']:.4f}")
        return {f"{source}_to_{target}": result}
    else:
        # Full matrix
        return evaluator.run_full_matrix(
            models_by_dataset=models_by_dataset,
            loaders_by_dataset=loaders_by_dataset,
            in_domain_metrics=in_domain_metrics,
        )


def print_matrix(results: dict):
    """Pretty-print the cross-dataset AUROC matrix."""
    if not results:
        return

    datasets = set()
    for key in results:
        parts = key.split("_to_")
        if len(parts) == 2:
            datasets.add(parts[0])
            datasets.add(parts[1])

    datasets = sorted(datasets)
    if not datasets:
        return

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-DATASET AUROC MATRIX (Train → Test)")
    logger.info("=" * 60)

    # Header
    header = f"{'Train\\Test':20}" + "".join(f"{d[:10]:12}" for d in datasets)
    logger.info(header)
    logger.info("-" * len(header))

    for src in datasets:
        row = f"{src[:20]:20}"
        for tgt in datasets:
            if src == tgt:
                row += f"{'---':12}"
            else:
                key = f"{src}_to_{tgt}"
                auroc = results.get(key, {}).get("auroc", None)
                row += f"{auroc:.4f}      " if auroc is not None else f"{'N/A':12}"
        logger.info(row)

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation matrix")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run demo with synthetic data (no real datasets needed)")
    parser.add_argument("--datasets", nargs="+",
                        choices=["faceforensics", "celebdf", "dfdc"],
                        default=None,
                        help="Datasets to use in the evaluation matrix")
    parser.add_argument("--source", type=str, default=None,
                        help="Source dataset (train) for single-pair evaluation")
    parser.add_argument("--target", type=str, default=None,
                        help="Target dataset (test) for single-pair evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/checkpoints",
                        help="Directory containing trained model checkpoints")
    parser.add_argument("--output", type=str,
                        default="artifacts/experiment_logs/cross_dataset_matrix.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("RobustDeep: Cross-Dataset Evaluation Matrix")
    logger.info("Protocol: NO target fine-tuning or threshold tuning")
    logger.info("=" * 70)

    if args.synthetic:
        results = run_synthetic_demo()
    elif args.datasets or (args.source and args.target):
        datasets = args.datasets or [args.source, args.target]
        datasets = list(dict.fromkeys(datasets))  # deduplicate
        results = run_real_cross_dataset(
            datasets=datasets,
            checkpoint_dir=args.checkpoint_dir,
            source=args.source,
            target=args.target,
        )
    else:
        logger.info("No mode specified. Running synthetic demo (use --datasets for real data).")
        results = run_synthetic_demo()

    print_matrix(results)

    # Save to JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def serialize(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'item'):
            return obj.item()
        return str(obj)

    with open(args.output, "w") as f:
        json.dump(serialize(results), f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")
    logger.info(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
