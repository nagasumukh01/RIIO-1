"""
Main training script.
Usage: python scripts/train.py --model efficientnet_b0 --dataset synthetic --epochs 10
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_all_configs, ModelConfig, TrainingConfig, get_device
from src.utils.reproducibility import set_seed
from src.utils.logging_utils import setup_logger
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.factory import create_model, get_model_size
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.experiments.tracker import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'efficientnet_b4', 'vit_small', 'vit_base'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Configuration directory')

    args = parser.parse_args()

    # Setup
    logger = setup_logger("train", log_dir="artifacts/experiment_logs")
    configs = load_all_configs(args.config_dir)

    training_config = configs.get('training', TrainingConfig())
    if args.epochs:
        training_config.epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.lr:
        training_config.learning_rate = args.lr
    training_config.seed = args.seed

    set_seed(args.seed)
    device = get_device(training_config.device)

    logger.info(f"Training {args.model} on {args.dataset}")
    logger.info(f"Device: {device}, Epochs: {training_config.epochs}")

    # Create datasets
    if args.dataset == "synthetic":
        train_ds = SyntheticDataset(200, seed=args.seed, split='train')
        val_ds = SyntheticDataset(50, seed=args.seed + 1, split='val')
        test_ds = SyntheticDataset(50, seed=args.seed + 2, split='test')
    else:
        logger.error(f"Dataset '{args.dataset}' not available. Use 'synthetic' or prepare real data.")
        return

    loaders = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=training_config.batch_size,
        num_workers=0, balanced=True,
    )

    # Create model
    model_config = configs.get('models', {}).get(args.model, ModelConfig())
    model = create_model(model_config, device)
    size_info = get_model_size(model)
    logger.info(f"Model size: {size_info['parameters_formatted']}")

    # Track experiment
    tracker = ExperimentTracker()
    exp_id = tracker.create_experiment(
        name=f"train_{args.model}_{args.dataset}",
        exp_type="training",
        model_name=model_config.name,
        model_type=model_config.type,
        source_dataset=args.dataset,
        config=vars(training_config),
        seed=args.seed,
        device=device,
    )

    # Train
    trainer = Trainer(model, training_config, device, exp_id)
    result = trainer.train(loaders['train'], loaders['val'])

    # Evaluate
    evaluator = Evaluator(model, device)
    test_metrics = evaluator.evaluate(
        loaders['test'], dataset_name=f"{args.dataset}_test",
        experiment_id=exp_id,
    )

    tracker.log_metrics(exp_id, test_metrics, dataset=args.dataset)
    tracker.complete_experiment(exp_id, status="completed",
                              duration_s=result.get('total_time_s'))

    logger.info(f"\nFinal Results:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  AUROC:     {test_metrics['auroc']:.4f}")
    logger.info(f"  F1:        {test_metrics['f1_score']:.4f}")
    logger.info(f"  EER:       {test_metrics['eer']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
