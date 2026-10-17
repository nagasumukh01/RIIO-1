"""
Experiment runner: orchestrates training, evaluation, and ablation experiments.
"""

import os
import json
import time
import copy
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

import torch
import numpy as np

from src.utils.config import (
    load_all_configs, ModelConfig, TrainingConfig, DataConfig, get_device
)
from src.utils.reproducibility import set_seed
from src.utils.logging_utils import get_logger
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.factory import create_model, save_checkpoint, load_checkpoint, get_model_size
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator, CrossDatasetEvaluator
from src.experiments.tracker import ExperimentTracker

logger = get_logger("experiment_runner")


class ExperimentRunner:
    """
    Orchestrates complete experimental workflows.
    """

    def __init__(self, config_dir: str = "configs"):
        self.configs = load_all_configs(config_dir)
        self.tracker = ExperimentTracker(
            self.configs.get('experiments', {}).db_path
            if hasattr(self.configs.get('experiments', {}), 'db_path')
            else "artifacts/experiment_logs/experiments.db"
        )
        self.device = get_device(
            self.configs.get('training', TrainingConfig()).device
        )
        logger.info(f"ExperimentRunner initialized. Device: {self.device}")

    def run_smoke_test(self, model_key: str = "efficientnet_b0") -> Dict:
        """
        Stage 1: Synthetic smoke test.
        Validates the entire pipeline end-to-end with synthetic data.
        NEVER report these as research results.
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: SYNTHETIC SMOKE TEST")
        logger.info("WARNING: These results are NOT research results!")
        logger.info("=" * 60)

        training_config = self.configs.get('training', TrainingConfig())
        seed = training_config.seed
        set_seed(seed)

        # Create synthetic datasets
        train_dataset = SyntheticDataset(num_samples=100, seed=seed, split='train')
        val_dataset = SyntheticDataset(num_samples=30, seed=seed + 1, split='val')
        test_dataset = SyntheticDataset(num_samples=30, seed=seed + 2, split='test')

        loaders = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=min(16, training_config.batch_size),
            num_workers=0,
            balanced=True,
        )

        # Create model
        model_config = self.configs.get('models', {}).get(
            model_key, ModelConfig()
        )
        model = create_model(model_config, self.device)
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}

        logger.info(f"Model: {model_config.name}")
        logger.info(f"Parameters: {model_info.get('total_parameters', 'N/A')}")

        # Create experiment
        exp_id = self.tracker.create_experiment(
            name=f"smoke_test_{model_key}",
            exp_type="smoke_test",
            model_name=model_config.name,
            model_type=model_config.type,
            source_dataset="synthetic",
            config=vars(training_config),
            seed=seed,
            device=self.device,
        )

        # Train (very few epochs for smoke test)
        smoke_training_config = copy.copy(training_config)
        smoke_training_config.epochs = 3
        smoke_training_config.early_stopping_enabled = False
        smoke_training_config.log_interval = 5

        trainer = Trainer(model, smoke_training_config, self.device, exp_id)
        train_result = trainer.train(loaders['train'], loaders['val'], num_epochs=3)

        # Evaluate
        evaluator = Evaluator(model, self.device)
        test_metrics = evaluator.evaluate(
            loaders['test'], dataset_name="synthetic_test",
            experiment_id=exp_id, save_predictions=True,
        )

        # Log results
        self.tracker.log_metrics(exp_id, test_metrics, dataset="synthetic_test")
        self.tracker.complete_experiment(
            exp_id,
            status="completed",
            checkpoint_path=train_result.get('checkpoint_dir'),
            duration_s=train_result.get('total_time_s'),
        )

        # Model size info
        size_info = get_model_size(model)

        result = {
            'experiment_id': exp_id,
            'model_name': model_config.name,
            'model_info': model_info,
            'model_size': size_info,
            'training_result': train_result,
            'test_metrics': test_metrics,
            'is_smoke_test': True,
            'warning': 'SYNTHETIC DATA - NOT RESEARCH RESULTS',
        }

        logger.info(f"\nSmoke test complete!")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test AUROC: {test_metrics['auroc']:.4f}")
        logger.info(f"Model size: {size_info['parameters_formatted']}")

        return result

    def run_within_dataset_baseline(
        self,
        model_key: str = "efficientnet_b0",
        dataset_name: str = "synthetic",
        num_epochs: Optional[int] = None,
    ) -> Dict:
        """
        Train and evaluate a model within a single dataset.
        """
        training_config = self.configs.get('training', TrainingConfig())
        set_seed(training_config.seed)

        # For now, use synthetic data (real data loaders ready for when datasets are available)
        if dataset_name == "synthetic":
            train_dataset = SyntheticDataset(num_samples=200, seed=training_config.seed, split='train')
            val_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 1, split='val')
            test_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 2, split='test')
        else:
            # Placeholder for real dataset loading
            raise NotImplementedError(
                f"Real dataset '{dataset_name}' loading not yet implemented. "
                f"Place dataset in data/{dataset_name}/ and run preprocessing first."
            )

        loaders = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=training_config.batch_size,
            num_workers=0,
            balanced=True,
        )

        # Create model
        model_config = self.configs.get('models', {}).get(model_key, ModelConfig())
        model = create_model(model_config, self.device)

        # Create experiment
        exp_id = self.tracker.create_experiment(
            name=f"baseline_{model_key}_{dataset_name}",
            exp_type="within_dataset",
            model_name=model_config.name,
            model_type=model_config.type,
            source_dataset=dataset_name,
            config=vars(training_config),
            seed=training_config.seed,
            device=self.device,
        )

        # Train
        epochs = num_epochs or training_config.epochs
        trainer = Trainer(model, training_config, self.device, exp_id)
        train_result = trainer.train(loaders['train'], loaders['val'], num_epochs=epochs)

        # Evaluate
        evaluator = Evaluator(model, self.device)
        test_metrics = evaluator.evaluate(
            loaders['test'], dataset_name=f"{dataset_name}_test",
            experiment_id=exp_id, save_predictions=True,
        )

        self.tracker.log_metrics(exp_id, test_metrics, dataset=dataset_name)
        self.tracker.complete_experiment(
            exp_id,
            status="completed",
            checkpoint_path=train_result.get('checkpoint_dir'),
            duration_s=train_result.get('total_time_s'),
        )

        return {
            'experiment_id': exp_id,
            'model': model_key,
            'dataset': dataset_name,
            'training': train_result,
            'test_metrics': test_metrics,
        }

    def run_augmentation_ablation(
        self,
        model_key: str = "efficientnet_b0",
        dataset_name: str = "synthetic",
        num_epochs: int = 5,
    ) -> List[Dict]:
        """
        Gap C: Augmentation ablation study.
        Compare different augmentation strategies with identical data and training budget.
        """
        logger.info("=" * 60)
        logger.info("AUGMENTATION ABLATION STUDY")
        logger.info("=" * 60)

        ablation_config = self.configs.get('augmentation_ablation', {})
        strategies = ablation_config.get('strategies', [
            {'name': 'none', 'config': {}},
            {'name': 'jpeg_only', 'config': {'jpeg_compression': {'enabled': True, 'quality_range': [30, 95]}}},
            {'name': 'blur_only', 'config': {'gaussian_blur': {'enabled': True, 'kernel_range': [3, 7]}}},
        ])

        results = []
        for strategy in strategies:
            logger.info(f"\nAugmentation strategy: {strategy['name']}")

            training_config = self.configs.get('training', TrainingConfig())
            set_seed(training_config.seed)

            # Create datasets with different augmentation
            from src.data.augmentation import build_augmentation_pipeline
            train_transform = build_augmentation_pipeline(strategy['config'], is_training=True)

            train_dataset = SyntheticDataset(
                num_samples=200, seed=training_config.seed,
                split='train', transform=train_transform,
            )
            val_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 1, split='val')
            test_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 2, split='test')

            loaders = create_dataloaders(
                train_dataset, val_dataset, test_dataset,
                batch_size=training_config.batch_size, num_workers=0, balanced=True,
            )

            model_config = self.configs.get('models', {}).get(model_key, ModelConfig())
            model = create_model(model_config, self.device)

            exp_id = self.tracker.create_experiment(
                name=f"aug_ablation_{strategy['name']}_{model_key}",
                exp_type="augmentation_ablation",
                model_name=model_config.name,
                model_type=model_config.type,
                source_dataset=dataset_name,
                config={**vars(training_config), 'augmentation': strategy},
                seed=training_config.seed,
                device=self.device,
            )

            trainer = Trainer(model, training_config, self.device, exp_id)
            train_result = trainer.train(loaders['train'], loaders['val'], num_epochs=num_epochs)

            evaluator = Evaluator(model, self.device)
            test_metrics = evaluator.evaluate(
                loaders['test'], dataset_name=f"aug_{strategy['name']}_test",
                experiment_id=exp_id,
            )

            self.tracker.log_metrics(exp_id, test_metrics, dataset=dataset_name)
            self.tracker.complete_experiment(exp_id, status="completed",
                                           duration_s=train_result.get('total_time_s'))

            result = {
                'strategy': strategy['name'],
                'experiment_id': exp_id,
                'test_metrics': test_metrics,
                'training': train_result,
            }
            results.append(result)

            logger.info(
                f"  {strategy['name']}: Acc={test_metrics['accuracy']:.4f}, "
                f"AUROC={test_metrics['auroc']:.4f}"
            )

        return results

    def run_loss_ablation(
        self,
        model_key: str = "efficientnet_b0",
        dataset_name: str = "synthetic",
        num_epochs: int = 5,
    ) -> List[Dict]:
        """
        Gap D: Loss function ablation study.
        Compare cross-entropy, focal loss, and weighted BCE.
        """
        logger.info("=" * 60)
        logger.info("LOSS FUNCTION ABLATION STUDY")
        logger.info("=" * 60)

        loss_configs = [
            {'name': 'cross_entropy', 'loss_function': 'cross_entropy'},
            {'name': 'focal_loss', 'loss_function': 'focal'},
            {'name': 'weighted_bce', 'loss_function': 'weighted_bce'},
        ]

        results = []
        for loss_cfg in loss_configs:
            logger.info(f"\nLoss function: {loss_cfg['name']}")

            training_config = self.configs.get('training', TrainingConfig())
            training_config.loss_function = loss_cfg['loss_function']
            set_seed(training_config.seed)

            train_dataset = SyntheticDataset(num_samples=200, seed=training_config.seed, split='train')
            val_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 1, split='val')
            test_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 2, split='test')

            loaders = create_dataloaders(
                train_dataset, val_dataset, test_dataset,
                batch_size=training_config.batch_size, num_workers=0, balanced=True,
            )

            model_config = self.configs.get('models', {}).get(model_key, ModelConfig())
            model = create_model(model_config, self.device)

            exp_id = self.tracker.create_experiment(
                name=f"loss_ablation_{loss_cfg['name']}_{model_key}",
                exp_type="loss_ablation",
                model_name=model_config.name,
                model_type=model_config.type,
                source_dataset=dataset_name,
                config={**vars(training_config), 'loss': loss_cfg},
                seed=training_config.seed,
                device=self.device,
            )

            trainer = Trainer(model, training_config, self.device, exp_id)
            train_result = trainer.train(loaders['train'], loaders['val'], num_epochs=num_epochs)

            evaluator = Evaluator(model, self.device)
            test_metrics = evaluator.evaluate(
                loaders['test'], dataset_name=f"loss_{loss_cfg['name']}_test",
                experiment_id=exp_id,
            )

            self.tracker.log_metrics(exp_id, test_metrics, dataset=dataset_name)
            self.tracker.complete_experiment(exp_id, status="completed",
                                           duration_s=train_result.get('total_time_s'))

            results.append({
                'loss_function': loss_cfg['name'],
                'experiment_id': exp_id,
                'test_metrics': test_metrics,
                'training': train_result,
            })

        return results

    def run_backbone_ablation(
        self,
        dataset_name: str = "synthetic",
        num_epochs: int = 5,
    ) -> List[Dict]:
        """
        Gap E: Backbone capacity ablation.
        Compare different model architectures and sizes.
        """
        logger.info("=" * 60)
        logger.info("BACKBONE CAPACITY ABLATION STUDY")
        logger.info("=" * 60)

        backbone_configs = self.configs.get('backbone_ablation', {}).get('backbones', [
            {'name': 'efficientnet_b0', 'model_config': 'efficientnet_b0'},
            {'name': 'vit_small', 'model_config': 'vit_small'},
        ])

        results = []
        for bb_cfg in backbone_configs:
            model_key = bb_cfg['model_config']
            logger.info(f"\nBackbone: {bb_cfg['name']}")

            training_config = self.configs.get('training', TrainingConfig())
            set_seed(training_config.seed)

            train_dataset = SyntheticDataset(num_samples=200, seed=training_config.seed, split='train')
            val_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 1, split='val')
            test_dataset = SyntheticDataset(num_samples=50, seed=training_config.seed + 2, split='test')

            loaders = create_dataloaders(
                train_dataset, val_dataset, test_dataset,
                batch_size=training_config.batch_size, num_workers=0, balanced=True,
            )

            model_config = self.configs.get('models', {}).get(model_key, ModelConfig())
            model = create_model(model_config, self.device)
            model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
            size_info = get_model_size(model)

            exp_id = self.tracker.create_experiment(
                name=f"backbone_ablation_{bb_cfg['name']}",
                exp_type="backbone_ablation",
                model_name=model_config.name,
                model_type=model_config.type,
                source_dataset=dataset_name,
                config=vars(training_config),
                seed=training_config.seed,
                device=self.device,
            )

            trainer = Trainer(model, training_config, self.device, exp_id)
            train_result = trainer.train(loaders['train'], loaders['val'], num_epochs=num_epochs)

            evaluator = Evaluator(model, self.device)
            test_metrics = evaluator.evaluate(
                loaders['test'], dataset_name=f"bb_{bb_cfg['name']}_test",
                experiment_id=exp_id,
            )

            self.tracker.log_metrics(exp_id, test_metrics, dataset=dataset_name)
            self.tracker.complete_experiment(exp_id, status="completed",
                                           duration_s=train_result.get('total_time_s'))

            results.append({
                'backbone': bb_cfg['name'],
                'experiment_id': exp_id,
                'model_info': model_info,
                'model_size': size_info,
                'test_metrics': test_metrics,
                'training': train_result,
            })

        return results

    def run_all_experiments(self) -> Dict:
        """Run the complete experimental pipeline."""
        logger.info("=" * 80)
        logger.info("RUNNING COMPLETE EXPERIMENTAL PIPELINE")
        logger.info("=" * 80)

        all_results = {}

        # Stage 1: Smoke tests
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: SMOKE TESTS")
        logger.info("=" * 60)
        try:
            cnn_smoke = self.run_smoke_test("efficientnet_b0")
            all_results['cnn_smoke_test'] = cnn_smoke
        except Exception as e:
            logger.error(f"CNN smoke test failed: {e}")
            all_results['cnn_smoke_test'] = {'error': str(e)}

        try:
            vit_smoke = self.run_smoke_test("vit_base")
            all_results['vit_smoke_test'] = vit_smoke
        except Exception as e:
            logger.error(f"ViT smoke test failed: {e}")
            all_results['vit_smoke_test'] = {'error': str(e)}

        # Stage 2: Baselines
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: WITHIN-DATASET BASELINES")
        logger.info("=" * 60)
        try:
            cnn_baseline = self.run_within_dataset_baseline("efficientnet_b0", "synthetic", 5)
            all_results['cnn_baseline'] = cnn_baseline
        except Exception as e:
            logger.error(f"CNN baseline failed: {e}")
            all_results['cnn_baseline'] = {'error': str(e)}

        try:
            vit_baseline = self.run_within_dataset_baseline("vit_base", "synthetic", 5)
            all_results['vit_baseline'] = vit_baseline
        except Exception as e:
            logger.error(f"ViT baseline failed: {e}")
            all_results['vit_baseline'] = {'error': str(e)}

        # Stage 3: Ablations
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: ABLATION STUDIES")
        logger.info("=" * 60)
        try:
            aug_results = self.run_augmentation_ablation(num_epochs=3)
            all_results['augmentation_ablation'] = aug_results
        except Exception as e:
            logger.error(f"Augmentation ablation failed: {e}")
            all_results['augmentation_ablation'] = {'error': str(e)}

        try:
            loss_results = self.run_loss_ablation(num_epochs=3)
            all_results['loss_ablation'] = loss_results
        except Exception as e:
            logger.error(f"Loss ablation failed: {e}")
            all_results['loss_ablation'] = {'error': str(e)}

        try:
            bb_results = self.run_backbone_ablation(num_epochs=3)
            all_results['backbone_ablation'] = bb_results
        except Exception as e:
            logger.error(f"Backbone ablation failed: {e}")
            all_results['backbone_ablation'] = {'error': str(e)}

        # Save all results
        results_path = Path("artifacts/experiment_logs/all_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        def make_serializable(obj):
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

        with open(results_path, 'w') as f:
            json.dump(make_serializable(all_results), f, indent=2, default=str)

        # Export comparison table
        self.tracker.export_results()

        logger.info("\n" + "=" * 80)
        logger.info("ALL EXPERIMENTS COMPLETE")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 80)

        return all_results
