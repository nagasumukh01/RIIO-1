"""
Robustness testing module.
Evaluation-time stress tests: compression, blur, resize.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.augmentation import build_robustness_transform, ComposeTransforms
from src.utils.metrics import compute_all_metrics, compute_robustness_drop
from src.utils.logging_utils import get_logger

logger = get_logger("robustness")


class RobustnessTester:
    """
    Robustness testing: evaluates model performance under various stress conditions.
    ALL perturbations are applied at evaluation time only.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        results_dir: str = "artifacts/experiment_logs",
    ):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate_with_perturbation(
        self,
        dataloader: DataLoader,
        perturbation_name: str,
        transform: ComposeTransforms,
        dataset_name: str = "test",
    ) -> Dict:
        """
        Evaluate model on perturbed data.

        The transform is applied at evaluation time to test robustness.
        This is a SEPARATE experiment from training augmentation.
        """
        self.model.eval()

        all_labels = []
        all_scores = []
        all_preds = []

        eval_start = time.time()

        for batch in dataloader:
            images = batch['image']
            labels = batch['label']

            # Apply perturbation to each image
            perturbed_images = []
            for img in images:
                img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)

                result = transform(image=img_np)
                perturbed = result['image']
                if perturbed.dtype == np.uint8:
                    perturbed = perturbed.astype(np.float32) / 255.0

                perturbed_tensor = torch.from_numpy(
                    perturbed.transpose(2, 0, 1)
                ).float()
                perturbed_images.append(perturbed_tensor)

            perturbed_batch = torch.stack(perturbed_images).to(self.device)

            logits = self.model(perturbed_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        eval_time = time.time() - eval_start

        metrics = compute_all_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_scores)
        )
        metrics['perturbation'] = perturbation_name
        metrics['eval_time_s'] = eval_time

        logger.info(
            f"Robustness [{perturbation_name}]: "
            f"Acc={metrics['accuracy']:.4f}, AUROC={metrics['auroc']:.4f}"
        )

        return metrics

    def run_stress_tests(
        self,
        dataloader: DataLoader,
        stress_configs: List[Dict],
        clean_metrics: Optional[Dict] = None,
        dataset_name: str = "test",
    ) -> Dict:
        """
        Run all configured stress tests.

        Args:
            dataloader: Test data loader
            stress_configs: List of stress test configurations
            clean_metrics: Metrics on clean (unperturbed) data for comparison
            dataset_name: Name of the dataset being tested
        """
        logger.info("=" * 50)
        logger.info("ROBUSTNESS STRESS TESTS")
        logger.info("=" * 50)

        results = {}

        # Evaluate on clean data first if not provided
        if clean_metrics is None:
            logger.info("Evaluating on clean data...")
            from src.evaluation.evaluator import Evaluator
            evaluator = Evaluator(self.model, self.device)
            clean_metrics = evaluator.evaluate(dataloader, dataset_name="clean")

        results['clean'] = clean_metrics

        # Run each stress test
        for stress_cfg in stress_configs:
            name = stress_cfg.get('name', 'unknown')
            transform_type = stress_cfg.get('transform', '')
            params = stress_cfg.get('params', {})

            logger.info(f"\nStress test: {name}")

            try:
                transform = build_robustness_transform(transform_type, params)
                perturbed_metrics = self.evaluate_with_perturbation(
                    dataloader, name, transform, dataset_name
                )

                # Compute performance drop
                drop = compute_robustness_drop(clean_metrics, perturbed_metrics, 'auroc')
                perturbed_metrics['robustness_drop_pct'] = drop
                logger.info(f"  AUROC drop: {drop:.1f}%")

                results[name] = perturbed_metrics

            except Exception as e:
                logger.error(f"Stress test {name} failed: {e}")
                results[name] = {'error': str(e)}

        # Save results
        results_path = self.results_dir / f"robustness_{dataset_name}.json"

        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()
                        if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        with open(results_path, 'w') as f:
            json.dump(clean_for_json(results), f, indent=2)

        logger.info(f"\nRobustness results saved to {results_path}")
        return results
