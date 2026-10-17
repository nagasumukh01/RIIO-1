"""
Evaluation engine for deepfake detection.
Handles within-dataset evaluation, cross-dataset evaluation, and prediction logging.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import compute_all_metrics, compute_cross_dataset_drop
from src.utils.logging_utils import get_logger

logger = get_logger("evaluator")


class Evaluator:
    """
    Model evaluator with comprehensive metric computation and prediction logging.
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
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_name: str = "test",
        experiment_id: Optional[str] = None,
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Returns comprehensive metrics + per-sample predictions.
        """
        self.model.eval()

        all_labels = []
        all_scores = []
        all_preds = []
        all_video_ids = []
        all_paths = []

        eval_start = time.time()

        for batch in dataloader:
            images = batch['image'].to(self.device)
            labels = batch['label']

            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_video_ids.extend(batch.get('video_id', [''] * len(labels)))
            all_paths.extend(batch.get('path', [''] * len(labels)))

        eval_time = time.time() - eval_start

        labels_arr = np.array(all_labels)
        scores_arr = np.array(all_scores)
        preds_arr = np.array(all_preds)

        # Compute metrics
        metrics = compute_all_metrics(labels_arr, preds_arr, scores_arr)
        metrics['eval_time_s'] = eval_time
        metrics['dataset_name'] = dataset_name

        # Compute per-sample latency
        if len(all_labels) > 0:
            metrics['avg_latency_ms'] = (eval_time / len(all_labels)) * 1000
        else:
            metrics['avg_latency_ms'] = 0

        # Video-level aggregation
        video_metrics = self._aggregate_video_level(
            all_video_ids, labels_arr, scores_arr
        )
        metrics['video_level'] = video_metrics

        # Save predictions
        if save_predictions and experiment_id:
            self._save_predictions(
                experiment_id=experiment_id,
                dataset_name=dataset_name,
                labels=all_labels,
                scores=all_scores,
                predictions=all_preds,
                video_ids=all_video_ids,
                paths=all_paths,
                metrics=metrics,
            )

        logger.info(
            f"Evaluation on {dataset_name}: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"AUROC={metrics['auroc']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"EER={metrics['eer']:.4f}"
        )

        return metrics

    def _aggregate_video_level(
        self,
        video_ids: List[str],
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> Dict:
        """Aggregate frame-level predictions to video level."""
        if not video_ids or all(v == '' for v in video_ids):
            return {}

        video_scores = {}
        video_labels = {}

        for vid, label, score in zip(video_ids, labels, scores):
            if vid not in video_scores:
                video_scores[vid] = []
                video_labels[vid] = label
            video_scores[vid].append(score)

        # Aggregate: mean score per video
        vid_labels = []
        vid_scores = []
        for vid in video_scores:
            vid_labels.append(int(video_labels[vid]))
            vid_scores.append(float(np.mean(video_scores[vid])))

        vid_preds = [1 if s >= 0.5 else 0 for s in vid_scores]

        if len(set(vid_labels)) < 2:
            return {'num_videos': len(vid_labels)}

        video_metrics = compute_all_metrics(
            np.array(vid_labels), np.array(vid_preds), np.array(vid_scores)
        )
        video_metrics['num_videos'] = len(vid_labels)
        return video_metrics

    def _save_predictions(
        self,
        experiment_id: str,
        dataset_name: str,
        labels: List,
        scores: List,
        predictions: List,
        video_ids: List,
        paths: List,
        metrics: Dict,
    ):
        """Save per-sample predictions for reproducibility."""
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_data = {
            'experiment_id': experiment_id,
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(labels),
            'predictions': [
                {
                    'label': int(l),
                    'score': float(s),
                    'prediction': int(p),
                    'video_id': v,
                    'path': pt,
                }
                for l, s, p, v, pt in zip(labels, scores, predictions, video_ids, paths)
            ],
        }

        pred_path = exp_dir / f"predictions_{dataset_name}.json"
        with open(pred_path, 'w') as f:
            json.dump(pred_data, f, indent=2)

        # Save metrics summary
        metrics_path = exp_dir / f"metrics_{dataset_name}.json"
        # Clean non-serializable items
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                clean_metrics[k] = v
        with open(metrics_path, 'w') as f:
            json.dump(clean_metrics, f, indent=2)

        logger.info(f"Saved predictions to {pred_path}")


class CrossDatasetEvaluator:
    """
    Evaluates models across dataset boundaries.
    Implements the 6-pair directional cross-dataset matrix.
    """

    def __init__(
        self,
        results_dir: str = "artifacts/experiment_logs",
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_pair(
        self,
        model: nn.Module,
        source_name: str,
        target_name: str,
        target_loader: DataLoader,
        device: str = "cpu",
        experiment_id: Optional[str] = None,
        in_domain_metrics: Optional[Dict] = None,
    ) -> Dict:
        """
        Evaluate model trained on source dataset against target dataset.
        No fine-tuning or threshold tuning on target.
        """
        evaluator = Evaluator(model, device, str(self.results_dir))

        pair_id = f"{source_name}_to_{target_name}"
        exp_id = experiment_id or f"cross_{pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Cross-dataset evaluation: {source_name} → {target_name}")

        metrics = evaluator.evaluate(
            target_loader,
            dataset_name=pair_id,
            experiment_id=exp_id,
            save_predictions=True,
        )

        # Compute generalization gap
        if in_domain_metrics:
            gap = compute_cross_dataset_drop(in_domain_metrics, metrics, 'auroc')
            metrics['generalization_gap_pct'] = gap
            logger.info(f"  Generalization gap (AUROC drop): {gap:.1f}%")

        metrics['source_dataset'] = source_name
        metrics['target_dataset'] = target_name

        return metrics

    def run_full_matrix(
        self,
        models_by_dataset: Dict[str, nn.Module],
        loaders_by_dataset: Dict[str, DataLoader],
        in_domain_metrics: Dict[str, Dict],
        device: str = "cpu",
    ) -> Dict:
        """
        Run the full 6-pair cross-dataset evaluation matrix.
        """
        datasets = list(models_by_dataset.keys())
        results = {}

        for source in datasets:
            model = models_by_dataset[source]
            model.eval()

            for target in datasets:
                if source == target:
                    continue

                pair_key = f"{source}_to_{target}"
                logger.info(f"\n{'='*50}")
                logger.info(f"Evaluating: {pair_key}")
                logger.info(f"{'='*50}")

                pair_metrics = self.evaluate_pair(
                    model=model,
                    source_name=source,
                    target_name=target,
                    target_loader=loaders_by_dataset[target],
                    device=device,
                    in_domain_metrics=in_domain_metrics.get(source),
                )

                results[pair_key] = pair_metrics

        # Save full matrix results
        matrix_path = self.results_dir / "cross_dataset_matrix.json"
        clean_results = {}
        for k, v in results.items():
            clean_results[k] = {
                mk: mv for mk, mv in v.items()
                if isinstance(mv, (int, float, str, bool, list, dict))
            }
        with open(matrix_path, 'w') as f:
            json.dump(clean_results, f, indent=2)

        logger.info(f"\nFull matrix saved to {matrix_path}")
        return results
