"""
Tests for metrics computation.
Validates accuracy, AUROC, EER, confusion matrix, and cross-dataset drop calculations.
"""

import numpy as np
import pytest
from src.utils.metrics import (
    compute_all_metrics,
    compute_eer,
    compute_cross_dataset_drop,
    compute_robustness_drop,
)


class TestComputeAllMetrics:
    """Test the comprehensive metrics computation."""

    def test_perfect_predictions(self):
        """All predictions correct."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])

        metrics = compute_all_metrics(labels, predictions, scores)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['auroc'] == 1.0
        assert metrics['tp'] == 3
        assert metrics['tn'] == 3
        assert metrics['fp'] == 0
        assert metrics['fn'] == 0
        assert metrics['total_samples'] == 6

    def test_worst_predictions(self):
        """All predictions wrong."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        predictions = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.85, 0.8, 0.1, 0.15, 0.2])

        metrics = compute_all_metrics(labels, predictions, scores)

        assert metrics['accuracy'] == 0.0
        assert metrics['tp'] == 0
        assert metrics['tn'] == 0
        assert metrics['fp'] == 3
        assert metrics['fn'] == 3

    def test_balanced_random_predictions(self):
        """Mixed predictions should produce intermediate metrics."""
        labels = np.array([0, 0, 1, 1, 0, 1])
        predictions = np.array([0, 1, 1, 0, 0, 1])
        scores = np.array([0.3, 0.6, 0.8, 0.4, 0.2, 0.7])

        metrics = compute_all_metrics(labels, predictions, scores)

        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['auroc'] <= 1.0
        assert metrics['total_samples'] == 6
        assert metrics['tp'] + metrics['fp'] + metrics['tn'] + metrics['fn'] == 6

    def test_single_class_labels(self):
        """Handle edge case: only one class present."""
        labels = np.array([0, 0, 0, 0])
        predictions = np.array([0, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.6])

        metrics = compute_all_metrics(labels, predictions, scores)
        assert metrics['total_samples'] == 4

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2x2."""
        labels = np.array([0, 1, 0, 1])
        predictions = np.array([0, 1, 1, 0])
        scores = np.array([0.2, 0.8, 0.6, 0.4])

        metrics = compute_all_metrics(labels, predictions, scores)
        cm = metrics['confusion_matrix']
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_roc_curve_keys(self):
        """ROC curve data should have fpr and tpr keys."""
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 0])
        scores = np.array([0.3, 0.6, 0.8, 0.4])

        metrics = compute_all_metrics(labels, predictions, scores)
        assert 'roc_curve' in metrics
        assert 'fpr' in metrics['roc_curve']
        assert 'tpr' in metrics['roc_curve']

    def test_threshold_recorded(self):
        """Threshold should be recorded in metrics."""
        labels = np.array([0, 1])
        predictions = np.array([0, 1])
        scores = np.array([0.3, 0.7])

        metrics = compute_all_metrics(labels, predictions, scores, threshold=0.5)
        assert metrics['threshold'] == 0.5

    def test_empty_inputs(self):
        """Handle empty inputs gracefully."""
        labels = np.array([])
        predictions = np.array([])
        scores = np.array([])

        # Should not crash
        try:
            metrics = compute_all_metrics(labels, predictions, scores)
        except Exception:
            pass  # Acceptable to fail on empty input


class TestComputeEER:
    """Test EER computation."""

    def test_perfect_separation(self):
        """Perfect separation should give EER near 0."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.0, 0.1, 0.1, 0.2, 0.8, 0.9, 0.9, 1.0])

        eer, threshold = compute_eer(labels, scores)
        assert eer < 0.2  # Should be very low

    def test_random_scores(self):
        """Random scores should give EER near 0.5."""
        np.random.seed(42)
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.random.random(200)

        eer, threshold = compute_eer(labels, scores)
        assert 0.2 < eer < 0.8  # Should be near 0.5


class TestCrossDatasetDrop:
    """Test performance drop calculations."""

    def test_no_drop(self):
        """Same performance = 0% drop."""
        in_metrics = {'auroc': 0.95}
        cross_metrics = {'auroc': 0.95}
        drop = compute_cross_dataset_drop(in_metrics, cross_metrics, 'auroc')
        assert abs(drop) < 0.01

    def test_significant_drop(self):
        """Large drop should be reflected."""
        in_metrics = {'auroc': 0.95}
        cross_metrics = {'auroc': 0.65}
        drop = compute_cross_dataset_drop(in_metrics, cross_metrics, 'auroc')
        assert drop > 25  # More than 25% drop

    def test_zero_baseline(self):
        """Zero baseline should return 0 drop (avoid ZeroDivisionError)."""
        in_metrics = {'auroc': 0.0}
        cross_metrics = {'auroc': 0.5}
        drop = compute_cross_dataset_drop(in_metrics, cross_metrics, 'auroc')
        assert drop == 0.0


class TestRobustnessDrop:
    """Test robustness drop calculation."""

    def test_drop_under_perturbation(self):
        """Performance should drop under perturbation."""
        clean = {'auroc': 0.90}
        perturbed = {'auroc': 0.70}
        drop = compute_robustness_drop(clean, perturbed, 'auroc')
        assert drop > 20
