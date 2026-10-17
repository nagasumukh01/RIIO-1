"""
Tests for experiment tracking and configuration.
"""

import os
import json
import tempfile
import pytest
from src.utils.config import (
    DataConfig, ModelConfig, TrainingConfig, ExperimentConfig,
    load_yaml, load_all_configs,
)
from src.experiments.tracker import ExperimentTracker


class TestConfig:
    """Test configuration loading and validation."""

    def test_data_config_defaults(self):
        """DataConfig should have sensible defaults."""
        cfg = DataConfig()
        assert cfg.face_size == 224
        assert cfg.frames_per_video == 16
        assert 0 < cfg.train_ratio < 1
        assert cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == 1.0

    def test_model_config_defaults(self):
        """ModelConfig should have sensible defaults."""
        cfg = ModelConfig()
        assert cfg.num_classes == 2
        assert cfg.input_size == 224
        assert cfg.pretrained is True

    def test_training_config_defaults(self):
        """TrainingConfig should have sensible defaults."""
        cfg = TrainingConfig()
        assert cfg.epochs > 0
        assert cfg.batch_size > 0
        assert cfg.learning_rate > 0
        assert cfg.seed == 42

    def test_config_from_dict(self):
        """Config should be creatable from dict."""
        d = {'face_size': 112, 'frames_per_video': 8}
        cfg = DataConfig.from_dict(d)
        assert cfg.face_size == 112
        assert cfg.frames_per_video == 8

    def test_config_ignores_unknown_keys(self):
        """Config should ignore unknown keys without crashing."""
        d = {'face_size': 112, 'unknown_key': 'value'}
        cfg = DataConfig.from_dict(d)
        assert cfg.face_size == 112

    def test_load_all_configs(self):
        """Loading all configs should not crash."""
        try:
            configs = load_all_configs("configs")
            assert isinstance(configs, dict)
        except FileNotFoundError:
            pytest.skip("Config directory not found")

    def test_split_ratios_sum_to_one(self):
        """Train/val/test ratios must sum to 1.0."""
        cfg = DataConfig()
        total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
        assert abs(total - 1.0) < 1e-6


class TestExperimentTracker:
    """Test experiment tracking database."""

    def setup_method(self):
        """Create temporary database for testing."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_experiments.db")
        self.tracker = ExperimentTracker(self.db_path)

    def test_create_experiment(self):
        """Creating an experiment should return a valid ID."""
        exp_id = self.tracker.create_experiment(
            name="test_exp",
            exp_type="smoke_test",
            model_name="EfficientNet-B0",
            model_type="cnn",
            source_dataset="synthetic",
        )
        assert exp_id is not None
        assert len(exp_id) > 0

    def test_get_experiment(self):
        """Should be able to retrieve created experiment."""
        exp_id = self.tracker.create_experiment(
            name="test_retrieval",
            exp_type="training",
            model_name="ViT-Base",
            model_type="vit",
            source_dataset="faceforensics",
        )

        exp = self.tracker.get_experiment(exp_id)
        assert exp is not None
        assert exp['name'] == "test_retrieval"
        assert exp['model_name'] == "ViT-Base"

    def test_log_metrics(self):
        """Should be able to log and retrieve metrics."""
        exp_id = self.tracker.create_experiment(
            name="metric_test",
            exp_type="training",
            model_name="test",
            model_type="cnn",
            source_dataset="synthetic",
        )

        self.tracker.log_metrics(
            exp_id,
            {'accuracy': 0.85, 'auroc': 0.92, 'f1_score': 0.87},
            dataset="synthetic",
        )

        metrics = self.tracker.get_metrics(exp_id)
        assert len(metrics) == 3

        metric_names = [m['metric_name'] for m in metrics]
        assert 'accuracy' in metric_names
        assert 'auroc' in metric_names

    def test_complete_experiment(self):
        """Should be able to mark experiment as completed."""
        exp_id = self.tracker.create_experiment(
            name="completion_test",
            exp_type="training",
            model_name="test",
            model_type="cnn",
            source_dataset="synthetic",
        )

        self.tracker.complete_experiment(exp_id, status="completed", duration_s=120.5)

        exp = self.tracker.get_experiment(exp_id)
        assert exp['status'] == "completed"
        assert exp['duration_s'] == 120.5

    def test_get_all_experiments(self):
        """Should list all experiments."""
        for i in range(3):
            self.tracker.create_experiment(
                name=f"exp_{i}",
                exp_type="training",
                model_name="test",
                model_type="cnn",
                source_dataset="synthetic",
            )

        all_exps = self.tracker.get_all_experiments()
        assert len(all_exps) == 3

    def test_filter_by_type(self):
        """Should filter experiments by type."""
        self.tracker.create_experiment(
            name="train_exp", exp_type="training",
            model_name="test", model_type="cnn", source_dataset="s",
        )
        self.tracker.create_experiment(
            name="smoke_exp", exp_type="smoke_test",
            model_name="test", model_type="cnn", source_dataset="s",
        )

        training_exps = self.tracker.get_all_experiments("training")
        assert len(training_exps) == 1
        assert training_exps[0]['name'] == "train_exp"

    def test_comparison_table(self):
        """Comparison table should include all experiments."""
        exp_id = self.tracker.create_experiment(
            name="comp_test", exp_type="training",
            model_name="EffNet", model_type="cnn", source_dataset="ff",
        )
        self.tracker.log_metrics(
            exp_id, {'accuracy': 0.9, 'auroc': 0.95}, dataset="ff",
        )

        table = self.tracker.get_comparison_table()
        assert len(table) >= 1
        assert table[0]['model'] == "EffNet"

    def test_nonexistent_experiment(self):
        """Getting a nonexistent experiment should return None."""
        exp = self.tracker.get_experiment("nonexistent_id")
        assert exp is None
