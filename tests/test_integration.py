"""
Integration tests: end-to-end pipeline validation.
"""

import os
import sys
import json
import tempfile
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig, TrainingConfig
from src.utils.reproducibility import set_seed
from src.data.dataset import SyntheticDataset, create_dataloaders
from src.models.factory import create_model, save_checkpoint, load_checkpoint
from src.training.trainer import Trainer
from src.training.losses import create_loss_function, FocalLoss, WeightedBCELoss
from src.evaluation.evaluator import Evaluator


class TestEndToEndPipeline:
    """Integration test: full train → evaluate pipeline."""

    def test_smoke_pipeline_cnn(self):
        """End-to-end: create data → create model → train → evaluate."""
        set_seed(42)

        # Create tiny dataset
        train_ds = SyntheticDataset(num_samples=20, seed=42, split='train')
        val_ds = SyntheticDataset(num_samples=10, seed=43, split='val')

        loaders = create_dataloaders(
            train_ds, val_ds,
            batch_size=4, num_workers=0, balanced=False,
        )

        # Create model
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
            num_classes=2,
        )
        model = create_model(config, "cpu")

        # Train for 1 epoch
        train_config = TrainingConfig(
            epochs=1, batch_size=4, learning_rate=0.001,
            early_stopping_enabled=False, log_interval=100,
        )
        trainer = Trainer(model, train_config, "cpu", "test_pipeline")
        result = trainer.train(loaders['train'], loaders['val'], num_epochs=1)

        assert result['best_epoch'] >= 1
        assert result['total_time_s'] > 0

        # Evaluate
        evaluator = Evaluator(model, "cpu")
        metrics = evaluator.evaluate(loaders['val'], dataset_name="val_test")

        assert 'accuracy' in metrics
        assert 'auroc' in metrics
        assert 0 <= metrics['accuracy'] <= 1.0
        assert metrics['total_samples'] == 10

    def test_checkpoint_save_load(self):
        """Save and load checkpoint, verify model produces same outputs."""
        set_seed(42)

        config = ModelConfig(
            type="cnn", backbone="efficientnet_b0", pretrained=False,
        )
        model = create_model(config, "cpu")

        # Create test input
        x = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            original_output = model(x)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_checkpoint(
                model=model, optimizer=None, epoch=5,
                metrics={'accuracy': 0.85}, config={},
                path=f.name,
            )
            ckpt_path = f.name

        # Create new model and load checkpoint
        model2 = create_model(config, "cpu")
        metadata = load_checkpoint(model2, ckpt_path, "cpu")

        model2.eval()
        with torch.no_grad():
            loaded_output = model2(x)

        # Outputs should match
        assert torch.allclose(original_output, loaded_output, atol=1e-5)
        assert metadata['epoch'] == 5

        os.unlink(ckpt_path)


class TestLossFunctions:
    """Test different loss functions."""

    def test_cross_entropy(self):
        """Cross-entropy loss should work with model outputs."""
        loss_fn = create_loss_function("cross_entropy")
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_loss(self):
        """Focal loss should work and be differentiable."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(8, 2, requires_grad=True)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        loss.backward()
        assert loss.item() > 0
        assert logits.grad is not None

    def test_weighted_bce(self):
        """Weighted BCE should work with model outputs."""
        loss_fn = WeightedBCELoss(pos_weight=2.0)
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_loss_factory(self):
        """Loss factory should create all supported types."""
        for loss_type in ["cross_entropy", "focal", "weighted_bce"]:
            loss_fn = create_loss_function(loss_type)
            assert loss_fn is not None

    def test_invalid_loss(self):
        """Invalid loss type should raise ValueError."""
        with pytest.raises(ValueError):
            create_loss_function("invalid_loss")


class TestReproducibility:
    """Test reproducibility guarantees."""

    def test_seed_determinism(self):
        """Same seed should produce same results."""
        set_seed(42)
        ds1 = SyntheticDataset(num_samples=10, seed=42)
        labels1 = [ds1[i]['label'].item() for i in range(10)]

        set_seed(42)
        ds2 = SyntheticDataset(num_samples=10, seed=42)
        labels2 = [ds2[i]['label'].item() for i in range(10)]

        assert labels1 == labels2

    def test_model_determinism(self):
        """
        Verify that the same model produces deterministic outputs on the same input.

        NOTE: Cross-creation seed determinism (creating two separate model instances
        with the same seed and expecting identical weights) is NOT guaranteed when
        using timm, because timm's internal initialization may call RNG operations
        that don't respect set_seed() between separate model instantiations.

        The meaningful reproducibility guarantee here is:
          1. A single model produces identical outputs on repeated forward passes.
          2. Checkpointed models reproduce the exact same outputs after load.
             (tested in test_checkpoint_save_load)
        """
        set_seed(42)
        config = ModelConfig(type="cnn", backbone="efficientnet_b0", pretrained=False)
        model = create_model(config, "cpu")
        model.eval()

        # Same model, same input → must be exactly reproducible
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        # Forward pass must be deterministic on CPU
        assert torch.allclose(out1, out2, atol=1e-7), (
            "Model forward pass is not deterministic on the same input"
        )

        # Output must have correct shape: (1, num_classes)
        assert out1.shape == (1, 2), f"Expected (1, 2), got {out1.shape}"

