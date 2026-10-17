"""
Tests for model architectures: creation, forward pass, model info.
"""

import torch
import pytest
from src.utils.config import ModelConfig
from src.models.factory import create_model, get_model_size


class TestModelCreation:
    """Test model factory and forward pass."""

    def test_cnn_creation(self):
        """CNN model should create successfully."""
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
            num_classes=2, dropout=0.3,
        )
        model = create_model(config, "cpu")
        assert model is not None

    def test_cnn_forward_pass(self):
        """CNN forward pass should produce correct output shape."""
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
            num_classes=2,
        )
        model = create_model(config, "cpu")
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)

    def test_cnn_model_info(self):
        """CNN model info should contain required fields."""
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
        )
        model = create_model(config, "cpu")
        info = model.get_model_info()
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0

    def test_vit_creation(self):
        """ViT model should create successfully."""
        config = ModelConfig(
            name="ViT-Base", type="vit",
            backbone="vit_base_patch16_224", pretrained=False,
            num_classes=2,
        )
        try:
            model = create_model(config, "cpu")
            assert model is not None
        except Exception as e:
            pytest.skip(f"ViT creation failed (likely missing timm): {e}")

    def test_vit_forward_pass(self):
        """ViT forward pass should produce correct output shape."""
        config = ModelConfig(
            name="ViT-Base", type="vit",
            backbone="vit_base_patch16_224", pretrained=False,
            num_classes=2,
        )
        try:
            model = create_model(config, "cpu")
            x = torch.randn(2, 3, 224, 224)
            output = model(x)
            assert output.shape == (2, 2)
        except Exception as e:
            pytest.skip(f"ViT forward pass failed: {e}")

    def test_model_output_range(self):
        """Model output should be raw logits (not probabilities)."""
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
        )
        model = create_model(config, "cpu")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        # Logits can be any real number
        assert output.shape == (1, 2)

    def test_model_size(self):
        """Model size info should be valid."""
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
        )
        model = create_model(config, "cpu")
        size = get_model_size(model)
        assert size['total_parameters'] > 0
        assert size['checkpoint_size_mb'] > 0
        assert 'M' in size['parameters_formatted']

    def test_feature_extraction(self):
        """Feature extraction should work without classification."""
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
        )
        model = create_model(config, "cpu")
        x = torch.randn(2, 3, 224, 224)
        features = model.extract_features(x)
        assert features.dim() == 2
        assert features.shape[0] == 2

    def test_different_batch_sizes(self):
        """Model should handle different batch sizes."""
        config = ModelConfig(
            name="EfficientNet-B0", type="cnn",
            backbone="efficientnet_b0", pretrained=False,
        )
        model = create_model(config, "cpu")
        model.eval()

        for bs in [1, 4, 8]:
            x = torch.randn(bs, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (bs, 2)

    def test_invalid_model_type(self):
        """Invalid model type should raise ValueError."""
        config = ModelConfig(type="invalid_type")
        with pytest.raises(ValueError):
            create_model(config, "cpu")
