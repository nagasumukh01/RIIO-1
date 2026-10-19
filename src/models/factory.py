"""
Model factory: create models from config.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from src.utils.config import ModelConfig


def create_model(config: ModelConfig, device: str = "cpu") -> nn.Module:
    """Create a model from configuration."""
    if config.type == "cnn":
        from src.models.cnn_baseline import EfficientNetDetector
        model = EfficientNetDetector(
            backbone=config.backbone,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    elif config.type == "vit":
        from src.models.vit_detector import ViTDetector
        model = ViTDetector(
            backbone=config.backbone,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    elif config.type == "hybrid":
        from src.models.vit_detector import HybridCNNViTDetector
        model = HybridCNNViTDetector(
            cnn_backbone=config.cnn_backbone or "efficientnet_b0",
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {config.type}")

    model = model.to(device)
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> Dict:
    """Load model checkpoint and return metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    metadata = {
        'epoch': checkpoint.get('epoch', -1),
        'best_metric': checkpoint.get('best_metric', None),
        'config': checkpoint.get('config', {}),
    }

    return metadata


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict,
    config: Dict,
    path: str,
):
    """Save model checkpoint with full metadata."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if hasattr(model, 'get_model_info'):
        checkpoint['model_info'] = model.get_model_info()

    torch.save(checkpoint, path)


def get_model_size(model: nn.Module) -> Dict:
    """Calculate model size information."""
    import tempfile
    import os
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate checkpoint size (Windows-compatible: delete=False + manual cleanup)
    checkpoint_size_mb = 0.0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp_path = f.name
        torch.save(model.state_dict(), tmp_path)
        checkpoint_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    except Exception:
        checkpoint_size_mb = total_params * 4 / (1024 * 1024)  # Estimate: 4 bytes per float32
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'checkpoint_size_mb': round(checkpoint_size_mb, 2),
        'parameters_formatted': f"{total_params / 1e6:.1f}M",
    }
