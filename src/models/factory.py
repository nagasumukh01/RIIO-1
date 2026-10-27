"""Model factory for EfficientNet and ViT backbones via timm."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required. Install with: pip install timm") from e

from src.utils.config import get_model_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("robustdeep.models")


class VideoDeepfakeClassifier(nn.Module):
    """
    Frame-level classifier that processes a video as a batch of frames
    and aggregates logits (mean by default).
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_type: str = "cnn",
        aggregation: str = "mean",
        topk: int = 5,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.aggregation = aggregation
        self.topk = topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W) or (B, C, H, W)
        Returns: (B,) logits for binary classification (real=0 / fake=1)
        """
        if x.dim() == 4:
            # Single frame
            logits = self.backbone(x)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            return logits

        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        frame_logits = self.backbone(frames)
        if frame_logits.dim() > 1:
            frame_logits = frame_logits.squeeze(-1)
        frame_logits = frame_logits.view(B, T)

        if self.aggregation == "mean":
            return frame_logits.mean(dim=1)
        if self.aggregation == "median":
            return frame_logits.median(dim=1).values
        if self.aggregation == "topk":
            k = min(self.topk, T)
            topk_vals, _ = frame_logits.topk(k, dim=1)
            return topk_vals.mean(dim=1)
        return frame_logits.mean(dim=1)

    def get_frame_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-frame logits (B, T) for explainability."""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        logits = self.backbone(frames)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        return logits.view(B, T)


def build_model(
    name: str = "efficientnet_b0",
    pretrained: bool = True,
    num_classes: int = 1,
    aggregation: str = "mean",
    topk: int = 5,
    **kwargs: Any,
) -> VideoDeepfakeClassifier:
    cfg = get_model_config()
    backbones = cfg.get("backbones", {})
    if name not in backbones:
        # Allow direct timm names
        model_name = name
        backbone_type = "vit" if "vit" in name.lower() else "cnn"
    else:
        entry = backbones[name]
        model_name = entry["name"]
        backbone_type = entry.get("type", "cnn")
        pretrained = entry.get("pretrained", pretrained)
        num_classes = entry.get("num_classes", num_classes)

    logger.info("Building backbone=%s (timm name=%s, type=%s, pretrained=%s)", name, model_name, backbone_type, pretrained)
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return VideoDeepfakeClassifier(
        backbone=backbone,
        backbone_type=backbone_type,
        aggregation=aggregation,
        topk=topk,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
