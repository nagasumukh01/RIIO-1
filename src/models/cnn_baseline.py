"""
CNN baseline model for deepfake detection.
EfficientNet-B0 / B4 with configurable classification head.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-based deepfake detector.
    Uses pretrained backbone with custom classification head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Load backbone via timm
        try:
            import timm
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg',
            )
            # Get feature dimension
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy)
                self.feature_dim = features.shape[-1]
        except ImportError:
            # Fallback: use torchvision
            import torchvision.models as models
            if backbone == "efficientnet_b0":
                self.backbone = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                )
            elif backbone == "efficientnet_b4":
                self.backbone = models.efficientnet_b4(
                    weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
                )
            else:
                self.backbone = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                )

            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # For Grad-CAM: store target layer reference
        self._target_layer = None

    def get_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        if self._target_layer is not None:
            return self._target_layer

        # Try to get the last convolutional layer
        try:
            import timm
            # For timm models, get the last feature block
            if hasattr(self.backbone, 'blocks'):
                self._target_layer = self.backbone.blocks[-1]
            elif hasattr(self.backbone, 'features'):
                self._target_layer = self.backbone.features[-1]
            elif hasattr(self.backbone, 'layer4'):
                self._target_layer = self.backbone.layer4[-1]
        except Exception:
            pass

        if self._target_layer is None:
            # Fallback: traverse and find last Conv2d
            for name, module in reversed(list(self.backbone.named_modules())):
                if isinstance(module, nn.Conv2d):
                    self._target_layer = module
                    break

        return self._target_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without classification."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return features

    def get_model_info(self) -> Dict:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'name': f'EfficientNet-{self.backbone_name}',
            'type': 'cnn',
            'backbone': self.backbone_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
        }
