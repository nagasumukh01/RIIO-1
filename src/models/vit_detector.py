"""
Vision Transformer model for deepfake detection.
ViT-Small/16 and ViT-Base/16 with configurable classification head.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class ViTDetector(nn.Module):
    """
    Vision Transformer-based deepfake detector.
    Uses pretrained ViT backbone with custom classification head.
    Supports attention extraction for explainability.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1,
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
            )
            # Get feature dimension
            if hasattr(self.backbone, 'embed_dim'):
                self.feature_dim = self.backbone.embed_dim
            elif hasattr(self.backbone, 'num_features'):
                self.feature_dim = self.backbone.num_features
            else:
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 224, 224)
                    self.feature_dim = self.backbone(dummy).shape[-1]

            self._use_timm = True
        except ImportError:
            # Fallback: use torchvision ViT
            import torchvision.models as models
            if "base" in backbone:
                self.backbone = models.vit_b_16(
                    weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
                )
                self.feature_dim = self.backbone.heads.head.in_features
            else:
                self.backbone = models.vit_b_16(
                    weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
                )
                self.feature_dim = self.backbone.heads.head.in_features

            self.backbone.heads = nn.Identity()
            self._use_timm = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        # Attention storage for explainability
        self._attention_weights = []
        self._hooks = []

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        self._clear_hooks()

        if self._use_timm:
            for block in self.backbone.blocks:
                hook = block.attn.attn_drop.register_forward_hook(
                    self._attention_hook
                )
                self._hooks.append(hook)
        else:
            for layer in self.backbone.encoder.layers:
                hook = layer.self_attention.register_forward_hook(
                    self._attention_hook
                )
                self._hooks.append(hook)

    def _attention_hook(self, module, input, output):
        """Hook function to capture attention weights."""
        if isinstance(output, tuple):
            self._attention_weights.append(output[1] if len(output) > 1 else output[0])
        else:
            self._attention_weights.append(output)

    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_weights = []

    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from all transformer layers.
        Useful for attention rollout visualization.
        """
        self._attention_weights = []
        self._register_attention_hooks()

        with torch.no_grad():
            _ = self.forward(x)

        attention_maps = self._attention_weights.copy()
        self._clear_hooks()
        return attention_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=1)  # Global average pooling
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without classification."""
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=1)
        return features

    def get_model_info(self) -> Dict:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'name': f'ViT-{self.backbone_name}',
            'type': 'vit',
            'backbone': self.backbone_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'uses_timm': self._use_timm,
        }


class HybridCNNViTDetector(nn.Module):
    """
    Hybrid CNN-ViT detector.
    Uses CNN as feature extractor and ViT for classification.
    """

    def __init__(
        self,
        cnn_backbone: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        patch_size: int = 7,
        vit_depth: int = 4,
        vit_heads: int = 4,
        vit_dim: int = 256,
    ):
        super().__init__()

        # CNN feature extractor
        try:
            import timm
            self.cnn = timm.create_model(
                cnn_backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=[-1],
            )
            # Get feature map size
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                feats = self.cnn(dummy)
                cnn_feat_dim = feats[-1].shape[1]
                spatial_size = feats[-1].shape[2]
        except ImportError:
            import torchvision.models as models
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.cnn = base.features
            cnn_feat_dim = 1280
            spatial_size = 7

        # Project CNN features to ViT dimension
        self.projection = nn.Conv2d(cnn_feat_dim, vit_dim, kernel_size=1)

        # Positional embedding
        num_patches = spatial_size * spatial_size
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, vit_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=vit_heads,
            dim_feedforward=vit_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Dropout(p=dropout),
            nn.Linear(vit_dim, num_classes),
        )

        self.feature_dim = vit_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features
        if isinstance(self.cnn, nn.Sequential):
            cnn_feats = self.cnn(x)
        else:
            cnn_feats = self.cnn(x)
            if isinstance(cnn_feats, list):
                cnn_feats = cnn_feats[-1]

        # Project and reshape to patches
        projected = self.projection(cnn_feats)  # [B, dim, H, W]
        B, C, H, W = projected.shape
        patches = projected.flatten(2).transpose(1, 2)  # [B, num_patches, dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        # Add positional embedding
        if x.shape[1] <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, :x.shape[1]]

        # Transformer
        x = self.transformer(x)

        # Classify using CLS token
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        return logits

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'name': 'Hybrid-CNN-ViT',
            'type': 'hybrid',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
        }
