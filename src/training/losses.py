"""
Loss functions for deepfake detection.
Supports cross-entropy, focal loss, and weighted BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard-to-classify examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    Applies class weights to handle imbalance.
    """

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities for the positive class
        if inputs.dim() == 2 and inputs.shape[1] == 2:
            probs = F.softmax(inputs, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(inputs.squeeze())

        targets_float = targets.float()

        if self.pos_weight is not None:
            weight = torch.where(targets_float == 1, self.pos_weight, 1.0)
        else:
            weight = torch.ones_like(targets_float)

        bce = F.binary_cross_entropy(probs, targets_float, weight=weight)
        return bce


def create_loss_function(
    loss_type: str = "cross_entropy",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    class_weights: Optional[list] = None,
    device: str = "cpu",
) -> nn.Module:
    """Create loss function from config."""
    if loss_type == "cross_entropy":
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
            return nn.CrossEntropyLoss(weight=weight)
        return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    elif loss_type == "weighted_bce":
        pos_weight = class_weights[1] / class_weights[0] if class_weights else 2.0
        return WeightedBCELoss(pos_weight=pos_weight)

    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
