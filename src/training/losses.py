"""Loss functions for ablation studies."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss(name: str = "bce", pos_weight: float | None = None, focal_gamma: float = 2.0) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma)
    raise ValueError(f"Unknown loss: {name}. Choose from bce, weighted_bce, focal.")
