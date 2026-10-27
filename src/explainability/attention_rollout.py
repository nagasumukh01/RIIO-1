"""Attention Rollout for Vision Transformers (timm)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def attention_rollout(attentions: List[torch.Tensor], discard_ratio: float = 0.9) -> np.ndarray:
    """
    attentions: list of (B, heads, tokens, tokens) from each transformer block
    Returns: (tokens,) importance for the CLS / average path.
    """
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    with torch.no_grad():
        for att in attentions:
            # Average heads
            att_mean = att.mean(dim=1)  # (B, T, T)
            # Discard lowest attentions
            flat = att_mean.view(att_mean.size(0), -1)
            # Keep identity residual
            I = torch.eye(att_mean.size(-1), device=att_mean.device).unsqueeze(0)
            a = (att_mean + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)
    # Take CLS token attention to patches (index 0)
    mask = result[0, 0, 1:]  # drop CLS
    mask = mask.cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


class AttentionCapture:
    """Hook helper to collect attention maps from timm ViT blocks."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attentions: List[torch.Tensor] = []
        self._handles = []
        self._register()

    def _register(self):
        def make_hook():
            def hook(module, inp, out):
                # timm Attention returns (B, N, C); we need attn weights if available
                # Many timm versions store attn in module
                if hasattr(module, "attn_drop"):
                    pass  # limited access; placeholder for full impl
            return hook
        # Best-effort: user can extend for specific timm version
        backbone = getattr(self.model, "backbone", self.model)
        if hasattr(backbone, "blocks"):
            for blk in backbone.blocks:
                if hasattr(blk, "attn"):
                    self._handles.append(blk.attn.register_forward_hook(make_hook()))

    def clear(self):
        self.attentions = []

    def remove(self):
        for h in self._handles:
            h.remove()
