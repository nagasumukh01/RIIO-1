"""Grad-CAM for CNN backbones."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register()

    def _register(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        x: (1, C, H, W) single frame
        Returns heatmap (H, W) in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model.backbone(x) if hasattr(self.model, "backbone") else self.model(x)
        if logits.dim() > 1:
            score = logits[0, 0] if logits.shape[-1] == 1 else logits[0].max()
        else:
            score = logits[0]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
