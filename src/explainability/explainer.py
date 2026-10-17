"""
Explainability module: Grad-CAM for CNNs, Attention Rollout for ViTs.
Frame-level heatmaps with perturbation-based faithfulness checks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import cv2
from PIL import Image

from src.utils.logging_utils import get_logger

logger = get_logger("explainability")


class GradCAM:
    """
    Grad-CAM for CNN models.
    Generates class-discriminative localization maps.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Find target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, 'get_target_layer'):
            self.target_layer = model.get_target_layer()
        else:
            # Find last Conv2d layer
            self.target_layer = self._find_last_conv(model)

        if self.target_layer is not None:
            self._register_hooks()

    def _find_last_conv(self, model: nn.Module) -> Optional[nn.Module]:
        """Find the last Conv2d layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Class to explain (default: predicted class)

        Returns:
            Heatmap numpy array [H, W] normalized to [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM: No gradients/activations captured")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # Only positive contributions
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.
    Aggregates attention across all layers to create attribution maps.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean"):
        self.model = model
        self.head_fusion = head_fusion
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture attention weights from ViT."""
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'attn_drop'):
                hook = module.register_forward_hook(self._save_attention)
                self.hooks.append(hook)
            elif isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._save_mha_attention)
                self.hooks.append(hook)

    def _save_attention(self, module, input, output):
        """Hook for timm-style attention."""
        # timm attention modules often reshape internally
        # We'll capture the output before dropout
        if isinstance(output, torch.Tensor) and output.dim() >= 3:
            self.attention_maps.append(output.detach().cpu())

    def _save_mha_attention(self, module, input, output):
        """Hook for nn.MultiheadAttention."""
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None:
                self.attention_maps.append(attn_weights.detach().cpu())

    def generate(
        self,
        input_tensor: torch.Tensor,
        discard_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Generate attention rollout map.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            discard_ratio: Fraction of lowest attention values to zero out

        Returns:
            Attention map [H, W] normalized to [0, 1]
        """
        self.attention_maps = []
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)

        if not self.attention_maps:
            logger.warning("No attention maps captured. Using fallback gradient-based method.")
            return self._gradient_fallback(input_tensor)

        # Process attention maps
        result = None
        for attention in self.attention_maps:
            if attention.dim() == 4:  # [B, H, N, N]
                if self.head_fusion == "mean":
                    attention = attention.mean(dim=1)  # [B, N, N]
                elif self.head_fusion == "max":
                    attention = attention.max(dim=1)[0]
                elif self.head_fusion == "min":
                    attention = attention.min(dim=1)[0]

            if attention.dim() == 3:
                attn = attention[0]  # Take first batch element
            elif attention.dim() == 2:
                attn = attention
            else:
                continue

            # Add identity matrix (residual connection)
            I = torch.eye(attn.size(-1))
            attn = (attn + I) / 2

            # Discard low-attention tokens
            if discard_ratio > 0:
                flat = attn.flatten()
                threshold = flat.kthvalue(int(flat.numel() * discard_ratio))[0]
                attn[attn < threshold] = 0

            # Re-normalize
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            if result is None:
                result = attn
            else:
                # Matrix multiplication for rollout
                if result.shape == attn.shape:
                    result = torch.matmul(attn, result)

        if result is None:
            return self._gradient_fallback(input_tensor)

        # Get CLS token attention to all patches
        if result.dim() == 2 and result.shape[0] > 1:
            cls_attention = result[0, 1:]  # Exclude CLS token itself
        else:
            cls_attention = result.flatten()

        # Reshape to spatial dimensions
        num_patches = cls_attention.numel()
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            grid_size = int(np.ceil(np.sqrt(num_patches)))
            # Pad if needed
            padded = torch.zeros(grid_size * grid_size)
            padded[:num_patches] = cls_attention
            cls_attention = padded

        attention_map = cls_attention.reshape(grid_size, grid_size).numpy()

        # Normalize
        if attention_map.max() > 0:
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Resize to input size
        attention_map = cv2.resize(
            attention_map,
            (input_tensor.shape[3], input_tensor.shape[2]),
            interpolation=cv2.INTER_LINEAR
        )

        return attention_map

    def _gradient_fallback(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Fallback: use input gradient magnitude as saliency."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        self.model.eval()

        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1)
        score = output[0, pred_class]

        self.model.zero_grad()
        score.backward()

        if input_tensor.grad is not None:
            saliency = input_tensor.grad.abs().mean(dim=1).squeeze().cpu().numpy()
            if saliency.max() > 0:
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            return saliency

        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create a heatmap overlay on the original image.

    Args:
        image: Original image [H, W, 3] uint8
        heatmap: Heatmap [H, W] float in [0, 1]
        alpha: Overlay transparency
        colormap: OpenCV colormap

    Returns:
        Overlay image [H, W, 3] uint8
    """
    # Ensure heatmap is same size as image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert heatmap to color
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    if image.shape[2] == 3 and heatmap_color.shape[2] == 3:
        # Convert BGR to RGB if needed
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (alpha * heatmap_color + (1 - alpha) * image).astype(np.uint8)
    return overlay


class FaithfulnessChecker:
    """
    Perturbation-based faithfulness check for explanations.
    Verifies that model predictions actually change when highlighted regions are perturbed.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def check(
        self,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        num_steps: int = 10,
    ) -> Dict:
        """
        Perform perturbation faithfulness check.

        Progressively masks the most important regions and measures prediction change.
        A faithful explanation should show prediction drop when important regions are masked.

        Returns:
            Dictionary with faithfulness score and per-step metrics.
        """
        self.model.eval()

        # Get original prediction
        original_output = self.model(input_tensor.to(self.device))
        original_prob = torch.softmax(original_output, dim=1)
        original_pred = original_prob.max(dim=1)
        original_class = original_pred.indices.item()
        original_conf = original_pred.values.item()

        # Sort heatmap values (most important first)
        flat_heatmap = heatmap.flatten()
        sorted_indices = np.argsort(flat_heatmap)[::-1]

        step_results = []
        pixels_per_step = max(1, len(sorted_indices) // num_steps)

        for step in range(num_steps):
            # Mask top-k% most important pixels
            mask_count = min((step + 1) * pixels_per_step, len(sorted_indices))
            mask_indices = sorted_indices[:mask_count]

            # Create masked input
            masked_input = input_tensor.clone()
            for idx in mask_indices:
                y = idx // heatmap.shape[1]
                x = idx % heatmap.shape[1]
                if y < masked_input.shape[2] and x < masked_input.shape[3]:
                    masked_input[0, :, y, x] = 0  # Zero out pixel

            # Get masked prediction
            masked_output = self.model(masked_input.to(self.device))
            masked_prob = torch.softmax(masked_output, dim=1)
            masked_conf = masked_prob[0, original_class].item()

            step_results.append({
                'mask_fraction': mask_count / len(sorted_indices),
                'original_confidence': original_conf,
                'masked_confidence': masked_conf,
                'confidence_drop': original_conf - masked_conf,
            })

        # Compute faithfulness score (area under the confidence drop curve)
        drops = [r['confidence_drop'] for r in step_results]
        faithfulness_score = float(np.mean(drops))

        return {
            'faithfulness_score': faithfulness_score,
            'is_faithful': faithfulness_score > 0.05,  # Threshold
            'original_class': original_class,
            'original_confidence': original_conf,
            'steps': step_results,
        }


def explain_prediction(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_type: str = "cnn",
    original_image: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict:
    """
    Generate explanation for a single prediction.

    Returns:
        Dictionary with heatmap, overlay, prediction info, and faithfulness check.
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_conf = probs[0, pred_class].item()

    # Generate heatmap
    if model_type == "cnn":
        explainer = GradCAM(model)
        heatmap = explainer.generate(input_tensor, target_class=pred_class)
        explainer.cleanup()
    else:
        explainer = AttentionRollout(model)
        heatmap = explainer.generate(input_tensor)
        explainer.cleanup()

    # Create overlay if original image provided
    overlay = None
    if original_image is not None:
        overlay = create_heatmap_overlay(original_image, heatmap)

    # Faithfulness check
    checker = FaithfulnessChecker(model, device)
    faithfulness = checker.check(input_tensor, heatmap)

    return {
        'prediction': 'FAKE' if pred_class == 1 else 'REAL',
        'confidence': pred_conf,
        'predicted_class': pred_class,
        'heatmap': heatmap,
        'overlay': overlay,
        'faithfulness': faithfulness,
        'model_type': model_type,
    }
