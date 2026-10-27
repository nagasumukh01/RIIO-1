"""Evaluation-time stress tests: JPEG compression and Gaussian blur."""
from __future__ import annotations

from typing import Callable, Dict, List

import cv2
import numpy as np
import torch


def jpeg_compress(img_tensor: torch.Tensor, quality: int = 40) -> torch.Tensor:
    """img_tensor: (C, H, W) float ImageNet-normalized → re-normalized after JPEG."""
    # De-normalize approx
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img_bgr, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    t = (t - mean) / std
    return t


def gaussian_blur_tensor(img_tensor: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ksize = max(3, int(sigma * 4) | 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    t = torch.from_numpy(blurred.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = (t - mean) / std
    return t


STRESS_FN: Dict[str, Callable] = {
    "jpeg_40": lambda t: jpeg_compress(t, 40),
    "jpeg_20": lambda t: jpeg_compress(t, 20),
    "gaussian_blur_1.5": lambda t: gaussian_blur_tensor(t, 1.5),
    "gaussian_blur_3.0": lambda t: gaussian_blur_tensor(t, 3.0),
}


def apply_stress_to_batch(video: torch.Tensor, stress_name: str) -> torch.Tensor:
    """video: (B, T, C, H, W)"""
    fn = STRESS_FN[stress_name]
    B, T, C, H, W = video.shape
    out = []
    for b in range(B):
        frames = []
        for t in range(T):
            frames.append(fn(video[b, t]))
        out.append(torch.stack(frames))
    return torch.stack(out)
