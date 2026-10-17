"""
Data augmentation transforms for deepfake detection.
Applied ONLY to training data. Never corrupt evaluation data.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, List
from io import BytesIO
from PIL import Image


class JpegCompression:
    """Simulate JPEG compression artifacts."""

    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        lo, hi = self.quality_range
        quality = lo if lo >= hi else int(np.random.randint(lo, hi))
        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = np.array(Image.open(buffer))
        return {'image': compressed}


class GaussianBlur:
    """Apply Gaussian blur."""

    def __init__(self, kernel_range=(3, 7), sigma_range=(0.1, 2.0)):
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        k_lo, k_hi = self.kernel_range
        kernel_vals = list(range(k_lo, k_hi + 1, 2))
        k = int(np.random.choice(kernel_vals)) if len(kernel_vals) > 1 else k_lo
        s_lo, s_hi = self.sigma_range
        sigma = float(np.random.uniform(s_lo, s_hi)) if s_lo < s_hi else s_lo
        blurred = cv2.GaussianBlur(image, (k, k), sigma)
        return {'image': blurred}


class ResizeReencode:
    """Downscale and upscale to simulate re-encoding artifacts."""

    def __init__(self, scale_range=(0.5, 0.9)):
        self.scale_range = scale_range

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        h, w = image.shape[:2]
        s_lo, s_hi = self.scale_range
        scale = float(np.random.uniform(s_lo, s_hi)) if s_lo < s_hi else s_lo
        new_h, new_w = int(h * scale), int(w * scale)
        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        return {'image': upscaled}


class ColorJitter:
    """Apply color jittering."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        # Brightness
        factor = 1 + np.random.uniform(-self.brightness, self.brightness)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Contrast
        factor = 1 + np.random.uniform(-self.contrast, self.contrast)
        mean = np.mean(image)
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        return {'image': image}


class HorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if np.random.random() < self.probability:
            image = np.fliplr(image).copy()
        return {'image': image}


class Normalize:
    """Normalize image with mean and std."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return {'image': image}


class ComposeTransforms:
    """Compose multiple transforms."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        for t in self.transforms:
            result = t(image=image, **kwargs)
            image = result['image']
        return {'image': image}


def build_augmentation_pipeline(config: Dict[str, Any], is_training: bool = True) -> ComposeTransforms:
    """
    Build augmentation pipeline from config.
    Augmentations are ONLY applied during training.
    """
    transforms = []

    if is_training:
        # JPEG compression
        jpeg_cfg = config.get('jpeg_compression', {})
        if jpeg_cfg.get('enabled', False):
            transforms.append(JpegCompression(
                quality_range=tuple(jpeg_cfg.get('quality_range', [30, 95]))
            ))

        # Gaussian blur
        blur_cfg = config.get('gaussian_blur', {})
        if blur_cfg.get('enabled', False):
            transforms.append(GaussianBlur(
                kernel_range=tuple(blur_cfg.get('kernel_range', [3, 7])),
                sigma_range=tuple(blur_cfg.get('sigma_range', [0.1, 2.0]))
            ))

        # Resize + reencode
        resize_cfg = config.get('resize_reencode', {})
        if resize_cfg.get('enabled', False):
            transforms.append(ResizeReencode(
                scale_range=tuple(resize_cfg.get('scale_range', [0.5, 0.9]))
            ))

        # Color jitter
        color_cfg = config.get('color_jitter', {})
        if color_cfg.get('enabled', False):
            transforms.append(ColorJitter(
                brightness=color_cfg.get('brightness', 0.2),
                contrast=color_cfg.get('contrast', 0.2),
                saturation=color_cfg.get('saturation', 0.2),
                hue=color_cfg.get('hue', 0.1),
            ))

        # Horizontal flip
        flip_cfg = config.get('horizontal_flip', {})
        if flip_cfg.get('enabled', True):
            transforms.append(HorizontalFlip(
                probability=flip_cfg.get('probability', 0.5)
            ))

    return ComposeTransforms(transforms)


def build_robustness_transform(transform_name: str, params: Dict) -> ComposeTransforms:
    """Build evaluation-time robustness transform for stress testing."""
    transforms = []

    if transform_name == "jpeg_compression":
        q = params.get('quality', 50)
        transforms.append(JpegCompression(quality_range=(q, q)))
    elif transform_name == "gaussian_blur":
        k = params.get('kernel_size', 5)
        s = params.get('sigma', 1.0)
        transforms.append(GaussianBlur(kernel_range=(k, k), sigma_range=(s, s)))
    elif transform_name == "resize":
        s = params.get('scale', 0.5)
        transforms.append(ResizeReencode(scale_range=(s, s)))
    elif transform_name == "combined":
        q = params.get('jpeg_quality', 50)
        k = params.get('blur_kernel', 5)
        s = params.get('resize_scale', 0.7)
        transforms.append(JpegCompression(quality_range=(q, q)))
        transforms.append(GaussianBlur(kernel_range=(k, k), sigma_range=(1.0, 1.0)))
        transforms.append(ResizeReencode(scale_range=(s, s)))

    return ComposeTransforms(transforms)
