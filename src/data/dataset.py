"""
Dataset classes for deepfake detection.
Supports real datasets (FF++, Celeb-DF, DFDC) and synthetic smoke-test data.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import cv2

from src.utils.config import DataConfig


class DeepfakeFrameDataset(Dataset):
    """
    Frame-level deepfake detection dataset.
    Each sample is a face crop image with a binary label (0=real, 1=fake).
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transform=None,
        face_size: int = 224,
    ):
        """
        Args:
            samples: List of dicts with keys: 'path', 'label', 'video_id', 'frame_idx'
            transform: torchvision/albumentations transform
            face_size: Target face crop size
        """
        self.samples = samples
        self.transform = transform
        self.face_size = face_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']

        # Load image
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.face_size, self.face_size), Image.LANCZOS)
            image = np.array(image)
        else:
            # Fallback: generate blank image if path doesn't exist
            image = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image'] if isinstance(transformed, dict) else transformed

        # Convert to tensor if not already
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = torch.tensor(sample['label'], dtype=torch.long)

        return {
            'image': image,
            'label': label,
            'video_id': sample.get('video_id', ''),
            'frame_idx': sample.get('frame_idx', 0),
            'path': img_path,
        }


class SyntheticDataset(Dataset):
    """
    Synthetic smoke-test dataset for pipeline validation.
    Generates colored rectangles with pattern differences for real vs fake.
    NEVER use these metrics as research results.
    """

    def __init__(
        self,
        num_samples: int = 200,
        face_size: int = 224,
        transform=None,
        seed: int = 42,
        split: str = 'train',
    ):
        self.num_samples = num_samples
        self.face_size = face_size
        self.transform = transform
        self.rng = np.random.RandomState(seed + hash(split) % 10000)

        # Generate deterministic samples
        self.labels = self.rng.randint(0, 2, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        rng = np.random.RandomState(idx * 1000 + label)

        # Create synthetic "face" image
        image = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)

        if label == 0:  # "Real" - smooth gradients, natural skin tones
            base_r = rng.randint(180, 230)
            base_g = rng.randint(140, 190)
            base_b = rng.randint(120, 170)
            for y in range(self.face_size):
                for x in range(self.face_size):
                    image[y, x] = [
                        min(255, base_r + rng.randint(-10, 10)),
                        min(255, base_g + rng.randint(-10, 10)),
                        min(255, base_b + rng.randint(-10, 10)),
                    ]
            # Add smooth ellipse (face shape)
            cy, cx = self.face_size // 2, self.face_size // 2
            cv2.ellipse(image, (cx, cy), (cx - 30, cy - 20), 0, 0, 360,
                       (base_r - 20, base_g - 20, base_b - 20), -1)
        else:  # "Fake" - sharper edges, color artifacts, grid patterns
            base_r = rng.randint(160, 220)
            base_g = rng.randint(120, 180)
            base_b = rng.randint(100, 160)
            image[:] = [base_r, base_g, base_b]
            # Add checkerboard artifact pattern (simulating compression artifacts)
            block = 16
            for y in range(0, self.face_size, block):
                for x in range(0, self.face_size, block):
                    if (y // block + x // block) % 2 == 0:
                        noise = rng.randint(10, 30)
                        image[y:y+block, x:x+block] += noise
            # Add sharp boundary artifact
            cv2.rectangle(image, (20, 20), (self.face_size-20, self.face_size-20),
                         (base_r + 40, base_g - 30, base_b + 20), 3)
            # Add color blending artifact at nose area
            cv2.circle(image, (self.face_size//2, self.face_size//2), 25,
                      (base_r + 60, base_g, base_b - 40), -1)

        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image'] if isinstance(transformed, dict) else transformed

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': image,
            'label': label_tensor,
            'video_id': f'synthetic_{idx}',
            'frame_idx': 0,
            'path': f'synthetic_{idx}.png',
        }


def create_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Create a weighted random sampler for class-balanced training."""
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.append(sample['label'].item() if isinstance(sample['label'], torch.Tensor)
                     else sample['label'])

    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
    balanced: bool = True,
) -> Dict[str, DataLoader]:
    """Create data loaders for train/val/test splits."""
    loaders = {}

    if balanced:
        sampler = create_weighted_sampler(train_dataset)
        loaders['train'] = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )
    else:
        loaders['train'] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        )

    loaders['val'] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return loaders
