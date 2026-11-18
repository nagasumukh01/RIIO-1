"""
Synthetic dataset for Stage-1 smoke tests and unit tests ONLY.

CRITICAL: This module must NEVER be imported by train.py, evaluate.py,
or any research experiment runner after Stage 1. Metrics produced with
this data MUST be labelled "SYNTHETIC – NOT A RESEARCH RESULT".
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDeepfakeDataset(Dataset):
    """Tiny random face-like tensors for pipeline smoke tests."""

    def __init__(
        self,
        num_videos: int = 20,
        frames_per_video: int = 8,
        image_size: int = 224,
        seed: int = 42,
        transform: Optional[Callable] = None,
        split: str = "train",
    ):
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.transform = transform
        self.split = split
        rng = np.random.RandomState(seed + hash(split) % 10000)
        # Roughly balanced labels
        self.labels = rng.randint(0, 2, size=num_videos).tolist()
        self.video_ids = [f"synth_{split}_{i:04d}" for i in range(num_videos)]

    def __len__(self) -> int:
        return self.num_videos

    def __getitem__(self, idx: int) -> dict:
        # Generate a simple "face-like" pattern: brighter center region
        rng = np.random.RandomState(idx + 12345)
        frames = []
        for _ in range(self.frames_per_video):
            img = rng.randn(3, self.image_size, self.image_size).astype(np.float32) * 0.1
            # Bright blob in center to mimic a face
            cy, cx = self.image_size // 2, self.image_size // 2
            y, x = np.ogrid[: self.image_size, : self.image_size]
            mask = ((y - cy) ** 2 + (x - cx) ** 2) < (self.image_size // 4) ** 2
            for c in range(3):
                img[c][mask] += 0.5 + 0.2 * self.labels[idx]
            img = np.clip(img, -2.0, 2.0)
            tensor = torch.from_numpy(img)
            if self.transform:
                tensor = self.transform(tensor)
            frames.append(tensor)
        # Stack frames: (T, C, H, W)
        video = torch.stack(frames, dim=0)
        return {
            "video": video,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "video_id": self.video_ids[idx],
            "dataset": "synthetic",
            "is_synthetic": True,
        }


def collate_synthetic(batch: List[dict]) -> dict:
    videos = torch.stack([b["video"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "video": videos,
        "label": labels,
        "video_id": [b["video_id"] for b in batch],
        "dataset": "synthetic",
        "is_synthetic": True,
    }
