"""
Real dataset loaders for FaceForensics++, Celeb-DF, and DFDC.

These loaders assume datasets have been downloaded and preprocessed.
Run `python scripts/prepare_data.py` for download instructions.

Directory structure expected:
    data/
    ├── faceforensics/
    │   ├── original_sequences/youtube/c23/videos/*.mp4
    │   └── manipulated_sequences/Deepfakes/c23/videos/*.mp4
    ├── celebdf/
    │   ├── Celeb-real/*.mp4
    │   ├── Celeb-synthesis/*.mp4
    │   └── YouTube-real/*.mp4
    └── dfdc/
        ├── train_sample_videos/*.mp4
        └── metadata.json

Can also use pre-extracted face crops in:
    data/{dataset}/faces/{video_id}/frame_{n:04d}.jpg
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.logging_utils import get_logger

logger = get_logger("real_dataset")


# ─── base ────────────────────────────────────────────────────────────────────


class BaseVideoDataset(Dataset):
    """
    Base class for video deepfake datasets.
    Loads pre-extracted frame-level face crops.
    Enforces video-level train/val/test splits (no frame leakage).
    """

    # Subclasses must override
    DATASET_NAME: str = "base"
    DATA_DIR: str = "data/base"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        frames_per_video: int = 10,
        face_size: int = 224,
        transform: Optional[Callable] = None,
        seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        """
        Args:
            root_dir: Path to dataset root (e.g. data/faceforensics)
            split: One of 'train', 'val', 'test'
            frames_per_video: Number of frames to sample per video
            face_size: Target face crop size
            transform: Optional augmentation transform
            seed: Random seed for reproducible splitting
            train_ratio: Fraction of videos for training
            val_ratio: Fraction of videos for validation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.frames_per_video = frames_per_video
        self.face_size = face_size
        self.transform = transform
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}\n"
                f"Run `python scripts/prepare_data.py --dataset {self.DATASET_NAME}` for instructions."
            )

        # Build sample list
        self.samples: List[Dict] = []
        self._build_samples()

        logger.info(
            f"{self.DATASET_NAME} [{split}]: {len(self.samples)} frames "
            f"from {len(set(s['video_id'] for s in self.samples))} videos"
        )

    def _build_samples(self):
        """Build (frame_path, label, video_id) list respecting video-level splits."""
        raise NotImplementedError

    def _split_videos(
        self, video_ids: List[str], labels: Dict[str, int]
    ) -> List[str]:
        """
        Split videos into train/val/test at video level.
        Same video's frames never appear in different splits.
        """
        rng = random.Random(self.seed)
        all_vids = sorted(video_ids)
        rng.shuffle(all_vids)

        n = len(all_vids)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        if self.split == "train":
            return all_vids[:n_train]
        elif self.split == "val":
            return all_vids[n_train : n_train + n_val]
        else:  # test
            return all_vids[n_train + n_val :]

    def _sample_frames(self, frame_dir: Path) -> List[Path]:
        """Sample frames uniformly from a video's face crop directory."""
        frames = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        if not frames:
            return []
        if len(frames) <= self.frames_per_video:
            return frames
        indices = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        return [frames[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img_path = sample["path"]
        label = sample["label"]
        video_id = sample["video_id"]

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.face_size, self.face_size))
            img_np = np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}. Returning blank frame.")
            img_np = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)

        # Apply transform
        if self.transform:
            result = self.transform(image=img_np)
            img_np = result["image"]

        # Normalize to [0, 1]
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std

        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
            "path": str(img_path),
            "dataset": self.DATASET_NAME,
        }


# ─── FaceForensics++ ─────────────────────────────────────────────────────────


class FaceForensicsDataset(BaseVideoDataset):
    """
    FaceForensics++ dataset loader.

    Expects pre-extracted face crops at:
        data/faceforensics/faces/{real|deepfakes|face2face|faceswap|neuraltextures}/{video_id}/frame_*.jpg

    Labels: 0=real, 1=fake
    """

    DATASET_NAME = "faceforensics"
    DATA_DIR = "data/faceforensics"

    MANIPULATION_METHODS = ["deepfakes", "face2face", "faceswap", "neuraltextures"]

    def __init__(
        self,
        root_dir: str = "data/faceforensics",
        manipulation_methods: Optional[List[str]] = None,
        compression: str = "c23",
        **kwargs,
    ):
        self.manipulation_methods = manipulation_methods or self.MANIPULATION_METHODS
        self.compression = compression
        super().__init__(root_dir, **kwargs)

    def _build_samples(self):
        faces_dir = self.root_dir / "faces"
        if not faces_dir.exists():
            logger.warning(
                f"Face crops not found at {faces_dir}. "
                f"Run preprocessing first: python scripts/preprocess_faces.py --dataset faceforensics"
            )
            return

        # Collect real video IDs
        real_dir = faces_dir / "real"
        real_vids = [d.name for d in real_dir.iterdir() if d.is_dir()] if real_dir.exists() else []

        # Collect fake video IDs (across all manipulation methods)
        fake_vids = set()
        for method in self.manipulation_methods:
            method_dir = faces_dir / method
            if method_dir.exists():
                fake_vids.update(d.name for d in method_dir.iterdir() if d.is_dir())
        fake_vids = list(fake_vids)

        # Labels map
        all_vids = [(v, 0) for v in real_vids] + [(v, 1) for v in fake_vids]
        labels_map = dict(all_vids)

        # Split at video level
        split_vids = self._split_videos(
            [v for v, _ in all_vids],
            labels_map,
        )
        split_vid_set = set(split_vids)

        # Build frame samples
        for vid_id, label in all_vids:
            if vid_id not in split_vid_set:
                continue

            if label == 0:
                frame_dir = real_dir / vid_id
            else:
                # Randomly choose one method's version of this video
                for method in self.manipulation_methods:
                    frame_dir = faces_dir / method / vid_id
                    if frame_dir.exists():
                        break

            frames = self._sample_frames(frame_dir)
            for frame in frames:
                self.samples.append({
                    "path": frame,
                    "label": label,
                    "video_id": vid_id,
                })


# ─── Celeb-DF ────────────────────────────────────────────────────────────────


class CelebDFDataset(BaseVideoDataset):
    """
    Celeb-DF v2 dataset loader.

    Expects pre-extracted face crops at:
        data/celebdf/faces/{real|fake}/{video_id}/frame_*.jpg

    Labels: 0=real, 1=fake
    """

    DATASET_NAME = "celebdf"
    DATA_DIR = "data/celebdf"

    def _build_samples(self):
        faces_dir = self.root_dir / "faces"
        if not faces_dir.exists():
            logger.warning(f"Face crops not found at {faces_dir}.")
            return

        all_vids = []
        labels_map = {}

        for class_dir, label in [("real", 0), ("fake", 1)]:
            class_path = faces_dir / class_dir
            if class_path.exists():
                for vid_dir in class_path.iterdir():
                    if vid_dir.is_dir():
                        all_vids.append((vid_dir.name, label))
                        labels_map[vid_dir.name] = label

        split_vids = self._split_videos([v for v, _ in all_vids], labels_map)
        split_vid_set = set(split_vids)

        for vid_id, label in all_vids:
            if vid_id not in split_vid_set:
                continue

            class_name = "real" if label == 0 else "fake"
            frame_dir = faces_dir / class_name / vid_id
            frames = self._sample_frames(frame_dir)

            for frame in frames:
                self.samples.append({
                    "path": frame,
                    "label": label,
                    "video_id": vid_id,
                })


# ─── DFDC ────────────────────────────────────────────────────────────────────


class DFDCDataset(BaseVideoDataset):
    """
    DeepFake Detection Challenge (DFDC) dataset loader.

    Expects:
        data/dfdc/metadata.json  (video_id -> {"label": "FAKE"|"REAL"})
        data/dfdc/faces/{video_id}/frame_*.jpg

    Labels: 0=real, 1=fake
    """

    DATASET_NAME = "dfdc"
    DATA_DIR = "data/dfdc"

    def _build_samples(self):
        faces_dir = self.root_dir / "faces"
        meta_path = self.root_dir / "metadata.json"

        if not faces_dir.exists():
            logger.warning(f"Face crops not found at {faces_dir}.")
            return

        # Load labels from metadata.json
        labels_map = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for vid_name, info in meta.items():
                vid_id = Path(vid_name).stem
                label = 1 if info.get("label", "REAL") == "FAKE" else 0
                labels_map[vid_id] = label
        else:
            logger.warning(f"metadata.json not found at {meta_path}. Inferring labels from directory.")
            # Fall back: assume all are unlabeled
            for vid_dir in faces_dir.iterdir():
                if vid_dir.is_dir():
                    labels_map[vid_dir.name] = -1  # Unknown

        all_vids = [(v, l) for v, l in labels_map.items() if l != -1]
        split_vids = self._split_videos([v for v, _ in all_vids], dict(all_vids))
        split_vid_set = set(split_vids)

        for vid_id, label in all_vids:
            if vid_id not in split_vid_set:
                continue

            frame_dir = faces_dir / vid_id
            if not frame_dir.exists():
                continue

            frames = self._sample_frames(frame_dir)
            for frame in frames:
                self.samples.append({
                    "path": frame,
                    "label": label,
                    "video_id": vid_id,
                })


# ─── Dataset Registry ────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, type] = {
    "faceforensics": FaceForensicsDataset,
    "celebdf": CelebDFDataset,
    "dfdc": DFDCDataset,
}


def load_real_dataset(
    dataset_name: str,
    split: str = "train",
    **kwargs,
) -> BaseVideoDataset:
    """
    Factory function to load a real dataset by name.

    Args:
        dataset_name: One of 'faceforensics', 'celebdf', 'dfdc'
        split: 'train', 'val', or 'test'
        **kwargs: Passed to the dataset class

    Returns:
        Dataset instance
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    cls = DATASET_REGISTRY[dataset_name]
    root_dir = kwargs.pop("root_dir", cls.DATA_DIR)

    logger.info(f"Loading {dataset_name} [{split}] from {root_dir}")
    return cls(root_dir=root_dir, split=split, **kwargs)


def check_dataset_availability() -> Dict[str, bool]:
    """Check which real datasets are available on disk."""
    status = {}
    for name, cls in DATASET_REGISTRY.items():
        face_dir = Path(cls.DATA_DIR) / "faces"
        status[name] = face_dir.exists() and any(face_dir.iterdir())
    return status


def create_real_dataloaders(
    dataset_name: str,
    batch_size: int = 16,
    num_workers: int = 4,
    frames_per_video: int = 10,
    **dataset_kwargs,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for a real dataset.

    Args:
        dataset_name: Dataset name
        batch_size: Batch size
        num_workers: DataLoader workers
        frames_per_video: Frames to sample per video
        **dataset_kwargs: Extra kwargs for dataset

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = load_real_dataset(
            dataset_name,
            split=split,
            frames_per_video=frames_per_video,
            **dataset_kwargs,
        )
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
        )

    return loaders
