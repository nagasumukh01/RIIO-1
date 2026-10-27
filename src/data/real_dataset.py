"""
Real dataset loaders for FaceForensics++, Celeb-DF, DFDC.

CRITICAL RULES:
- Never silently fall back to synthetic data.
- If expected face crops are missing, raise FileNotFoundError with exact instructions.
- Used by train.py, evaluate.py, and all research runners.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.config import PROJECT_ROOT, get_data_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("robustdeep.data")


class RealFaceVideoDataset(Dataset):
    """
    Loads pre-extracted face crops organized as:
      data/faces/<dataset>/<split>/<video_id>/*.jpg  (or .png)
    Each video directory contains ordered face frames.
    A manifest JSON under data/processed/ lists video_id, label, split, path.
    """

    SUPPORTED = ("faceforensics", "celebdf", "dfdc")

    def __init__(
        self,
        dataset: str,
        split: str = "train",
        frames_per_video: int = 16,
        transform: Optional[Callable] = None,
        max_videos: Optional[int] = None,
        faces_root: Optional[str | Path] = None,
        require_faces: bool = True,
    ):
        if dataset not in self.SUPPORTED:
            raise ValueError(f"Unknown dataset '{dataset}'. Supported: {self.SUPPORTED}")
        self.dataset = dataset
        self.split = split
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.max_videos = max_videos

        cfg = get_data_config()
        self.faces_root = Path(faces_root) if faces_root else PROJECT_ROOT / cfg["paths"]["faces_root"]
        self.processed_root = PROJECT_ROOT / cfg["paths"]["processed_root"]
        self.image_size = cfg["sampling"]["face_size"]

        self.samples = self._load_manifest_or_scan()
        if len(self.samples) == 0:
            self._raise_missing_faces()

        if max_videos is not None and len(self.samples) > max_videos:
            self.samples = self.samples[:max_videos]

        logger.info(
            "Loaded %d videos for dataset=%s split=%s (max_videos=%s)",
            len(self.samples),
            dataset,
            split,
            max_videos,
        )

    def _manifest_path(self) -> Path:
        return self.processed_root / self.dataset / f"{self.split}_manifest.json"

    def _load_manifest_or_scan(self) -> List[Dict[str, Any]]:
        man = self._manifest_path()
        if man.exists():
            with open(man, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", [])
            # Verify paths still exist
            valid = []
            for s in samples:
                p = Path(s["path"])
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                if p.exists() and (any(p.glob("*.jpg")) or any(p.glob("*.png"))):
                    s["path"] = str(p)
                    valid.append(s)
            if valid:
                return valid
            logger.warning("Manifest exists but no valid face directories found; rescanning.")

        # Scan directory structure
        split_dir = self.faces_root / self.dataset / self.split
        if not split_dir.exists():
            return []

        samples = []
        for video_dir in sorted(split_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frames = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
            if len(frames) == 0:
                continue
            # Infer label from parent name or video_id prefix
            label = self._infer_label(video_dir.name)
            samples.append(
                {
                    "video_id": video_dir.name,
                    "path": str(video_dir),
                    "label": label,
                    "num_frames": len(frames),
                }
            )
        return samples

    def _infer_label(self, video_id: str) -> int:
        vid = video_id.lower()
        if any(k in vid for k in ("real", "original", "pristine", "authentic")):
            return 0
        if any(k in vid for k in ("fake", "deepfake", "manip", "synth", "df_", "fs_", "f2f", "nt_")):
            return 1
        # Default heuristic: if starts with 0_ or real_ → real
        if vid.startswith("0_") or vid.startswith("real"):
            return 0
        return 1

    def _raise_missing_faces(self) -> None:
        expected = self.faces_root / self.dataset / self.split
        msg = (
            f"\n{'='*70}\n"
            f"REAL FACE CROPS NOT FOUND for dataset='{self.dataset}' split='{self.split}'.\n"
            f"Expected directory: {expected}\n\n"
            f"This is a RESEARCH pipeline. Synthetic data is FORBIDDEN for training/evaluation.\n\n"
            f"To fix:\n"
            f"  1. Place raw videos under data/raw/{self.dataset}/ following docs/dataset_protocol.md\n"
            f"  2. Run: python scripts/prepare_data.py --dataset {self.dataset}\n"
            f"  3. Run: python scripts/preprocess_faces.py --dataset {self.dataset} --max-videos 100\n"
            f"  4. Re-run training.\n\n"
            f"See docs/REAL_DATA_QUICKSTART.md for the exact 4-command flow.\n"
            f"{'='*70}\n"
        )
        raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, video_dir: Path) -> List[Image.Image]:
        frames = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
        if len(frames) == 0:
            raise FileNotFoundError(f"No frames in {video_dir}")
        # Uniform temporal sampling
        if len(frames) <= self.frames_per_video:
            indices = list(range(len(frames)))
            # Pad by repeating last frame
            while len(indices) < self.frames_per_video:
                indices.append(indices[-1])
        else:
            indices = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int).tolist()
        images = []
        for i in indices:
            img = Image.open(frames[i]).convert("RGB")
            images.append(img)
        return images

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        video_dir = Path(sample["path"])
        images = self._load_frames(video_dir)
        tensors = []
        for img in images:
            if self.transform is not None:
                # Albumentations or torchvision-style
                try:
                    # Albumentations expects numpy HWC
                    arr = np.array(img)
                    out = self.transform(image=arr)
                    t = out["image"]
                    if isinstance(t, np.ndarray):
                        t = torch.from_numpy(t).permute(2, 0, 1).float()
                    tensors.append(t)
                except TypeError:
                    # torchvision transform
                    tensors.append(self.transform(img))
            else:
                # Default: to tensor + ImageNet norm
                arr = np.array(img).astype(np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                t = (t - mean) / std
                tensors.append(t)
        video = torch.stack(tensors, dim=0)  # (T, C, H, W)
        return {
            "video": video,
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "video_id": sample["video_id"],
            "dataset": self.dataset,
            "is_synthetic": False,
        }


def collate_real(batch: List[dict]) -> dict:
    videos = torch.stack([b["video"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "video": videos,
        "label": labels,
        "video_id": [b["video_id"] for b in batch],
        "dataset": batch[0]["dataset"],
        "is_synthetic": False,
    }


def get_real_dataset(
    dataset: str,
    split: str = "train",
    frames_per_video: int = 16,
    transform: Optional[Callable] = None,
    max_videos: Optional[int] = None,
) -> RealFaceVideoDataset:
    """Factory used by train.py / evaluate.py. Raises FileNotFoundError if faces missing."""
    return RealFaceVideoDataset(
        dataset=dataset,
        split=split,
        frames_per_video=frames_per_video,
        transform=transform,
        max_videos=max_videos,
        require_faces=True,
    )
