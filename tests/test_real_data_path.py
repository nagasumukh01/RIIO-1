"""
Critical tests: real-data path must NOT fall back to synthetic.
"""
from __future__ import annotations

import sys
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real_dataset import RealFaceVideoDataset


def test_missing_faces_raises_file_not_found(tmp_path, monkeypatch):
    """If faces are absent, FileNotFoundError with clear instructions must be raised."""
    import src.data.real_dataset as real_dataset

    monkeypatch.setattr(real_dataset, "PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        RealFaceVideoDataset(dataset="dfdc", split="train", max_videos=5)
    msg = str(exc_info.value)
    assert "REAL FACE CROPS NOT FOUND" in msg
    assert "preprocess_faces.py" in msg
    assert "SYNTHETIC" in msg or "synthetic" in msg.lower()
    assert "dfdc" in msg


def test_unsupported_dataset():
    with pytest.raises(ValueError):
        RealFaceVideoDataset(dataset="nonexistent_dataset", split="train")


def test_synthetic_module_exists_but_is_separate():
    """Synthetic module must exist for Stage-1 but is not used by real loader."""
    from src.data import synthetic
    assert hasattr(synthetic, "SyntheticDeepfakeDataset")
    # Real loader must not import synthetic
    import src.data.real_dataset as rd
    assert not hasattr(rd, "SyntheticDeepfakeDataset")


def test_max_videos_selects_a_balanced_subset(tmp_path, monkeypatch):
    import src.data.real_dataset as real_dataset

    monkeypatch.setattr(real_dataset, "PROJECT_ROOT", tmp_path)
    split_dir = tmp_path / "data" / "faces" / "dfdc" / "train"
    samples = []
    for label in (0, 1):
        for index in range(10):
            video_id = f"{label}_{index}"
            video_dir = split_dir / video_id
            video_dir.mkdir(parents=True)
            (video_dir / "frame.jpg").touch()
            samples.append({
                "video_id": video_id,
                "path": str(video_dir),
                "label": label,
            })

    manifest_path = tmp_path / "data" / "processed" / "dfdc" / "train_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    dataset = RealFaceVideoDataset(dataset="dfdc", split="train", max_videos=6)

    assert len(dataset) == 6
    assert Counter(sample["label"] for sample in dataset.samples) == {0: 3, 1: 3}
