"""
Critical tests: real-data path must NOT fall back to synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real_dataset import RealFaceVideoDataset


def test_missing_faces_raises_file_not_found():
    """If faces are absent, FileNotFoundError with clear instructions must be raised."""
    with pytest.raises(FileNotFoundError) as exc_info:
        RealFaceVideoDataset(dataset="faceforensics", split="train", max_videos=5)
    msg = str(exc_info.value)
    assert "REAL FACE CROPS NOT FOUND" in msg
    assert "preprocess_faces.py" in msg
    assert "SYNTHETIC" in msg or "synthetic" in msg.lower()
    assert "faceforensics" in msg


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
