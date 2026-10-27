from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real_dataset import RealFaceVideoDataset


def test_only_three_research_datasets_are_supported():
    assert RealFaceVideoDataset.SUPPORTED == ("faceforensics", "celebdf", "dfdc")
    for dataset in RealFaceVideoDataset.SUPPORTED:
        with pytest.raises(FileNotFoundError):
            RealFaceVideoDataset(dataset=dataset, split="train", max_videos=1)
    with pytest.raises(ValueError):
        RealFaceVideoDataset(dataset="synthetic", split="train")
    with pytest.raises(ValueError):
        RealFaceVideoDataset(dataset="diffusion", split="test")
