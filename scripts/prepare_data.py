#!/usr/bin/env python3
"""
Create the expected folder structure for real datasets and print instructions.
Does NOT download any data (license restrictions).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_data_config, ensure_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Prepare RobustDeep data directories")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "faceforensics", "celebdf", "dfdc"])
    parser.add_argument("--max-videos", type=int, default=None, help="Document the intended small-subset size")
    args = parser.parse_args()

    cfg = get_data_config()
    raw_root = ROOT / cfg["paths"]["raw_root"]
    faces_root = ROOT / cfg["paths"]["faces_root"]
    processed_root = ROOT / cfg["paths"]["processed_root"]

    datasets = ["faceforensics", "celebdf", "dfdc"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        entry = cfg["datasets"][ds]
        methods = entry["methods"]
        for m in methods:
            ensure_dir(raw_root / entry["raw_subdir"] / m)
        for split in ("train", "val", "test"):
            ensure_dir(faces_root / entry["faces_subdir"] / split)
        ensure_dir(processed_root / ds)

    print("\n" + "=" * 70)
    print("RobustDeep data directories created.")
    print("=" * 70)
    print(f"\nRaw videos go under:  {raw_root}/<dataset>/...")
    print(f"Face crops will go under: {faces_root}/<dataset>/(train|val|test)/<video_id>/")
    print("\nOnly these research datasets are supported: faceforensics, celebdf, dfdc.")
    print("Recommended starting size: 50-200 videos per dataset, balanced across real/fake classes.")
    print("\nOfficial acquisition and license instructions:")
    print("  FaceForensics++: https://github.com/ondyari/FaceForensics")
    print("    Request access and accept the academic/research agreement; do not redistribute videos.")
    print("    Kaggle discovery: kaggle datasets list -s faceforensics")
    print("  Celeb-DF v2: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("    Request access under the dataset's research-use terms; do not redistribute videos.")
    print("    Kaggle discovery: kaggle datasets list -s celeb-df")
    print("  DFDC: https://ai.facebook.com/datasets/dfdc/")
    print("    Use the competition/dataset terms and download only a legal research subset.")
    print("    Kaggle command: kaggle competitions download -c deepfake-detection-challenge")
    print("\nNext steps:")
    print("  1. Place licensed videos into the matching data/raw/<dataset>/real-or-method folder.")
    print("  2. Run preprocess_faces.py for each dataset with --max-videos 100.")
    print("  3. Train with train.py; real checkpoints are stored under artifacts/models/<experiment_id>/.")
    print("\nIf you skip step 2–3, train.py will raise a clear FileNotFoundError (no synthetic fallback).")
    print("See docs/REAL_DATA_QUICKSTART.md for the full 4-command flow.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
