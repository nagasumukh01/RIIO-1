"""
Data preparation script.
Usage: python scripts/prepare_data.py [--dataset faceforensics|celebdf|dfdc|all]
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger

logger = setup_logger("prepare_data")


def prepare_faceforensics():
    """Prepare FaceForensics++ dataset."""
    logger.info("=" * 50)
    logger.info("FaceForensics++ Dataset Preparation")
    logger.info("=" * 50)
    logger.info("")
    logger.info("FaceForensics++ requires an academic agreement.")
    logger.info("Steps:")
    logger.info("1. Visit: https://github.com/ondyari/FaceForensics")
    logger.info("2. Fill out the Google Form to request access")
    logger.info("3. Download using the provided script")
    logger.info("4. Place videos in: data/faceforensics/")
    logger.info("   - data/faceforensics/original_sequences/youtube/c23/videos/")
    logger.info("   - data/faceforensics/manipulated_sequences/Deepfakes/c23/videos/")
    logger.info("   - data/faceforensics/manipulated_sequences/Face2Face/c23/videos/")
    logger.info("   - data/faceforensics/manipulated_sequences/FaceSwap/c23/videos/")
    logger.info("   - data/faceforensics/manipulated_sequences/NeuralTextures/c23/videos/")
    logger.info("")
    logger.info("License: Academic use only")

    os.makedirs("data/faceforensics", exist_ok=True)
    logger.info("Created: data/faceforensics/")


def prepare_celebdf():
    """Prepare Celeb-DF dataset."""
    logger.info("=" * 50)
    logger.info("Celeb-DF v2 Dataset Preparation")
    logger.info("=" * 50)
    logger.info("")
    logger.info("Steps:")
    logger.info("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
    logger.info("2. Download the dataset following the instructions")
    logger.info("3. Place videos in: data/celebdf/")
    logger.info("   - data/celebdf/Celeb-real/")
    logger.info("   - data/celebdf/Celeb-synthesis/")
    logger.info("   - data/celebdf/YouTube-real/")
    logger.info("")
    logger.info("License: Research use only")

    os.makedirs("data/celebdf", exist_ok=True)
    logger.info("Created: data/celebdf/")


def prepare_dfdc():
    """Prepare DFDC dataset."""
    logger.info("=" * 50)
    logger.info("DFDC Dataset Preparation")
    logger.info("=" * 50)
    logger.info("")
    logger.info("Steps:")
    logger.info("1. Visit: https://www.kaggle.com/c/deepfake-detection-challenge")
    logger.info("2. Accept the competition rules")
    logger.info("3. Download using Kaggle CLI:")
    logger.info("   kaggle competitions download -c deepfake-detection-challenge")
    logger.info("4. Place videos in: data/dfdc/")
    logger.info("   - data/dfdc/train_sample_videos/")
    logger.info("   - data/dfdc/metadata.json")
    logger.info("")
    logger.info("Note: Full dataset is ~500 GB. Consider using a subset.")
    logger.info("License: Kaggle competition agreement")

    os.makedirs("data/dfdc", exist_ok=True)
    logger.info("Created: data/dfdc/")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for RobustDeep")
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['faceforensics', 'celebdf', 'dfdc', 'all'],
                       help='Dataset to prepare')
    args = parser.parse_args()

    if args.dataset in ('faceforensics', 'all'):
        prepare_faceforensics()
    if args.dataset in ('celebdf', 'all'):
        prepare_celebdf()
    if args.dataset in ('dfdc', 'all'):
        prepare_dfdc()

    logger.info("")
    logger.info("=" * 50)
    logger.info("After downloading datasets, run preprocessing:")
    logger.info("  python scripts/train.py --model efficientnet_b0 --dataset <name>")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
