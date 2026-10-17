"""
Face preprocessing / extraction pipeline for real video datasets.
Extracts faces from videos using MTCNN or RetinaFace, saves as JPEG crops.

Usage:
    python scripts/preprocess_faces.py --dataset faceforensics
    python scripts/preprocess_faces.py --dataset celebdf
    python scripts/preprocess_faces.py --dataset dfdc --max_videos 500

Output:
    data/{dataset}/faces/{class}/{video_id}/frame_{n:04d}.jpg

Requirements:
    pip install facenet-pytorch opencv-python
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger

logger = setup_logger("preprocess_faces")


# ─── Detector ────────────────────────────────────────────────────────────────

def get_face_detector():
    """Initialize face detector (MTCNN preferred, cv2 DNN fallback)."""
    try:
        from facenet_pytorch import MTCNN
        detector = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            device="cpu",
            post_process=False,
        )
        logger.info("Using MTCNN face detector")
        return detector, "mtcnn"
    except ImportError:
        logger.warning("facenet-pytorch not installed. Falling back to center-crop.")
        return None, "center_crop"


def extract_face_mtcnn(detector, frame_bgr, target_size: int = 224):
    """Extract aligned face using MTCNN."""
    import cv2
    import numpy as np

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    img_pil = Image.fromarray(frame_rgb)

    face = detector(img_pil)
    if face is None:
        return None

    # Convert from tensor [C, H, W] in [0, 255] to numpy
    face_np = face.permute(1, 2, 0).numpy().astype("uint8")
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return face_bgr


def extract_face_center_crop(frame_bgr, target_size: int = 224):
    """Center-crop fallback when no face detector available."""
    import cv2
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame_bgr[top:top+side, left:left+side]
    return cv2.resize(crop, (target_size, target_size))


# ─── Video Processing ─────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    output_dir: Path,
    detector,
    detector_type: str,
    frames_to_extract: int = 20,
    face_size: int = 224,
    min_face_size: int = 80,
) -> int:
    """
    Extract faces from a video file.

    Returns:
        Number of successfully extracted frames
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    # Sample frame indices uniformly
    import numpy as np
    sample_indices = np.linspace(0, max(total_frames - 1, 0), frames_to_extract, dtype=int)
    saved = 0

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Extract face
        if detector_type == "mtcnn" and detector is not None:
            face = extract_face_mtcnn(detector, frame, face_size)
        else:
            face = extract_face_center_crop(frame, face_size)

        if face is None:
            continue

        out_path = output_dir / f"frame_{saved:04d}.jpg"
        cv2.imwrite(str(out_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


# ─── Dataset-specific processors ─────────────────────────────────────────────

def process_faceforensics(
    root_dir: str,
    compression: str = "c23",
    max_videos: Optional[int] = None,
    frames_per_video: int = 20,
):
    """Process FaceForensics++ dataset."""
    root = Path(root_dir)
    faces_out = root / "faces"

    detector, det_type = get_face_detector()

    manipulation_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    sources = [
        (root / "original_sequences" / "youtube" / compression / "videos", "real"),
    ]
    for method in manipulation_methods:
        sources.append(
            (root / "manipulated_sequences" / method / compression / "videos",
             method.lower())
        )

    total_processed = 0
    for video_dir, class_name in sources:
        if not video_dir.exists():
            logger.warning(f"Not found: {video_dir}")
            continue

        videos = list(video_dir.glob("*.mp4"))
        if max_videos:
            videos = videos[:max_videos]

        logger.info(f"Processing {class_name}: {len(videos)} videos")

        for video_path in videos:
            vid_id = video_path.stem
            out_dir = faces_out / class_name / vid_id
            if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= frames_per_video // 2:
                continue  # Already processed

            n = process_video(video_path, out_dir, detector, det_type, frames_per_video)
            total_processed += n

        logger.info(f"  {class_name}: done")

    logger.info(f"FaceForensics++ processing complete. {total_processed} frames extracted.")


def process_celebdf(
    root_dir: str,
    max_videos: Optional[int] = None,
    frames_per_video: int = 20,
):
    """Process Celeb-DF v2 dataset."""
    root = Path(root_dir)
    faces_out = root / "faces"

    detector, det_type = get_face_detector()

    sources = [
        (root / "Celeb-real", "real"),
        (root / "YouTube-real", "real"),
        (root / "Celeb-synthesis", "fake"),
    ]

    for video_dir, class_name in sources:
        if not video_dir.exists():
            logger.warning(f"Not found: {video_dir}")
            continue

        videos = list(video_dir.glob("*.mp4"))
        if max_videos:
            videos = videos[:max_videos]

        logger.info(f"Processing Celeb-DF [{class_name}]: {len(videos)} videos")
        for video_path in videos:
            vid_id = video_path.stem
            out_dir = faces_out / class_name / vid_id
            if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= 5:
                continue
            process_video(video_path, out_dir, detector, det_type, frames_per_video)

    logger.info("Celeb-DF processing complete.")


def process_dfdc(
    root_dir: str,
    max_videos: Optional[int] = None,
    frames_per_video: int = 20,
):
    """Process DFDC dataset."""
    root = Path(root_dir)
    faces_out = root / "faces"
    meta_path = root / "metadata.json"

    if not meta_path.exists():
        logger.error(f"metadata.json not found at {meta_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    detector, det_type = get_face_detector()

    video_dir = root / "train_sample_videos"
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return

    videos = [(video_dir / vid_name) for vid_name in meta.keys() if (video_dir / vid_name).exists()]
    if max_videos:
        videos = videos[:max_videos]

    logger.info(f"Processing DFDC: {len(videos)} videos")
    for video_path in videos:
        vid_id = video_path.stem
        out_dir = faces_out / vid_id
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= 5:
            continue
        process_video(video_path, out_dir, detector, det_type, frames_per_video)

    logger.info("DFDC processing complete.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract face crops from video datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["faceforensics", "celebdf", "dfdc"],
                        help="Dataset to process")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Dataset root directory (default: data/{dataset})")
    parser.add_argument("--compression", type=str, default="c23",
                        help="FF++ compression level (c0, c23, c40)")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to process (useful for testing)")
    parser.add_argument("--frames_per_video", type=int, default=20,
                        help="Number of frames to extract per video")
    args = parser.parse_args()

    root = args.root_dir or f"data/{args.dataset}"
    t0 = time.time()

    if args.dataset == "faceforensics":
        process_faceforensics(
            root_dir=root,
            compression=args.compression,
            max_videos=args.max_videos,
            frames_per_video=args.frames_per_video,
        )
    elif args.dataset == "celebdf":
        process_celebdf(
            root_dir=root,
            max_videos=args.max_videos,
            frames_per_video=args.frames_per_video,
        )
    elif args.dataset == "dfdc":
        process_dfdc(
            root_dir=root,
            max_videos=args.max_videos,
            frames_per_video=args.frames_per_video,
        )

    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
