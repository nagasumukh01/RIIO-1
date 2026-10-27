#!/usr/bin/env python3
"""
Extract and align faces from raw videos. Resumable. Supports --max-videos.
Requires OpenCV. Uses a simple detector cascade + optional MTCNN/RetinaFace if installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_data_config, ensure_dir, PROJECT_ROOT
from src.utils.logging_utils import setup_logger

logger = setup_logger("preprocess_faces")


def get_face_detector():
    """Try RetinaFace → MTCNN → OpenCV Haar."""
    try:
        from retinaface import RetinaFace
        logger.info("Using RetinaFace detector")
        return ("retinaface", RetinaFace)
    except Exception:
        pass
    try:
        from mtcnn import MTCNN
        logger.info("Using MTCNN detector")
        return ("mtcnn", MTCNN())
    except Exception:
        pass
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("Using OpenCV Haar cascade (fallback)")
    return ("haar", cascade)


def detect_faces(detector_type, detector, frame_bgr, min_size=64, conf=0.7):
    if detector_type == "retinaface":
        from retinaface import RetinaFace
        resp = RetinaFace.detect_faces(frame_bgr)
        boxes = []
        if isinstance(resp, dict):
            for k, v in resp.items():
                if v.get("score", 0) >= conf:
                    x1, y1, x2, y2 = v["facial_area"]
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
    if detector_type == "mtcnn":
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)
        boxes = []
        for r in results:
            if r.get("confidence", 0) >= conf:
                x, y, w, h = r["box"]
                boxes.append((max(0, x), max(0, y), w, h))
        return boxes
    # Haar
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
    return [(x, y, w, h) for (x, y, w, h) in faces]


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    detector_type,
    detector,
    max_frames: int = 32,
    face_size: int = 224,
    min_face: int = 64,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= max_frames // 2:
        return len(existing)  # already processed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s", video_path)
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, total - 1, min(max_frames * 2, total), dtype=int)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        boxes = detect_faces(detector_type, detector, frame, min_size=min_face)
        if not boxes:
            continue
        # Largest face
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        # Expand a bit
        cx, cy = x + w // 2, y + h // 2
        side = int(max(w, h) * 1.3)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(frame.shape[1], x1 + side)
        y2 = min(frame.shape[0], y1 + side)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (face_size, face_size), interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"frame_{saved:04d}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        if saved >= max_frames:
            break
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["faceforensics", "celebdf", "dfdc"])
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--frames-per-video", type=int, default=32)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = get_data_config()
    raw_root = PROJECT_ROOT / cfg["paths"]["raw_root"] / cfg["datasets"][args.dataset]["raw_subdir"]
    faces_root = PROJECT_ROOT / cfg["paths"]["faces_root"] / cfg["datasets"][args.dataset]["faces_subdir"]
    processed = PROJECT_ROOT / cfg["paths"]["processed_root"] / args.dataset
    ensure_dir(processed)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_root}\n"
            f"Run: python scripts/prepare_data.py --dataset {args.dataset}\n"
            f"Then place videos into the created folders. See docs/dataset_protocol.md"
        )

    # Collect all videos
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    all_videos = []
    for method_dir in sorted(raw_root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for vp in sorted(method_dir.rglob("*")):
            if vp.suffix.lower() in video_exts:
                label = 0 if method.lower() in ("original", "real") else 1
                all_videos.append({"path": vp, "method": method, "label": label, "video_id": f"{method}_{vp.stem}"})

    if len(all_videos) == 0:
        raise FileNotFoundError(
            f"No video files found under {raw_root}.\n"
            f"Place .mp4/.avi videos into subfolders (e.g. original/, deepfakes/ or real/, fake/)."
        )

    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_videos)
    all_videos = all_videos[: args.max_videos]
    n = len(all_videos)
    n_train = int(n * args.split_ratios[0])
    n_val = int(n * args.split_ratios[1])
    splits = {
        "train": all_videos[:n_train],
        "val": all_videos[n_train : n_train + n_val],
        "test": all_videos[n_train + n_val :],
    }

    detector_type, detector = get_face_detector()
    face_size = cfg["sampling"]["face_size"]
    min_face = cfg["sampling"]["min_face_size"]

    for split_name, videos in splits.items():
        manifest_samples = []
        for item in tqdm(videos, desc=f"Processing {split_name}"):
            out_dir = faces_root / split_name / item["video_id"]
            n_saved = extract_from_video(
                item["path"], out_dir, detector_type, detector,
                max_frames=args.frames_per_video, face_size=face_size, min_face=min_face,
            )
            if n_saved > 0:
                manifest_samples.append({
                    "video_id": item["video_id"],
                    "path": str(out_dir.relative_to(PROJECT_ROOT)),
                    "label": item["label"],
                    "num_frames": n_saved,
                    "method": item["method"],
                    "source_video": str(item["path"]),
                })
        man_path = processed / f"{split_name}_manifest.json"
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "split": split_name, "samples": manifest_samples}, f, indent=2)
        logger.info("Split %s: %d videos with faces → %s", split_name, len(manifest_samples), man_path)

    print("\nPreprocessing complete. You can now run:")
    print(f"  python scripts/train.py --dataset {args.dataset} --model efficientnet_b0 --epochs 5")


if __name__ == "__main__":
    main()
