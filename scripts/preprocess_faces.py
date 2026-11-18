#!/usr/bin/env python3
"""
Extract and align faces from raw videos. Resumable. Supports --max-videos.
Requires OpenCV. Uses a simple detector cascade + optional MTCNN/RetinaFace if installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def collect_repository_videos(dataset: str) -> list[dict]:
    """Read datasets stored in the repository's native archive layout."""
    datasets_root = PROJECT_ROOT / "datasets"
    videos = []

    if dataset == "dfdc":
        dfdc_root = datasets_root / "DFDC"
        for part_dir in sorted(dfdc_root.glob("dfdc_train_part_*")):
            metadata_path = part_dir / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"DFDC metadata file not found: {metadata_path}")
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            for filename, entry in metadata.items():
                video_path = part_dir / filename
                if not video_path.is_file():
                    logger.warning("DFDC video listed in metadata but not found: %s", video_path)
                    continue
                label_name = str(entry.get("label", "")).upper()
                if label_name not in {"REAL", "FAKE"}:
                    raise ValueError(f"Unexpected DFDC label {label_name!r} for {video_path}")
                original = entry.get("original") or filename
                videos.append({
                    "path": video_path,
                    "method": label_name.lower(),
                    "label": int(label_name == "FAKE"),
                    "video_id": f"{part_dir.name}_{video_path.stem}",
                    "group_id": f"{part_dir.name}_{Path(original).stem}",
                })
        if videos:
            return videos

    if dataset == "faceforensics":
        ff_root = datasets_root / "FaceForensics++"
        original_root = ff_root / "original_sequences"
        if original_root.exists():
            for video_path in sorted(original_root.glob("*/c23/videos/*.mp4")):
                videos.append({
                    "path": video_path,
                    "method": "original",
                    "label": 0,
                    "video_id": f"original_{video_path.stem}",
                    "group_id": f"original_{video_path.stem}",
                })
            manipulated_root = ff_root / "manipulated_sequences"
            method_names = {
                "Deepfakes": "deepfakes",
                "Face2Face": "face2face",
                "FaceSwap": "faceswap",
                "NeuralTextures": "neuraltextures",
            }
            for source_name, method in method_names.items():
                method_root = manipulated_root / source_name
                for video_path in sorted(method_root.glob("c23/videos/*.mp4")):
                    videos.append({
                        "path": video_path,
                        "method": method,
                        "label": 1,
                        "video_id": f"{method}_{video_path.stem}",
                        "group_id": f"{method}_{video_path.stem}",
                    })
            if videos:
                return videos

    return []


def split_videos(videos: list[dict], ratios: list[float], seed: int, max_videos: int | None):
    """Split related source videos together and balance capped subsets by label."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for item in videos:
        groups[item.get("group_id", item["video_id"])].append(item)

    labels = sorted({item["label"] for item in videos})
    label_targets = {
        label: np.asarray(
            [sum(item["label"] == label for item in videos) * ratio for ratio in ratios],
            dtype=float,
        )
        for label in labels
    }
    group_ids = list(groups)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda group_id: len(groups[group_id]), reverse=True)
    splits = [[], [], []]
    split_label_counts = [defaultdict(int) for _ in splits]
    for group_id in group_ids:
        group_counts = defaultdict(int)
        for item in groups[group_id]:
            group_counts[item["label"]] += 1
        deficits = [
            sum(
                min(group_counts[label], max(0, label_targets[label][index] - split_label_counts[index][label]))
                for label in labels
            )
            for index in range(len(splits))
        ]
        split_index = int(np.argmax(deficits))
        splits[split_index].extend(groups[group_id])
        for label, count in group_counts.items():
            split_label_counts[split_index][label] += count

    results = {}
    for split_name, split_ratio, samples in zip(("train", "val", "test"), ratios, splits):
        rng.shuffle(samples)
        if max_videos is not None and len(samples) > 0:
            limit = max(1, round(max_videos * split_ratio))
            by_label = defaultdict(list)
            for item in samples:
                by_label[item["label"]].append(item)
            labels = sorted(by_label)
            per_label = max(1, limit // max(1, len(labels)))
            selected = []
            for label in labels:
                selected.extend(by_label[label][:per_label])
            samples = selected
        results[split_name] = samples
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["faceforensics", "celebdf", "dfdc", "diffusion"])
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

    all_videos = collect_repository_videos(args.dataset)
    if not all_videos and not raw_root.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_root}\n"
            f"Run: python scripts/prepare_data.py --dataset {args.dataset}\n"
            f"Then place videos into the created folders. See docs/dataset_protocol.md"
        )

    if not all_videos:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        for method_dir in sorted(raw_root.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            for vp in sorted(method_dir.rglob("*")):
                if vp.suffix.lower() in video_exts:
                    label = 0 if method.lower() in ("original", "real") else 1
                    all_videos.append({
                        "path": vp,
                        "method": method,
                        "label": label,
                        "video_id": f"{method}_{vp.stem}",
                    })

    if len(all_videos) == 0:
        raise FileNotFoundError(
            f"No video files found under {raw_root}.\n"
            f"Place .mp4/.avi videos into subfolders (e.g. original/, deepfakes/ or real/, fake/)."
        )

    splits = split_videos(all_videos, args.split_ratios, args.seed, args.max_videos)
    logger.info("Found %d source videos; selected %d for face extraction",
                len(all_videos), sum(map(len, splits.values())))

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
        logger.info("Split %s: %d videos with faces -> %s", split_name, len(manifest_samples), man_path)

    print("\nPreprocessing complete. You can now run:")
    print(f"  python scripts/train.py --dataset {args.dataset} --model efficientnet_b0 --epochs 5")


if __name__ == "__main__":
    main()
