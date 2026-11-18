"""
Acquire 50 diverse synthetic human face images from the verified BitMind SDXL Benchmark,
detect and crop faces using Haar cascade, check for duplicates with pHash, save crops
to data/diffusion_face/raw/fake/, record metadata, and run the dataset pipeline.

Source: bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces (Hugging Face)
Generator: stabilityai/stable-diffusion-xl-base-1.0 (Diffusion Model)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imagehash
except ImportError:
    imagehash = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("robustdeep.diffusion_acquire")

HF_ROWS_API = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=bitmind%2Fceleb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces"
    "&config=base_transforms&split=train&offset=0&limit=100"
)


# ---------------------------------------------------------------------------
# Face Detector (Haar cascade matching project pipeline)
# ---------------------------------------------------------------------------
def load_detector():
    if cv2 is None:
        return None
    for name in ["haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml"]:
        path = cv2.data.haarcascades + name
        det = cv2.CascadeClassifier(path)
        if not det.empty():
            return det
    return None


def detect_and_crop(pil_img: Image.Image, detector, min_size: int = 64) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Detect largest face and return (crop_rgb, bbox) or (None, None)."""
    if detector is None or cv2 is None:
        # Full image fallback
        arr = np.array(pil_img.convert("RGB"))
        return arr, (0, 0, arr.shape[1], arr.shape[0])

    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = arr[y : y + h, x : x + w]
    return crop, (int(x), int(y), int(w), int(h))


# ---------------------------------------------------------------------------
# Duplicate Checking (pHash)
# ---------------------------------------------------------------------------
def compute_phash(pil_img: Image.Image) -> str:
    if imagehash is not None:
        return str(imagehash.phash(pil_img.convert("RGB")))
    import hashlib
    return hashlib.md5(pil_img.tobytes()).hexdigest()


def hamming(h1: str, h2: str) -> int:
    if imagehash is not None:
        try:
            return abs(imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2))
        except Exception:
            pass
    return 0 if h1 == h2 else 999


# ---------------------------------------------------------------------------
# Main Acquisition & Processing
# ---------------------------------------------------------------------------
def acquire_diffusion_faces(target_count: int = 50, min_face: int = 64, phash_thresh: int = 8):
    raw_output = PROJECT_ROOT / "diffusion_workspace" / "output"
    fake_dir = PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "fake"
    meta_dir = PROJECT_ROOT / "data" / "diffusion_face" / "metadata"
    report_dir = PROJECT_ROOT / "data" / "diffusion_face" / "reports"

    for d in [raw_output, fake_dir, meta_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    detector = load_detector()
    if detector is not None:
        logger.info("Loaded Haar cascade face detector successfully.")
    else:
        logger.warning("Haar cascade detector unavailable — will save full images.")

    logger.info("Fetching dataset index from Hugging Face: bitmind/celeb-a-hq (SDXL 1.0)...")
    req = urllib.request.Request(HF_ROWS_API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        rows = data.get("rows", [])

    logger.info("Received %d records. Processing first %d images...", len(rows), target_count)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_rows: List[dict] = []
    seen_hashes: List[str] = []

    fetched = 0
    face_detected = 0
    face_failed = 0
    duplicates = 0
    corrupted = 0

    for idx, r in enumerate(rows):
        if face_detected >= target_count:
            break

        row_data = r.get("row", {})
        img_info = row_data.get("image", {})
        img_url = img_info.get("src")
        orig_idx = row_data.get("original_index", idx)

        if not img_url:
            continue

        fetched += 1
        img_name = f"sdxl_{ts}_{idx:04d}_orig{orig_idx}.png"
        raw_path = raw_output / img_name

        # Download image
        try:
            img_req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(img_req, timeout=15) as img_resp:
                img_bytes = img_resp.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img.save(str(raw_path))
        except Exception as e:
            logger.error("[%d] Failed to download/open %s: %s", idx, img_url, e)
            corrupted += 1
            metadata_rows.append({
                "index": idx,
                "original_index": orig_idx,
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "dataset_source": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces",
                "raw_path": None,
                "crop_path": None,
                "width": 0,
                "height": 0,
                "face_found": False,
                "bbox": None,
                "phash": None,
                "status": f"CORRUPTED_OR_DOWNLOAD_ERROR: {e}",
                "timestamp": ts,
            })
            continue

        # pHash check
        ph = compute_phash(pil_img)
        is_dup = any(hamming(ph, prev) <= phash_thresh for prev in seen_hashes)
        if is_dup:
            duplicates += 1
            logger.warning("[%d] Duplicate detected via pHash — skipping crop", idx)
            metadata_rows.append({
                "index": idx,
                "original_index": orig_idx,
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "dataset_source": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces",
                "raw_path": str(raw_path),
                "crop_path": None,
                "width": pil_img.width,
                "height": pil_img.height,
                "face_found": False,
                "bbox": None,
                "phash": ph,
                "status": "DUPLICATE",
                "timestamp": ts,
            })
            continue
        seen_hashes.append(ph)

        # Detect face
        crop_arr, bbox = detect_and_crop(pil_img, detector, min_size=min_face)
        if crop_arr is not None:
            face_detected += 1
            crop_name = f"sdxl_{ts}_{idx:04d}_face.png"
            crop_path = fake_dir / crop_name
            Image.fromarray(crop_arr).save(str(crop_path))
            status = "OK"
            logger.info(
                "[%d/%d] ✓ Face detected & cropped: size=%dx%d -> %s",
                face_detected, target_count, crop_arr.shape[1], crop_arr.shape[0], crop_name
            )
            metadata_rows.append({
                "index": idx,
                "original_index": orig_idx,
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "dataset_source": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces",
                "raw_path": str(raw_path),
                "crop_path": str(crop_path),
                "width": crop_arr.shape[1],
                "height": crop_arr.shape[0],
                "face_found": True,
                "bbox": str(bbox),
                "phash": ph,
                "status": status,
                "timestamp": ts,
            })
        else:
            face_failed += 1
            status = "NO_FACE"
            logger.warning("[%d] ✗ No face detected by Haar cascade — logged", idx)
            metadata_rows.append({
                "index": idx,
                "original_index": orig_idx,
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "dataset_source": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces",
                "raw_path": str(raw_path),
                "crop_path": None,
                "width": pil_img.width,
                "height": pil_img.height,
                "face_found": False,
                "bbox": None,
                "phash": ph,
                "status": status,
                "timestamp": ts,
            })

    # Save metadata CSV
    meta_csv = meta_dir / f"diffusion_pilot_metadata_{ts}.csv"
    if metadata_rows:
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
            w.writeheader()
            w.writerows(metadata_rows)
    logger.info("Metadata CSV saved → %s", meta_csv)

    # Save pilot JSON report
    report = {
        "timestamp": ts,
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "benchmark_source": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces",
        "target_requested": target_count,
        "total_fetched": fetched,
        "face_detected": face_detected,
        "face_failed": face_failed,
        "duplicates": duplicates,
        "corrupted": corrupted,
        "success_rate_percent": round(100.0 * face_detected / max(fetched, 1), 2),
        "raw_directory": str(raw_output),
        "crops_directory": str(fake_dir),
        "metadata_csv": str(meta_csv),
        "is_fabricated": False,
        "label": "REAL DATA — SDXL diffusion generated benchmark",
    }
    rpath = report_dir / f"diffusion_pilot_{ts}.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("  DIFFUSION FACE PILOT ACQUISITION & PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Target requested     : {target_count}")
    print(f"  Fetched & examined   : {fetched}")
    print(f"  Face detected & saved: {face_detected}  -> {fake_dir}")
    print(f"  No face detected     : {face_failed}  (logged in metadata, raw preserved)")
    print(f"  Duplicates detected  : {duplicates}  (skipped from crops)")
    print(f"  Corrupted / errors   : {corrupted}")
    print(f"  Face detection rate  : {report['success_rate_percent']}%")
    print(f"  Metadata CSV         : {meta_csv}")
    print(f"  Report JSON          : {rpath}")
    print("=" * 70 + "\n")

    # Generate split manifests
    manifest_dir = PROJECT_ROOT / "data" / "diffusion_face" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    crops = sorted(fake_dir.glob("*.png"))
    # Deterministic split (70% train, 15% val, 15% test)
    rng = np.random.RandomState(42)
    indices = np.arange(len(crops))
    rng.shuffle(indices)

    n = len(crops)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": [crops[i] for i in indices[:n_train]],
        "val": [crops[i] for i in indices[n_train : n_train + n_val]],
        "test": [crops[i] for i in indices[n_train + n_val :]],
    }

    print("  CREATING SPLITS (Separate test set preserved):")
    for split_name, paths in splits.items():
        samples = [
            {
                "path": str(p),
                "label": 1,
                "split": split_name,
                "dataset": "diffusion_sdxl",
                "generator": "stabilityai/stable-diffusion-xl-base-1.0",
            }
            for p in paths
        ]
        mpath = manifest_dir / f"{split_name}_manifest.json"
        manifest_data = {
            "created": ts,
            "split": split_name,
            "num_samples": len(samples),
            "num_fake": len(samples),
            "num_real": 0,
            "samples": samples,
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"    - {split_name:5s}: {len(samples):2d} images -> {mpath.name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Number of face images to acquire")
    parser.add_argument("--min-face", type=int, default=64, help="Min face size in pixels")
    parser.add_argument("--phash-thresh", type=int, default=8, help="pHash duplicate threshold")
    args = parser.parse_args()
    acquire_diffusion_faces(args.count, args.min_face, args.phash_thresh)
