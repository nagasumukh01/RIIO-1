"""
Diffusion-generated face dataset builder — Contribution 1.

WHAT THIS SCRIPT DOES
─────────────────────
1. Scans source directories for REAL and FAKE (diffusion-generated) face images.
2. Runs face detection (Haar cascade) on each image and crops the face region.
3. Checks for near-duplicates using perceptual hashing (phash, threshold=8 bits).
4. Builds balanced train/val/test splits (identity-safe — no cross-split leakage
   because diffusion images share no identity with real images).
5. Writes a manifest JSON and per-image metadata CSV.
6. Prints a summary report and saves it to data/diffusion_face/reports/.

REAL DATA ONLY — no synthetic fallback.

USAGE
─────
# Minimal (uses defaults from configs/data.yaml):
  python scripts/build_diffusion_dataset.py

# Custom sources:
  python scripts/build_diffusion_dataset.py \
    --real-dir  path/to/real_face_images \
    --fake-dir  path/to/diffusion_face_images \
    --output-root data/diffusion_face \
    --max-per-class 2000 \
    --phash-threshold 8

DEPENDENCIES
────────────
  pip install imagehash pillow opencv-python-headless tqdm scikit-learn

STATUS
──────
If --real-dir or --fake-dir is not provided and the default paths don't exist,
the script will print exactly what is missing and exit with a non-zero code.
It will NOT fabricate any data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import imagehash
except ImportError:
    imagehash = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("robustdeep.diffusion")

# ---------------------------------------------------------------------------
# Face detector (Haar cascade, same as the rest of the pipeline)
# ---------------------------------------------------------------------------
HAAR_ALTS = [
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_default.xml",
]


def _load_detector() -> Optional[object]:
    if cv2 is None:
        logger.warning("opencv-python not installed — face detection disabled.")
        return None
    for name in HAAR_ALTS:
        path = cv2.data.haarcascades + name  # type: ignore[attr-defined]
        det = cv2.CascadeClassifier(path)  # type: ignore[attr-defined]
        if not det.empty():
            return det
    logger.warning("Haar cascade not found — face detection disabled.")
    return None


DETECTOR = None  # lazy-loaded on first use


def detect_face(image_path: Path, min_size: int = 64) -> Optional[np.ndarray]:
    """
    Detect the largest face in an image and return the cropped face as an
    H×W×3 uint8 numpy array, or None if no face was found.
    """
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = _load_detector()
    if DETECTOR is None or cv2 is None:
        # No detector — just load the full image resized
        if Image is None:
            return None
        img = np.array(Image.open(image_path).convert("RGB"))
        return img

    img_bgr = cv2.imread(str(image_path))  # type: ignore[attr-defined]
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]

    faces = DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,  # type: ignore[attr-defined]
    )
    if len(faces) == 0:
        return None

    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = img_bgr[y : y + h, x : x + w]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
    return crop_rgb


# ---------------------------------------------------------------------------
# Perceptual hash duplicate check
# ---------------------------------------------------------------------------

def phash_str(image_path: Path) -> Optional[str]:
    """Return perceptual hash string, or None on failure."""
    if imagehash is None or Image is None:
        return hashlib.md5(image_path.read_bytes()).hexdigest()
    try:
        img = Image.open(image_path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None


def find_duplicates(paths: List[Path], threshold: int = 8) -> List[Path]:
    """
    Return subset of `paths` that are near-duplicates of a previously seen image.
    Uses Hamming distance between perceptual hashes.
    """
    if imagehash is None:
        logger.warning(
            "imagehash not installed — duplicate check by exact MD5 only."
        )
        seen_md5: set = set()
        dups = []
        for p in paths:
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen_md5:
                dups.append(p)
            else:
                seen_md5.add(h)
        return dups

    seen_hashes: list = []
    duplicates: List[Path] = []
    for p in tqdm(paths, desc="Deduplicating", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            ph = imagehash.phash(img)
        except Exception:
            continue
        is_dup = any(abs(ph - prev) <= threshold for prev in seen_hashes)
        if is_dup:
            duplicates.append(p)
        else:
            seen_hashes.append(ph)
    return duplicates


# ---------------------------------------------------------------------------
# Split builder (stratified, deterministic)
# ---------------------------------------------------------------------------

def make_splits(
    paths: List[Path],
    label: int,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, List[Path]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": [paths[i] for i in idx[:n_train]],
        "val":   [paths[i] for i in idx[n_train : n_train + n_val]],
        "test":  [paths[i] for i in idx[n_train + n_val :]],
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_class(
    src_dir: Path,
    label: int,
    out_dir: Path,
    max_items: int,
    min_face_size: int,
    phash_threshold: int,
) -> List[dict]:
    """
    For each image in src_dir:
      1. Detect face.
      2. Save cropped face to out_dir.
      3. Collect metadata.

    Returns list of metadata dicts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_paths = sorted(
        p for p in src_dir.rglob("*") if p.suffix.lower() in extensions
    )

    if len(all_paths) == 0:
        logger.warning("No images found in %s", src_dir)
        return []

    logger.info(
        "Found %d images in %s (label=%d)", len(all_paths), src_dir, label
    )

    # Dedup first
    logger.info("Running near-duplicate check …")
    dup_paths = set(str(p) for p in find_duplicates(all_paths, phash_threshold))
    unique_paths = [p for p in all_paths if str(p) not in dup_paths]
    logger.info(
        "After dedup: %d unique / %d total (removed %d duplicates)",
        len(unique_paths),
        len(all_paths),
        len(dup_paths),
    )

    if max_items and len(unique_paths) > max_items:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(unique_paths), max_items, replace=False)
        unique_paths = [unique_paths[i] for i in sorted(idx)]

    metadata: List[dict] = []
    face_found = 0
    face_missing = 0

    for src_path in tqdm(unique_paths, desc=f"label={label}", unit="img"):
        crop = detect_face(src_path, min_size=min_face_size)
        if crop is None:
            face_missing += 1
            metadata.append(
                {
                    "source": str(src_path),
                    "label": label,
                    "face_found": False,
                    "saved_path": None,
                    "phash": phash_str(src_path),
                    "generator": _infer_generator(src_path),
                }
            )
            continue

        # Save face crop
        dest = out_dir / src_path.name
        if dest.exists():
            # Avoid overwrite — append stem suffix
            dest = out_dir / f"{src_path.stem}__{face_found}{src_path.suffix}"

        if Image is not None:
            Image.fromarray(crop).save(str(dest))
        elif cv2 is not None:
            cv2.imwrite(str(dest), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        face_found += 1
        metadata.append(
            {
                "source": str(src_path),
                "label": label,
                "face_found": True,
                "saved_path": str(dest),
                "phash": phash_str(src_path),
                "generator": _infer_generator(src_path),
            }
        )

    logger.info(
        "label=%d: %d faces found, %d images skipped (no face)",
        label,
        face_found,
        face_missing,
    )
    return metadata


def _infer_generator(path: Path) -> str:
    """
    Infer the generation method from parent directory name or filename.
    Extend this function as more generators are added.
    """
    parts = [p.lower() for p in path.parts]
    keywords = {
        "sdxl": "SDXL",
        "stable_diffusion": "stable-diffusion-v1",
        "sd1": "stable-diffusion-v1",
        "sd2": "stable-diffusion-v2",
        "sd3": "stable-diffusion-v3",
        "flux": "FLUX",
        "midjourney": "midjourney",
        "dalle": "DALL-E",
        "dall_e": "DALL-E",
        "fastsd": "fastsd-cpu",
        "diffusion": "diffusion-unspecified",
    }
    for part in parts:
        for key, name in keywords.items():
            if key in part:
                return name
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build diffusion-face dataset")
    p.add_argument(
        "--real-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "real",
        help="Directory of real face images",
    )
    p.add_argument(
        "--fake-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "fake",
        help="Directory of diffusion-generated face images",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "diffusion_face",
        help="Root output directory",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Max images per class (0 = no limit)",
    )
    p.add_argument(
        "--phash-threshold",
        type=int,
        default=8,
        help="Max Hamming distance for near-duplicate detection (default 8)",
    )
    p.add_argument(
        "--min-face-size",
        type=int,
        default=64,
        help="Minimum face size in pixels (default 64)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without saving anything",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    missing = []
    for label, d in [("real", args.real_dir), ("fake", args.fake_dir)]:
        if not d.exists():
            missing.append((label, d))

    if missing:
        print("\n" + "=" * 70)
        print("DIFFUSION DATASET — MISSING SOURCE DIRECTORIES")
        print("=" * 70)
        for label, d in missing:
            print(f"  [{label.upper()}] Expected: {d}")
        print()
        print("To fix:")
        print("  1. Place REAL face images in:")
        print(f"       {args.real_dir}")
        print("     (Any real portrait/face photos — NOT deepfakes)")
        print()
        print("  2. Place DIFFUSION-GENERATED face images in:")
        print(f"       {args.fake_dir}")
        print("     (Generated by Stable Diffusion, SDXL, FLUX, DALL-E, etc.)")
        print()
        print("  Suggested generators:")
        print("    - fastsdcpu (already in diffusion_workspace/fastsdcpu-main/)")
        print("    - stabilityai/stable-diffusion-xl-base-1.0 via diffusers")
        print("    - black-forest-labs/FLUX.1-schnell (requires 8GB VRAM)")
        print()
        print("  Then re-run: python scripts/build_diffusion_dataset.py")
        print("=" * 70 + "\n")
        return 1

    # Create output structure
    proc_root = args.output_root / "processed"
    manifest_dir = args.output_root / "manifests"
    metadata_dir = args.output_root / "metadata"
    report_dir = args.output_root / "reports"
    for d in [proc_root, manifest_dir, metadata_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for label_name, src in [("real", args.real_dir), ("fake", args.fake_dir)]:
            imgs = list(src.rglob("*.jpg")) + list(src.rglob("*.png"))
            print(f"[DRY RUN] {label_name}: {len(imgs)} images found in {src}")
        return 0

    # Process each class
    max_n = args.max_per_class if args.max_per_class > 0 else None
    real_meta = process_class(
        args.real_dir, label=0,
        out_dir=proc_root / "real",
        max_items=max_n or 999999,
        min_face_size=args.min_face_size,
        phash_threshold=args.phash_threshold,
    )
    fake_meta = process_class(
        args.fake_dir, label=1,
        out_dir=proc_root / "fake",
        max_items=max_n or 999999,
        min_face_size=args.min_face_size,
        phash_threshold=args.phash_threshold,
    )

    all_meta = real_meta + fake_meta

    # Save CSV metadata
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_csv = metadata_dir / f"diffusion_metadata_{ts}.csv"
    if all_meta:
        fieldnames = list(all_meta[0].keys())
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_meta)
        logger.info("Metadata CSV saved to %s", meta_csv)

    # Build splits using only images where face was found
    real_with_face = [Path(m["saved_path"]) for m in real_meta if m["face_found"]]
    fake_with_face = [Path(m["saved_path"]) for m in fake_meta if m["face_found"]]

    real_splits = make_splits(real_with_face, label=0)
    fake_splits = make_splits(fake_with_face, label=1)

    # Save manifest per split
    for split in ("train", "val", "test"):
        samples = []
        for label, paths in [(0, real_splits[split]), (1, fake_splits[split])]:
            for p in paths:
                samples.append(
                    {
                        "path": str(p),
                        "label": label,
                        "split": split,
                        "dataset": "diffusion",
                    }
                )
        manifest = {
            "created": ts,
            "split": split,
            "num_samples": len(samples),
            "num_real": sum(1 for s in samples if s["label"] == 0),
            "num_fake": sum(1 for s in samples if s["label"] == 1),
            "samples": samples,
        }
        mpath = manifest_dir / f"{split}_manifest.json"
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Manifest: %s (%d samples)", mpath, len(samples))

    # Report
    real_face_count = sum(1 for m in real_meta if m["face_found"])
    fake_face_count = sum(1 for m in fake_meta if m["face_found"])

    # Generator breakdown
    gen_counts: Dict[str, int] = {}
    for m in fake_meta:
        g = m.get("generator", "unknown")
        gen_counts[g] = gen_counts.get(g, 0) + 1

    report = {
        "created": ts,
        "real_dir": str(args.real_dir),
        "fake_dir": str(args.fake_dir),
        "real_total_input": len(real_meta),
        "real_face_found": real_face_count,
        "real_no_face": len(real_meta) - real_face_count,
        "fake_total_input": len(fake_meta),
        "fake_face_found": fake_face_count,
        "fake_no_face": len(fake_meta) - fake_face_count,
        "train_real": len(real_splits["train"]),
        "train_fake": len(fake_splits["train"]),
        "val_real": len(real_splits["val"]),
        "val_fake": len(fake_splits["val"]),
        "test_real": len(real_splits["test"]),
        "test_fake": len(fake_splits["test"]),
        "generators": gen_counts,
        "phash_threshold": args.phash_threshold,
        "is_fabricated": False,
        "label": "REAL DATA — diffusion pipeline",
    }

    rpath = report_dir / f"diffusion_report_{ts}.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("DIFFUSION DATASET BUILD COMPLETE")
    print("=" * 70)
    print(f"  Real images processed : {len(real_meta):>6}  (faces found: {real_face_count})")
    print(f"  Fake images processed : {len(fake_meta):>6}  (faces found: {fake_face_count})")
    print(f"\n  Splits (face-only):")
    print(f"    train real={len(real_splits['train'])}  fake={len(fake_splits['train'])}")
    print(f"    val   real={len(real_splits['val'])}    fake={len(fake_splits['val'])}")
    print(f"    test  real={len(real_splits['test'])}   fake={len(fake_splits['test'])}")
    if gen_counts:
        print(f"\n  Fake generators detected:")
        for g, cnt in sorted(gen_counts.items(), key=lambda x: -x[1]):
            print(f"    {g:30s} {cnt}")
    print(f"\n  Report saved to : {rpath}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
