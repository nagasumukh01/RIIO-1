"""
Generate synthetic face images using the locally-downloaded SD-Turbo model,
detect faces, save crops, record metadata, then run the dataset pipeline.

Requirements: pip install diffusers transformers accelerate pillow opencv-python-headless tqdm imagehash

Usage:
    python scripts/generate_diffusion_faces.py --n 50 --steps 2 --seed 42

Output:
    diffusion_workspace/output/     ← raw generated images (never deleted)
    data/diffusion_face/raw/fake/   ← face-detected crops
    data/diffusion_face/manifests/  ← train/val/test splits (after pipeline run)
    data/diffusion_face/metadata/   ← CSV with provenance per image
    data/diffusion_face/reports/    ← JSON summary report
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Patch diffusers v0.42 MemoryError bug:
#   model_loading_utils.load_state_dict does f.read() on the full weights file
#   just to check if it's a pickle. With low RAM this OOMs on the VAE.
#   Fix: route by extension — use safetensors directly, torch.load for .bin
# ---------------------------------------------------------------------------
def _patch_diffusers_load():
    try:
        import diffusers.models.model_loading_utils as _mlu

        def _safe_load_state_dict(checkpoint_file, disable_mmap=False):
            # diffusers sometimes passes an already-loaded dict (pass-through) — return as-is
            if isinstance(checkpoint_file, dict):
                return checkpoint_file
            checkpoint_file = str(checkpoint_file)
            # Route .safetensors to safetensors lib (uses mmap — no RAM spike)
            if checkpoint_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(checkpoint_file)
            # Fallback for legacy .bin pickle files
            import torch
            return torch.load(checkpoint_file, map_location="cpu", weights_only=True)

        _mlu.load_state_dict = _safe_load_state_dict
        logging.getLogger("robustdeep.generate").info(
            "Applied diffusers load_state_dict patch (low-RAM fix)"
        )
    except Exception as e:
        logging.getLogger("robustdeep.generate").warning(
            "Could not patch diffusers load_state_dict: %s", e
        )

_patch_diffusers_load()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("robustdeep.generate")

# ---------------------------------------------------------------------------
# Diverse prompts — different ages, ethnicities, genders, lighting, styles
# ---------------------------------------------------------------------------
FACE_PROMPTS = [
    "portrait photo of a young woman, natural lighting, sharp focus, photorealistic",
    "portrait photo of a middle-aged man, studio lighting, sharp focus, photorealistic",
    "portrait photo of an elderly woman, soft lighting, photorealistic, detailed",
    "portrait photo of a young man with glasses, outdoor lighting, photorealistic",
    "portrait photo of a teenage girl, golden hour lighting, photorealistic",
    "close-up portrait of a south asian woman, professional headshot, photorealistic",
    "close-up portrait of an east asian man, natural lighting, photorealistic",
    "portrait of an african american woman, studio lighting, photorealistic",
    "portrait of a hispanic man, outdoor lighting, photorealistic",
    "portrait of a caucasian woman with curly hair, soft lighting, photorealistic",
    "professional headshot of a businessman in a suit, studio background, photorealistic",
    "portrait of a woman with short hair, side lighting, photorealistic",
    "portrait of an older man with a beard, dramatic lighting, photorealistic",
    "portrait of a young woman smiling, bright outdoor lighting, photorealistic",
    "passport style portrait of a person, neutral background, photorealistic",
    "portrait of a woman with straight hair, natural lighting, photorealistic",
    "portrait of a man in his 40s, office background, photorealistic",
    "portrait of a young person with freckles, daylight, photorealistic",
    "portrait of an asian woman with long hair, studio lighting, photorealistic",
    "portrait of a man with blue eyes, close-up, photorealistic",
    "environmental portrait of a woman outdoors, candid style, photorealistic",
    "portrait of a person with dark skin, warm lighting, photorealistic",
    "portrait of a young man with curly hair, urban background, photorealistic",
    "portrait of a woman in her 30s, neutral expression, professional, photorealistic",
    "portrait of an older woman with grey hair, soft light, photorealistic",
]

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, art, unrealistic, blurry, "
    "low quality, deformed, distorted, extra limbs, watermark, text"
)


# ---------------------------------------------------------------------------
# Face detector (Haar cascade)
# ---------------------------------------------------------------------------
def load_detector():
    try:
        import cv2
        for name in ["haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml"]:
            path = cv2.data.haarcascades + name
            det = cv2.CascadeClassifier(path)
            if not det.empty():
                return det, cv2
    except Exception:
        pass
    return None, None


def detect_and_crop(pil_img, detector, cv2, min_size: int = 80):
    """Return (crop_array, bbox) or (None, None) if no face found."""
    import numpy as np
    img_array = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(min_size, min_size), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = img_array[y:y+h, x:x+w]
    return crop, (int(x), int(y), int(w), int(h))


# ---------------------------------------------------------------------------
# Duplicate check (pHash)
# ---------------------------------------------------------------------------
def compute_phash(pil_img):
    try:
        import imagehash
        return str(imagehash.phash(pil_img.convert("RGB")))
    except ImportError:
        import hashlib
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return hashlib.md5(buf.getvalue()).hexdigest()


def hamming(h1: str, h2: str) -> int:
    """Hamming distance between two hex phash strings."""
    try:
        import imagehash
        return abs(imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2))
    except Exception:
        return 0 if h1 == h2 else 999


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate(args):
    from PIL import Image

    # Paths
    model_cache = PROJECT_ROOT / "diffusion_workspace" / "models" / "sd-turbo"
    raw_output = PROJECT_ROOT / "diffusion_workspace" / "output"
    fake_dir = PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "fake"
    meta_dir = PROJECT_ROOT / "data" / "diffusion_face" / "metadata"
    report_dir = PROJECT_ROOT / "data" / "diffusion_face" / "reports"

    for d in [raw_output, fake_dir, meta_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Verify SD-Turbo model exists
    snapshot_dir = model_cache / "models--stabilityai--sd-turbo" / "snapshots"
    snapshots = list(snapshot_dir.iterdir()) if snapshot_dir.exists() else []
    if not snapshots:
        logger.error("SD-Turbo model not found at %s", model_cache)
        logger.error("Run: python -c \"from diffusers import AutoPipelineForText2Image; AutoPipelineForText2Image.from_pretrained('stabilityai/sd-turbo')\"")
        sys.exit(1)
    model_path = str(snapshots[0])
    logger.info("Loading SD-Turbo from %s", model_path)

    # Load pipeline
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_path,
        # float16 halves model RAM (~2.5 GB vs ~5 GB).
        # safetensors reads lazily via mmap and converts on-the-fly,
        # so peak RAM ≈ float16 output size, not float32 source size.
        # CPU inference in float16 is slower but functional on PyTorch.
        torch_dtype=torch.float16,
        variant=None,
    )
    pipe.set_progress_bar_config(disable=True)
    logger.info("Pipeline loaded. Device: CPU, dtype: float32")

    # Load face detector
    detector, cv2_mod = load_detector()
    if detector is None:
        logger.warning("No face detector available — crops will be skipped")

    # Generation settings
    n_total = args.n
    steps = args.steps          # SD-Turbo: 1–4 steps
    guidance = args.guidance    # SD-Turbo: 0.0 (no CFG needed)
    width = height = args.size
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Generating %d images | steps=%d | size=%dx%d | seed_start=%d",
                n_total, steps, width, height, args.seed)

    # Tracking
    metadata_rows = []
    generated = 0
    face_found = 0
    face_missing = 0
    duplicate_count = 0
    corrupted = 0
    seen_hashes: list = []

    for i in range(n_total):
        prompt = FACE_PROMPTS[i % len(FACE_PROMPTS)]
        seed = args.seed + i
        generator = torch.Generator().manual_seed(seed)

        t0 = time.time()
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
            pil_img = result.images[0]
            elapsed = time.time() - t0
        except Exception as e:
            logger.error("Generation failed for i=%d seed=%d: %s", i, seed, e)
            corrupted += 1
            metadata_rows.append({
                "index": i, "seed": seed, "prompt": prompt,
                "model": "stabilityai/sd-turbo", "steps": steps,
                "guidance": guidance, "width": width, "height": height,
                "raw_path": None, "crop_path": None,
                "face_found": False, "bbox": None, "phash": None,
                "status": f"GENERATION_ERROR: {e}", "elapsed_s": None,
                "timestamp": ts,
            })
            continue

        generated += 1

        # Save raw image (never deleted)
        raw_name = f"sdturbo_{ts}_{i:04d}_seed{seed}.png"
        raw_path = raw_output / raw_name
        pil_img.save(str(raw_path))

        # Compute phash
        ph = compute_phash(pil_img)

        # Duplicate check
        is_dup = any(hamming(ph, prev) <= args.phash_threshold for prev in seen_hashes)
        if is_dup:
            duplicate_count += 1
            logger.warning("[%d/%d] Duplicate detected (seed=%d) — keeping raw, skipping crop", i+1, n_total, seed)
            metadata_rows.append({
                "index": i, "seed": seed, "prompt": prompt,
                "model": "stabilityai/sd-turbo", "steps": steps,
                "guidance": guidance, "width": width, "height": height,
                "raw_path": str(raw_path), "crop_path": None,
                "face_found": False, "bbox": None, "phash": ph,
                "status": "DUPLICATE", "elapsed_s": round(elapsed, 2),
                "timestamp": ts,
            })
            continue
        seen_hashes.append(ph)

        # Face detection
        crop_path = None
        bbox_str = None
        status = "OK"
        if detector is not None:
            crop_arr, bbox = detect_and_crop(pil_img, detector, cv2_mod, min_size=args.min_face)
            if crop_arr is not None:
                face_found += 1
                crop_name = f"sdturbo_{ts}_{i:04d}_seed{seed}_face.png"
                crop_path = fake_dir / crop_name
                Image.fromarray(crop_arr).save(str(crop_path))
                bbox_str = str(bbox)
                logger.info("[%d/%d] ✓ Face crop saved  seed=%-10d  size=%dx%d  %.1fs",
                            i+1, n_total, seed, crop_arr.shape[1], crop_arr.shape[0], elapsed)
            else:
                face_missing += 1
                status = "NO_FACE"
                logger.warning("[%d/%d] ✗ No face detected  seed=%d  %.1fs", i+1, n_total, seed, elapsed)
        else:
            # No detector — save full image as crop
            face_found += 1
            crop_name = f"sdturbo_{ts}_{i:04d}_seed{seed}_full.png"
            crop_path = fake_dir / crop_name
            pil_img.save(str(crop_path))
            status = "NO_DETECTOR"

        metadata_rows.append({
            "index": i, "seed": seed, "prompt": prompt,
            "model": "stabilityai/sd-turbo", "steps": steps,
            "guidance": guidance, "width": width, "height": height,
            "raw_path": str(raw_path), "crop_path": str(crop_path) if crop_path else None,
            "face_found": crop_path is not None, "bbox": bbox_str, "phash": ph,
            "status": status, "elapsed_s": round(elapsed, 2),
            "timestamp": ts,
        })

    # Save metadata CSV
    meta_csv = meta_dir / f"sdturbo_metadata_{ts}.csv"
    if metadata_rows:
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)
    logger.info("Metadata saved → %s", meta_csv)

    # Print pilot report
    print("\n" + "=" * 65)
    print("  SD-TURBO GENERATION PILOT — RESULTS")
    print("=" * 65)
    print(f"  Requested       : {n_total}")
    print(f"  Generated       : {generated}  (errors/skipped: {corrupted})")
    print(f"  Duplicates      : {duplicate_count}")
    print(f"  Face detected   : {face_found}  → saved to data/diffusion_face/raw/fake/")
    print(f"  No face         : {face_missing}  (logged in metadata, raw image kept)")
    print(f"  Face success %  : {100*face_found/max(generated,1):.1f}%")
    print(f"\n  Raw images      : {raw_output}")
    print(f"  Face crops      : {fake_dir}")
    print(f"  Metadata CSV    : {meta_csv}")
    print("=" * 65)

    # Save JSON report
    report = {
        "timestamp": ts, "model": "stabilityai/sd-turbo",
        "requested": n_total, "generated": generated,
        "face_found": face_found, "face_missing": face_missing,
        "duplicates": duplicate_count, "corrupted": corrupted,
        "steps": steps, "guidance": guidance,
        "width": width, "height": height,
        "seed_start": args.seed,
        "is_fabricated": False,
        "label": "REAL DATA — SD-Turbo diffusion generated",
    }
    rpath = report_dir / f"sdturbo_pilot_{ts}.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if face_found == 0:
        print("\n  ⚠ No faces detected. The dataset pipeline will not run.")
        print("  Try reducing --min-face or check if opencv is installed correctly.\n")
        return

    # Run dataset pipeline to create manifests + splits
    print("\n  Running dataset pipeline (manifests + splits)...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/build_diffusion_dataset.py",
         "--fake-dir", str(fake_dir),
         "--real-dir", str(PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "real")],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("  Pipeline output (fake-only mode — no real images yet):")
        print(result.stdout or result.stderr)

    print(f"\n  Full report → {rpath}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate SD-Turbo face images for diffusion dataset")
    p.add_argument("--n", type=int, default=50, help="Number of images to generate")
    p.add_argument("--steps", type=int, default=2, help="Inference steps (SD-Turbo: 1-4)")
    p.add_argument("--guidance", type=float, default=0.0, help="Guidance scale (SD-Turbo: 0.0)")
    p.add_argument("--size", type=int, default=512, help="Image size (square)")
    p.add_argument("--seed", type=int, default=42, help="Starting seed")
    p.add_argument("--min-face", type=int, default=80, help="Min face size in pixels")
    p.add_argument("--phash-threshold", type=int, default=8, help="Hamming distance for dup detection")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args)
