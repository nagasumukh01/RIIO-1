"""
Single-image / single-video inference CLI for RobustDeep.

Usage:
    # Predict a single image
    python scripts/infer.py --input path/to/face.jpg

    # Predict multiple images
    python scripts/infer.py --input path/to/images/ --model vit_base

    # Predict from a video (extracts frames, runs inference)
    python scripts/infer.py --input path/to/video.mp4 --frames 20

    # Specify checkpoint
    python scripts/infer.py --input face.jpg --checkpoint artifacts/checkpoints/exp_001/best_model.pt
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import setup_logger

logger = setup_logger("infer")


def load_model(model_key: str, checkpoint_path: str = None, device: str = "cpu"):
    """Load model for inference."""
    from src.utils.config import ModelConfig
    from src.models.factory import create_model, load_checkpoint

    config = ModelConfig()
    if "vit" in model_key.lower():
        config.type = "vit"
        config.backbone = "vit_base_patch16_224"
        config.name = "ViT-Base-Detector"
    elif "hybrid" in model_key.lower():
        config.type = "hybrid"
        config.backbone = "efficientnet_b0"
        config.name = "Hybrid-CNN-ViT"
    else:
        config.type = "cnn"
        config.backbone = "efficientnet_b0"
        config.name = "EfficientNet-B0-Detector"

    has_ckpt = checkpoint_path and Path(checkpoint_path).exists()
    # If checkpoint exists, skip downloading ImageNet weights since checkpoint overrides them
    if has_ckpt:
        config.pretrained = False

    try:
        model = create_model(config, device)
    except Exception as e:
        logger.warning(f"Failed to create model with pretrained={config.pretrained}: {e}. Retrying with pretrained=False...")
        config.pretrained = False
        model = create_model(config, device)

    if has_ckpt:
        meta = load_checkpoint(model, checkpoint_path, device=device, strict=True)
        logger.info(f"Loaded checkpoint from epoch {meta.get('epoch', '?')}")
    else:
        logger.info("Using initialized weights. Ready for inference.")

    model.eval()
    return model, config


def preprocess_image(image_path: str, face_size: int = 224):
    """Load and preprocess an image for inference."""
    import torch
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((face_size, face_size))
    img_np = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std

    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def extract_video_frames(video_path: str, num_frames: int = 5, face_size: int = 224):
    """Extract frames from a video file with robust codec-agnostic sequential reading."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames) if total_frames > 0 else 1
    frames = []
    count = 0

    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            side = min(h, w)
            top, left = (h - side) // 2, (w - side) // 2
            crop = frame_rgb[top:top+side, left:left+side]
            crop = cv2.resize(crop, (face_size, face_size))
            frames.append(crop)
        count += 1

    cap.release()

    # If step reading yielded fewer than expected, read whatever frames are available
    if not frames:
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            side = min(h, w)
            top, left = (h - side) // 2, (w - side) // 2
            crop = frame_rgb[top:top+side, left:left+side]
            crop = cv2.resize(crop, (face_size, face_size))
            frames.append(crop)
        cap.release()

    return frames


@torch.no_grad()
def predict_image(model, tensor, device: str = "cpu"):
    """Run inference on a single image tensor."""
    import torch
    import torch.nn.functional as F

    tensor = tensor.to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]

    real_prob = float(probs[0].cpu())
    fake_prob = float(probs[1].cpu())
    prediction = "FAKE" if fake_prob >= 0.5 else "REAL"

    return {
        "prediction": prediction,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "confidence": round(max(real_prob, fake_prob), 4),
    }


def predict_single_image(model, image_path: str, device: str):
    """Predict a single image file."""
    import torch
    tensor = preprocess_image(image_path)
    result = predict_image(model, tensor, device)
    result["file"] = str(image_path)
    return result


def predict_video(model, video_path: str, num_frames: int, device: str):
    """Predict a video by aggregating frame-level scores."""
    import torch
    import numpy as np

    logger.info(f"Extracting {num_frames} frames from video...")
    frames = extract_video_frames(video_path, num_frames)

    if not frames:
        return {"error": "No frames could be extracted"}

    fake_probs = []
    for frame_np in frames:
        frame_np = frame_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_np = (frame_np - mean) / std
        tensor = torch.from_numpy(frame_np.transpose(2, 0, 1)).float().unsqueeze(0)
        result = predict_image(model, tensor, device)
        fake_probs.append(result["fake_probability"])

    mean_fake_prob = float(np.mean(fake_probs))
    prediction = "FAKE" if mean_fake_prob >= 0.5 else "REAL"

    return {
        "file": str(video_path),
        "prediction": prediction,
        "fake_probability": round(mean_fake_prob, 4),
        "real_probability": round(1 - mean_fake_prob, 4),
        "confidence": round(max(mean_fake_prob, 1 - mean_fake_prob), 4),
        "frames_analyzed": len(fake_probs),
        "frame_scores": [round(p, 4) for p in fake_probs],
    }


# Need torch for the @torch.no_grad decorator
try:
    import torch
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="RobustDeep Inference — predict real/fake on images or videos"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to image, video, or directory of images")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "vit_base", "hybrid"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--frames", type=int, default=20,
                        help="Number of frames to sample per video")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame scores for videos")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RobustDeep Inference")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    logger.info("⚠️  DISCLAIMER: This is a research system and must NOT be")
    logger.info("   treated as conclusive forensic evidence.")
    logger.info("")

    # Load model
    model, config = load_model(args.model, args.checkpoint, args.device)
    logger.info(f"Model loaded: {config.name}")

    input_path = Path(args.input)
    results = []
    t0 = time.time()

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    # Determine input type
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if input_path.is_dir():
        # Process all images in directory
        image_files = []
        for ext in image_exts:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_files)} images in directory")
        for img_path in sorted(image_files):
            result = predict_single_image(model, str(img_path), args.device)
            results.append(result)
            logger.info(
                f"  {img_path.name:40s} → {result['prediction']} "
                f"(fake_prob={result['fake_probability']:.4f})"
            )

    elif input_path.suffix.lower() in video_exts:
        result = predict_video(model, str(input_path), args.frames, args.device)
        results.append(result)
        logger.info(f"\nVideo Prediction: {result['prediction']}")
        logger.info(f"  Mean fake probability: {result['fake_probability']:.4f}")
        logger.info(f"  Frames analyzed: {result.get('frames_analyzed', 0)}")
        if args.verbose and "frame_scores" in result:
            logger.info(f"  Frame scores: {result['frame_scores']}")

    elif input_path.suffix.lower() in image_exts:
        result = predict_single_image(model, str(input_path), args.device)
        results.append(result)
        logger.info(f"\nImage Prediction: {result['prediction']}")
        logger.info(f"  Fake probability: {result['fake_probability']:.4f}")
        logger.info(f"  Real probability: {result['real_probability']:.4f}")
        logger.info(f"  Confidence: {result['confidence']:.4f}")

    else:
        logger.error(f"Unsupported file type: {input_path.suffix}")
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info(f"\nCompleted in {elapsed:.2f}s")

    # Save results
    if args.output:
        output = {
            "model": config.name,
            "checkpoint": args.checkpoint,
            "results": results,
            "elapsed_s": round(elapsed, 2),
            "disclaimer": "Research system — not for forensic or legal use.",
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
