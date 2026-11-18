"""FastAPI inference service for the RobustDeep React frontend."""
from __future__ import annotations

import base64
import dataclasses
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.data.real_dataset import PROJECT_ROOT
from src.ensemble.fusion import (
    EnsembleVerdict,
    FaceResult,
    ModelResult,
    analyze_image_faces,
    analyze_video_frames,
)
from src.ensemble.registry import EnsembleRegistry, get_registry
from src.explainability.gradcam import GradCAM
from src.models.factory import build_model
from src.utils.config import load_yaml
from src.utils.logging_utils import setup_logger

logger = setup_logger("robustdeep.api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log_checkpoint_status()
    # Pre-load ensemble registry so models are cached on startup
    try:
        reg = get_registry()
        loaded = reg.load_all()
        logger.info("Ensemble registry: %d/%d models loaded", len(loaded), len(reg.enabled_configs()))
    except Exception as exc:
        logger.warning("Ensemble registry pre-load failed: %s", exc)
    yield


app = FastAPI(
    title="RobustDeep Inference API",
    version="1.2.0",
    description="Cross-Dataset Deepfake Detection with Verified Multi-Snapshot Face Biometrics",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "cross_ensemble": {
        "name": "cross_ensemble",
        "display_name": "Cross-Architecture Hybrid Ensemble",
        "architecture": "Hybrid Ensemble (CNN + ViT)",
        "dataset": "Celeb-DF v2 & FaceForensics++",
        "checkpoint": None,
        "description": "Equal-weight average of fake probabilities from whichever of the four registered CNN/ViT checkpoints are present. Missing members are listed in model status and excluded from fusion.",
    },
    "celebdf_b0": {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "architecture": "CNN",
        "dataset": "Celeb-DF v2",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "celebdf_b0_balanced_pilot8_cpu_20261109" / "best.pt",
        "description": "EfficientNet-B0 pilot trained on Celeb-DF v2; this small checkpoint is experimental.",
    },
    "ff_b0": {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "architecture": "CNN",
        "dataset": "FaceForensics++",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "ff_c23_b0_balanced200_4f_20261109" / "best.pt",
        "description": "EfficientNet-B0 trained on a FaceForensics++ c23 subset.",
    },
    "celebdf_vit": {
        "name": "vit_small",
        "display_name": "ViT-Small",
        "architecture": "Vision Transformer",
        "dataset": "Celeb-DF v2",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "celebdf_vit_small_subset50_f4" / "best.pt",
        "description": "Vision Transformer (ViT-Small/16) trained on Celeb-DF v2.",
    },
    "ff_vit": {
        "name": "vit_small",
        "display_name": "ViT-Small",
        "architecture": "Vision Transformer",
        "dataset": "FaceForensics++",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "ff_c23_vit_small_subset50_f4" / "best.pt",
        "description": "Vision Transformer (ViT-Small/16) trained on FaceForensics++.",
    },
    "dfdc_b0": {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0 (Pilot)",
        "architecture": "CNN",
        "dataset": "DFDC (8-video pilot)",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "dfdc_b0_balanced_pilot8_cpu_20261109" / "best.pt",
        "description": "Experimental DFDC pilot checkpoint trained for one epoch on a tiny subset. Results are not representative of full-dataset performance.",
    },
    "combined_multidomain_b0": {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0 (Universal Multi-Domain)",
        "architecture": "CNN",
        "dataset": "All 4 Combined (FF++, Celeb-DF, DFDC, Diffusion)",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "combined_multidomain_b0" / "best.pt",
        "description": "Universal multi-domain detector trained across FaceForensics++, Celeb-DF v2, DFDC, and Diffusion synthetic faces (SDXL + CelebA-HQ).",
    },
}

MODEL_INFERENCE_CONFIG: dict[str, dict[str, Any]] = load_yaml("configs/inference.yaml")["models"]
BASE_MODEL_KEYS = ["celebdf_b0", "ff_b0", "celebdf_vit", "ff_vit"]
_MODELS: dict[str, torch.nn.Module] = {}
_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def _checkpoint_path(model_id: str) -> Path:
    checkpoint = MODEL_REGISTRY[model_id].get("checkpoint")
    if checkpoint is None:
        raise ValueError(f"Model {model_id!r} does not have an individual checkpoint.")
    return Path(checkpoint)


def _inference_config(model_id: str) -> dict[str, Any]:
    try:
        return MODEL_INFERENCE_CONFIG[model_id]
    except KeyError as exc:
        raise ValueError(f"No inference configuration is registered for model {model_id!r}.") from exc


def _fake_probability(logits: torch.Tensor, model_id: str) -> float:
    config = _inference_config(model_id)
    class_count = int(config["num_classes"])
    output_units = int(config["output_units"])
    fake_index = int(config["fake_index"])
    if not 0 <= fake_index < class_count:
        raise ValueError(f"Invalid fake_index for model {model_id!r}: {fake_index}")
    if output_units == 1:
        if logits.numel() != 1:
            raise ValueError(
                f"Model {model_id!r} is configured for one binary logit but returned shape {tuple(logits.shape)}."
            )
        fake_logit = logits.reshape(-1)[0]
        if fake_index == 0:
            fake_logit = -fake_logit
        return float(torch.sigmoid(fake_logit).item())
    if output_units != class_count:
        raise ValueError(
            f"Model {model_id!r} must have either one binary output or one output per class; "
            f"configured {output_units} outputs for {class_count} classes."
        )
    if logits.shape[-1] != output_units:
        raise ValueError(
            f"Model {model_id!r} is configured for {output_units} outputs but returned shape {tuple(logits.shape)}."
        )
    return float(torch.softmax(logits, dim=-1)[..., fake_index].reshape(-1)[0].item())


def _is_fake(probability: float, model_id: str) -> bool:
    return probability >= float(_inference_config(model_id)["threshold"])


def _normalise(image: np.ndarray, model_id: str) -> torch.Tensor:
    config = _inference_config(model_id)
    if config["color_order"] != "RGB":
        raise ValueError(f"Unsupported inference color order for {model_id!r}: {config['color_order']!r}")
    interpolation = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }
    try:
        resize_interpolation = interpolation[config["interpolation"]]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported interpolation for {model_id!r}: {config['interpolation']!r}"
        ) from exc
    size = int(config["input_size"])
    image = cv2.resize(image, (size, size), interpolation=resize_interpolation).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    mean = torch.tensor(config["mean"], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config["std"], dtype=tensor.dtype).view(3, 1, 1)
    if torch.any(std <= 0):
        raise ValueError(f"Normalization standard deviations must be positive for model {model_id!r}.")
    return (tensor - mean) / std


def _model_input(raw_snapshots: list[dict[str, Any]], model_id: str, is_video: bool) -> torch.Tensor:
    config = _inference_config(model_id)
    view_policy = config["video_views"] if is_video else config["image_views"]
    if view_policy == "primary_face":
        selected = raw_snapshots[:1]
    elif view_policy == "all_face_crops":
        selected = raw_snapshots
        max_views = config.get("video_max_views")
        if is_video and max_views is not None:
            selected = selected[:int(max_views)]
    else:
        raise ValueError(f"Unsupported view policy for {model_id!r}: {view_policy!r}")
    if not selected:
        raise ValueError(f"No input snapshots available for model {model_id!r}.")
    return torch.stack([_normalise(snapshot["crop"], model_id) for snapshot in selected]).unsqueeze(0)


def _available_ensemble_members() -> list[str]:
    return [key for key in BASE_MODEL_KEYS if _checkpoint_path(key).is_file()]


def _missing_ensemble_members() -> list[str]:
    return [key for key in BASE_MODEL_KEYS if not _checkpoint_path(key).is_file()]


def log_checkpoint_status() -> None:
    for model_id, config in MODEL_REGISTRY.items():
        if model_id == "cross_ensemble":
            continue
        checkpoint_path = _checkpoint_path(model_id)
        status = "FOUND" if checkpoint_path.is_file() else "MISSING"
        logger.info(
            "Checkpoint %s | model=%s (%s) | path=%s",
            status,
            model_id,
            config["display_name"],
            checkpoint_path,
        )
    available = _available_ensemble_members()
    missing = _missing_ensemble_members()
    logger.info(
        "Ensemble members available=%d/%d | missing=%s",
        len(available),
        len(BASE_MODEL_KEYS),
        ", ".join(
            f"{MODEL_REGISTRY[key]['display_name']} ({key}): {_checkpoint_path(key)}"
            for key in missing
        ) or "none",
    )


def _get_cascades() -> dict[str, cv2.CascadeClassifier]:
    global _CASCADES
    if not _CASCADES:
        _CASCADES["alt2"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        _CASCADES["default"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _CASCADES["profile"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        _CASCADES["eye"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return _CASCADES


def _get_model(model_id: str) -> torch.nn.Module:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model selection '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}",
        )
    if model_id in _MODELS:
        return _MODELS[model_id]
    entry = MODEL_REGISTRY[model_id]
    if entry.get("checkpoint") is None:
        raise HTTPException(status_code=400, detail="Ensemble model must be invoked via composite execution.")
    checkpoint_path = _checkpoint_path(model_id)
    if not checkpoint_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Weights missing for {entry['display_name']} ({entry['dataset']}); "
                f"expected checkpoint: {checkpoint_path}"
            ),
        )
    model = build_model(entry["name"], pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatibility = model.load_state_dict(checkpoint["model_state"], strict=True)
    logger.info(
        "Checkpoint state validated | model=%s | architecture=%s | missing_keys=%s | unexpected_keys=%s",
        model_id,
        entry["name"],
        incompatibility.missing_keys,
        incompatibility.unexpected_keys,
    )
    model.eval()
    _MODELS[model_id] = model
    return model


def _detect_and_verify_faces(image: np.ndarray) -> list[dict[str, Any]]:
    """
    Scans, verifies, and rates human faces in the given RGB image.
    Uses multi-pass cascaded detection with geometric constraints, eye confirmation,
    and Laplacian sharpness assessment to reject non-face textures.
    """
    if image is None or image.size == 0:
        return []

    h_orig, w_orig = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    cascades = _get_cascades()

    # Pass 1: High-precision frontal face (alt2)
    raw_boxes = list(cascades["alt2"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(44, 44)))

    # Pass 2: Broader recall frontal face
    if not raw_boxes:
        raw_boxes = list(cascades["default"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(44, 44)))

    # Pass 3: Profile and angled face check
    if not raw_boxes:
        profiles = list(cascades["profile"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(44, 44)))
        if not profiles:
            flipped_gray = cv2.flip(gray, 1)
            f_profiles = cascades["profile"].detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=4, minSize=(44, 44))
            for fx, fy, fw, fh in f_profiles:
                profiles.append((w_orig - fx - fw, fy, fw, fh))
        raw_boxes = profiles

    if not raw_boxes:
        return []

    verified_faces: list[dict[str, Any]] = []
    for (x, y, w, h) in raw_boxes:
        # A. Aspect ratio validation: natural human faces have 0.75 <= h/w <= 1.60
        aspect = h / max(1, w)
        if aspect < 0.75 or aspect > 1.60:
            continue

        # B. Area / dimension check: ignore tiny artifacts
        if w < 40 or h < 40:
            continue

        # C. Margin crop with safety bounds
        margin = int(max(w, h) * 0.20)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_orig, x + w + margin)
        y2 = min(h_orig, y + h + margin)
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # D. Eye landmark verification inside upper 60% of candidate box
        face_gray = gray[y:y+h, x:x+w]
        upper_half = face_gray[:max(10, int(h * 0.60)), :]
        eyes = cascades["eye"].detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=2, minSize=(14, 14))
        has_eyes = len(eyes) > 0

        # E. Sharpness / clarity check using Laplacian variance
        sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        if sharpness > 90.0:
            clarity = "HIGH CLARITY"
        elif sharpness > 30.0:
            clarity = "ACCEPTABLE"
        else:
            clarity = "MOTION BLURRED"

        size_factor = min(1.0, (w * h) / (180.0 * 180.0))
        quality_score = sharpness * 0.4 + (100.0 if has_eyes else 30.0) + (size_factor * 50.0)

        verified_faces.append({
            "box": (int(x), int(y), int(w), int(h)),
            "crop": face_crop,
            "sharpness": round(sharpness, 1),
            "clarity": clarity,
            "eyes_detected": has_eyes,
            "quality_score": round(quality_score, 1),
            "size": f"{w}x{h}px",
        })

    verified_faces.sort(key=lambda f: f["quality_score"], reverse=True)
    return verified_faces


def _crop_face(image: np.ndarray) -> tuple[np.ndarray, bool]:
    verified = _detect_and_verify_faces(image)
    if verified:
        return verified[0]["crop"], True
    return image, False


def _extract_image_snapshots(image: np.ndarray) -> tuple[list[dict[str, Any]], bool]:
    h_orig, w_orig = image.shape[:2]
    verified_faces = _detect_and_verify_faces(image)
    face_detected = len(verified_faces) > 0
    snapshots: list[dict[str, Any]] = []

    if face_detected:
        primary = verified_faces[0]
        x, y, w, h = primary["box"]
        crop1 = primary["crop"]

        # Snapshot 1: Verified Primary Face
        snapshots.append({
            "name": "Snapshot 1 · Primary Verified Face",
            "region": f"Verified Face ({primary['clarity']}, {primary['size']})",
            "crop": crop1,
            "face_detected": True,
            "face_quality_score": primary["quality_score"],
            "face_clarity": primary["clarity"],
            "eyes_verified": primary["eyes_detected"],
        })

        # Snapshot 2: Contextual Face-Swap Seam & Hairline (45% margin)
        m2 = int(max(w, h) * 0.45)
        crop2 = image[max(0, y - m2):min(h_orig, y + h + m2), max(0, x - m2):min(w_orig, x + w + m2)]
        snapshots.append({
            "name": "Snapshot 2 · Face-Swap Boundary & Hairline Seam",
            "region": "Perimeter Seam & Blending Contour",
            "crop": crop2,
            "face_detected": True,
            "face_quality_score": primary["quality_score"],
            "face_clarity": primary["clarity"],
            "eyes_verified": primary["eyes_detected"],
        })

        # Snapshot 3: Ocular / Eye & Iris Gaze Region (upper 55%)
        h_c1 = crop1.shape[0]
        ocular_crop = crop1[:max(10, int(h_c1 * 0.55)), :]
        snapshots.append({
            "name": "Snapshot 3 · Ocular / Gaze Alignment",
            "region": "Eyes, Brow & Specular Reflection",
            "crop": ocular_crop,
            "face_detected": True,
            "face_quality_score": primary["quality_score"],
            "face_clarity": primary["clarity"],
            "eyes_verified": primary["eyes_detected"],
        })

        # Snapshot 4: Perioral / Mouth, Teeth & Speech Region (lower 55%)
        perioral_crop = crop1[max(0, int(h_c1 * 0.45)):, :]
        snapshots.append({
            "name": "Snapshot 4 · Perioral & Speech Articulation",
            "region": "Mouth, Teeth Realism & Chin",
            "crop": perioral_crop,
            "face_detected": True,
            "face_quality_score": primary["quality_score"],
            "face_clarity": primary["clarity"],
            "eyes_verified": primary["eyes_detected"],
        })

        # Snapshot 5: Secondary Face or Bilateral Invariance Check
        if len(verified_faces) > 1:
            sec = verified_faces[1]
            snapshots.append({
                "name": "Snapshot 5 · Secondary Verified Face",
                "region": f"Co-occurring Face ({sec['clarity']})",
                "crop": sec["crop"],
                "face_detected": True,
                "face_quality_score": sec["quality_score"],
                "face_clarity": sec["clarity"],
                "eyes_verified": sec["eyes_detected"],
            })
        else:
            snapshots.append({
                "name": "Snapshot 5 · Bilateral Invariance Check",
                "region": "Horizontal Symmetry Check",
                "crop": cv2.flip(crop1, 1),
                "face_detected": True,
                "face_quality_score": primary["quality_score"],
                "face_clarity": primary["clarity"],
                "eyes_verified": primary["eyes_detected"],
            })
    else:
        # Fallback when no human face is detected
        min_dim = min(h_orig, w_orig)
        cx, cy = w_orig // 2, h_orig // 2
        half = min_dim // 2
        center_crop = image[max(0, cy - half):min(h_orig, cy + half), max(0, cx - half):min(w_orig, cx + half)]

        snapshots.append({
            "name": "Snapshot 1 · Center Region [Unverified]",
            "region": "Center Region (No Face Located)",
            "crop": center_crop,
            "face_detected": False,
            "face_quality_score": 0.0,
            "face_clarity": "NO FACE DETECTED",
            "eyes_verified": False,
        })
        snapshots.append({
            "name": "Snapshot 2 · Global Overview [Unverified]",
            "region": "Full Frame (No Face Located)",
            "crop": image,
            "face_detected": False,
            "face_quality_score": 0.0,
            "face_clarity": "NO FACE DETECTED",
            "eyes_verified": False,
        })
        snapshots.append({
            "name": "Snapshot 3 · Upper Frame [Unverified]",
            "region": "Upper Section (No Face Located)",
            "crop": image[:int(h_orig * 0.65), :],
            "face_detected": False,
            "face_quality_score": 0.0,
            "face_clarity": "NO FACE DETECTED",
            "eyes_verified": False,
        })
        snapshots.append({
            "name": "Snapshot 4 · Lower Frame [Unverified]",
            "region": "Lower Section (No Face Located)",
            "crop": image[int(h_orig * 0.35):, :],
            "face_detected": False,
            "face_quality_score": 0.0,
            "face_clarity": "NO FACE DETECTED",
            "eyes_verified": False,
        })
        snapshots.append({
            "name": "Snapshot 5 · Mirror Overview [Unverified]",
            "region": "Symmetry Check (No Face Located)",
            "crop": cv2.flip(center_crop, 1),
            "face_detected": False,
            "face_quality_score": 0.0,
            "face_clarity": "NO FACE DETECTED",
            "eyes_verified": False,
        })

    return snapshots, face_detected


def _extract_video_snapshots(temp_path: Path) -> tuple[list[dict[str, Any]], bool]:
    """
    Intelligently scans the video timeline for candidate frames, detects and verifies
    faces in each candidate, filters out non-face frames/blurs, and extracts the 6 best
    diverse face snapshots across time.
    """
    capture = cv2.VideoCapture(str(temp_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Cannot decode video file. Format may be unsupported.")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    # Broad candidate scan (up to 24 frames across the video)
    sample_budget = min(24, max(8, total))
    candidate_indices = np.linspace(0, total - 1, sample_budget, dtype=int)

    face_candidates: list[dict[str, Any]] = []
    fallback_frames: list[dict[str, Any]] = []

    for idx in candidate_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_sec = round(float(idx) / float(fps), 2) if fps > 0 else 0.0

        verified = _detect_and_verify_faces(rgb)
        if verified:
            best_face = verified[0]
            face_candidates.append({
                "frame_index": int(idx),
                "timestamp_sec": t_sec,
                "crop": best_face["crop"],
                "quality_score": best_face["quality_score"],
                "sharpness": best_face["sharpness"],
                "clarity": best_face["clarity"],
                "eyes_detected": best_face["eyes_detected"],
                "size": best_face["size"],
                "face_detected": True,
            })
        else:
            fallback_frames.append({
                "frame_index": int(idx),
                "timestamp_sec": t_sec,
                "crop": rgb,
                "quality_score": 0.0,
                "sharpness": 0.0,
                "clarity": "NO FACE DETECTED",
                "eyes_detected": False,
                "size": "0x0px",
                "face_detected": False,
            })

    capture.release()

    snapshots: list[dict[str, Any]] = []
    face_detected_any = len(face_candidates) > 0

    if face_detected_any:
        # Sort face candidates chronologically by frame index
        face_candidates.sort(key=lambda c: c["frame_index"])

        # Pick 6 diverse verified face snapshots across time
        if len(face_candidates) >= 6:
            chosen_indices = np.linspace(0, len(face_candidates) - 1, 6, dtype=int)
            chosen_candidates = [face_candidates[i] for i in chosen_indices]
        else:
            chosen_candidates = list(face_candidates)
            while len(chosen_candidates) < 6:
                chosen_candidates.append(chosen_candidates[-1])

        for i, cand in enumerate(chosen_candidates):
            snapshots.append({
                "name": f"Snapshot {i + 1} · {cand['timestamp_sec']}s",
                "region": f"Verified Face ({cand['clarity']}, {cand['size']})",
                "crop": cand["crop"],
                "timestamp_sec": cand["timestamp_sec"],
                "frame_index": cand["frame_index"],
                "face_detected": True,
                "face_quality_score": cand["quality_score"],
                "face_clarity": cand["clarity"],
                "eyes_verified": cand["eyes_detected"],
            })
    else:
        # If no faces were detected anywhere in the entire video
        chosen_fallbacks = fallback_frames[:6]
        while len(chosen_fallbacks) < 6 and chosen_fallbacks:
            chosen_fallbacks.append(chosen_fallbacks[-1])
        if not chosen_fallbacks:
            raise HTTPException(status_code=400, detail="No readable frames found in uploaded video.")

        for i, cand in enumerate(chosen_fallbacks):
            snapshots.append({
                "name": f"Snapshot {i + 1} · {cand['timestamp_sec']}s [Unverified]",
                "region": "Unverified Frame (No Face Located)",
                "crop": cand["crop"],
                "timestamp_sec": cand["timestamp_sec"],
                "frame_index": cand["frame_index"],
                "face_detected": False,
                "face_quality_score": 0.0,
                "face_clarity": "NO FACE DETECTED",
                "eyes_verified": False,
            })

    return snapshots, face_detected_any


def _extract_snapshots_from_upload(
    filename: str, payload: bytes
) -> tuple[list[dict[str, Any]], bool, np.ndarray]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        try:
            image = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {e}")
        snapshots, detected = _extract_image_snapshots(image)
        primary_preview = snapshots[0]["crop"] if snapshots else image
        return snapshots, detected, primary_preview

    # Video handling with Windows-safe temporary file management
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / f"upload_{int(time.time()*1000)}{suffix or '.mp4'}"

    try:
        temp_file.write_bytes(payload)
        snapshots, detected = _extract_video_snapshots(temp_file)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        try:
            Path(temp_dir).rmdir()
        except Exception:
            pass

    primary_preview = snapshots[0]["crop"] if snapshots else np.zeros((224, 224, 3), dtype=np.uint8)
    return snapshots, detected, primary_preview


def _data_uri(image: np.ndarray) -> str:
    image = np.clip(image, 0, 1)
    buffer = io.BytesIO()
    Image.fromarray((image * 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _explain(model: torch.nn.Module, model_id: str, frame: torch.Tensor) -> tuple[str, str, float]:
    backbone = getattr(model, "backbone", model)
    layers = [module for module in backbone.modules() if isinstance(module, torch.nn.Conv2d)]
    
    if not layers:
        return "", "Attention Saliency", 0.0

    target_layer = layers[-1]
    cam = GradCAM(model, target_layer)
    frame_input = frame.detach().clone().unsqueeze(0)
    attribution = cam(frame_input)
    cam.remove_hooks()

    # Create visualization
    image = frame.detach().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    figure, axis = plt.subplots(figsize=(4.8, 4.8))
    axis.imshow(image)
    axis.imshow(attribution, cmap="magma", alpha=0.55)
    axis.set_title("Attribution Heatmap (Model Focus)")
    axis.axis("off")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    plt.close(figure)

    # Faithfulness drop: mask the top 10% attribution pixels
    faithfulness_mask = torch.from_numpy(attribution >= np.quantile(attribution, 0.9))
    masked = frame.detach().clone().unsqueeze(0)
    masked[0, :, faithfulness_mask] = 0
    with torch.no_grad():
        original = _fake_probability(model(frame.detach().clone().unsqueeze(0)), model_id)
        perturbed = _fake_probability(model(masked), model_id)

    method_name = "Grad-CAM"
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), method_name, max(0.0, original - perturbed)


@app.get("/api/health")
def health() -> dict[str, Any]:
    available_models = {}
    for key, val in MODEL_REGISTRY.items():
        if key == "cross_ensemble":
            available_models[key] = bool(_available_ensemble_members())
        else:
            available_models[key] = _checkpoint_path(key).is_file()

    available_count = sum(1 for v in available_models.values() if v)
    return {
        "status": "ok",
        "service": "RobustDeep Inference API",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_available": available_count,
        "models": available_models,
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    models_info = []
    for model_id, config in MODEL_REGISTRY.items():
        if model_id == "cross_ensemble":
            available_members = _available_ensemble_members()
            missing_members = _missing_ensemble_members()
            ckpt_exists = bool(available_members)
            member_status = [
                {
                    "id": member_id,
                    "name": MODEL_REGISTRY[member_id]["display_name"],
                    "dataset": MODEL_REGISTRY[member_id]["dataset"],
                    "checkpoint_path": str(_checkpoint_path(member_id)),
                    "available": member_id in available_members,
                }
                for member_id in BASE_MODEL_KEYS
            ]
        else:
            checkpoint_path = _checkpoint_path(model_id)
            ckpt_exists = checkpoint_path.is_file()
            member_status = None
            available_members = []
            missing_members = []

        models_info.append({
            "id": model_id,
            "display_name": config["display_name"],
            "dataset": config["dataset"],
            "architecture": config["architecture"],
            "available": ckpt_exists,
            "description": config["description"] if model_id != "cross_ensemble" else (
                f"{config['description']} Available members: {len(available_members)}/{len(BASE_MODEL_KEYS)}."
            ),
            "is_ensemble": model_id == "cross_ensemble",
            "checkpoint_path": None if model_id == "cross_ensemble" else str(_checkpoint_path(model_id)),
            "members": member_status,
            "missing_members": missing_members,
            "inference_config": dict(_inference_config(model_id)),
        })
    return {"models": models_info}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: Optional[str] = Query(None),
    model: Optional[str] = Form(None),
) -> dict[str, Any]:
    start_time = time.time()
    chosen_model = model or model_id or "cross_ensemble"
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video or image file is required.")
    
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if chosen_model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model selection '{chosen_model}'. Available: {list(MODEL_REGISTRY.keys())}",
        )

    if chosen_model == "cross_ensemble":
        available_members = _available_ensemble_members()
        missing_members = _missing_ensemble_members()
        if not available_members:
            missing_details = "; ".join(
                f"{MODEL_REGISTRY[key]['display_name']} ({key}): {_checkpoint_path(key)}"
                for key in missing_members
            )
            raise HTTPException(
                status_code=503,
                detail=f"Ensemble has no available checkpoints. Missing members: {missing_details}",
            )
        sub_models = {key: _get_model(key) for key in available_members}
        expl_model = next(iter(sub_models.values()))
    else:
        loaded_model = _get_model(chosen_model)
        available_members = []
        missing_members = []
        sub_models = {}
        expl_model = loaded_model

    # 1. Verified Multi-Snapshot Face Extraction
    raw_snapshots, face_detected, primary_crop = _extract_snapshots_from_upload(file.filename, payload)
    suffix = Path(file.filename).suffix.lower()
    is_video = suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    model_ids = available_members if chosen_model == "cross_ensemble" else [chosen_model]
    model_inputs = {
        model_id: _model_input(raw_snapshots, model_id, is_video)
        for model_id in model_ids
    }
    decision_view_count = max(int(model_input.shape[1]) for model_input in model_inputs.values())
    display_model_id = model_ids[0]
    for snap in raw_snapshots:
        snap["preview"] = _data_uri(snap["crop"].astype(np.float32) / 255.0)

    entry = MODEL_REGISTRY[chosen_model]
    ensemble_breakdown = None
    consensus_text = None
    display_threshold_model_id = chosen_model

    if chosen_model == "cross_ensemble":
        breakdown = []
        for key in available_members:
            sub_m = sub_models[key]
            sub_entry = MODEL_REGISTRY[key]
            with torch.no_grad():
                sub_logits = sub_m(model_inputs[key])
                sub_score = _fake_probability(sub_logits, key)
            breakdown.append({
                "id": key,
                "name": sub_entry["display_name"],
                "architecture": sub_entry["architecture"],
                "dataset": sub_entry["dataset"],
                "score": round(sub_score, 4),
                "label": "FAKE" if _is_fake(sub_score, key) else "REAL",
                "weight": round(1.0 / len(available_members), 4),
                "status": "available",
                "checkpoint_path": str(_checkpoint_path(key)),
            })

        overall_score = float(np.mean([item["score"] for item in breakdown]))
        ensemble_threshold = float(_inference_config(chosen_model)["threshold"])
        for key in missing_members:
            sub_entry = MODEL_REGISTRY[key]
            breakdown.append({
                "id": key,
                "name": sub_entry["display_name"],
                "architecture": sub_entry["architecture"],
                "dataset": sub_entry["dataset"],
                "score": None,
                "label": "MISSING",
                "weight": 0.0,
                "status": "missing",
                "checkpoint_path": str(_checkpoint_path(key)),
            })
        ensemble_breakdown = breakdown
        fake_votes = sum(1 for item in breakdown if item["label"] == "FAKE")
        real_votes = len(available_members) - fake_votes
        if overall_score >= ensemble_threshold:
            consensus_text = (
                f"{fake_votes}/{len(available_members)} available models vote FAKE "
                f"({len(available_members)}/{len(BASE_MODEL_KEYS)} ensemble members available)"
            )
        else:
            consensus_text = (
                f"{real_votes}/{len(available_members)} available models vote REAL "
                f"({len(available_members)}/{len(BASE_MODEL_KEYS)} ensemble members available)"
            )

        processed_snapshots = []
        for i, snap in enumerate(raw_snapshots):
            t_input = snap["tensor"].unsqueeze(0)
            with torch.no_grad():
                snap_sub_scores = [
                    float(torch.sigmoid(sub_models[k](t_input)).view(-1)[0].item())
                for k in available_members
                ]
            snap_score = float(np.mean(snap_sub_scores))
            snap_label = "FAKE" if snap_score >= 0.5 else "REAL"

            expl_img, method_name, faith_drop = _explain(expl_model, snap["tensor"])

            processed_snapshots.append({
                "id": i,
                "name": snap["name"],
                "region": snap.get("region", "Face Crop"),
                "timestamp_sec": snap.get("timestamp_sec"),
                "frame_index": snap.get("frame_index"),
                "preview": snap["preview"],
                "explanation_image": expl_img,
                "score": round(snap_score, 4),
                "confidence": round(max(snap_score, 1.0 - snap_score), 4),
                "label": snap_label,
                "faithfulness_confidence_drop": round(faith_drop, 4),
                "face_detected": snap["face_detected"],
                "face_quality_score": snap.get("face_quality_score", 0.0),
                "face_clarity": snap.get("face_clarity", "VERIFIED"),
                "eyes_verified": snap.get("eyes_verified", False),
                "is_peak_anomaly": False,
            })
    else:
        with torch.no_grad():
            seq_logits = loaded_model(model_inputs[chosen_model])
            overall_score = _fake_probability(seq_logits, chosen_model)

    processed_snapshots = []
    for i, snap in enumerate(raw_snapshots):
        display_tensor = _normalise(snap["crop"], display_model_id)
        if chosen_model == "cross_ensemble":
            with torch.no_grad():
                snap_sub_scores = [
                    _fake_probability(
                        sub_models[key](_normalise(snap["crop"], key).unsqueeze(0)),
                        key,
                    )
                    for key in available_members
                ]
            snap_score = float(np.mean(snap_sub_scores))
        else:
            with torch.no_grad():
                snap_logits = loaded_model(display_tensor.unsqueeze(0))
                snap_score = _fake_probability(snap_logits, chosen_model)
        snap_label = "FAKE" if _is_fake(snap_score, chosen_model) else "REAL"

        expl_img, method_name, faith_drop = _explain(
            expl_model,
            display_model_id,
            display_tensor,
        )

        image_decision_view = i == 0
        processed_snapshots.append({
            "id": i,
            "name": snap["name"],
            "region": snap.get("region", "Face Crop"),
            "timestamp_sec": snap.get("timestamp_sec"),
            "frame_index": snap.get("frame_index"),
            "preview": snap["preview"],
            "explanation_image": expl_img,
            "score": round(snap_score, 4),
            "confidence": round(max(snap_score, 1.0 - snap_score), 4),
            "label": snap_label,
            "faithfulness_confidence_drop": round(faith_drop, 4),
            "face_detected": snap["face_detected"],
            "face_quality_score": snap.get("face_quality_score", 0.0),
            "face_clarity": snap.get("face_clarity", "VERIFIED"),
            "eyes_verified": snap.get("eyes_verified", False),
            "used_for_decision": i < decision_view_count if is_video else image_decision_view,
            "is_peak_anomaly": False,
        })

    decision_snapshots = [snapshot for snapshot in processed_snapshots if snapshot["used_for_decision"]]
    if decision_snapshots:
        max_decision_idx = int(np.argmax([snapshot["score"] for snapshot in decision_snapshots]))
        peak_snapshot = decision_snapshots[max_decision_idx]
        max_snap_idx = peak_snapshot["id"]
    elif processed_snapshots:
        max_snap_idx = int(np.argmax([s["score"] for s in processed_snapshots]))
        processed_snapshots[max_snap_idx]["is_peak_anomaly"] = True
        peak_snapshot = processed_snapshots[max_snap_idx]
    else:
        max_snap_idx = 0
        peak_snapshot = None
    if peak_snapshot:
        peak_snapshot["is_peak_anomaly"] = True

    total_snaps = len(decision_snapshots)
    fake_snaps = sum(1 for s in decision_snapshots if s["label"] == "FAKE")
    real_snaps = total_snaps - fake_snaps
    agreement_snaps = max(fake_snaps, real_snaps)
    agreement_pct = round((agreement_snaps / total_snaps) * 100, 1) if total_snaps > 0 else 100.0

    all_scores = [s["score"] for s in decision_snapshots]
    verified_face_count = sum(1 for s in decision_snapshots if s["face_detected"])

    multi_snapshot_stats = {
        "total_snapshots": total_snaps,
        "verified_faces_count": verified_face_count,
        "fake_count": fake_snaps,
        "real_count": real_snaps,
        "agreement_pct": agreement_pct,
        "agreement_text": f"{agreement_snaps} of {total_snaps} decision snapshots agree ({agreement_pct}% consistency)",
        "mean_score": round(float(np.mean(all_scores)), 4) if all_scores else round(overall_score, 4),
        "max_score": round(float(np.max(all_scores)), 4) if all_scores else round(overall_score, 4),
        "min_score": round(float(np.min(all_scores)), 4) if all_scores else round(overall_score, 4),
        "variance": round(float(np.var(all_scores)), 5) if all_scores else 0.0,
        "peak_anomaly_index": max_snap_idx,
    }

    overall_is_fake = _is_fake(overall_score, chosen_model)
    primary_snap = peak_snapshot if (overall_is_fake and peak_snapshot) else processed_snapshots[0]
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "model": chosen_model,
        "model_name": entry["display_name"],
        "architecture": entry["architecture"],
        "dataset": entry["dataset"],
        "label": "FAKE" if overall_is_fake else "REAL",
        "score": round(overall_score, 4),
        "confidence": round(max(overall_score, 1.0 - overall_score), 4),
        "inference_config": dict(_inference_config(chosen_model)),
        "face_detected": face_detected,
        "explanation_method": method_name if "method_name" in locals() else "Grad-CAM",
        "explanation_image": primary_snap["explanation_image"] if primary_snap else "",
        "input_preview": primary_snap["preview"] if primary_snap else _data_uri(primary_crop.astype(np.float32) / 255.0),
        "faithfulness_confidence_drop": primary_snap["faithfulness_confidence_drop"] if primary_snap else 0.0,
        "latency_ms": elapsed_ms,
        "ensemble_breakdown": ensemble_breakdown,
        "consensus": consensus_text,
        "ensemble_members_used": available_members if chosen_model == "cross_ensemble" else [],
        "missing_ensemble_members": [
            {
                "id": key,
                "name": MODEL_REGISTRY[key]["display_name"],
                "checkpoint_path": str(_checkpoint_path(key)),
            }
            for key in missing_members
        ] if chosen_model == "cross_ensemble" else [],
        "snapshots": processed_snapshots,
        "multi_snapshot_analysis": multi_snapshot_stats,
        "disclaimer": "Model attribution is research evidence, not forensic proof.",
    }


# =============================================================================
# NEW: Ensemble "Analyze All Models" endpoint  (requirement §1–§5)
# =============================================================================

def _ensemble_extract_video_face_frames(
    payload: bytes,
    filename: str,
) -> tuple[list[dict[str, Any]], int]:
    """
    Sample N frames from a video, detect faces in each.

    Returns (frame_face_crops, total_frames_sampled).
    Each item in frame_face_crops is {"frame_index": int, "faces": [...]}.
    """
    suffix = Path(filename).suffix.lower()
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / f"upload_{int(time.time() * 1000)}{suffix or '.mp4'}"
    try:
        temp_file.write_bytes(payload)
        cap = cv2.VideoCapture(str(temp_file))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot decode video file.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        registry = get_registry()
        n_frames = min(registry.max_video_frames, max(registry.min_video_frames, total))
        indices = np.linspace(0, total - 1, n_frames, dtype=int)

        frame_face_crops: list[dict[str, Any]] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                frame_face_crops.append({"frame_index": int(idx), "faces": []})
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            verified = _detect_and_verify_faces(rgb)
            faces = [{"crop": v["crop"], "box": v["box"]} for v in verified]
            frame_face_crops.append({"frame_index": int(idx), "faces": faces})
        cap.release()
        return frame_face_crops, len(indices)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        try:
            Path(temp_dir).rmdir()
        except Exception:
            pass


def _verdict_to_dict(v: EnsembleVerdict) -> dict[str, Any]:
    """Serialise an EnsembleVerdict to a JSON-friendly dict."""
    model_details = []
    for mr in v.model_details:
        d: dict[str, Any] = {
            "model_id": mr.model_id,
            "display_name": mr.display_name,
            "dataset": mr.dataset,
            "architecture": mr.architecture,
            "probability": mr.probability,
            "weight": mr.weight,
            "val_auc": mr.val_auc,
            "included": mr.included,
            "excluded_reason": mr.excluded_reason,
        }
        if mr.frame_probs is not None:
            d["frames_flagged_fake"] = mr.frames_flagged_fake
            d["total_frames"] = mr.total_frames
            d["frame_variance"] = mr.frame_variance
        model_details.append(d)

    per_face = []
    for fr in v.per_face:
        per_face.append({
            "face_index": fr.face_index,
            "box": fr.box,
            "fused_score": fr.fused_score,
            "verdict": fr.verdict,
            "confidence": fr.confidence,
            "model_results": [
                {
                    "model_id": mr.model_id,
                    "probability": mr.probability,
                    "weight": mr.weight,
                    "included": mr.included,
                }
                for mr in fr.model_results
            ],
        })

    return {
        "verdict": v.verdict,
        "confidence": v.confidence,
        "fused_score": v.fused_score,
        "face_count": v.face_count,
        "frames_analyzed": v.frames_analyzed,
        "threshold_used": v.threshold_used,
        "disagreement_std": v.disagreement_std,
        "face_frame_fraction": v.face_frame_fraction,
        "frame_variance": v.frame_variance,
        "models_loaded": v.models_loaded,
        "models_total": v.models_total,
        "models_excluded": v.models_excluded,
        "model_details": model_details,
        "per_face": per_face,
        "disclaimer": "Model attribution is research evidence, not forensic proof.",
    }


@app.post("/api/ensemble-predict")
async def ensemble_predict(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Analyze All Models — the primary endpoint.

    Detects faces first; if no face → returns "No face detected" without
    running any model.  For images with multiple faces, reports per face and
    uses the highest-risk face for the overall verdict.  For videos, samples
    16–32 frames and averages model probabilities across frames.
    """
    start = time.time()
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = Path(file.filename).suffix.lower()
    is_video = suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    registry = get_registry()

    if is_video:
        # --- Video path ---
        frame_face_crops, total_sampled = _ensemble_extract_video_face_frames(
            payload, file.filename,
        )
        n_face_frames = sum(1 for f in frame_face_crops if f["faces"])
        if n_face_frames == 0:
            return {
                "verdict": "NO_FACE",
                "message": "No face detected - cannot analyze",
                "face_count": 0,
                "frames_analyzed": total_sampled,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "disclaimer": "Model attribution is research evidence, not forensic proof.",
            }
        if n_face_frames < registry.min_face_frames:
            return {
                "verdict": "INSUFFICIENT_FRAMES",
                "message": (
                    f"Not enough face frames for reliable analysis "
                    f"({n_face_frames}/{total_sampled}, need ≥{registry.min_face_frames})"
                ),
                "face_count": 0,
                "frames_analyzed": total_sampled,
                "face_frames": n_face_frames,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "disclaimer": "Model attribution is research evidence, not forensic proof.",
            }
        result = analyze_video_frames(frame_face_crops, registry)
    else:
        # --- Image path ---
        try:
            image = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

        verified_faces = _detect_and_verify_faces(image)
        if not verified_faces:
            return {
                "verdict": "NO_FACE",
                "message": "No face detected - cannot analyze",
                "face_count": 0,
                "frames_analyzed": 1,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "disclaimer": "Model attribution is research evidence, not forensic proof.",
            }

        face_crops = [
            {"crop": v["crop"], "box": v["box"]}
            for v in verified_faces
        ]
        result = analyze_image_faces(face_crops, registry)

    out = _verdict_to_dict(result)
    out["latency_ms"] = round((time.time() - start) * 1000, 1)
    out["is_video"] = is_video
    return out


@app.get("/api/ensemble-status")
def ensemble_status() -> dict[str, Any]:
    """Return which models are available, their weights and AUCs."""
    registry = get_registry()
    models = []
    for cfg in registry.all_configs():
        models.append({
            "model_id": cfg.model_id,
            "display_name": cfg.display_name,
            "dataset": cfg.dataset,
            "architecture": cfg.architecture,
            "checkpoint": str(cfg.checkpoint),
            "checkpoint_exists": cfg.checkpoint.is_file(),
            "val_auc": cfg.val_auc,
            "weight": cfg.weight,
            "threshold": cfg.threshold,
            "enabled": cfg.enabled,
            "load_error": cfg.load_error,
            "included_in_fusion": cfg.val_auc >= 0.6 and cfg.weight > 0,
        })
    return {
        "ensemble_threshold": registry.ensemble_threshold,
        "models": models,
        "total": len(models),
        "available": sum(1 for m in models if m["checkpoint_exists"]),
    }


@app.get("/api/diffusion-samples")
def list_diffusion_samples() -> dict[str, Any]:
    """Return count and list of available diffusion synthetic and real face samples."""
    fake_dir = PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "fake"
    real_dir = PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "real"
    fake_samples = sorted(p.name for p in fake_dir.glob("*.png")) if fake_dir.exists() else []
    real_samples = sorted(p.name for p in real_dir.glob("*.png")) if real_dir.exists() else []
    return {
        "total": len(fake_samples) + len(real_samples),
        "fake_count": len(fake_samples),
        "real_count": len(real_samples),
        "fake_samples": fake_samples,
        "real_samples": real_samples,
    }


@app.get("/api/diffusion-sample/random")
def get_random_diffusion_sample():
    """Return a random diffusion synthetic face crop image for live testing."""
    import random
    fake_dir = PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "fake"
    if not fake_dir.exists():
        raise HTTPException(status_code=404, detail="Diffusion face directory not found.")
    samples = list(fake_dir.glob("*.png"))
    if not samples:
        raise HTTPException(status_code=404, detail="No diffusion face samples available.")
    chosen = random.choice(samples)
    return FileResponse(
        str(chosen),
        media_type="image/png",
        filename=chosen.name,
        headers={"X-Sample-Name": chosen.name, "Access-Control-Expose-Headers": "X-Sample-Name"},
    )


@app.get("/api/diffusion-sample/random-real")
def get_random_real_sample():
    """Return a random real face crop image for live testing."""
    import random
    real_dir = PROJECT_ROOT / "data" / "diffusion_face" / "raw" / "real"
    if not real_dir.exists():
        raise HTTPException(status_code=404, detail="Real face directory not found.")
    samples = list(real_dir.glob("*.png"))
    if not samples:
        raise HTTPException(status_code=404, detail="No real face samples available.")
    chosen = random.choice(samples)
    return FileResponse(
        str(chosen),
        media_type="image/png",
        filename=chosen.name,
        headers={"X-Sample-Name": chosen.name, "Access-Control-Expose-Headers": "X-Sample-Name"},
    )


# Serve React build assets in production if built
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "service": "RobustDeep Inference API",
            "status": "online",
            "message": "Frontend build is not present. Run 'npm run build' inside frontend/ to serve the UI here.",
            "docs_url": "/docs",
            "health_url": "/api/health",
            "models_url": "/api/models",
        }
