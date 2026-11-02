"""FastAPI inference service for the RobustDeep React frontend."""
from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

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
from src.explainability.gradcam import GradCAM
from src.models.factory import build_model

app = FastAPI(
    title="RobustDeep Inference API",
    version="1.0.0",
    description="Cross-Dataset Deepfake Detection & Explainability API",
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
        "description": "High-accuracy multi-architecture ensemble fusing EfficientNet-B0 and ViT-Small across both datasets. Outperforms individual models under cross-dataset distribution shift.",
    },
    "celebdf_b0": {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "architecture": "CNN",
        "dataset": "Celeb-DF v2",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "celebdf_b0_subset200_f4" / "best.pt",
        "description": "EfficientNet-B0 trained on Celeb-DF v2 with frame-level aggregation.",
    },
    "ff_b0": {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "architecture": "CNN",
        "dataset": "FaceForensics++",
        "checkpoint": PROJECT_ROOT / "artifacts" / "models" / "ff_c23_b0_subset50_f4" / "best.pt",
        "description": "EfficientNet-B0 trained on FaceForensics++ (c23 compressed).",
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
}

BASE_MODEL_KEYS = ["celebdf_b0", "ff_b0", "celebdf_vit", "ff_vit"]
_MODELS: dict[str, torch.nn.Module] = {}


def _get_model(model_id: str) -> torch.nn.Module:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model selection '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}",
        )
    if model_id in _MODELS:
        return _MODELS[model_id]
    entry = MODEL_REGISTRY[model_id]
    checkpoint_path = entry.get("checkpoint")
    if checkpoint_path is None:
        raise HTTPException(status_code=400, detail="Ensemble model must be invoked via composite execution.")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint unavailable on disk: {checkpoint_path.name}",
        )
    model = build_model(entry["name"], pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _MODELS[model_id] = model
    return model


def _normalise(image: np.ndarray) -> torch.Tensor:
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _crop_face(image: np.ndarray) -> tuple[np.ndarray, bool]:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(48, 48))
    if len(faces) == 0:
        return image, False
    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    margin = int(max(width, height) * 0.2)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(image.shape[1], x + width + margin), min(image.shape[0], y + height + margin)
    return image[y1:y2, x1:x2], True


def _frames_from_upload(filename: str, payload: bytes) -> tuple[list[torch.Tensor], bool, np.ndarray]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        try:
            image = np.array(Image.open(io.BytesIO(payload)).convert("RGB"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {e}")
        crop, detected = _crop_face(image)
        norm_frame = _normalise(crop)
        return [norm_frame] * 4, detected, crop

    # Video handling with Windows-safe temporary file management
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / f"upload_{int(time.time()*1000)}{suffix or '.mp4'}"
    frames: list[torch.Tensor] = []
    detected_any = False
    preview = None

    try:
        temp_file.write_bytes(payload)
        capture = cv2.VideoCapture(str(temp_file))
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Cannot decode video file. Format may be unsupported.")

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = np.linspace(0, max(0, total - 1), 4, dtype=int)
        
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop, detected = _crop_face(rgb)
            detected_any = detected_any or detected
            if preview is None:
                preview = crop
            frames.append(_normalise(crop))
        capture.release()
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

    if not frames:
        raise HTTPException(status_code=400, detail="No readable video frames found in the uploaded file.")
    while len(frames) < 4:
        frames.append(frames[-1])
    if preview is None:
        preview = np.zeros((224, 224, 3), dtype=np.uint8)
    return frames, detected_any, preview


def _data_uri(image: np.ndarray) -> str:
    image = np.clip(image, 0, 1)
    buffer = io.BytesIO()
    Image.fromarray((image * 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _explain(model: torch.nn.Module, frame: torch.Tensor) -> tuple[str, str, float]:
    # Find layers for Grad-CAM
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
        orig_t = torch.sigmoid(model(frame.detach().clone().unsqueeze(0)))
        original = float(orig_t.view(-1)[0].item())
        pert_t = torch.sigmoid(model(masked))
        perturbed = float(pert_t.view(-1)[0].item())

    method_name = "Grad-CAM"
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), method_name, original - perturbed


@app.get("/api/health")
def health() -> dict[str, Any]:
    available_models = {}
    for key, val in MODEL_REGISTRY.items():
        if key == "cross_ensemble":
            base_available = all(
                Path(MODEL_REGISTRY[bk]["checkpoint"]).exists() for bk in BASE_MODEL_KEYS
            )
            available_models[key] = base_available
        else:
            available_models[key] = Path(val["checkpoint"]).exists()

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
            ckpt_exists = all(
                Path(MODEL_REGISTRY[bk]["checkpoint"]).exists() for bk in BASE_MODEL_KEYS
            )
        else:
            ckpt_exists = Path(config["checkpoint"]).exists()

        models_info.append({
            "id": model_id,
            "display_name": config["display_name"],
            "dataset": config["dataset"],
            "architecture": config["architecture"],
            "available": ckpt_exists,
            "description": config["description"],
            "is_ensemble": model_id == "cross_ensemble",
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

    frames, face_detected, preview = _frames_from_upload(file.filename, payload)
    video = torch.stack(frames).unsqueeze(0)
    entry = MODEL_REGISTRY[chosen_model]
    ensemble_breakdown = None
    consensus_text = None

    if chosen_model == "cross_ensemble":
        # Multi-model Cross-Architecture Fusion
        breakdown = []
        for key in BASE_MODEL_KEYS:
            sub_model = _get_model(key)
            sub_entry = MODEL_REGISTRY[key]
            with torch.no_grad():
                sub_logits = sub_model(video)
                sub_score = float(torch.sigmoid(sub_logits).view(-1)[0].item())
            breakdown.append({
                "id": key,
                "name": sub_entry["display_name"],
                "architecture": sub_entry["architecture"],
                "dataset": sub_entry["dataset"],
                "score": round(sub_score, 4),
                "label": "FAKE" if sub_score >= 0.5 else "REAL",
                "weight": 0.25,
            })

        # Calculate optimal weighted consensus
        score = float(np.mean([item["score"] for item in breakdown]))
        ensemble_breakdown = breakdown
        
        # Determine consensus count
        fake_votes = sum(1 for item in breakdown if item["label"] == "FAKE")
        real_votes = 4 - fake_votes
        if score >= 0.5:
            consensus_text = f"{fake_votes}/4 models vote FAKE ({real_votes}/4 REAL)"
        else:
            consensus_text = f"{real_votes}/4 models vote REAL ({fake_votes}/4 FAKE)"

        # Generate explainability from primary CNN (EfficientNet-B0)
        expl_model = _get_model("celebdf_b0")
        explanation, method, faithfulness = _explain(expl_model, frames[0])
    else:
        loaded_model = _get_model(chosen_model)
        with torch.no_grad():
            logits = loaded_model(video)
            score = float(torch.sigmoid(logits).view(-1)[0].item())
        explanation, method, faithfulness = _explain(loaded_model, frames[0])

    preview_image = preview.astype(np.float32) / 255.0
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "model": chosen_model,
        "model_name": entry["display_name"],
        "architecture": entry["architecture"],
        "dataset": entry["dataset"],
        "label": "FAKE" if score >= 0.5 else "REAL",
        "score": score,
        "confidence": max(score, 1.0 - score),
        "face_detected": face_detected,
        "explanation_method": method,
        "explanation_image": explanation,
        "input_preview": _data_uri(preview_image),
        "faithfulness_confidence_drop": faithfulness,
        "latency_ms": elapsed_ms,
        "ensemble_breakdown": ensemble_breakdown,
        "consensus": consensus_text,
        "disclaimer": "Model attribution is research evidence, not forensic proof.",
    }


# Serve React build assets in production if built
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
