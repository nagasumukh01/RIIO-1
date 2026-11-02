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
    version="1.1.0",
    description="Cross-Dataset Deepfake Detection & Multi-Snapshot Explainability API",
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


def _extract_image_snapshots(image: np.ndarray) -> tuple[list[dict[str, Any]], bool]:
    h_orig, w_orig = image.shape[:2]
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(48, 48))
    
    snapshots: list[dict[str, Any]] = []
    face_detected = len(faces) > 0
    
    if face_detected:
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        
        # Snapshot 1: Primary Tight Face Crop (margin 15%)
        m1 = int(max(w, h) * 0.15)
        crop1 = image[max(0, y - m1):min(h_orig, y + h + m1), max(0, x - m1):min(w_orig, x + w + m1)]
        snapshots.append({
            "name": "Snapshot 1 · Primary Face Geometry",
            "region": "Tight Facial Region",
            "crop": crop1,
            "face_detected": True,
        })
        
        # Snapshot 2: Contextual Boundary (wider margin 45% for hairline/jaw)
        m2 = int(max(w, h) * 0.45)
        crop2 = image[max(0, y - m2):min(h_orig, y + h + m2), max(0, x - m2):min(w_orig, x + w + m2)]
        snapshots.append({
            "name": "Snapshot 2 · Contextual Blending Boundary",
            "region": "Jawline & Hair Boundary",
            "crop": crop2,
            "face_detected": True,
        })
        
        # Snapshot 3: Upper Facial / Ocular Region (eyes, gaze, brow)
        h_c1 = crop1.shape[0]
        ocular_crop = crop1[:max(10, int(h_c1 * 0.60)), :]
        snapshots.append({
            "name": "Snapshot 3 · Ocular / Gaze Alignment",
            "region": "Eyes & Forehead",
            "crop": ocular_crop,
            "face_detected": True,
        })
        
        # Snapshot 4: Perioral / Lip & Mouth Region (lip-sync, chin artifacts)
        perioral_crop = crop1[max(0, int(h_c1 * 0.40)):, :]
        snapshots.append({
            "name": "Snapshot 4 · Perioral / Speech Articulation",
            "region": "Mouth & Chin",
            "crop": perioral_crop,
            "face_detected": True,
        })
        
        # Snapshot 5: Secondary Face or Horizontal Flip TTA Check
        if len(faces_sorted) > 1:
            xs, ys, ws, hs = faces_sorted[1]
            ms = int(max(ws, hs) * 0.2)
            crop_sec = image[max(0, ys - ms):min(h_orig, ys + hs + ms), max(0, xs - ms):min(w_orig, xs + ws + ms)]
            snapshots.append({
                "name": "Snapshot 5 · Secondary Subject Face",
                "region": "Co-occurring Subject",
                "crop": crop_sec,
                "face_detected": True,
            })
        else:
            snapshots.append({
                "name": "Snapshot 5 · Horizontal Mirror Check (TTA)",
                "region": "Bilateral Symmetry Check",
                "crop": cv2.flip(crop1, 1),
                "face_detected": True,
            })
    else:
        # Fallback for non-frontal or artistic media: multi-region spatial decomposition
        min_dim = min(h_orig, w_orig)
        cx, cy = w_orig // 2, h_orig // 2
        half = min_dim // 2
        center_crop = image[max(0, cy - half):min(h_orig, cy + half), max(0, cx - half):min(w_orig, cx + half)]
        
        snapshots.append({
            "name": "Snapshot 1 · Center Region",
            "region": "Central Focus",
            "crop": center_crop,
            "face_detected": False,
        })
        snapshots.append({
            "name": "Snapshot 2 · Global Media View",
            "region": "Full Frame Context",
            "crop": image,
            "face_detected": False,
        })
        snapshots.append({
            "name": "Snapshot 3 · Upper Frame Region",
            "region": "Upper Section",
            "crop": image[:int(h_orig * 0.65), :],
            "face_detected": False,
        })
        snapshots.append({
            "name": "Snapshot 4 · Lower Frame Region",
            "region": "Lower Section",
            "crop": image[int(h_orig * 0.35):, :],
            "face_detected": False,
        })
        snapshots.append({
            "name": "Snapshot 5 · Bilateral Inversion",
            "region": "Symmetry Check",
            "crop": cv2.flip(center_crop, 1),
            "face_detected": False,
        })
        
    return snapshots, face_detected


def _extract_video_snapshots(temp_path: Path) -> tuple[list[dict[str, Any]], bool]:
    capture = cv2.VideoCapture(str(temp_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Cannot decode video file. Format may be unsupported.")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    # Sample 6 distinct temporal snapshots across the video timeline
    num_samples = 6
    if total <= num_samples:
        indices = np.arange(total)
    else:
        indices = np.linspace(0, total - 1, num_samples, dtype=int)

    snapshots: list[dict[str, Any]] = []
    face_detected_any = False

    for i, idx in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop, detected = _crop_face(rgb)
        face_detected_any = face_detected_any or detected
        t_sec = round(float(idx) / float(fps), 2) if fps > 0 else 0.0

        snapshots.append({
            "name": f"Snapshot {i + 1} · {t_sec}s",
            "region": f"Timestamp {t_sec}s (Frame #{int(idx)})",
            "crop": crop,
            "timestamp_sec": t_sec,
            "frame_index": int(idx),
            "face_detected": detected,
        })

    capture.release()

    if not snapshots:
        raise HTTPException(status_code=400, detail="No readable video frames found in the uploaded file.")

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


def _explain(model: torch.nn.Module, frame: torch.Tensor) -> tuple[str, str, float]:
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
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), method_name, max(0.0, original - perturbed)


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

    # 1. Extract multi-snapshot sequence (multi-crop for images, multi-timestamp for videos)
    raw_snapshots, face_detected, primary_crop = _extract_snapshots_from_upload(file.filename, payload)

    # Prepare normalized tensors for all snapshots
    for snap in raw_snapshots:
        snap["tensor"] = _normalise(snap["crop"])
        snap["preview"] = _data_uri(snap["crop"].astype(np.float32) / 255.0)

    # Sequence tensor for sequence-level video aggregation (B=1, T=N, C=3, H=224, W=224)
    sequence_tensor = torch.stack([s["tensor"] for s in raw_snapshots]).unsqueeze(0)
    entry = MODEL_REGISTRY[chosen_model]
    ensemble_breakdown = None
    consensus_text = None

    # Explanation model reference (CNN is optimal for Grad-CAM)
    expl_model = _get_model("celebdf_b0") if chosen_model == "cross_ensemble" else _get_model(chosen_model)

    if chosen_model == "cross_ensemble":
        # 2a. Multi-model Cross-Architecture Fusion across sequence
        sub_models = {k: _get_model(k) for k in BASE_MODEL_KEYS}
        breakdown = []
        for key in BASE_MODEL_KEYS:
            sub_m = sub_models[key]
            sub_entry = MODEL_REGISTRY[key]
            with torch.no_grad():
                sub_logits = sub_m(sequence_tensor)
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

        overall_score = float(np.mean([item["score"] for item in breakdown]))
        ensemble_breakdown = breakdown
        fake_votes = sum(1 for item in breakdown if item["label"] == "FAKE")
        real_votes = 4 - fake_votes
        if overall_score >= 0.5:
            consensus_text = f"{fake_votes}/4 models vote FAKE ({real_votes}/4 REAL)"
        else:
            consensus_text = f"{real_votes}/4 models vote REAL ({fake_votes}/4 FAKE)"

        # 2b. Evaluate each individual snapshot across the 4 ensemble models
        processed_snapshots = []
        for i, snap in enumerate(raw_snapshots):
            t_input = snap["tensor"].unsqueeze(0)
            with torch.no_grad():
                snap_sub_scores = [
                    float(torch.sigmoid(sub_models[k](t_input)).view(-1)[0].item())
                    for k in BASE_MODEL_KEYS
                ]
            snap_score = float(np.mean(snap_sub_scores))
            snap_label = "FAKE" if snap_score >= 0.5 else "REAL"
            
            # Generate Grad-CAM for this snapshot
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
                "is_peak_anomaly": False,
            })
    else:
        # Single model evaluation
        loaded_model = _get_model(chosen_model)
        with torch.no_grad():
            seq_logits = loaded_model(sequence_tensor)
            overall_score = float(torch.sigmoid(seq_logits).view(-1)[0].item())

        processed_snapshots = []
        for i, snap in enumerate(raw_snapshots):
            t_input = snap["tensor"].unsqueeze(0)
            with torch.no_grad():
                s_logits = loaded_model(t_input)
                snap_score = float(torch.sigmoid(s_logits).view(-1)[0].item())
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
                "is_peak_anomaly": False,
            })

    # 3. Identify Peak Anomaly Snapshot (highest fake score)
    if processed_snapshots:
        max_snap_idx = int(np.argmax([s["score"] for s in processed_snapshots]))
        processed_snapshots[max_snap_idx]["is_peak_anomaly"] = True
        peak_snapshot = processed_snapshots[max_snap_idx]
    else:
        max_snap_idx = 0
        peak_snapshot = None

    # Multi-snapshot summary analytics
    total_snaps = len(processed_snapshots)
    fake_snaps = sum(1 for s in processed_snapshots if s["label"] == "FAKE")
    real_snaps = total_snaps - fake_snaps
    agreement_snaps = max(fake_snaps, real_snaps)
    agreement_pct = round((agreement_snaps / total_snaps) * 100, 1) if total_snaps > 0 else 100.0

    all_scores = [s["score"] for s in processed_snapshots]
    multi_snapshot_stats = {
        "total_snapshots": total_snaps,
        "fake_count": fake_snaps,
        "real_count": real_snaps,
        "agreement_pct": agreement_pct,
        "agreement_text": f"{agreement_snaps} of {total_snaps} snapshots agree ({agreement_pct}% temporal consistency)",
        "mean_score": round(float(np.mean(all_scores)), 4) if all_scores else round(overall_score, 4),
        "max_score": round(float(np.max(all_scores)), 4) if all_scores else round(overall_score, 4),
        "min_score": round(float(np.min(all_scores)), 4) if all_scores else round(overall_score, 4),
        "variance": round(float(np.var(all_scores)), 5) if all_scores else 0.0,
        "peak_anomaly_index": max_snap_idx,
    }

    # Primary representation: default to peak anomaly if fake, else primary crop
    primary_snap = peak_snapshot if (overall_score >= 0.5 and peak_snapshot) else processed_snapshots[0]
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "model": chosen_model,
        "model_name": entry["display_name"],
        "architecture": entry["architecture"],
        "dataset": entry["dataset"],
        "label": "FAKE" if overall_score >= 0.5 else "REAL",
        "score": round(overall_score, 4),
        "confidence": round(max(overall_score, 1.0 - overall_score), 4),
        "face_detected": face_detected,
        "explanation_method": method_name if "method_name" in locals() else "Grad-CAM",
        "explanation_image": primary_snap["explanation_image"] if primary_snap else "",
        "input_preview": primary_snap["preview"] if primary_snap else _data_uri(primary_crop.astype(np.float32) / 255.0),
        "faithfulness_confidence_drop": primary_snap["faithfulness_confidence_drop"] if primary_snap else 0.0,
        "latency_ms": elapsed_ms,
        "ensemble_breakdown": ensemble_breakdown,
        "consensus": consensus_text,
        "snapshots": processed_snapshots,
        "multi_snapshot_analysis": multi_snapshot_stats,
        "disclaimer": "Model attribution is research evidence, not forensic proof.",
    }


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
