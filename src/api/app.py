"""
RobustDeep FastAPI Backend
Provides REST API endpoints for inference, experiment results, and system info.

Run with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Or from project root:
    python -m uvicorn src.api.app:app --reload
"""

import os
import sys
import json
import time
import base64
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# FastAPI
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging_utils import get_logger

logger = get_logger("api")

# ─── App Initialization ──────────────────────────────────────────────────────

app = FastAPI(
    title="RobustDeep API",
    description="Deepfake Detection Research API — explainable ViT-based detection under cross-dataset distribution shift",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── State ───────────────────────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {}


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: str          # "real" or "fake"
    confidence: float        # probability of fake class [0, 1]
    fake_probability: float
    real_probability: float
    model_used: str
    inference_time_ms: float
    disclaimer: str = (
        "This is a probabilistic research system and must NOT be treated as "
        "conclusive forensic evidence."
    )


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    backbone: str
    total_parameters: str
    is_cached: bool


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    status: str
    model_name: str
    source_dataset: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    torch_available: bool
    gpu_available: bool
    version: str


# ─── Helper: Load Model ──────────────────────────────────────────────────────

def _load_model(model_key: str = "efficientnet_b0", device: str = "cpu"):
    """Load model, caching after first load."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    try:
        from src.utils.config import ModelConfig
        from src.models.factory import create_model

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

        try:
            model = create_model(config, device)
        except Exception as e:
            logger.warning(f"create_model with pretrained={config.pretrained} failed: {e}. Retrying with pretrained=False...")
            config.pretrained = False
            model = create_model(config, device)

        model.eval()

        _model_cache[model_key] = {
            "model": model,
            "config": config,
            "device": device,
        }
        logger.info(f"Model loaded: {config.name}")
        return _model_cache[model_key]

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")


def _preprocess_image(image_bytes: bytes):
    """Preprocess uploaded image for inference."""
    import torch
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, img


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", response_model=Dict)
async def root():
    """API root — returns basic info."""
    return {
        "name": "RobustDeep API",
        "version": "1.0.0",
        "description": "Deepfake detection research API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "models": "/models",
            "experiments": "/experiments",
            "docs": "/docs",
        },
        "disclaimer": (
            "Research system only. Not for forensic or legal use."
        ),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        import torch
        torch_ok = True
        gpu_ok = torch.cuda.is_available()
    except ImportError:
        torch_ok = False
        gpu_ok = False

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        torch_available=torch_ok,
        gpu_available=gpu_ok,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = "efficientnet_b0",
):
    """
    Predict whether an uploaded image is real or deepfake.

    - **file**: Image file (JPEG, PNG, WebP)
    - **model**: Model to use (efficientnet_b0, vit_base, hybrid)

    Returns prediction, confidence, and inference time.
    """
    import torch

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    try:
        image_bytes = await file.read()
        tensor, _ = _preprocess_image(image_bytes)

        model_data = _load_model(model)
        net = model_data["model"]
        device = model_data["device"]

        tensor = tensor.to(device)

        t0 = time.time()
        with torch.no_grad():
            logits = net(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        inference_ms = (time.time() - t0) * 1000

        real_prob = float(probs[0].cpu())
        fake_prob = float(probs[1].cpu())
        prediction = "fake" if fake_prob >= 0.5 else "real"
        confidence = max(real_prob, fake_prob)

        logger.info(
            f"Prediction: {prediction} (fake_prob={fake_prob:.4f}) "
            f"in {inference_ms:.1f}ms [{model}]"
        )

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            fake_probability=fake_prob,
            real_probability=real_prob,
            model_used=model_data["config"].name,
            inference_time_ms=round(inference_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model: str = "efficientnet_b0",
):
    """
    Batch prediction for multiple images (max 8 images per request).
    """
    import torch

    if len(files) > 8:
        raise HTTPException(status_code=400, detail="Maximum 8 images per batch.")

    model_data = _load_model(model)
    net = model_data["model"]
    device = model_data["device"]

    results = []
    t0 = time.time()

    tensors = []
    valid_indices = []

    for i, f in enumerate(files):
        try:
            image_bytes = await f.read()
            tensor, _ = _preprocess_image(image_bytes)
            tensors.append(tensor)
            valid_indices.append(i)
        except Exception as e:
            results.append({"file": f.filename, "error": str(e)})

    if tensors:
        batch = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            logits = net(batch)
            probs = torch.softmax(logits, dim=1).cpu()

        for j, idx in enumerate(valid_indices):
            real_prob = float(probs[j][0])
            fake_prob = float(probs[j][1])
            results.append({
                "file": files[idx].filename,
                "prediction": "fake" if fake_prob >= 0.5 else "real",
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "confidence": round(max(real_prob, fake_prob), 4),
            })

    total_ms = (time.time() - t0) * 1000
    return {
        "results": results,
        "batch_size": len(files),
        "total_inference_ms": round(total_ms, 2),
        "model_used": model_data["config"].name,
        "disclaimer": "Research system — not for forensic use.",
    }


@app.get("/models", response_model=List[ModelInfoResponse])
async def list_models():
    """List available models and their info."""
    models = [
        {
            "model_name": "EfficientNet-B0 CNN Detector",
            "model_type": "cnn",
            "backbone": "efficientnet_b0",
            "total_parameters": "~5.3M",
            "is_cached": "efficientnet_b0" in _model_cache,
            "key": "efficientnet_b0",
        },
        {
            "model_name": "ViT-Base/16 Detector",
            "model_type": "vit",
            "backbone": "vit_base_patch16_224",
            "total_parameters": "~86M",
            "is_cached": "vit_base" in _model_cache,
            "key": "vit_base",
        },
        {
            "model_name": "Hybrid CNN-ViT Detector",
            "model_type": "hybrid",
            "backbone": "efficientnet_b0 + transformer",
            "total_parameters": "~6M",
            "is_cached": "hybrid" in _model_cache,
            "key": "hybrid",
        },
    ]
    return [ModelInfoResponse(**{k: v for k, v in m.items() if k != "key"}) for m in models]


@app.get("/models/{model_key}/info")
async def model_info(model_key: str):
    """Get detailed info about a specific model."""
    try:
        model_data = _load_model(model_key)
        net = model_data["model"]

        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        return {
            "model_key": model_key,
            "model_name": model_data["config"].name,
            "model_type": model_data["config"].type,
            "backbone": model_data["config"].backbone,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameters_formatted": f"{total_params / 1e6:.1f}M",
            "num_classes": model_data["config"].num_classes,
            "device": model_data["device"],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/experiments", response_model=List[Dict])
async def list_experiments(
    experiment_type: Optional[str] = None,
    limit: int = 50,
):
    """List all logged experiments from the SQLite tracker."""
    try:
        from src.experiments.tracker import ExperimentTracker
        tracker = ExperimentTracker()
        experiments = tracker.get_all_experiments(exp_type=experiment_type)
        return experiments[:limit]
    except Exception as e:
        logger.warning(f"Could not load experiments: {e}")
        return []


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get details of a specific experiment."""
    try:
        from src.experiments.tracker import ExperimentTracker
        tracker = ExperimentTracker()
        exp = tracker.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        metrics = tracker.get_metrics(experiment_id)
        exp["metrics"] = metrics
        return exp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _sanitize_for_json(obj):
    """Recursively sanitize float inf and nan values for compliant JSON serialization."""
    if isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@app.get("/results/cross_dataset")
async def cross_dataset_results():
    """Return cross-dataset evaluation matrix results."""
    matrix_path = Path("artifacts/experiment_logs/cross_dataset_matrix.json")
    if not matrix_path.exists():
        return {
            "status": "no_data",
            "message": "Cross-dataset results not yet generated. Run: python scripts/run_cross_dataset.py",
            "data": None,
        }
    with open(matrix_path) as f:
        data = json.load(f)
    return {"status": "ok", "data": _sanitize_for_json(data)}


@app.get("/results/robustness")
async def robustness_results():
    """Return robustness stress test results."""
    results_dir = Path("artifacts/experiment_logs")
    rob_files = list(results_dir.glob("robustness*.json"))

    if not rob_files:
        return {
            "status": "no_data",
            "message": "No robustness test results yet.",
            "data": None,
        }

    all_results = {}
    for p in rob_files:
        with open(p) as f:
            all_results[p.stem] = json.load(f)

    return {"status": "ok", "data": _sanitize_for_json(all_results)}


@app.get("/results/summary")
async def results_summary():
    """Return summary of all experiment results."""
    results_path = Path("artifacts/experiment_logs/all_results.json")
    if not results_path.exists():
        return {
            "status": "no_data",
            "message": "No results yet. Run: python scripts/run_all.py",
        }
    with open(results_path) as f:
        data = json.load(f)

    # Build summary
    summary = {}
    for key, result in data.items():
        if isinstance(result, dict) and "test_metrics" in result:
            m = result["test_metrics"]
            summary[key] = {
                "accuracy": m.get("accuracy"),
                "auroc": m.get("auroc"),
                "f1_score": m.get("f1_score"),
                "eer": m.get("eer"),
            }
        elif isinstance(result, list):
            summary[key] = f"{len(result)} experiments"

    return {"status": "ok", "summary": _sanitize_for_json(summary), "num_experiments": len(data)}


@app.post("/explain")
async def explain_prediction(
    file: UploadFile = File(...),
    model: str = "efficientnet_b0",
    method: str = "gradcam",
):
    """
    Generate explanation heatmap for a prediction.

    - **file**: Image file
    - **model**: Model to use
    - **method**: Explanation method (gradcam | attention_rollout)

    Returns prediction + base64-encoded heatmap image.
    """
    try:
        import torch
        from src.explainability.explainer import explain_prediction as _explain

        image_bytes = await file.read()
        tensor, img_pil = _preprocess_image(image_bytes)

        img_np = np.array(img_pil)

        model_data = _load_model(model)
        net = model_data["model"]
        device = model_data["device"]
        model_type = model_data["config"].type

        result = _explain(net, tensor, model_type, img_np, device)

        # Encode overlay as base64 PNG
        import io
        from PIL import Image

        overlay_b64 = None
        if result.get("overlay") is not None:
            overlay_img = Image.fromarray(result["overlay"])
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        heatmap_b64 = None
        if result.get("heatmap") is not None:
            from matplotlib import cm
            heatmap_colored = (cm.jet(result["heatmap"])[:, :, :3] * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap_colored)
            buf = io.BytesIO()
            heatmap_img.save(buf, format="PNG")
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "faithfulness": result.get("faithfulness", {}),
            "heatmap_base64": heatmap_b64,
            "overlay_base64": overlay_b64,
            "model_used": model_data["config"].name,
            "explanation_method": result.get("method", method),
            "disclaimer": "Attribution maps show model focus areas, NOT forensic proof of manipulation.",
        }

    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/cache")
async def clear_model_cache():
    """Clear cached models from memory."""
    global _model_cache
    count = len(_model_cache)
    _model_cache = {}
    return {"cleared": count, "message": f"Cleared {count} cached models."}


# ─── Background experiment runner ────────────────────────────────────────────

_running_experiments: Dict[str, str] = {}


@app.post("/experiments/run/smoke_test")
async def run_smoke_test_async(
    background_tasks: BackgroundTasks,
    model: str = "efficientnet_b0",
):
    """Start a smoke test experiment in the background."""
    task_id = f"smoke_{model}_{int(time.time())}"

    def _run():
        try:
            from src.experiments.runner import ExperimentRunner
            _running_experiments[task_id] = "running"
            runner = ExperimentRunner()
            result = runner.run_smoke_test(model)
            _running_experiments[task_id] = "completed"
            logger.info(f"Smoke test {task_id} complete: acc={result['test_metrics']['accuracy']:.4f}")
        except Exception as e:
            _running_experiments[task_id] = f"failed: {str(e)}"
            logger.error(f"Smoke test {task_id} failed: {e}")

    background_tasks.add_task(_run)
    _running_experiments[task_id] = "started"

    return {
        "task_id": task_id,
        "status": "started",
        "message": f"Smoke test for {model} started in background.",
        "check_status": f"/experiments/status/{task_id}",
    }


@app.get("/experiments/status/{task_id}")
async def experiment_status(task_id: str):
    """Check status of a background experiment."""
    status = _running_experiments.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return {"task_id": task_id, "status": status}


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
