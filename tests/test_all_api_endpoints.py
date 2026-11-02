"""Comprehensive end-to-end tests for all FastAPI endpoints and models."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

try:
    from fastapi.testclient import TestClient
    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False
    TestClient = None

from src.api.server import app, MODEL_REGISTRY

# Check if model checkpoints are present in current environment (git-ignored in CI)
CHECKPOINTS_PRESENT = bool(
    MODEL_REGISTRY.get("celebdf_b0", {}).get("checkpoint")
    and Path(MODEL_REGISTRY["celebdf_b0"]["checkpoint"]).exists()
)


@pytest.fixture(scope="module")
def client():
    if not HAS_TESTCLIENT:
        pytest.skip("TestClient requires httpx package")
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    img = Image.new("RGB", (224, 224), color=(130, 160, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_video_bytes(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (224, 224))
    for i in range(8):
        frame = np.full((224, 224, 3), (40 + i * 15, 80 + i * 10, 120), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path.read_bytes()


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_available" in data
    assert data["models_available"] >= 0
    assert "celebdf_b0" in data["models"]
    assert "cross_ensemble" in data["models"]


def test_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 5
    model_ids = [m["id"] for m in data["models"]]
    assert "cross_ensemble" in model_ids
    assert "celebdf_b0" in model_ids
    assert "ff_b0" in model_ids
    assert "celebdf_vit" in model_ids
    assert "ff_vit" in model_ids


@pytest.mark.skipif(not CHECKPOINTS_PRESENT, reason="Model .pt checkpoints not present in CI runner")
def test_predict_cross_ensemble(client, sample_image_bytes):
    response = client.post(
        "/api/predict",
        params={"model_id": "cross_ensemble"},
        files={"file": ("sample.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "cross_ensemble"
    assert data["architecture"] == "Hybrid Ensemble (CNN + ViT)"
    assert data["label"] in {"REAL", "FAKE"}
    assert 0.0 <= data["score"] <= 1.0
    assert data["ensemble_breakdown"] is not None
    assert len(data["ensemble_breakdown"]) == 4
    assert data["consensus"] is not None


@pytest.mark.skipif(not CHECKPOINTS_PRESENT, reason="Model .pt checkpoints not present in CI runner")
def test_predict_image_cnn(client, sample_image_bytes):
    response = client.post(
        "/api/predict",
        params={"model_id": "celebdf_b0"},
        files={"file": ("test_face.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "celebdf_b0"
    assert data["label"] in {"REAL", "FAKE"}
    assert 0.0 <= data["score"] <= 1.0
    assert 0.5 <= data["confidence"] <= 1.0
    assert "input_preview" in data
    assert data["input_preview"].startswith("data:image/png;base64,")
    assert "explanation_image" in data
    assert "latency_ms" in data


@pytest.mark.skipif(not CHECKPOINTS_PRESENT, reason="Model .pt checkpoints not present in CI runner")
def test_predict_image_vit(client, sample_image_bytes):
    response = client.post(
        "/api/predict",
        params={"model_id": "celebdf_vit"},
        files={"file": ("test_vit_face.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "celebdf_vit"
    assert data["architecture"] == "Vision Transformer"
    assert data["label"] in {"REAL", "FAKE"}
    assert 0.0 <= data["score"] <= 1.0


@pytest.mark.skipif(not CHECKPOINTS_PRESENT, reason="Model .pt checkpoints not present in CI runner")
def test_predict_video_upload(client, sample_video_bytes):
    response = client.post(
        "/api/predict",
        params={"model_id": "ff_b0"},
        files={"file": ("clip.mp4", sample_video_bytes, "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "ff_b0"
    assert data["label"] in {"REAL", "FAKE"}
    assert 0.0 <= data["score"] <= 1.0
    assert "explanation_image" in data


def test_predict_invalid_model(client, sample_image_bytes):
    response = client.post(
        "/api/predict",
        params={"model_id": "unknown_model_xyz"},
        files={"file": ("face.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported model selection" in response.json()["detail"]


def test_predict_empty_file(client):
    response = client.post(
        "/api/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_static_frontend_serving(client):
    response = client.get("/")
    assert response.status_code == 200
    if "<!doctype html>" in response.text.lower():
        assert True
    else:
        data = response.json()
        assert data.get("service") == "RobustDeep Inference API" or "status" in data
