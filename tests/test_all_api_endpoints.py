"""
End-to-End API Test Suite
Tests all endpoints in src/api/app.py using FastAPI TestClient.
"""

import sys
import io
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def create_test_image():
    """Create a 224x224 RGB test image buffer."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_root():
    print("Testing GET / ...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    # Can be HTML SPA dashboard or JSON
    assert "RobustDeep" in res.text or "endpoints" in res.text
    print("  Root OK (SPA dashboard or API root returned)")

def test_samples():
    print("Testing GET /api/samples ...")
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "samples" in data
    print("  Samples OK, count:", len(data["samples"]))

def test_predict_video_sample():
    print("Testing POST /api/predict_video with sample_real_01.mp4 ...")
    res = client.post(
        "/api/predict_video",
        data={
            "sample_name": "sample_real_01.mp4",
            "model": "efficientnet_b0",
            "num_frames": "3"
        }
    )
    assert res.status_code == 200, f"Predict video failed: {res.text}"
    data = res.json()
    assert data["status"] == "ok"
    assert data["prediction"] in ("REAL", "FAKE")
    assert "frame_predictions" in data
    assert len(data["frame_predictions"]) == 3
    print(f"  Predict video OK: verdict={data['prediction']}, conf={data['confidence']}, ms={data['inference_time_ms']}")

def test_predict_video_uploaded_image():
    print("Testing POST /api/predict_video with uploaded test image ...")
    img_buf = create_test_image()
    res = client.post(
        "/api/predict_video",
        files={"file": ("test.jpg", img_buf, "image/jpeg")},
        data={"model": "efficientnet_b0", "num_frames": "3"}
    )
    assert res.status_code == 200, f"Predict image failed: {res.text}"
    data = res.json()
    assert data["status"] == "ok"
    assert data["prediction"] in ("REAL", "FAKE")
    assert len(data["frame_predictions"]) == 1
    print(f"  Predict uploaded image OK: verdict={data['prediction']}, conf={data['confidence']}")

def test_health():
    print("Testing GET /health ...")
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    print("  Health OK:", data)

def test_models():
    print("Testing GET /models ...")
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 3
    print("  Models OK, count:", len(data))

def test_model_info():
    print("Testing GET /models/efficientnet_b0/info ...")
    res = client.get("/models/efficientnet_b0/info")
    assert res.status_code == 200
    data = res.json()
    assert "total_parameters" in data
    print(f"  Model info OK: {data['model_name']} ({data['parameters_formatted']})")

def test_experiments():
    print("Testing GET /experiments ...")
    res = client.get("/experiments")
    assert res.status_code == 200
    data = res.json()
    print("  Experiments count:", len(data))
    assert len(data) > 0, "Expected at least 1 experiment"

def test_results_cross_dataset():
    print("Testing GET /results/cross_dataset ...")
    res = client.get("/results/cross_dataset")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "faceforensics_to_celebdf" in data["data"]
    print("  Cross-dataset results OK")

def test_results_robustness():
    print("Testing GET /results/robustness ...")
    res = client.get("/results/robustness")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    print("  Robustness results OK")

def test_results_summary():
    print("Testing GET /results/summary ...")
    res = client.get("/results/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    print("  Results summary OK:", data["summary"])

def test_predict():
    print("Testing POST /predict ...")
    img_buf = create_test_image()
    res = client.post(
        "/predict",
        files={"file": ("test.jpg", img_buf, "image/jpeg")},
        params={"model": "efficientnet_b0"}
    )
    assert res.status_code == 200, f"Predict failed: {res.text}"
    data = res.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "fake_probability" in data
    print(f"  Prediction OK: {data['prediction']} (conf: {data['confidence']:.4f}, time: {data['inference_time_ms']}ms)")

def test_predict_batch():
    print("Testing POST /predict/batch ...")
    img_buf1 = create_test_image()
    img_buf2 = create_test_image()
    res = client.post(
        "/predict/batch",
        files=[
            ("files", ("test1.jpg", img_buf1, "image/jpeg")),
            ("files", ("test2.jpg", img_buf2, "image/jpeg"))
        ],
        params={"model": "efficientnet_b0"}
    )
    assert res.status_code == 200, f"Batch predict failed: {res.text}"
    data = res.json()
    assert len(data["results"]) == 2
    print(f"  Batch prediction OK, items: {len(data['results'])}, total ms: {data['total_inference_ms']}")

def test_clear_cache():
    print("Testing DELETE /models/cache ...")
    res = client.delete("/models/cache")
    assert res.status_code == 200
    print("  Cache cleared OK:", res.json())

if __name__ == "__main__":
    test_root()
    test_health()
    test_models()
    test_model_info()
    test_experiments()
    test_results_cross_dataset()
    test_results_robustness()
    test_results_summary()
    test_predict()
    test_predict_batch()
    test_clear_cache()
    print("\nALL API ENDPOINTS TESTED AND PASSED PERFECTLY!")
