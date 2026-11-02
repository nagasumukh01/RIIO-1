# RobustDeep

**Explainable Vision Transformer & CNN Deepfake Detection Under Cross-Dataset Distribution Shift**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646cff.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RobustDeep is a research-grade, end-to-end framework and interactive forensic platform for evaluating and deploying deepfake detectors under **cross-dataset distribution shifts**. It benchmarked Convolutional Neural Networks (EfficientNet-B0) against Vision Transformers (ViT-Small/16) across **FaceForensics++** and **Celeb-DF v2**, offering an interactive React UI, a high-throughput FastAPI inference server, visual explainability (Grad-CAM), and a **Cross-Architecture Hybrid Ensemble**.

> Primary reference (spirit, not copy): Petmezas, G. (2026). *Cross-dataset video deepfake detection using Transformer and CNN architectures*. Machine Vision and Applications.  
> We do **not** claim novelty for ViTs, cross-dataset evaluation, or temporal modeling.

---

## Interactive Forensic Console

RobustDeep includes a web console with a dark-mode UI (`DM Mono` & `Space Grotesk`) designed for media forensics:

- **Dynamic Model Selection**: Instant switching between single-architecture baselines and the fused ensemble.
- **Cross-Architecture Consensus**: Real-time breakdown of all 4 sub-model predictions and collective agreement percentage.
- **Visual Explainability**: Side-by-side display of the input crop and Grad-CAM spatial activation heatmaps.
- **Faithfulness Verification**: Quantifies model explanation validity by measuring confidence drops upon masking highlighted regions.
- **Dual Media Pipeline**: Seamless support for static images (`.jpg`, `.png`, `.webp`) and multi-frame video uploads (`.mp4`, `.avi`, `.mov`) with automatic Haar-Cascade face detection and temporal aggregation.

---

## Model Suite & Cross-Architecture Ensemble

The system exposes five detector configurations via its model registry:

| Model ID | Display Name | Architecture | Training Dataset | Specialization / Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`cross_ensemble`** | **★ Cross-Architecture Hybrid Ensemble** | **Hybrid (CNN + ViT)** | **Celeb-DF v2 & FF++** | **High-accuracy composite fusion. Fuses all 4 detectors to maximize resilience against cross-dataset distribution shift.** |
| `celebdf_b0` | CNN · EfficientNet-B0 | CNN (`timm`) | Celeb-DF v2 | Mobile-optimized CNN baseline trained on high-quality celebrity deepfakes. |
| `ff_b0` | CNN · EfficientNet-B0 | CNN (`timm`) | FaceForensics++ | CNN baseline trained on H.264 compressed manipulation sequences (c23). |
| `celebdf_vit` | Vision Transformer · ViT-Small | ViT-Small/16 | Celeb-DF v2 | Self-attention detector capturing long-range spatial patch correlations. |
| `ff_vit` | Vision Transformer · ViT-Small | ViT-Small/16 | FaceForensics++ | Self-attention detector robust against compression artifacts. |

### How the Cross-Architecture Ensemble Works

When running inference with `cross_ensemble`:
1. The input image or sampled video frames are passed concurrently through all four base models.
2. Individual predictions, confidence scores, and classifications (`REAL` vs. `FAKE`) are calculated for each model.
3. Predictions are fused via a weighted consensus algorithm:
   $$\bar{p} = \frac{1}{4} \sum_{i=1}^4 p_i$$
4. Consensus metrics (e.g., *4 of 4 models agree (100% agreement)*) are presented alongside an individual sub-model score matrix.
5. The highest-resolution feature representation is utilized to generate an explanatory Grad-CAM heatmap.

---

## Quick Start

### 1. Prerequisites & Environment Setup

Clone the repository and install backend dependencies:

```bash
git clone https://github.com/nagasumukh01/RIIO-1.git -b just-face
cd RIIO-1

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install Python requirements
pip install -r requirements.txt
```

### 2. Launch the Interactive Application

#### Option A: Development Mode (Vite Hot-Reload + FastAPI)

**Terminal 1 — FastAPI Backend Server:**
```bash
python -m uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — React Console:**
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

#### Option B: Production Embedded Mode

Build the frontend bundle to let FastAPI serve the static assets directly:

```bash
cd frontend
npm install
npm run build
cd ..

python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## API Documentation

FastAPI provides automatic interactive Swagger documentation at **`http://127.0.0.1:8000/docs`**.

### Core Endpoints

#### 1. System Health
```http
GET /api/health
```
Returns system status and number of loaded model checkpoints.

#### 2. Model Catalog
```http
GET /api/models
```
Returns metadata, architecture specifications, and dataset pairings for all registered models.

#### 3. Run Inference & Attribution
```http
POST /api/predict?model_id=cross_ensemble
Content-Type: multipart/form-data
```
**Form Parameters:**
- `file` *(required)*: Binary image or video upload (`image/jpeg`, `image/png`, `video/mp4`, etc.).
- `model_id` *(optional, default: `cross_ensemble`)*: Target model identifier (`cross_ensemble`, `celebdf_b0`, `ff_b0`, `celebdf_vit`, `ff_vit`).

**Example Response:**
```json
{
  "model": "cross_ensemble",
  "architecture": "Hybrid Ensemble (CNN + ViT)",
  "dataset": "Celeb-DF v2 & FaceForensics++",
  "label": "FAKE",
  "score": 0.8842,
  "confidence": 0.8842,
  "explanation_method": "Grad-CAM (Ensemble-Fused)",
  "explanation_image": "data:image/png;base64,...",
  "input_preview": "data:image/png;base64,...",
  "faithfulness_confidence_drop": 0.2415,
  "latency_ms": 142.6,
  "ensemble_breakdown": [
    {
      "key": "celebdf_b0",
      "name": "CNN · EfficientNet-B0 (Celeb-DF v2)",
      "score": 0.8621,
      "label": "FAKE",
      "architecture": "CNN"
    },
    {
      "key": "ff_b0",
      "name": "CNN · EfficientNet-B0 (FaceForensics++)",
      "score": 0.9104,
      "label": "FAKE",
      "architecture": "CNN"
    },
    {
      "key": "celebdf_vit",
      "name": "Vision Transformer · ViT-Small (Celeb-DF v2)",
      "score": 0.8315,
      "label": "FAKE",
      "architecture": "Vision Transformer"
    },
    {
      "key": "ff_vit",
      "name": "Vision Transformer · ViT-Small (FaceForensics++)",
      "score": 0.9328,
      "label": "FAKE",
      "architecture": "Vision Transformer"
    }
  ],
  "consensus": "4 of 4 models agree (100% agreement)",
  "disclaimer": "Model attribution is research evidence, not forensic proof."
}
```

---

## Research Pipeline & Reproducibility

### Critical Design Rule
> **Synthetic data is permitted only for Stage-1 smoke tests.**  
> From Stage 2 onward, `train.py` and `evaluate.py` require **real** face crops extracted from licensed research datasets. Missing face directories raise a strict `FileNotFoundError` with clear setup instructions.

### Research Workflow

```bash
# 1. Run unit test suite
pytest tests/ -v

# 2. Stage 1 — Synthetic smoke test (labelled SYNTHETIC)
python scripts/smoke_test.py

# 3. Stage 2 — Prepare real dataset directory structures
python scripts/prepare_data.py --dataset faceforensics
python scripts/prepare_data.py --dataset celebdf

# 4. Extract face crops with face alignment
python scripts/preprocess_faces.py --dataset faceforensics --max-videos 100
python scripts/preprocess_faces.py --dataset celebdf --max-videos 100

# Stage 3 - Model evaluation
python scripts/evaluate.py --dataset faceforensics --split test --checkpoint artifacts/models/ff_c23_subset10/best.pt --model efficientnet_b0

# Stage 4 - Explainability
python scripts/run_explainability.py --checkpoint artifacts/models/ff_c23_subset10/best.pt --dataset faceforensics --split test --model efficientnet_b0 --max-videos 32
# Dashboard
streamlit run app.py
```
# 5. Train CNN and ViT detectors
python scripts/train.py --dataset faceforensics --model efficientnet_b0 --epochs 10
python scripts/train.py --dataset celebdf --model vit_small --epochs 10

# 6. Evaluate cross-dataset generalization (zero target-tuning)
python scripts/run_cross_dataset.py

# 7. Generate explainability & robustness artifacts
python scripts/run_explainability.py
python scripts/run_robustness.py

# 8. Launch Streamlit Research Dashboard
streamlit run app.py
```

---

## Repository Layout

```
RIIO-1/
├── frontend/                     # Interactive React + Vite Forensic Console
│   ├── src/
│   │   ├── App.jsx               # Forensic dashboard with dynamic model discovery & consensus
│   │   ├── styles.css            # Dark-mode styling (DM Mono + Space Grotesk)
│   │   └── main.jsx              # React root entrypoint
│   ├── index.html                # Single-page application template
│   ├── package.json              # NPM dependencies & scripts
│   └── vite.config.js            # Vite configuration with API proxying
├── src/
│   ├── api/
│   │   └── server.py             # FastAPI multi-model inference server & static file host
│   ├── data/
│   │   ├── real_dataset.py       # Canonical real face dataset loader
│   │   └── synthetic.py          # Stage-1 smoke test data generator
│   ├── models/
│   │   └── factory.py            # Model factory (EfficientNet-B0/B4, ViT-Small/Base)
│   ├── training/
│   │   ├── trainer.py            # Mixed-precision training loop with early stopping
│   │   └── losses.py             # Focal & weighted binary cross-entropy losses
│   ├── evaluation/
│   │   ├── metrics.py            # AUROC, EER, F1, confusion matrix, generalization gap
│   │   └── artifacts.py          # Experiment serialization utilities
│   ├── explainability/
│   │   ├── gradcam.py            # Grad-CAM implementation with faithfulness check
│   │   └── attention_rollout.py  # ViT attention rollout mechanism
│   ├── robustness/
│   │   └── stress.py             # JPEG compression & Gaussian blur perturbation tests
│   └── utils/
│       ├── config.py             # YAML configuration parsers
│       └── logging_utils.py      # Structured experiment logging
├── configs/                      # YAML configs for data, models, training & experiments
├── scripts/                      # Preprocessing, training, cross-eval & report scripts
├── tests/                        # Comprehensive pytest suite (API, metrics, contracts)
├── docs/                         # Architecture, literature review, and experiment protocols
├── artifacts/                    # Checkpoints, logs, figures, and evaluations
├── app.py                        # Streamlit 12-page research dashboard
├── pytest.ini                    # Pytest configuration with root pythonpath
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Project documentation
```

---

## Testing & Quality Assurance

To execute the test suite across all API endpoints, contracts, metrics, and dataset boundaries:

```bash
pytest tests/ -v
```

CI runs automatically on all pull requests and pushes via GitHub Actions (`.github/workflows/ci.yml`).

---

## Research Protocol & Ethics

- **Authentic Face Attribution**: Real evaluations utilize authentic face crops with a fixed 0.50 decision threshold.
- **Zero Target Tuning**: Cross-dataset evaluations preserve strict isolation between source training distributions and target test domains.
- **Forensic Disclaimer**: Predictions, Grad-CAM heatmaps, and confidence scores provided by this software are research diagnostic aids and do not constitute self-sufficient legal evidence.
- **Data Governance**: This repository does not redistribute copyrighted video material from FaceForensics++, Celeb-DF, or DFDC. Users must obtain datasets under their respective academic research licenses.

---

## License

This project is licensed under the **MIT License**.
