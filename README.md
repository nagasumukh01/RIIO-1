# RobustDeep

**Explainable Vision Transformer & CNN Deepfake Detection Under Cross-Dataset Distribution Shift**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646cff.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RobustDeep is a research prototype for inspecting image and video deepfake detector outputs. It includes a React/Vite console, a FastAPI inference API, EfficientNet-B0 and ViT-Small models, an optional four-member ensemble, and experimental checkpoints trained on FaceForensics++, Celeb-DF v2, and DFDC.

The checkpoints currently in this workspace have different training histories and evaluation coverage. Some are small pilots, and the detector outputs are not forensic proof or a guarantee of accuracy. See [Checkpoint status and evaluation notes](#checkpoint-status-and-evaluation-notes) before interpreting results.

---

## Interactive Forensic Console

RobustDeep includes a web console with a dark-mode UI:

- **Model Selection**: Select an individual detector, the four-member ensemble, or the combined-domain EfficientNet checkpoint when available.
- **Ensemble Breakdown**: Displays member predictions and reports missing ensemble checkpoints.
- **Visual Explanations**: Shows detected face crops and attribution visualizations where supported.
- **Image and Video Input**: Accepts common image formats and video uploads for face-crop inference.

---

## Model Suite & Cross-Architecture Ensemble

The API registry currently defines these model configurations. Checkpoint availability is determined at runtime by checking the listed file paths.

| Model ID | Architecture | Dataset | Checkpoint path (relative to repository root) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `cross_ensemble` | EfficientNet-B0 + ViT-Small | Celeb-DF v2 and FaceForensics++ | No separate checkpoint; combines the four members below | Averages available member fake probabilities; it is not itself a trained model. |
| `celebdf_b0` | EfficientNet-B0 | Celeb-DF v2 | `artifacts/models/celebdf_b0_balanced_pilot8_cpu_20261109/best.pt` | Eight-video pilot; experimental. |
| `ff_b0` | EfficientNet-B0 | FaceForensics++ | `artifacts/models/ff_c23_b0_balanced200_4f_20261109/best.pt` | Subset checkpoint; a training summary is not present at this path. |
| `celebdf_vit` | ViT-Small/16 | Celeb-DF v2 | `artifacts/models/celebdf_vit_small_subset50_f4/best.pt` | Saved summary records five epochs; test accuracy 64% on 100 videos. This is not a full-scale benchmark. |
| `ff_vit` | ViT-Small/16 | FaceForensics++ | `artifacts/models/ff_c23_vit_small_subset50_f4/best.pt` | Checkpoint and saved training summary have differed during prior runs; verify they match before citing summary metrics. |
| `dfdc_b0` | EfficientNet-B0 | DFDC pilot | `artifacts/models/dfdc_b0_balanced_pilot8_cpu_20261109/best.pt` | Eight-video pilot; not representative of full-dataset performance. |
| `combined_multidomain_b0` | EfficientNet-B0 | Combined-domain | `artifacts/models/combined_multidomain_b0/best.pt` | A checkpoint is present in the workspace; a training summary is not present, so its training and accuracy are not recorded here. |

### How the Ensemble Modes Work

The advanced single-model endpoint `/api/predict?model_id=cross_ensemble` averages fake probabilities from whichever of the four registered CNN/ViT members are available. The console's **Analyze All Models** mode uses `/api/ensemble-predict` and evaluates six configured checkpoints. It displays each result, but only members with a non-zero fusion weight and validation AUROC of at least 0.6 contribute to that verdict. Excluded models do not affect its fusion or disagreement score.

The two modes are separate inference paths. A checkpoint being present and loadable does not establish that it predicts accurately. Current experiments are small or have incomplete/inconsistent evaluation artifacts; treat individual scores and ensemble verdicts as research outputs, not forensic evidence or a guarantee of accuracy. See [Checkpoint status and evaluation notes](#checkpoint-status-and-evaluation-notes).

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
**Parameters:**
- `file` *(required)*: Image or video file upload.
- `model_id` *(optional query parameter, default: `cross_ensemble`)*: `cross_ensemble`, `celebdf_b0`, `ff_b0`, `celebdf_vit`, `ff_vit`, `dfdc_b0`, or `combined_multidomain_b0`.

Example request:
```bash
curl -X POST "http://127.0.0.1:8000/api/predict?model_id=celebdf_vit" \
  -F "file=@path/to/face.jpg"
```

The response includes the selected model, predicted label, fake probability,
input preview, attribution when supported, and ensemble member information for
`cross_ensemble`. Values depend on the uploaded file and are not represented by
fixed example scores in this README.

### Inference preprocessing

Per-model input size, interpolation, RGB normalization, binary label mapping,
threshold, and view policy are configured in [`configs/inference.yaml`](configs/inference.yaml).
The current models use 224×224 RGB inputs, ImageNet mean/std normalization,
two semantic labels (real index 0, fake index 1), and one binary logit. The
default 0.5 threshold is not calibrated to a held-out evaluation set.

Still-image decisions use the primary detected face crop. Video handling
aggregates face crops and respects configured per-model frame limits. Training
and inference face-crop geometry may differ: preprocessing prefers RetinaFace,
then MTCNN, then Haar, while inference uses its verified Haar-based detector.
The preprocessing detector used for every existing manifest is not recorded.

---

## Training and Reproducibility

Training and evaluation use real face crops listed in video-level manifests.
Synthetic data is reserved for smoke tests and must not be used to claim model
accuracy. You must obtain licensed datasets yourself; this repository does not
redistribute them.

```bash
# Create dataset folders and print acquisition guidance
python scripts/prepare_data.py --dataset all --max-videos 100

# Extract faces (run only for datasets whose face crops are not already prepared)
python scripts/preprocess_faces.py --dataset faceforensics --max-videos 100
python scripts/preprocess_faces.py --dataset celebdf --max-videos 100
python scripts/preprocess_faces.py --dataset dfdc --max-videos 100

# Example: train a ViT on the prepared Celeb-DF split
python scripts/train.py --dataset celebdf --model vit_small --epochs 5 --frames 4 --experiment-id celebdf_vit_small_subset50_f4

# Run the real-data path checks
pytest tests/test_real_data_path.py -v
```

`scripts/train.py` supports `efficientnet_b0`, `efficientnet_b4`, `vit_small`,
and `vit_base`. Omit `--max-videos` to use all videos in the manifest. The
training script writes checkpoints, a `training_summary.json`, and evaluation
artifacts under `artifacts/models/<experiment_id>/`. Training may require a
GPU for reasonable ViT training times; an epoch count is not a guarantee of
good accuracy.

### Checkpoint status and evaluation notes

- Checkpoints are separate files under `artifacts/models/`; the API registry
  paths are listed in the [model table](#model-suite--cross-architecture-ensemble).
- The Celeb-DF ViT saved summary records five epochs and 64% test accuracy on
  100 videos. This is a limited experiment, not a production-quality estimate.
- The DFDC and Celeb-DF EfficientNet pilots each used eight videos; their saved
  test accuracies are 37.5% and 50%, respectively. Do not generalize those
  measurements to the full datasets.
- A training summary is absent for the registered FaceForensics++ EfficientNet
  subset checkpoint. The FaceForensics++ ViT checkpoint and its saved summary
  have not always matched; verify checkpoint epoch and metadata before citing
  the summary's metrics.
- The combined-domain EfficientNet checkpoint is present, but its training
  summary and test metrics are not recorded in this workspace.
- Preprocessing and evaluation settings are configured per model in
  `configs/inference.yaml`. Defaults such as a 0.5 threshold are not calibrated
  to test data and must not be presented as an accuracy improvement.

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
├── configs/                      # Data, backbone, training, experiment, and inference YAML
│   └── inference.yaml            # Per-model preprocessing, labels, thresholds, and views
├── scripts/                      # Preprocessing, training, cross-eval & report scripts
├── tests/                        # Comprehensive pytest suite (API, metrics, contracts)
├── docs/                         # Architecture, literature review, and experiment protocols
├── artifacts/                    # Checkpoints, logs, figures, and evaluations
├── app.py                        # Streamlit research dashboard
├── pytest.ini                    # Pytest configuration with root pythonpath
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Project documentation
```

---

## Testing & Quality Assurance

Run the focused inference configuration and API tests:

```bash
pytest tests/test_inference_config.py tests/test_all_api_endpoints.py::test_models_endpoint tests/test_all_api_endpoints.py::test_predict_image_vit tests/test_all_api_endpoints.py::test_predict_image_cnn -q
```

Run the complete suite when the environment and datasets required by tests are available:

```bash
pytest tests/ -v
```

---

## Research Protocol & Ethics

- **Per-Model Inference Settings**: Input normalization, label mapping, and decision thresholds are stored per model in `configs/inference.yaml`. Defaults are not evidence of calibration.
- **Zero Target Tuning**: Cross-dataset evaluations preserve strict isolation between source training distributions and target test domains.
- **Forensic Disclaimer**: Predictions, Grad-CAM heatmaps, and confidence scores provided by this software are research diagnostic aids and do not constitute self-sufficient legal evidence.
- **Data Governance**: This repository does not redistribute copyrighted video material from FaceForensics++, Celeb-DF, or DFDC. Users must obtain datasets under their respective academic research licenses.

---

## License

This project is licensed under the **MIT License**.
