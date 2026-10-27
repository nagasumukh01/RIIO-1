# RobustDeep

**Explainable Vision Transformer-Based Video Deepfake Detection Under Cross-Dataset Distribution Shift**

Research-grade, reproducible pipeline for studying generalization of CNN and Vision Transformer deepfake detectors across FaceForensics++, Celeb-DF, and DFDC.

> Primary reference (spirit, not copy): Petmezas, G. (2026). Cross-dataset video deepfake detection using Transformer and CNN architectures. *Machine Vision and Applications*.  
> We do **not** claim novelty for ViTs, cross-dataset evaluation, or temporal modeling.

## Critical Design Rule

**Synthetic data is allowed only for Stage-1 smoke tests.**  
From Stage 2 onward, `train.py` / `evaluate.py` load **real** face crops only. Missing data → clear `FileNotFoundError` with exact fix instructions. No silent fallback.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stage 1 — synthetic smoke test (metrics labelled SYNTHETIC)
python scripts/smoke_test.py
pytest tests/ -v

# Stage 2 — real data path
python scripts/prepare_data.py --dataset faceforensics
# → place videos, then:
python scripts/preprocess_faces.py --dataset faceforensics --max-videos 100
python scripts/train.py --dataset faceforensics --model efficientnet_b0 --epochs 5

# Dashboard
streamlit run app.py
```

See **docs/REAL_DATA_QUICKSTART.md** for the exact 4-command research flow.

### What still requires your data

This repository does not download or redistribute FF++, Celeb-DF, or DFDC. Real AUROC, cross-dataset, ablation, and explainability outputs remain pending until you provide licensed videos, preprocess them into `data/faces/`, and run the commands in `docs/experiment_protocol.md`. Smoke-test numbers remain explicitly non-research results.

## Repository Layout

```
robustdeep/
├── app.py                 # Streamlit research dashboard
├── configs/               # data / models / training / experiments
├── src/
│   ├── data/              # real_dataset.py (canonical) + synthetic (Stage-1 only)
│   ├── models/            # EfficientNet + ViT factory
│   ├── training/          # trainer, losses
│   ├── evaluation/        # metrics (AUROC primary)
│   ├── experiments/       # cross-dataset, ablations
│   ├── explainability/    # Grad-CAM, Attention Rollout
│   └── robustness/        # JPEG / blur stress tests
├── scripts/               # prepare_data, preprocess_faces, train, evaluate, runners
├── tests/                 # includes real-data path leakage checks
├── docs/                  # literature, gaps, protocol, architecture, quickstart
└── artifacts/             # models, smoke checkpoints, logs, figures (git-ignored content)
```

## Experimental Design

1. Within-dataset baselines (CNN + ViT on each dataset)
2. 6-pair cross-dataset matrix (zero target tuning)
3. Augmentation / loss / capacity ablations
4. Real-data explainability under distribution shift
5. Explainability under distribution shift + faithfulness
6. Evaluation-time compression & blur stress tests

## Primary Metrics

AUROC (primary), Accuracy, Precision, Recall, F1, EER, FPR/FNR, confusion matrix, generalization gap, parameter count, latency.

## Documentation

| Document | Purpose |
|----------|---------|
| docs/architecture.md | System design |
| docs/research_gaps.md | Differences from Petmezas 2026 + hypotheses |
| docs/dataset_protocol.md | Data layout, splitting, feasibility |
| docs/REAL_DATA_QUICKSTART.md | 4-command real-data flow |
| docs/literature_review.md | Focused review |

## License & Data

Code: MIT (or project choice).  
Datasets: obtain under their original research licenses. This repository does **not** redistribute FF++, Celeb-DF, or DFDC.
