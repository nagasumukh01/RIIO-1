# RobustDeep Architecture Plan

## High-Level Design

RobustDeep is a modular, research-first video deepfake detection pipeline centered on **cross-dataset generalization**. The system separates concerns strictly:

- **Data layer** (`src/data/`): Real dataset loaders only. Synthetic data exists solely for Stage-1 unit/smoke tests and is never reachable from training/evaluation entry points after Stage 1.
- **Model layer** (`src/models/`): CNN baselines (EfficientNet-B0/B4) and Vision Transformers (ViT-Small/Base) via `timm`. Optional hybrid interface.
- **Training / Evaluation** (`src/training/`, `src/evaluation/`): Config-driven, seed-controlled, checkpointed. Strict video-level splits. Zero target-set tuning.
- **Experiments** (`src/experiments/`): Orchestrates within-dataset baselines, 6-pair cross-dataset matrix, ablations, diffusion interface.
- **Explainability & Robustness** (`src/explainability/`, `src/robustness/`): Grad-CAM, Attention Rollout, faithfulness checks, compression/blur stress tests.
- **UI / API** (`src/ui/`, `src/api/`, `app.py`): Streamlit research dashboard (12 pages) + optional FastAPI. All metrics shown must come from saved experiment artifacts or be labelled SYNTHETIC.

## Data Flow (Mandatory)

```
Raw Video
  → Integrity + metadata validation
  → Video-/identity-aware split (no leakage)
  → Frame sampling (fixed strategy across models)
  → Face detection + alignment (RetinaFace primary, MTCNN fallback)
  → Normalized 224×224 crops (ImageNet mean/std)
  → Training-only augmentation (Albumentations)
  → Model (frame-level logits)
  → Video-level aggregation (mean / median / top-k)
  → Metrics (AUROC primary) + uncertainty
  → Explanation maps + faithfulness
  → Experiment record (JSON + CSV)
```

All comparable models share identical sampling, crop size, normalization, and aggregation.

## Key Design Decisions

1. **No silent synthetic fallback**  
   `train.py`, `evaluate.py`, and runners raise `FileNotFoundError` with exact instructions if real face crops are missing.

2. **Config-driven everything**  
   `configs/*.yaml` control data paths, model names, training hyper-parameters, experiment matrices.

3. **Reproducibility first**  
   Seeded splits, pinned requirements, experiment IDs containing config hash + seed, full prediction dumps.

4. **Research vs Demo separation**  
   Dashboard pages that show live predictions on user uploads are clearly labelled “Demo”. Research metrics load only from `artifacts/experiment_logs/`.

5. **Extensibility**  
   New backbones register via a simple factory. New datasets implement a common `VideoDeepfakeDataset` interface.

## Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `src/data/real_dataset.py` | Canonical loader used by all training/eval |
| `src/data/synthetic.py` | Stage-1 only; never imported by train/eval after Stage 1 |
| `src/models/factory.py` | Builds EfficientNet / ViT / hybrid from config |
| `src/training/trainer.py` | Mixed-precision, early stopping, class weighting, checkpointing |
| `src/evaluation/metrics.py` | AUROC, EER, F1, confusion, cross-dataset gap |
| `src/experiments/cross_dataset.py` | 6-pair matrix runner, zero target tuning |
| `src/explainability/` | Grad-CAM + Attention Rollout + faithfulness |
| `src/robustness/` | Compression / blur stress evaluation |
| `scripts/prepare_data.py` | Creates folder layout + instructions |
| `scripts/preprocess_faces.py` | Resumable face extraction with `--max-videos` |
| `scripts/train.py` | Entry point; requires `--dataset faceforensics\|celebdf\|dfdc` |
| `app.py` | Streamlit research dashboard |

## Stage Progression

- **Stage 1**: Synthetic smoke test + unit tests. Metrics labelled SYNTHETIC.
- **Stage 2**: Real loaders + preprocess + train on small real subset. FileNotFoundError if faces missing.
- **Stage 3**: Within-dataset + 6-pair cross-dataset.
- **Stage 4**: Ablations (aug, loss, capacity).
- **Stage 5**: Diffusion interface + explainability + robustness.
- **Stage 6**: Full dashboard + documentation package.

## Non-Goals

- Claiming novelty for ViT, cross-dataset evaluation, or temporal modeling.
- Auto-downloading licensed datasets.
- Hard-coded impressive metrics in the UI.
- Target-test-set threshold tuning.
