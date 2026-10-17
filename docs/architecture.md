# System Architecture

## Overview

RobustDeep is a modular, research-oriented deepfake detection system designed for cross-dataset evaluation and explainability analysis.

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                          │
│  Video → Validate → Sample Frames → Detect Faces → Crop/Norm  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   TRAINING ENGINE   │
                │  Loss / Optimizer / │
                │  Checkpointing      │
                └──────────┬──────────┘
                           │
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────▼─────┐    ┌────▼─────┐    ┌────▼──────┐
    │ CNN (EN) │    │ ViT-B/16 │    │  Hybrid   │
    └────┬─────┘    └────┬─────┘    └────┬──────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
              ┌───────────▼────────────┐
              │    EVALUATION ENGINE   │
              │  Metrics / Predictions │
              │  Cross-Dataset Matrix  │
              └───────────┬────────────┘
                          │
         ┌────────────────┼──────────────────┐
         │                │                  │
    ┌────▼──────┐   ┌────▼───────┐    ┌────▼──────┐
    │Experiment │   │Explain-    │    │Robustness │
    │Tracker DB │   │ability     │    │Tester     │
    └────┬──────┘   └────┬───────┘    └────┬──────┘
         │                │                │
         └────────────────┼──────────────────┘
                          │
              ┌───────────▼────────────┐
              │   STREAMLIT DASHBOARD  │
              │  11 Research Pages     │
              └────────────────────────┘
```

## Module Descriptions

### `src/data/`
- **dataset.py**: Frame-level dataset classes (real + synthetic), weighted sampling
- **preprocessing.py**: Video-to-face-crop pipeline with MTCNN/Haar detection
- **augmentation.py**: Training-only augmentations (JPEG, blur, resize, color jitter)

### `src/models/`
- **cnn_baseline.py**: EfficientNet-B0/B4 with custom classification head + Grad-CAM target layer
- **vit_detector.py**: ViT-B/16 and ViT-S/16 with attention hooks + Hybrid CNN-ViT
- **factory.py**: Model creation, checkpoint save/load, size estimation

### `src/training/`
- **trainer.py**: Training loop with early stopping, gradient clipping, metric tracking
- **losses.py**: Cross-entropy, focal loss, weighted BCE

### `src/evaluation/`
- **evaluator.py**: Frame-level + video-level evaluation, cross-dataset matrix runner

### `src/experiments/`
- **tracker.py**: SQLite experiment database with metric logging and export
- **runner.py**: Orchestration of complete experimental workflows

### `src/explainability/`
- **explainer.py**: Grad-CAM (CNN), attention rollout (ViT), heatmap overlay, faithfulness check

### `src/robustness/`
- **tester.py**: Evaluation-time perturbation stress tests

## Design Decisions

1. **CPU-compatible**: All code works without GPU; GPU is optional
2. **No data fabrication**: Dashboard only shows actual experiment results
3. **Config-driven**: All hyperparameters in YAML configs
4. **Modular**: Each component independently testable
5. **SQLite tracking**: Lightweight, server-free experiment database
6. **Separation of concerns**: ML pipeline independent of UI
