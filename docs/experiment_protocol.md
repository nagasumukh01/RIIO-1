# Experiment Protocol

## Overview

This document defines the experimental protocol ensuring scientific validity and reproducibility.

## Experimental Design

### A. Within-Dataset Baselines
- Train and evaluate each model on each dataset separately
- Disjoint video-level splits: 70/15/15
- Seeds fixed at 42 (configurable)
- Report: Accuracy, AUROC, F1, EER, Precision, Recall

### B. Cross-Dataset Matrix (6 Pairs)
All directional pairs with NO target data leakage:

| # | Train | Test |
|---|-------|------|
| 1 | FF++ | Celeb-DF |
| 2 | FF++ | DFDC |
| 3 | Celeb-DF | FF++ |
| 4 | Celeb-DF | DFDC |
| 5 | DFDC | FF++ |
| 6 | DFDC | Celeb-DF |

**Rules**:
- Train model to convergence on source dataset
- Evaluate as-is on target test set
- NO fine-tuning on target
- NO threshold tuning on target
- NO hyperparameter selection using target metrics
- Use threshold=0.5 for binary decisions (or EER threshold from source validation)

### C. Diffusion-Generated Evaluation
- Completely separate from training
- Same preprocessing pipeline
- Report results with clear provenance

### D. Augmentation Ablation
6 strategies compared with identical data, seeds, architecture, and training budget:
1. None (baseline)
2. JPEG compression (quality 30–95)
3. Gaussian blur (kernel 3–7)
4. Resize + re-encoding (scale 0.5–0.9)
5. Combined compression + blur
6. Full augmentation (all + color jitter + flip)

### E. Loss Function Ablation
3 loss functions with identical everything else:
1. Cross-entropy
2. Focal loss (γ=2.0, α=0.25)
3. Weighted BCE

### F. Backbone Capacity Ablation
4 architectures with identical training conditions:
1. EfficientNet-B0 (5.3M)
2. EfficientNet-B4 (19.3M)
3. ViT-Small/16 (22.1M)
4. ViT-Base/16 (86.6M)

### G. Explainability Evaluation
- Generate attribution maps for correct and incorrect predictions
- In-domain and cross-dataset examples
- Perturbation-based faithfulness check (progressive masking)

### H. Robustness Stress Tests
Applied at evaluation time ONLY:
- JPEG Q30, Q50
- Gaussian blur K5, K7
- Resize 0.5×
- Combined stress

## Metric Definitions

| Metric | Formula | Range |
|--------|---------|-------|
| Accuracy | (TP+TN)/Total | [0,1] |
| Precision | TP/(TP+FP) | [0,1] |
| Recall | TP/(TP+FN) | [0,1] |
| F1 | 2×P×R/(P+R) | [0,1] |
| AUROC | Area under ROC curve | [0,1] |
| EER | FPR = FNR crossover | [0,1] |
| FPR | FP/(FP+TN) | [0,1] |
| FNR | FN/(FN+TP) | [0,1] |
| Gen. Gap | (in - cross)/in × 100% | [0,100] |

## Per-Experiment Recording

Every experiment records:
- Unique experiment ID
- Timestamp (start + end)
- Source and target dataset names
- Model architecture + config
- Random seed
- Data split IDs
- Hardware info (CPU/GPU, memory)
- Training time
- All metrics listed above
- Per-sample predictions (for recomputation)
- Checkpoint path

## Reporting Rules

1. Never merge train/val/test scores
2. Never use test data for model or threshold selection
3. Report per-dataset AND macro-averaged results
4. Include seed variability when compute permits
5. Report negative/failed results honestly
6. Never compare metrics from incompatible protocols
