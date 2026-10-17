# RobustDeep — Explainable ViT-Based Video Deepfake Detection Under Cross-Dataset Distribution Shift

A complete, reproducible, research-oriented video deepfake detection system comparing CNN and Vision Transformer architectures under cross-dataset evaluation with explainability analysis.

## 🔬 Research Focus

Deepfake detectors often achieve high accuracy within their training distribution but collapse when tested on unseen datasets. This system systematically investigates:

1. **Cross-dataset generalization**: Full 6-pair directional evaluation matrix (FF++ ↔ Celeb-DF ↔ DFDC)
2. **CNN vs ViT**: Whether Vision Transformers generalize better than CNN baselines
3. **Augmentation robustness**: Impact of compression-aware training on cross-dataset performance
4. **Explainability**: Grad-CAM (CNN) and attention rollout (ViT) under distribution shift
5. **Controlled ablations**: Augmentation strategy, loss function, and backbone capacity

## 📋 Quick Start

### 1. Install Dependencies

```bash
cd "Rio project"
pip install -r requirements.txt
```

### 2. Run Smoke Test (Validates Pipeline)

```bash
python scripts/run_all.py
```

This runs CNN + ViT smoke tests on synthetic data, augmentation/loss/backbone ablations, and generates all result files.

> ⚠️ Smoke test uses synthetic data. These metrics validate pipeline correctness, NOT research performance.

### 3. Launch Dashboard

```bash
streamlit run app.py
```

### 4. Run Specific Experiments

```bash
# Train individual model
python scripts/train.py --model efficientnet_b0 --dataset synthetic --epochs 10

# Run ablation studies
python scripts/run_ablations.py --type augmentation --epochs 5
python scripts/run_ablations.py --type loss --epochs 5
python scripts/run_ablations.py --type backbone --epochs 5
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## 🏗 Architecture

```
robustdeep/
├── app.py                  # Streamlit research dashboard (11 pages)
├── requirements.txt        # Pinned dependencies
├── configs/                # YAML configuration
│   ├── data.yaml          # Data pipeline + augmentation
│   ├── models.yaml        # CNN/ViT/Hybrid architectures
│   ├── training.yaml      # Training hyperparameters
│   └── experiments.yaml   # 6-pair matrix + ablation configs
├── src/
│   ├── data/              # Dataset, preprocessing, augmentation
│   ├── models/            # EfficientNet, ViT, Hybrid CNN-ViT
│   ├── training/          # Training engine + loss functions
│   ├── evaluation/        # Evaluator + cross-dataset matrix
│   ├── experiments/       # SQLite tracker + experiment runner
│   ├── explainability/    # Grad-CAM, attention rollout, faithfulness
│   └── robustness/        # Compression/blur stress tests
├── scripts/               # CLI scripts (train, evaluate, ablate)
├── tests/                 # Unit + integration tests
├── docs/                  # Research documentation
│   ├── literature_review.md
│   ├── research_gaps.md
│   ├── dataset_protocol.md
│   ├── experiment_protocol.md
│   ├── architecture.md
│   ├── limitations.md
│   └── results_template.md
└── artifacts/             # Checkpoints, logs, figures
```

## 📊 Experimental Design

### Models
| Model | Type | Parameters | Input |
|-------|------|------------|-------|
| EfficientNet-B0 | CNN | 5.3M | 224×224 |
| EfficientNet-B4 | CNN | 19.3M | 224×224 |
| ViT-Small/16 | Transformer | 22.1M | 224×224 |
| ViT-Base/16 | Transformer | 86.6M | 224×224 |
| Hybrid CNN-ViT | Hybrid | ~10M | 224×224 |

### Cross-Dataset Matrix
All 6 directional pairs with **NO target fine-tuning**:
- FF++ → Celeb-DF, FF++ → DFDC
- Celeb-DF → FF++, Celeb-DF → DFDC
- DFDC → FF++, DFDC → Celeb-DF

### Ablation Studies
- **Augmentation**: None / JPEG / Blur / Resize / Combined / Full
- **Loss Function**: Cross-entropy / Focal / Weighted BCE
- **Backbone Capacity**: B0 / B4 / ViT-S / ViT-B

### Metrics
AUROC, Accuracy, Precision, Recall, F1, EER, FPR, FNR, confusion matrix, ROC curve, generalization gap, robustness drop.

## 📄 Reference

Petmezas, G. (2026). "Cross-dataset video deepfake detection using Transformer and CNN architectures." *Machine Vision and Applications*. DOI: 10.1007/s00138-026-01809-w

This project extends the methodology by adding explainability (Gap F), diffusion evaluation support (Gap B), controlled ablations (Gaps C/D/E), and a stricter 6-pair cross-dataset matrix (Gap A).

## ⚠️ Disclaimer

This is a **probabilistic research system**. It must NOT be treated as conclusive forensic evidence. All metrics come from actual model predictions, never fabricated.

## 📝 License

Research use only. Datasets have their own licenses (see docs/dataset_protocol.md).
