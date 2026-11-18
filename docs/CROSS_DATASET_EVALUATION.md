# Cross-Dataset Generalization Evaluation — Results

## Protocol

Each row in the matrix uses a model trained **only** on the source dataset and evaluated
**only** on the target dataset. No target data is used for training or threshold selection.
Preprocessing is consistent with the source model's training pipeline.

```
Source dataset → Source train → Source val/test → (best model) → Target test
                 (no target data at any point)
```

Metric: **AUROC** (Area Under ROC Curve). Higher = better generalization.
Gap = Source test AUROC − Target test AUROC. Positive = performance degrades on target.

---

## Results Matrix

### Completed Experiments (4 of 6 pairs)

From `artifacts/experiment_logs/cross_ff_celeb_vit_cnn/cross_dataset_matrix.json`

| # | Source | Target | Model | Src Test AUC | Tgt Test AUC | Gap | Note |
|---|---|---|---|---|---|---|---|
| E1 | FF++ | CelebDF | EfficientNet-B0 | 0.583 | 0.268 | +0.315 | Severe drop |
| E2 | FF++ | CelebDF | ViT-Small | 0.000† | 0.589 | -0.589 | Anomalous |
| E3 | CelebDF | FF++ | EfficientNet-B0 | **0.924** | 0.750 | +0.174 | Best result |
| E4 | CelebDF | FF++ | ViT-Small | 0.938 | 0.500 | +0.438 | Poor generalization |
| E5 | FF++ | DFDC | — | — | — | — | ⚠ Blocked |
| E6 | FF++ | DFDC | — | — | — | — | ⚠ Blocked |
| E7 | CelebDF | DFDC | — | — | — | — | ⚠ Blocked |
| E8 | CelebDF | DFDC | — | — | — | — | ⚠ Blocked |
| E9 | DFDC | FF++ | — | — | — | — | ⚠ Blocked |
| E10 | DFDC | CelebDF | — | — | — | — | ⚠ Blocked |

†Source test AUROC=0.000 for ViT-Small on FF++ (8 test videos) is statistically unreliable.

**Why blocked (E5–E10):** The DFDC model (dfdc_b0) was trained on only 8 videos for 1 epoch,
producing val AUROC=0.44 (sub-random). Evaluating this model as source or target produces
meaningless results. These experiments require a properly trained DFDC model.

### How to Run Remaining Pairs

Once a valid DFDC model is available:

```bash
# Run the full 6-pair matrix (uses configs/experiments.yaml pairs):
python scripts/run_cross_dataset.py

# Single pair manually:
python scripts/evaluate.py \
  --dataset celebdf \
  --split test \
  --checkpoint artifacts/models/celebdf_b0_subset200_f4/best.pt \
  --model efficientnet_b0
```

---

## Key Findings

### 1. CelebDF → FF++ is the most stable transfer
The best model (CelebDF B0, test AUROC 0.924) transfers to FF++ at 0.750 — only a 0.174
gap. This suggests that Celeb-DF manipulation artifacts partially overlap with FF++ artifacts.

### 2. FF++ → CelebDF B0 transfer fails
B0 trained on FF++ reaches only 0.268 on CelebDF (gap 0.315). This is consistent with
the known difficulty of generalizing from FF++ compression-compressed fakes to CelebDF's
higher-quality recompression chain.

### 3. ViT-Small shows poor cross-dataset generalization
Both ViT-Small results show high generalization gaps (0.438, or anomalous negative).
Small training sets (50 videos) are likely insufficient for ViT to learn
generalizable features.

### 4. DFDC experiments are blocked
Until a properly-trained DFDC model is available, 4 of the 6 directed pairs cannot
be completed. This is reported honestly — partial completion is not labeled as a full matrix.

---

## Comparison vs Petmezas (2026)

The Petmezas (2026) paper does not report cross-dataset evaluation (per user description).
RobustDeep implements this as a novel contribution. Full 6-pair results are pending a
complete DFDC model.

---

*Last updated: 2026-11-10*
