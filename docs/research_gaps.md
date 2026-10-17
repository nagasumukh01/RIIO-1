# Research Gaps — Differences from Petmezas (2026)

This document records the explicit differences between this project and the primary reference paper, along with the research gaps we investigate.

## Summary of Differences

| Aspect | Petmezas 2026 | This Project (RobustDeep) |
|--------|---------------|--------------------------|
| **Datasets** | FF++, Celeb-DF, DFD, ReenactFaces | FF++, Celeb-DF, DFDC + diffusion set |
| **Cross-dataset protocol** | Train on FF++ + Celeb-DF combined → test on DFD + ReenactFaces combined; optional fine-tuning | Full 6-pair directional matrix; NO target fine-tuning; NO threshold tuning on target |
| **Ablation** | Fine-tuning % and clip length | Augmentation strategy, loss function, backbone capacity |
| **Explainability** | None (listed as future work) | Grad-CAM (CNN) + attention rollout (ViT) + faithfulness check |
| **Robustness** | Mentioned as future work only | Compression + blur augmentation during training; evaluation-time stress tests |
| **Diffusion evaluation** | Not covered | Interface implemented; pending data access |

---

## Gap A — Broader Cross-Dataset Evaluation

**Hypothesis**: Performance drops systematically when models are tested on datasets they have never seen, with some dataset-pair directions being harder than others.

**Implementation**:
- Full 6-pair matrix: FF++ ↔ Celeb-DF ↔ DFDC
- NO target fine-tuning, NO threshold tuning, NO hyperparameter adjustment on target
- Record AUROC, Accuracy, F1, EER for each direction
- Compute generalization gap as percentage drop from within-dataset performance

**Status**: ✅ Infrastructure implemented. Pending real dataset access.

---

## Gap B — Diffusion-Generated Manipulation Evaluation

**Hypothesis**: Models trained on traditional face-swap/reenactment manipulations may fail against diffusion-generated manipulations, which produce fundamentally different artifacts.

**Implementation**:
- Search for suitable public datasets (e.g., from recent diffusion model benchmarks)
- Keep completely separate from training and model selection
- Evaluate CNN and ViT models under identical preprocessing
- Report results separately with full provenance documentation

**Status**: ⏳ Evaluation interface implemented. Data collection pending.

---

## Gap C — Controlled Augmentation Ablation

**Hypothesis**: Training with compression-aware augmentation improves robustness to post-processing degradation without significantly hurting within-dataset performance.

**Strategies compared**:
1. No extra augmentation (baseline)
2. JPEG compression only (quality 30–95)
3. Gaussian blur only (kernel 3–7)
4. Resize + re-encoding
5. Combined compression + blur
6. Full augmentation (compression + blur + color + flip)

**Controls**:
- Same data splits, same seed, same training budget
- Augmentation applied ONLY to training data
- Evaluation always on clean data (unless running stress tests)

**Status**: ✅ Implemented and runnable.

---

## Gap D — Loss Function Ablation

**Hypothesis**: The choice of loss function affects both within-dataset performance and cross-dataset generalization, particularly for imbalanced datasets.

**Functions compared**:
1. Standard binary cross-entropy
2. Focal loss (γ=2.0, α=0.25) — emphasizes hard examples
3. Weighted BCE — compensates for class imbalance

**Controls**:
- Identical data splits, seeds, and training budgets
- Same model architecture (EfficientNet-B0)

**Status**: ✅ Implemented and runnable.

---

## Gap E — Backbone Capacity Ablation

**Hypothesis**: Larger models may overfit more on within-dataset evaluation but could either generalize better or worse across datasets due to capacity/regularization tradeoffs.

**Models compared**:
1. EfficientNet-B0 (5.3M params) — lightweight CNN
2. EfficientNet-B4 (19.3M params) — larger CNN
3. ViT-Small/16 (22.1M params) — compact ViT
4. ViT-Base/16 (86.6M params) — standard ViT

**Measured**:
- Parameter count, checkpoint size, peak memory
- Training time, inference latency
- All evaluation metrics

**Status**: ✅ Implemented and runnable.

---

## Gap F — Explainability Under Distribution Shift

**Hypothesis**: Model explanations (attention/gradient maps) may highlight different regions when the model is applied to in-domain vs. out-of-domain data, and the faithfulness of these explanations degrades under distribution shift.

**Methods**:
- Grad-CAM for CNN models
- Attention rollout for ViT models
- Frame-level heatmaps + video-level summary
- Perturbation-based faithfulness check

**Evaluation cases**:
- Correct predictions (true positives, true negatives)
- False positives, false negatives
- Uncertain cases (near-threshold confidence)
- In-domain and cross-dataset examples

**Important**: All visualizations are labeled as "model attribution maps" — NOT forensic proof.

**Status**: ✅ Implemented and runnable.

---

## Gap G — Reproducibility and Failure Analysis

**Hypothesis**: Full reproducibility requires not just code sharing but also fixed splits, configurable seeds, saved predictions, and explicit failure documentation.

**Implementation**:
- SQLite experiment tracker with full metadata
- Saved configs, seeds, split IDs per experiment
- Per-sample prediction logs (not just aggregate metrics)
- Failed/negative results reported alongside successes
- Environment documentation (Python version, CUDA, hardware)

**Status**: ✅ Implemented.

---

## Important Notes

1. We do NOT claim any of these gaps as novel contributions by themselves. Cross-dataset evaluation, ViTs, and Grad-CAM are all well-established techniques.
2. The contribution is the **systematic combination** of these techniques into a reproducible experimental framework, with controlled ablations and measured generalization.
3. We never claim accuracy improvement over Petmezas (or any other paper) until a fair, documented comparison is completed.
4. A scientifically valid negative result (e.g., poor cross-dataset performance) is acceptable and reported honestly.
