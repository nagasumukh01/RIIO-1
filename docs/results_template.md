# Results Template

## Experiment Report: [Experiment Name]

**Date**: [YYYY-MM-DD]  
**Experiment ID**: [auto-generated]  
**Status**: [Completed / Failed / Pending]

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | [EfficientNet-B0 / ViT-Base / ...] |
| Source Dataset | [FF++ / Celeb-DF / DFDC / Synthetic] |
| Target Dataset | [same / other / N/A] |
| Seed | [42] |
| Epochs | [N] |
| Batch Size | [16] |
| Learning Rate | [0.0001] |
| Loss Function | [cross_entropy / focal / weighted_bce] |
| Augmentation | [none / jpeg / blur / combined / ...] |
| Device | [cpu / cuda] |

---

## Results

### Within-Dataset Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | | | |
| AUROC | | | |
| F1 | | | |
| Precision | | | |
| Recall | | | |
| EER | | | |
| FPR | | | |
| FNR | | | |

### Cross-Dataset Performance (if applicable)

| Train → Test | AUROC | Accuracy | F1 | Gen. Gap (%) |
|-------------|-------|----------|------|-------------|
| FF++ → Celeb-DF | | | | |
| FF++ → DFDC | | | | |
| ... | | | | |

### Model Information

| Property | Value |
|----------|-------|
| Total Parameters | |
| Trainable Parameters | |
| Checkpoint Size (MB) | |
| Training Time (s) | |
| Inference Latency (ms/sample) | |
| Peak Memory (MB) | |

---

## Training Curves

[Insert training loss / validation AUROC plots here]

---

## Observations

- [Key finding 1]
- [Key finding 2]
- [Unexpected behavior / failure mode]

## Limitations of This Experiment

- [Specific limitation 1]
- [Specific limitation 2]

## Negative Results (if any)

- [Document poor performance honestly]
- [Explain possible reasons]

---

*This report was generated from actual model predictions stored in `artifacts/experiment_logs/[experiment_id]/`.*
*Metrics can be independently recomputed from saved prediction logs.*
