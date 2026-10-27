# Experiment Protocol

## Seeds & Reproducibility
- Global seed from `configs/training.yaml` (default 42).
- Video-level splits fixed by seed; manifests saved under `data/processed/`.
- Full config + prediction dumps stored with each experiment ID.

## Within-Dataset Baselines
For each dataset ∈ {faceforensics, celebdf, dfdc} and model ∈ {efficientnet_b0, vit_small}:
1. Train on train split.
2. Early-stop on val AUROC.
3. Evaluate once on held-out test split (no threshold tuning).

## Cross-Dataset Matrix
For each ordered pair (source, target) in the 6-pair matrix:
1. Train exclusively on source train split.
2. Select checkpoint by source val AUROC only.
3. Evaluate on target test split with the **same** decision threshold (default 0.5) or pure AUROC (threshold-free).
4. Record generalization gap = source_test_AUROC − target_test_AUROC. A positive value is a target-domain drop.

Run with a small real subset:

```bash
python scripts/run_cross_dataset.py --dry-run --max-videos 100
python scripts/run_cross_dataset.py --execute --max-videos 100 --epochs 5
```

The executable command trains each source independently, evaluates only the target test split, and writes CSV/JSON under `artifacts/experiment_logs/`.

## Ablations
- **Augmentation**: none / jpeg / blur / resize-reencode / combined (train only).
- **Loss**: BCE / Focal / class-weighted BCE.
- **Capacity**: EfficientNet-B0 vs B4; ViT-Small vs Base.

```bash
python scripts/run_ablations.py --dry-run --dataset faceforensics --max-videos 100
python scripts/run_ablations.py --execute --dataset faceforensics --max-videos 100 --epochs 5
```

Outputs are written as `ablation_results.csv` and `ablation_results.json`.

## Artifact and Holdout Commands

```bash
python scripts/train.py --dataset faceforensics --model efficientnet_b0 --epochs 5 --max-videos 80
python scripts/evaluate.py --dataset faceforensics --split test --checkpoint artifacts/models/<experiment_id>/best.pt --model efficientnet_b0
python scripts/run_explainability.py --checkpoint artifacts/models/<experiment_id>/best.pt --dataset faceforensics --max-videos 32
```

Real checkpoints and summaries are stored under `artifacts/models/<experiment_id>/`. The legacy `artifacts/checkpoints/` directory remains reserved for the Stage-1 smoke test.

## Explainability (Gap F)
- Grad-CAM on CNN correct & incorrect predictions (in-domain + out-of-domain).
- Attention Rollout on ViT.
- Simple faithfulness: delete top-k% salient regions and measure confidence drop.

## Robustness
- Train-time: controlled JPEG/blur augmentation.
- Eval-time stress: JPEG q=40/20, Gaussian σ=1.5/3.0; measure AUROC drop.
