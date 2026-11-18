# Results Template (Research Report)

## 1. Verified Results (from actual real-data runs only)

| Experiment ID | Dataset | Model | Split | AUROC | F1 | EER | Notes |
|---------------|---------|-------|-------|-------|----|-----|-------|
| pending | pending | pending | pending | pending | pending | pending | Fill only from `training_summary.json` after a real run |

## 2. Cross-Dataset Matrix (AUROC)

| Source \ Target | FF++ | Celeb-DF | DFDC |
|-----------------|------|----------|------|
| FF++            | –    |          |      |
| Celeb-DF        |      | –        |      |
| DFDC            |      |          | –    |

Generalization gaps to be filled after runs.

## 3. Ablation Summaries
Populate only from `ablation_results.csv` / `ablation_results.json` produced by an executed run.

## 4. Pending Experiments
- Full 6-pair matrix on complete datasets
- Diffusion held-out evaluation
- Full capacity + loss ablations
- Large-scale faithfulness study

## 5. Limitations
See `docs/limitations.md`.

## 6. Synthetic / Smoke Results (NON-RESEARCH)
Any numbers produced by `scripts/smoke_test.py` must be labelled:

> **SYNTHETIC – NOT A RESEARCH RESULT**
