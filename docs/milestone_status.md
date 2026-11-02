# Milestone Status (as of implementation)

## Stage 1 – Smoke test only ✅ COMPLETE
- [x] Synthetic dataset (`src/data/synthetic.py`)
- [x] Unit tests including real-data path guard (`tests/`)
- [x] `scripts/smoke_test.py` runs end-to-end, labels metrics **SYNTHETIC – NOT A RESEARCH RESULT**
- [x] 5/5 pytest passed
- [x] Training summary written under `artifacts/checkpoints/smoke_synthetic_stage1/`

## Stage 2 – Real data path working ✅ COMPLETE (interface + guard)
- [x] `scripts/prepare_data.py` creates correct folder structure + instructions
- [x] `scripts/preprocess_faces.py` (resumable, `--max-videos`, RetinaFace/MTCNN/Haar)
- [x] `src/data/real_dataset.py` — canonical loader used by train/evaluate
- [x] `scripts/train.py` supports `--dataset faceforensics|celebdf|dfdc`
- [x] Missing faces → clear `FileNotFoundError` with exact 4-step instructions (verified)
- [x] **No synthetic fallback** in any research entry point

## Stage 3 – Within-dataset + cross-dataset evaluation ✅ PARTIALLY EXECUTED
- [x] `scripts/evaluate.py`
- [x] `scripts/run_cross_dataset.py` (dry-run and executable six-pair matrix)
- [x] FF++ and Celeb-DF CNN/ViT cross-evaluations written to `artifacts/experiment_logs/cross_ff_celeb_vit_cnn/`
- [ ] DFDC-dependent pairs require official DFDC training data, which is currently unavailable

## Stage 4 – Ablations ✅ EXECUTABLE WHEN FACES EXIST
- [x] `scripts/run_ablations.py` (augmentation / loss / capacity execution and CSV/JSON export)
- [ ] Full ablation tables require dedicated compute runs; no fabricated comparison values are included

## Stage 5 – Explainability + Robustness ✅ EXECUTED FOR AVAILABLE REAL CHECKPOINTS
- [x] Grad-CAM module
- [x] Attention Rollout helper
- [x] Stress tests (JPEG, blur)
- [x] `scripts/run_explainability.py` (attribution figures and perturbation report)
- [x] `scripts/run_robustness.py` (JPEG/blur stress JSON and CSV reports)
- [x] Separate diffusion holdout evaluator with explicit pending status when no samples exist

## Stage 6 – Dashboard + Docs ✅ COMPLETE
- [x] Streamlit research dashboard (`app.py`) with mandatory disclaimer
- [x] Full documentation package under `docs/`
- [x] `docs/REAL_DATA_QUICKSTART.md` (exact 4-command flow)
- [x] requirements.txt, README, CI workflow, .gitignore

## Verified Current Results
- [x] FF++ real-data EfficientNet-B0 and ViT-Small checkpoints and held-out artifacts
- [x] Celeb-DF real-data EfficientNet-B0 and ViT-Small checkpoints and held-out artifacts
- [x] Matched FF++/Celeb-DF cross-model matrix
- [x] Assignment report updated with metrics, figures, cross-validation, robustness, and diffusion status
- [x] Synthetic smoke test remains isolated under `artifacts/checkpoints/smoke_synthetic_stage1/`

## Remaining External Blockers
1. DFDC labeled training data is unavailable because the official competition is closed.
2. Diffusion metrics require user-provided licensed diffusion samples.
3. Full six-pair matrix, complete ablation tables, and paper-level comparison require those data and additional compute.

## What cannot be completed without external data / GPU
- DFDC-dependent within-dataset and cross-dataset numeric results
- Diffusion-heldout numeric results without user-provided samples
- Fair comparison numbers vs Petmezas 2026
- Full ViT-Base training runs
