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

## Stage 3 – Within-dataset + 6-pair evaluation ✅ EXECUTABLE WHEN FACES EXIST
- [x] `scripts/evaluate.py`
- [x] `scripts/run_cross_dataset.py` (dry-run and executable six-pair matrix)
- [ ] Full runs require user-provided face crops

## Stage 4 – Ablations ✅ EXECUTABLE WHEN FACES EXIST
- [x] `scripts/run_ablations.py` (augmentation / loss / capacity execution and CSV/JSON export)

## Stage 5 – Explainability + Robustness ✅ EXECUTABLE INTERFACES
- [x] Grad-CAM module
- [x] Attention Rollout helper
- [x] Stress tests (JPEG, blur)
- [x] `scripts/run_explainability.py` (attribution figures and perturbation report)

## Stage 6 – Dashboard + Docs ✅ COMPLETE
- [x] Streamlit research dashboard (`app.py`) with mandatory disclaimer
- [x] Full documentation package under `docs/`
- [x] `docs/REAL_DATA_QUICKSTART.md` (exact 4-command flow)
- [x] requirements.txt, README, CI workflow, .gitignore

## Next concrete task for a user with data
1. Obtain 50–200 videos of FF++ / Celeb-DF / DFDC under research license.
2. Follow `docs/REAL_DATA_QUICKSTART.md`.
3. Run the exact commands in `docs/experiment_protocol.md`.
4. Expand to cross-dataset matrix and ablations only after the required datasets have face crops.

## What cannot be completed without external data / GPU
- Actual within-dataset and cross-dataset numeric results on real videos
- Fair comparison numbers vs Petmezas 2026
- Full ViT-Base training runs
