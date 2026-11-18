# Real Data Quickstart (4 Commands)

This is the **only** supported research path. Synthetic data is for Stage-1 smoke tests only.

## Prerequisites
- Python ≥ 3.10
- GPU recommended but not required for small subsets
- 50–200 videos from FF++ / Celeb-DF / DFDC (research license)

## Exact Three-Dataset Flow

```bash
# 1. Create all three folder layouts + print official acquisition instructions
python scripts/prepare_data.py --dataset all --max-videos 100

# 2. Place licensed videos into the matching raw folders, unless they are already
#    in the repository's native dataset folders:
#    data/raw/faceforensics/{original,deepfakes,face2face,faceswap,neuraltextures}/
#    data/raw/celebdf/{real,fake}/
#    data/raw/dfdc/{real,fake}/
#    Existing datasets/FaceForensics++ c23 videos and datasets/DFDC shard folders
#    with metadata.json are read directly by preprocess_faces.py.

# 3. Extract faces from a balanced, group-preserving subset (resumable)
python scripts/preprocess_faces.py --dataset faceforensics --max-videos 100
python scripts/preprocess_faces.py --dataset celebdf --max-videos 100
python scripts/preprocess_faces.py --dataset dfdc --max-videos 100

# 4. Train one real model per official dataset
python scripts/train.py --dataset faceforensics --model efficientnet_b0 --epochs 5 --max-videos 80
python scripts/train.py --dataset celebdf --model efficientnet_b0 --epochs 5 --max-videos 80
python scripts/train.py --dataset dfdc --model efficientnet_b0 --epochs 5 --max-videos 80
```

Real checkpoints are stored separately under `artifacts/models/<experiment_id>/`. The legacy `artifacts/checkpoints/` path is reserved for the synthetic Stage-1 smoke test.

## Forensic Console Model Checkpoints

The console reads checkpoints from the paths reported by `GET /api/models`. If a
registered checkpoint is missing, train the matching model using its registered
experiment ID. These commands produce the two ViT checkpoints currently expected
by the registry:

```bash
python scripts/train.py --dataset celebdf --model vit_small --epochs 5 --frames 4 --experiment-id celebdf_vit_small_subset50_f4
python scripts/train.py --dataset faceforensics --model vit_small --epochs 5 --frames 4 --experiment-id ff_c23_vit_small_subset50_f4
```

The resulting files are respectively
`artifacts/models/celebdf_vit_small_subset50_f4/best.pt` and
`artifacts/models/ff_c23_vit_small_subset50_f4/best.pt`. The ensemble averages
the available CNN/ViT model probabilities and reports any missing members; it
never substitutes weights from another dataset or architecture.

## Inference Preprocessing and Checkpoint Validation

`configs/inference.yaml` stores preprocessing, label indices, and decision
thresholds independently for each registered model. The current training path
uses RGB 224x224 face crops, ImageNet mean/std normalization, and a single
binary logit where real is label 0 and fake is label 1. Inference uses the same
normalization and interpolation. For still images, the overall decision uses
the primary face crop rather than averaging the extra context/eye/mouth
visualization crops; for videos it aggregates the detected face crops.
Crop geometry is not guaranteed identical: preprocessing prefers RetinaFace,
then MTCNN, then Haar and expands the detected square by 1.3x, while API
inference uses its verified Haar face box and margins. The manifests do not
record which preprocessing detector was selected for each dataset, so that
part of the train/inference path cannot be confirmed from saved metadata.
Models trained with four frames (`ff_b0`, `celebdf_vit`, and `ff_vit`) are
capped at four inference face crops per video; other models retain their
registered video view policy.

The configured 0.5 thresholds are the uncalibrated binary-logit defaults, not
thresholds optimized for a test set or a guarantee of accuracy. Keep each
model's settings aligned with the preprocessing used for its checkpoint.
Checkpoint tensors are loaded strictly; the API logs missing/unexpected state
keys when each checkpoint is first loaded and rejects incompatible weights.
The model catalog endpoint includes each model's effective inference config.

## Verify Real Path
```bash
pytest tests/test_real_data_path.py -v
```

## Cross-Dataset (after faces exist for multiple datasets)
```bash
python scripts/run_cross_dataset.py --dry-run --max-videos 100
python scripts/run_cross_dataset.py --execute --max-videos 100 --epochs 5
```

## Dashboard
```bash
streamlit run app.py
```

All metrics from real runs are labelled **REAL DATA**. Smoke-test metrics are labelled **SYNTHETIC – NOT A RESEARCH RESULT**.
