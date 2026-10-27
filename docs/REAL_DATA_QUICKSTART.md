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

# 2. Place licensed videos into the matching raw folders:
#    data/raw/faceforensics/{original,deepfakes,face2face,faceswap,neuraltextures}/
#    data/raw/celebdf/{real,fake}/
#    data/raw/dfdc/{real,fake}/

# 3. Extract faces from a small subset (resumable, limited)
python scripts/preprocess_faces.py --dataset faceforensics --max-videos 100
python scripts/preprocess_faces.py --dataset celebdf --max-videos 100
python scripts/preprocess_faces.py --dataset dfdc --max-videos 100

# 4. Train one real model per official dataset
python scripts/train.py --dataset faceforensics --model efficientnet_b0 --epochs 5 --max-videos 80
python scripts/train.py --dataset celebdf --model efficientnet_b0 --epochs 5 --max-videos 80
python scripts/train.py --dataset dfdc --model efficientnet_b0 --epochs 5 --max-videos 80
```

Real checkpoints are stored separately under `artifacts/models/<experiment_id>/`. The legacy `artifacts/checkpoints/` path is reserved for the synthetic Stage-1 smoke test.

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
