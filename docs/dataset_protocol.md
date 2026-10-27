# Dataset Protocol & Feasibility Assessment

## Supported Datasets

1. **FaceForensics++ (FF++)**  
   - Real + 4 manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)  
   - Compression levels: c0 (raw), c23, c40  
   - Official access: https://github.com/ondyari/FaceForensics
   - License: research use only; request access and accept the academic agreement.
   - Small run: 50-200 balanced videos, preferably c23 for feasible storage.

2. **Celeb-DF (v2 recommended)**  
   - High-quality celebrity deepfakes  
   - Official access: https://github.com/yuezunli/celeb-deepfakeforensics
   - License: research use; download only after accepting the dataset agreement.
   - Small run: 50-200 balanced real/fake videos.

3. **DFDC (Deepfake Detection Challenge)**  
   - Large-scale, diverse identities and methods  
   - Official access: https://ai.facebook.com/datasets/dfdc/
   - License: follow the competition/dataset terms and do not redistribute videos.
   - Small run: 50-200 balanced real/fake videos.


## Folder Convention (after `prepare_data.py`)

```
data/
├── raw/
│   ├── faceforensics/
│   │   ├── original/
│   │   ├── deepfakes/
│   │   ├── face2face/
│   │   ├── faceswap/
│   │   └── neuraltextures/
│   ├── celebdf/
│   │   ├── real/
│   │   └── fake/
│   ├── dfdc/
│   │   ├── real/
│   │   └── fake/
├── faces/                  # output of preprocess_faces.py
│   ├── faceforensics/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── celebdf/
│   └── dfdc/
└── processed/              # manifests, splits, caches
```

## How a User Obtains a Small Usable Subset (50–200 videos)

### FaceForensics++
1. Request access: https://github.com/ondyari/FaceForensics  
2. Download only the **c23** compressed subset (smaller).  
3. Place a few dozen original + manipulated videos into the folders above.  
4. Run `python scripts/preprocess_faces.py --dataset faceforensics --max-videos 100`.

### Celeb-DF
1. Request: https://github.com/yuezunli/celeb-deepfakeforensics  
2. Download a subset of real + fake videos.  
3. Same preprocess command with `--dataset celebdf`.

### DFDC
1. Official page: https://ai.facebook.com/datasets/dfdc/  
2. Optional Kaggle command after accepting Kaggle terms: `kaggle competitions download -c deepfake-detection-challenge`.  
3. Extract 50–100 videos per class into `data/raw/dfdc/{real,fake}/`.  
4. Preprocess with `--max-videos 100`.

**Important**: The code never auto-downloads. All scripts document the expected layout and raise clear `FileNotFoundError` with instructions when faces or raw videos are missing.

## Splitting Rules (Leakage Prevention)

- **Video-level** primary split (train / val / test).  
- Where identity metadata is available (FF++, Celeb-DF), enforce **identity-disjoint** splits.  
- Frame sampling is performed **after** split; no frame from a test video ever appears in train.  
- Fixed random seed for split reproducibility.  
- Manifests saved under `data/processed/` for audit.

## Face Extraction

- Detector: RetinaFace (primary) → MTCNN → Haar cascade fallback.  
- Alignment: 5-point or approximate eye-based.  
- Output: 224×224 RGB, ImageNet normalization statistics.  
- Caching: per-video face directories; resumable via `--max-videos` and existing-file skip.  
- Quality filter: minimum face size, detection confidence threshold.

## Sampling Strategy (Shared Across Models)

- Fixed number of frames per video (configurable, default 16 or 32).  
- Uniform temporal sampling (or middle-biased for short clips).  
- Same strategy for train / val / test / cross-dataset evaluation.

## Aggregation

- Frame-level scores → video-level score via mean (default), median, or top-k mean.  
- Documented and fixed for all comparable experiments.
