# Dataset Protocol

## Overview

This document describes the datasets used in RobustDeep, their access procedures, licenses, preprocessing steps, and splitting protocols.

## Datasets

### 1. FaceForensics++ (FF++)

- **Source**: [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- **Size**: ~1000 original videos, ~4000 manipulated (4 methods × 1000)
- **Methods**: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- **Compression levels**: c0 (raw), c23 (light), c40 (heavy)
- **License**: Academic use only; requires filling Google Form agreement
- **Access**: Request download script via official GitHub; download via provided script
- **Storage estimate**: ~50 GB for c23 compression

### 2. Celeb-DF (v2)

- **Source**: [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- **Size**: 590 real videos + 5639 fake videos
- **Methods**: Improved DeepFake synthesis with visual quality enhancements
- **License**: Research use only
- **Access**: Direct download via provided links after reading terms
- **Storage estimate**: ~15 GB

### 3. DFDC (Deepfake Detection Challenge)

- **Source**: [kaggle.com/c/deepfake-detection-challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
- **Size**: 100,000+ videos (using documented subset of ~5000)
- **Methods**: Multiple methods, paid actors, diverse demographics
- **License**: Kaggle competition agreement
- **Access**: Download via Kaggle CLI after accepting terms
- **Storage estimate**: ~500 GB full; ~25 GB for subset

### 4. Diffusion-Generated Set

- **Source**: To be determined
- **Size**: TBD
- **Methods**: Diffusion-based face manipulation / generation
- **License**: TBD
- **Status**: Interface implemented; data collection pending
- **Notes**: Will be kept completely separate from training. Must document generation method, provenance, sample counts, and known limitations.

## Preprocessing Pipeline

```
Video → Metadata Validation → Frame Sampling → Face Detection → Alignment → Crop → Normalize
```

### Step 1: Metadata Validation
- Verify file integrity (can open, can read frames)
- Log corrupted files
- Record resolution, FPS, frame count, duration

### Step 2: Frame Sampling
- **Strategy**: Uniform sampling (configurable)
- **Default**: 16 frames per video
- **Alternatives**: Random, keyframe-based

### Step 3: Face Detection
- **Primary**: MTCNN (via facenet-pytorch)
- **Fallback**: OpenCV Haar cascade
- **Final fallback**: Center crop
- **Confidence threshold**: 0.9
- **Selection**: Largest face in frame

### Step 4: Alignment & Cropping
- Add margin (30% default) around detected face
- Resize to 224×224 (configurable)
- Maintain aspect ratio within margin

### Step 5: Normalization
- Convert to float32 [0, 1]
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Splitting Protocol

### Rules
1. **Video-level splits**: ALL frames from a single video belong to the SAME split
2. **Identity-aware**: Where metadata is available, same identity stays in same split
3. **No leakage**: Split validation before any frame extraction or augmentation
4. **Fixed ratios**: 70% train / 15% validation / 15% test
5. **Fixed seed**: seed=42 for all splits (configurable)

### Cross-Dataset Protocol
- Models trained on source dataset evaluated directly on target test set
- **NO** fine-tuning on target data
- **NO** threshold tuning on target data
- **NO** hyperparameter selection using target performance
- Same preprocessing and normalization applied to all datasets

## Data Download Scripts

Download scripts are provided in `scripts/prepare_data.py` with clear instructions.
Each script:
1. Shows license/terms information before downloading
2. Requires explicit user confirmation
3. Validates downloaded files
4. Logs any issues

## Ethical Considerations

1. All datasets used for research purposes only
2. No personal data collected beyond what datasets provide
3. No re-distribution of dataset videos
4. Results reported on aggregate metrics, not individual identities
5. System explicitly disclaims forensic validity
6. Temporary storage with deletion controls for user-uploaded content
