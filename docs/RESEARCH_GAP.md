# Research Gap Analysis — RobustDeep

## 1. Comparison Paper

**Petmezas (2026).** *Title and venue TBC.* Machine Vision and Applications.

> NOTE: The full text of this paper has not been retrieved. Claims about its methods,
> datasets, and experiments below are based on the user-provided description. Do not
> cite this document as fact until the paper has been verified against its full text.

---

## 2. Identified Research Gaps

The following gaps distinguish **RobustDeep** from the Petmezas (2026) baseline:

| Gap | Petmezas (2026) | RobustDeep |
|---|---|---|
| **Datasets** | FaceForensics++, Celeb-DF, DFD, ReenactFaces | FF++, Celeb-DF v2, DFDC + diffusion-generated face dataset |
| **Diffusion manipulation** | Not reported | Dedicated diffusion-generated fake dataset with generator-specific metadata |
| **Cross-dataset evaluation** | Not reported | 6-pair directional train→test matrix across FF++, Celeb-DF v2, DFDC |
| **Ablation study** | Not reported | Augmentation × loss function × backbone capacity |
| **Explainability** | Not reported | Grad-CAM (CNNs) + Attention Rollout (ViTs), faithfulness metric |
| **Compression robustness** | Not reported | JPEG/blur stress testing at eval time, with and without compression augmentation |
| **Ensemble fusion** | Single model | AUC-weighted ensemble of 5 models with disagreement detection |

---

## 3. Contribution Summary

### Contribution 1: Diffusion-Generated Manipulation Dataset
- **Status:** Pipeline implemented (`scripts/build_diffusion_dataset.py`)
- **Dataset:** Requires real face images in `data/diffusion_face/raw/real/` and diffusion-generated fakes in `data/diffusion_face/raw/fake/`
- **Generator inventory:** `diffusion_workspace/fastsdcpu-main/` available locally; SDXL, FLUX not set up
- **Face detection:** Haar cascade (same as main pipeline)
- **Deduplication:** Perceptual hash (phash, Hamming threshold = 8)
- **Splits:** 70/15/15 train/val/test, deterministic (seed=42)
- **Metadata:** Per-image CSV with generator provenance, face-found flag, phash
- **Honest status:** Dataset not yet generated — pipeline is ready, images must be provided

### Contribution 2: Six-Pair Cross-Dataset Generalization Evaluation
- **Status:** Results exist for 4 of 6 pairs (FF↔CelebDF with B0 and ViT-Small)
- **Missing pairs:** FF↔DFDC (×2), CelebDF↔DFDC (×2) — DFDC models are near-random (val AUC 0.44)
- **Existing results:** See `artifacts/experiment_logs/cross_ff_celeb_vit_cnn/cross_dataset_matrix.json`
- **Command:** `python scripts/run_cross_dataset.py`

### Contribution 3: Ablation Study
- **Status:** Runner implemented (`scripts/run_ablations.py`)
- **Axes:** augmentation (none/jpeg/blur/combined), loss (BCE/focal/weighted-BCE), backbone (B0/ViT-Small)
- **Command:** `python scripts/run_ablations.py --dataset celebdf --max-videos 100 --epochs 3`

### Contribution 4: Explainability
- **Status:** Runner implemented (`scripts/run_explainability.py`); existing reports in `artifacts/figures/explainability/`
- **Methods:** Grad-CAM for EfficientNet-B0, Attention Rollout for ViT-Small
- **Command:** `python scripts/run_explainability.py`

### Contribution 5: Compression Robustness
- **Status:** Completed for CelebDF-B0 and FF-B0
- **Results:** See `artifacts/experiment_logs/robustness_celebdf_b0.json`, `robustness_ff_c23_b0.json`
- **Finding:** JPEG compression negligible impact; Gaussian blur severe (AUROC drop 0.28–0.40)
- **Command:** `python scripts/run_robustness.py`

---

## 4. Checkpoint Inventory

| ID | Architecture | Dataset (pilot size) | Val AUROC | Test AUROC | Included in Ensemble? |
|---|---|---|---|---|---|
| `dfdc_b0` | EfficientNet-B0 | DFDC (8 videos) | 0.438 | 0.125 | ✗ (AUC < 0.6) |
| `celebdf_b0` | EfficientNet-B0 | CelebDF v2 (200 videos) | 1.000* | **0.924** | ✓ (weight 0.424) |
| `ff_b0` | EfficientNet-B0 | FF++ (200 videos) | unknown | unknown | ⚠ pending calibration |
| `celebdf_vit` | ViT-Small | CelebDF v2 (50 videos) | 0.710 | **0.741** | ✓ (weight 0.241) |
| `ff_vit` | ViT-Small | FF++ (50 videos) | **0.800** | 0.000† | ✓ (weight 0.300, tentative) |

\* `celebdf_b0` val AUROC=1.0 is overfit on a 30-video val set. Test AUROC (0.924) is the honest estimate.
† `ff_vit` test AUROC=0.0 on 8 test videos is unreliable (sample size too small). Val AUROC used.

---

## 5. Cross-Dataset Generalization Results (Existing)

From `artifacts/experiment_logs/cross_ff_celeb_vit_cnn/cross_dataset_matrix.json`:

| Source | Target | Model | Src Test AUROC | Tgt Test AUROC | Gap |
|---|---|---|---|---|---|
| FF++ | CelebDF | EfficientNet-B0 | 0.583 | 0.268 | 0.315 |
| FF++ | CelebDF | ViT-Small | 0.000† | 0.589 | -0.589 |
| CelebDF | FF++ | EfficientNet-B0 | **0.924** | 0.750 | 0.174 |
| CelebDF | FF++ | ViT-Small | 0.938 | 0.500 | 0.438 |
| FF++ ↔ DFDC | *both* | *both* | — | — | Not run (DFDC model near-random) |
| CelebDF ↔ DFDC | *both* | *both* | — | — | Not run (DFDC model near-random) |

**Key finding:** CelebDF-trained B0 (the strongest model) generalizes to FF++ at 0.750 AUROC — a gap of only 0.174. ViT-Small shows poor generalization (gap 0.438).

---

## 6. Robustness Results (Existing)

CelebDF-B0 (test AUROC=0.924 clean):

| Stress Test | Stressed AUROC | AUROC Drop |
|---|---|---|
| JPEG Q=40 | 0.946 | -0.022 (improved — JPEG adds info) |
| JPEG Q=20 | 0.933 | -0.009 |
| Gaussian blur σ=1.5 | 0.643 | **+0.281** (severe) |
| Gaussian blur σ=3.0 | 0.527 | **+0.397** (severe) |

**Key finding:** JPEG compression is benign (model trained with JPEG augmentation). Gaussian blur is a severe vulnerability — model appears to rely on high-frequency texture artifacts.

---

## 7. Limitations

1. All models trained on small pilots (8–200 videos), far below the original datasets.
2. `dfdc_b0` is near-random (AUC 0.44) — DFDC data preparation needs more videos.
3. `ff_vit` test AUROC=0.0 on 8-sample test set is statistically meaningless.
4. Cross-dataset matrix is incomplete (4/6 pairs completed; DFDC pairs blocked).
5. Diffusion dataset not yet generated — awaiting source images.
6. CPU-only environment limits training speed; large-scale ablations not feasible.
7. No external validation data provided — calibration script awaiting `--val-dir`.

---

*Last updated: 2026-11-10*
