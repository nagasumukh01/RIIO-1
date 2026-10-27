# Research Gaps & Explicit Differences from Petmezas (2026)

Primary reference: Georgios Petmezas, “Cross-dataset video deepfake detection using Transformer and CNN architectures”, Machine Vision and Applications, 2026.  
DOI: https://doi.org/10.1007/s00138-026-01809-w

We reproduce the **spirit** of standardized preprocessing, cross-dataset evaluation, and temporal context where practical. We do **not** claim novelty for Vision Transformers, cross-dataset protocols, or temporal modeling themselves.

## Explicit Differences

### Datasets
| Aspect | Petmezas 2026 | RobustDeep |
|--------|---------------|------------|
| Training / Test sets | FF++, Celeb-DF, DFD, ReenactFaces | FaceForensics++ (FF++), Celeb-DF, DFDC + separate diffusion-generated held-out set |
| Diffusion / modern generative | Not covered | Dedicated evaluation interface + documentation |

### Cross-dataset Protocol
| Aspect | Petmezas 2026 | RobustDeep |
|--------|---------------|------------|
| Protocol | Train on FF++ + Celeb-DF combined; test on DFD + ReenactFaces (optional limited fine-tuning) | Full 6-pair **directional** matrix with **ZERO** target fine-tuning and **ZERO** threshold / hyper-parameter tuning on the target |
| Pairs | Combined sets | FF++→Celeb-DF, FF++→DFDC, Celeb-DF→FF++, Celeb-DF→DFDC, DFDC→FF++, DFDC→Celeb-DF |

### Ablation Study
| Aspect | Petmezas 2026 | RobustDeep |
|--------|---------------|------------|
| Focus | Primarily fine-tuning % and clip length | Controlled ablations of (a) augmentation strategy, (b) loss function, (c) backbone capacity |

### Explainability
| Aspect | Petmezas 2026 | RobustDeep |
|--------|---------------|------------|
| Coverage | None (explicitly listed as open gap) | Grad-CAM (CNNs) + Attention Rollout (ViTs) as first-class pipeline; evaluated on correct/incorrect predictions from seen and unseen domains; simple perturbation-based faithfulness check |

### Robustness
| Aspect | Petmezas 2026 | RobustDeep |
|--------|---------------|------------|
| Coverage | Compression / blur mentioned only as future work | Compression-artifact and blur augmentation during training as controlled experiment + separate evaluation-time stress tests |

## Hypotheses Under Investigation (Gaps A–G)

- **Gap A** — Broader cross-dataset evaluation (6-pair matrix, no target tuning)
- **Gap B** — Diffusion-generated manipulation evaluation (separate held-out set)
- **Gap C** — Controlled augmentation ablation (JPEG, blur, resize-reencode, combined)
- **Gap D** — Loss-function ablation (BCE vs Focal vs class-weighted BCE)
- **Gap E** — Backbone-capacity ablation (small vs larger EfficientNet / ViT)
- **Gap F** — Explainability under distribution shift
- **Gap G** — Full reproducibility and failure analysis

Primary scientific objective = measurable improvement in **generalization and robustness**, not merely highest within-dataset accuracy.
