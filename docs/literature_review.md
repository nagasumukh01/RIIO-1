# Literature Review (Focused)

## Primary Reference

Georgios Petmezas, “Cross-dataset video deepfake detection using Transformer and CNN architectures”, *Machine Vision and Applications*, 2026.  
DOI: https://doi.org/10.1007/s00138-026-01809-w

**Key contributions of the paper (from abstract and publicly available descriptions):**
- Comparative study of CNN and Transformer architectures for video deepfake detection.
- Emphasis on cross-dataset evaluation.
- Standardized preprocessing pipeline and temporal context modeling.
- Reports that models trained on one set of datasets often degrade on others.

**Limitations / open gaps noted by the paper or by us:**
- Limited directional cross-dataset matrix.
- Limited or no dedicated explainability analysis under distribution shift.
- Compression and post-processing robustness left largely as future work.
- Diffusion-based or modern generative methods not the focus.

## Broader Context

Deepfake detection literature has evolved from early CNN frame classifiers (Xception, MesoNet, EfficientNet) to:
- Spatio-temporal models (I3D, SlowFast, Video Transformers).
- Face-centric pipelines with careful identity and video-level splitting.
- Growing recognition that **in-distribution accuracy is a poor proxy** for real-world robustness.
- Recent interest in foundation-model features, self-supervised pretraining, and explainable detection.

RobustDeep positions itself as a **controlled experimental platform** that:
1. Makes the 6-pair zero-shot cross-dataset matrix first-class.
2. Adds systematic ablations on augmentation, loss, and capacity.
3. Integrates explainability (Grad-CAM + Attention Rollout) and faithfulness checks.
4. Provides a clear path for diffusion-set evaluation.
5. Prioritizes reproducibility and honest failure analysis over leaderboard chasing.

We do **not** claim that using a Vision Transformer or performing cross-dataset evaluation is novel. The value lies in the strict protocol, the ablation design, the explainability under shift, and the fully documented, runnable research pipeline.
