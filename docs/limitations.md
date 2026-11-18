# Limitations

1. **Dataset access** — Full FF++, Celeb-DF, and DFDC are large and license-restricted. Users must obtain subsets themselves. The code provides clear paths and errors but cannot ship the data.

2. **Compute** — Full 6-pair matrix + ablations with ViT-Base is GPU-intensive. Small-subset runs are supported via `--max-videos`.

3. **Face detector quality** — Haar cascade is a weak fallback. RetinaFace/MTCNN give better crops when installed.

4. **Temporal modeling** — Current models aggregate independent frame logits. True video transformers / temporal attention are future extensions.

5. **Diffusion set** — Interface only; no public standardized diffusion deepfake benchmark is bundled.

6. **Explainability** — Grad-CAM and Attention Rollout are first-class but faithfulness is a simple perturbation check, not a complete XAI suite.

7. **No claim of superiority** — We do not claim accuracy improvement over Petmezas (2026) until a fair, documented comparison is completed on matching protocols.
