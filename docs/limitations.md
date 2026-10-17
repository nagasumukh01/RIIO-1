# Known Limitations

## Compute Constraints

1. **CPU-only environment**: Current development environment lacks GPU, limiting model sizes and training duration.
2. **Memory**: Large batch sizes not feasible; training uses batch_size=16 or smaller.
3. **Training time**: Full training on real datasets may take hours/days on CPU; GPU strongly recommended.

## Dataset Limitations

1. **FF++ access**: Requires academic agreement; may take days/weeks to get approved.
2. **DFDC size**: Full dataset is ~500 GB; only a documented subset is used.
3. **Diffusion set**: No established public benchmark exists yet; evaluation interface implemented but data pending.
4. **Demographic bias**: All three datasets have known demographic biases in actor/identity selection.
5. **Temporal scope**: Datasets represent manipulation technology of their era; newer methods may differ.

## Methodological Limitations

1. **Frame-level only**: No temporal modeling across frames (TimeSformer/SlowFast not implemented).
2. **Single face**: Only the largest detected face per frame is processed.
3. **Center crop fallback**: When face detection fails, center crop may include non-face content.
4. **No adversarial testing**: System is not evaluated against adversarial attacks.
5. **No identity-aware splitting** for datasets without identity metadata.

## Model Limitations

1. **Pretrained weights**: Models use ImageNet pretraining; domain gap exists between natural images and face crops.
2. **Binary classification**: Only real vs. fake; no multi-class manipulation detection.
3. **No confidence calibration**: Output probabilities are not formally calibrated.
4. **ViT data hunger**: ViTs may underperform CNNs on small datasets due to lack of inductive bias.

## Evaluation Limitations

1. **Synthetic smoke tests**: Pipeline is validated on synthetic data that does not represent real deepfake challenges.
2. **Single seed**: Most experiments use a single seed; variability not fully quantified.
3. **Threshold sensitivity**: EER-based threshold may not be optimal for all applications.
4. **Video aggregation**: Mean-score aggregation may not capture temporal patterns.

## Explainability Limitations

1. **Attribution ≠ explanation**: Heatmaps show what the model looks at, not why it makes a decision.
2. **Not forensic evidence**: Explicitly labeled as model attribution maps, not proof of manipulation.
3. **Attention rollout approximation**: May not perfectly represent ViT's decision process.
4. **Perturbation faithfulness**: Simple progressive masking may not capture all aspects of explanation quality.

## Deployment Limitations

1. **Research-grade only**: Not production-ready; no security hardening.
2. **No real-time processing**: Designed for batch evaluation, not streaming video.
3. **No user authentication**: Dashboard has no access control.
4. **Privacy**: Video upload feature stores files temporarily; no encryption at rest.

## Important Disclaimers

- This system is a **probabilistic research tool** and must never be treated as conclusive forensic evidence.
- No third-party video upload occurs without explicit user action and disclosure.
- Results should be interpreted by researchers with appropriate domain expertise.
- False positives and false negatives are expected and must be accounted for in any application.
