"""
Ensemble fusion: run all models, combine probabilities, produce one verdict.

Handles images (per-face) and videos (multi-frame per-face).
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.ensemble.registry import EnsembleRegistry, ModelConfig, get_registry
from src.utils.logging_utils import setup_logger

logger = setup_logger("robustdeep.ensemble.fusion")


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelResult:
    """One model's output on one face (or averaged across frames for video)."""
    model_id: str
    display_name: str
    dataset: str
    architecture: str
    probability: float          # P(fake) ∈ [0,1]
    weight: float               # fusion weight (may be 0)
    val_auc: float
    included: bool              # whether it contributes to the verdict
    excluded_reason: Optional[str] = None  # why it was excluded
    # Video-specific
    frame_probs: Optional[List[float]] = None   # per-frame probs
    frames_flagged_fake: Optional[int] = None
    total_frames: Optional[int] = None
    frame_variance: Optional[float] = None


@dataclasses.dataclass
class FaceResult:
    """Ensemble result for one detected face."""
    face_index: int
    box: Optional[Tuple[int, int, int, int]]   # (x, y, w, h)
    model_results: List[ModelResult]
    fused_score: float
    verdict: str               # REAL / FAKE / UNCERTAIN
    confidence: float          # percentage 0–100


@dataclasses.dataclass
class EnsembleVerdict:
    """Top-level verdict for a single upload (image or video)."""
    verdict: str               # REAL / FAKE / UNCERTAIN
    confidence: float          # percentage 0–100
    fused_score: float         # weighted-average P(fake)
    face_count: int
    frames_analyzed: int       # 1 for images
    per_face: List[FaceResult]
    model_details: List[ModelResult]   # from the highest-risk face
    models_loaded: int
    models_excluded: List[Dict[str, str]]   # [{id, reason}]
    models_total: int
    threshold_used: float
    disagreement_std: float
    # Video-specific
    face_frame_fraction: Optional[str] = None
    frame_variance: Optional[float] = None


# ---------------------------------------------------------------------------
# Core fusion functions
# ---------------------------------------------------------------------------

def _weighted_average(probs: List[float], weights: List[float]) -> float:
    """Weighted average of probabilities; falls back to uniform if all weights 0."""
    total_w = sum(weights)
    if total_w <= 0:
        return float(np.mean(probs)) if probs else 0.5
    return sum(p * w for p, w in zip(probs, weights)) / total_w


def _fuse_model_results(
    model_results: List[ModelResult],
    registry: EnsembleRegistry,
) -> Tuple[float, str, float, float]:
    """Fuse eligible models and calculate disagreement only among contributors."""
    included = [result for result in model_results if result.included]
    if not included:
        return 0.5, "UNCERTAIN", 0.0, 0.0

    probs = [result.probability for result in included]
    weights = [result.weight for result in included]
    fused = _weighted_average(probs, weights)
    verdict, confidence = _compute_verdict(fused, probs, registry)
    disagreement = float(np.std(probs)) if probs else 0.0
    return fused, verdict, confidence, disagreement


def _compute_verdict(
    fused_score: float,
    model_probs: List[float],
    registry: EnsembleRegistry,
) -> Tuple[str, float]:
    """
    Determine REAL / FAKE / UNCERTAIN and confidence.

    Rules:
    1. If models disagree heavily (high std), → UNCERTAIN
    2. If fused_score is in the uncertainty band, → UNCERTAIN
    3. Otherwise FAKE or REAL based on ensemble threshold.
    """
    threshold = registry.ensemble_threshold
    low = registry.uncertain_low
    high = registry.uncertain_high
    std_thresh = registry.disagreement_std_threshold

    std_val = float(np.std(model_probs)) if len(model_probs) > 1 else 0.0

    # Heavy disagreement → uncertain
    if std_val > std_thresh:
        confidence = max(0.0, (1.0 - 2.0 * std_val)) * 100.0
        return "UNCERTAIN", round(confidence, 1)

    # In the uncertainty band
    if low <= fused_score <= high:
        # Still might be classifiable if models agree
        if std_val < std_thresh * 0.5:
            # Models agree despite borderline score — use threshold
            pass
        else:
            confidence = max(0.0, (1.0 - 2.0 * abs(fused_score - 0.5))) * 100.0
            return "UNCERTAIN", round(confidence, 1)

    # Clear verdict
    if fused_score >= threshold:
        raw_conf = fused_score
    else:
        raw_conf = 1.0 - fused_score

    # Reduce confidence if there is disagreement
    if std_val > std_thresh:
        raw_conf *= max(0.5, 1.0 - std_val)

    verdict = "FAKE" if fused_score >= threshold else "REAL"
    return verdict, round(raw_conf * 100.0, 1)


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

def analyze_image_faces(
    face_crops: List[Dict[str, Any]],
    registry: EnsembleRegistry | None = None,
) -> EnsembleVerdict:
    """
    Run all models on every detected face crop in an image.

    face_crops: list of dicts with at least {"crop": np.ndarray (RGB HWC uint8),
                "box": (x,y,w,h)} — from _detect_and_verify_faces.
    """
    if registry is None:
        registry = get_registry()

    # Load all models
    loaded = registry.load_all()
    if not loaded:
        return EnsembleVerdict(
            verdict="UNCERTAIN",
            confidence=0.0,
            fused_score=0.5,
            face_count=len(face_crops),
            frames_analyzed=1,
            per_face=[],
            model_details=[],
            models_loaded=0,
            models_excluded=[
                {"id": c.model_id, "reason": c.load_error or "not loaded"}
                for c in registry.enabled_configs()
            ],
            models_total=len(registry.enabled_configs()),
            threshold_used=registry.ensemble_threshold,
            disagreement_std=0.0,
        )

    face_results: List[FaceResult] = []

    for fi, face_info in enumerate(face_crops):
        crop = face_info["crop"]
        box = face_info.get("box")
        model_results: List[ModelResult] = []

        for model_id, model in loaded.items():
            cfg = registry.get(model_id)
            try:
                tensor = registry.normalise(crop, model_id).unsqueeze(0)  # 1×C×H×W
                with torch.no_grad():
                    logits = model(tensor)
                prob = registry.fake_probability(logits, model_id)
            except Exception as exc:
                logger.error("Model %s failed on face %d: %s", model_id, fi, exc)
                prob = 0.5  # neutral on error

            included = cfg.val_auc >= 0.6 and cfg.weight > 0
            reason = None
            if cfg.val_auc < 0.6:
                reason = f"val_auc={cfg.val_auc:.3f} < 0.6 (near chance)"
            elif cfg.weight <= 0:
                reason = "weight is 0"

            model_results.append(ModelResult(
                model_id=model_id,
                display_name=cfg.display_name,
                dataset=cfg.dataset,
                architecture=cfg.architecture,
                probability=round(prob, 5),
                weight=cfg.weight,
                val_auc=cfg.val_auc,
                included=included,
                excluded_reason=reason,
            ))

        fused, verdict, confidence, _ = _fuse_model_results(model_results, registry)

        face_results.append(FaceResult(
            face_index=fi,
            box=box,
            model_results=model_results,
            fused_score=round(fused, 5),
            verdict=verdict,
            confidence=confidence,
        ))

    # Overall verdict: use the highest-risk face
    if face_results:
        highest_risk = max(face_results, key=lambda f: f.fused_score)
    else:
        highest_risk = None

    excluded_list = []
    for cfg in registry.enabled_configs():
        if cfg.load_error:
            excluded_list.append({"id": cfg.model_id, "reason": cfg.load_error})

    # Also list models excluded from fusion (loaded but low AUC)
    if highest_risk:
        for mr in highest_risk.model_results:
            if not mr.included and mr.excluded_reason:
                excluded_list.append({"id": mr.model_id, "reason": mr.excluded_reason})

    overall_score = highest_risk.fused_score if highest_risk else 0.5
    overall_verdict = highest_risk.verdict if highest_risk else "UNCERTAIN"
    overall_conf = highest_risk.confidence if highest_risk else 0.0
    _, _, _, disagree_std = (
        _fuse_model_results(highest_risk.model_results, registry)
        if highest_risk
        else (0.5, "UNCERTAIN", 0.0, 0.0)
    )

    return EnsembleVerdict(
        verdict=overall_verdict,
        confidence=overall_conf,
        fused_score=overall_score,
        face_count=len(face_crops),
        frames_analyzed=1,
        per_face=face_results,
        model_details=highest_risk.model_results if highest_risk else [],
        models_loaded=len(loaded),
        models_excluded=excluded_list,
        models_total=len(registry.enabled_configs()),
        threshold_used=registry.ensemble_threshold,
        disagreement_std=round(disagree_std, 5),
    )


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

def analyze_video_frames(
    frame_face_crops: List[Dict[str, Any]],
    registry: EnsembleRegistry | None = None,
) -> EnsembleVerdict:
    """
    Run all models on every face crop extracted from video frames.

    frame_face_crops: list of dicts, each with:
        - "frame_index": int
        - "faces": list of dicts with {"crop": np.ndarray, "box": (x,y,w,h)}
          (may be empty if no face found in that frame)
    """
    if registry is None:
        registry = get_registry()

    loaded = registry.load_all()
    if not loaded:
        return EnsembleVerdict(
            verdict="UNCERTAIN", confidence=0.0, fused_score=0.5,
            face_count=0, frames_analyzed=len(frame_face_crops),
            per_face=[], model_details=[],
            models_loaded=0,
            models_excluded=[
                {"id": c.model_id, "reason": c.load_error or "not loaded"}
                for c in registry.enabled_configs()
            ],
            models_total=len(registry.enabled_configs()),
            threshold_used=registry.ensemble_threshold,
            disagreement_std=0.0,
        )

    total_frames = len(frame_face_crops)
    frames_with_face = [f for f in frame_face_crops if f.get("faces")]
    n_face_frames = len(frames_with_face)

    # Check minimum face frames
    if n_face_frames < registry.min_face_frames:
        return EnsembleVerdict(
            verdict="UNCERTAIN",
            confidence=0.0,
            fused_score=0.5,
            face_count=0,
            frames_analyzed=total_frames,
            per_face=[],
            model_details=[],
            models_loaded=len(loaded),
            models_excluded=[],
            models_total=len(registry.enabled_configs()),
            threshold_used=registry.ensemble_threshold,
            disagreement_std=0.0,
            face_frame_fraction=f"{n_face_frames}/{total_frames}",
        )

    # For each model, collect per-frame fake probabilities (primary face per frame)
    per_model_frame_probs: Dict[str, List[float]] = {mid: [] for mid in loaded}

    for frame_info in frames_with_face:
        faces = frame_info["faces"]
        if not faces:
            continue
        primary_crop = faces[0]["crop"]  # highest quality face

        for model_id, model in loaded.items():
            try:
                tensor = registry.normalise(primary_crop, model_id).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor)
                prob = registry.fake_probability(logits, model_id)
            except Exception as exc:
                logger.error("Model %s frame error: %s", model_id, exc)
                prob = 0.5
            per_model_frame_probs[model_id].append(prob)

    # Average each model's probability across frames
    model_results: List[ModelResult] = []
    for model_id in loaded:
        cfg = registry.get(model_id)
        frame_probs = per_model_frame_probs[model_id]
        avg_prob = float(np.mean(frame_probs)) if frame_probs else 0.5
        variance = float(np.var(frame_probs)) if len(frame_probs) > 1 else 0.0
        frames_fake = sum(1 for p in frame_probs if p >= cfg.threshold)

        included = cfg.val_auc >= 0.6 and cfg.weight > 0
        reason = None
        if cfg.val_auc < 0.6:
            reason = f"val_auc={cfg.val_auc:.3f} < 0.6"
        elif cfg.weight <= 0:
            reason = "weight is 0"

        model_results.append(ModelResult(
            model_id=model_id,
            display_name=cfg.display_name,
            dataset=cfg.dataset,
            architecture=cfg.architecture,
            probability=round(avg_prob, 5),
            weight=cfg.weight,
            val_auc=cfg.val_auc,
            included=included,
            excluded_reason=reason,
            frame_probs=[round(p, 5) for p in frame_probs],
            frames_flagged_fake=frames_fake,
            total_frames=len(frame_probs),
            frame_variance=round(variance, 6),
        ))

    fused, verdict, confidence, disagree_std = _fuse_model_results(model_results, registry)

    # Frame variance (across all models combined)
    all_frame_variances = [r.frame_variance for r in model_results if r.frame_variance is not None]
    overall_frame_var = float(np.mean(all_frame_variances)) if all_frame_variances else 0.0

    excluded_list = []
    for cfg in registry.enabled_configs():
        if cfg.load_error:
            excluded_list.append({"id": cfg.model_id, "reason": cfg.load_error})
    for mr in model_results:
        if not mr.included and mr.excluded_reason:
            excluded_list.append({"id": mr.model_id, "reason": mr.excluded_reason})

    return EnsembleVerdict(
        verdict=verdict,
        confidence=confidence,
        fused_score=round(fused, 5),
        face_count=1,  # primary face tracked
        frames_analyzed=total_frames,
        per_face=[],   # not per-face for video, per-frame instead
        model_details=model_results,
        models_loaded=len(loaded),
        models_excluded=excluded_list,
        models_total=len(registry.enabled_configs()),
        threshold_used=registry.ensemble_threshold,
        disagreement_std=round(disagree_std, 5),
        face_frame_fraction=f"{n_face_frames}/{total_frames}",
        frame_variance=round(overall_frame_var, 6),
    )
