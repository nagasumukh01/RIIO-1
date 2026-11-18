"""
Model registry for ensemble inference.

Each model has its own checkpoint, architecture, preprocessing pipeline,
label semantics, threshold, and fusion weight.  Nothing is shared across
models — every model is normalised, resized, and thresholded independently.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required: pip install timm") from exc

from src.utils.config import PROJECT_ROOT, load_yaml
from src.utils.logging_utils import setup_logger

logger = setup_logger("robustdeep.ensemble.registry")


# ---------------------------------------------------------------------------
# Data class for one model's full configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelConfig:
    """All metadata and runtime state for a single model in the ensemble."""

    model_id: str
    display_name: str
    architecture: str          # key in models.yaml (e.g. "efficientnet_b0")
    timm_name: str             # actual timm model name
    backbone_type: str         # "cnn" or "vit"
    dataset: str
    checkpoint: Path
    input_size: int
    interpolation: str
    color_order: str
    mean: List[float]
    std: List[float]
    num_classes: int
    output_units: int
    fake_index: int
    threshold: float
    val_auc: float
    weight: float
    enabled: bool

    # Runtime (not serialised)
    _model: Optional[nn.Module] = dataclasses.field(default=None, repr=False)
    load_error: Optional[str] = dataclasses.field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Registry — loads config and lazily loads models
# ---------------------------------------------------------------------------
class EnsembleRegistry:
    """Loads configs/ensemble.yaml and manages per-model lifecycle."""

    def __init__(self, config_path: str | Path = "configs/ensemble.yaml"):
        raw = load_yaml(config_path)
        self.ensemble_cfg = raw.get("ensemble", {})
        self.video_cfg = raw.get("video", {})
        self._models: Dict[str, ModelConfig] = {}

        for model_id, mcfg in raw.get("models", {}).items():
            ckpt_path = mcfg["checkpoint"]
            if not Path(ckpt_path).is_absolute():
                ckpt_path = PROJECT_ROOT / ckpt_path
            self._models[model_id] = ModelConfig(
                model_id=model_id,
                display_name=mcfg["display_name"],
                architecture=mcfg["architecture"],
                timm_name=mcfg["timm_name"],
                backbone_type=mcfg["backbone_type"],
                dataset=mcfg["dataset"],
                checkpoint=Path(ckpt_path),
                input_size=int(mcfg["input_size"]),
                interpolation=mcfg["interpolation"],
                color_order=mcfg["color_order"],
                mean=list(mcfg["mean"]),
                std=list(mcfg["std"]),
                num_classes=int(mcfg["num_classes"]),
                output_units=int(mcfg["output_units"]),
                fake_index=int(mcfg["fake_index"]),
                threshold=float(mcfg["threshold"]),
                val_auc=float(mcfg.get("val_auc", 0.0)),
                weight=float(mcfg.get("weight", 0.0)),
                enabled=bool(mcfg.get("enabled", True)),
            )

    # -- accessors -----------------------------------------------------------

    @property
    def model_ids(self) -> List[str]:
        return list(self._models.keys())

    def get(self, model_id: str) -> ModelConfig:
        return self._models[model_id]

    def all_configs(self) -> List[ModelConfig]:
        return list(self._models.values())

    def enabled_configs(self) -> List[ModelConfig]:
        return [m for m in self._models.values() if m.enabled]

    @property
    def ensemble_threshold(self) -> float:
        return float(self.ensemble_cfg.get("threshold", 0.5))

    @property
    def uncertain_low(self) -> float:
        return float(self.ensemble_cfg.get("uncertain_low", 0.40))

    @property
    def uncertain_high(self) -> float:
        return float(self.ensemble_cfg.get("uncertain_high", 0.60))

    @property
    def disagreement_std_threshold(self) -> float:
        return float(self.ensemble_cfg.get("disagreement_std_threshold", 0.25))

    @property
    def min_video_frames(self) -> int:
        return int(self.video_cfg.get("min_frames", 16))

    @property
    def max_video_frames(self) -> int:
        return int(self.video_cfg.get("max_frames", 32))

    @property
    def min_face_frames(self) -> int:
        return int(self.video_cfg.get("min_face_frames", 4))

    # -- model loading -------------------------------------------------------

    def load_model(self, model_id: str) -> Optional[nn.Module]:
        """Load one model.  Returns None and records error if loading fails."""
        cfg = self._models[model_id]
        if cfg._model is not None:
            return cfg._model

        if not cfg.checkpoint.is_file():
            msg = f"Checkpoint not found: {cfg.checkpoint}"
            logger.warning("SKIP %s — %s", model_id, msg)
            cfg.load_error = msg
            return None

        try:
            backbone = timm.create_model(
                cfg.timm_name, pretrained=False, num_classes=cfg.output_units,
            )
            # Wrap in VideoDeepfakeClassifier to match training wrapper
            from src.models.factory import VideoDeepfakeClassifier
            model = VideoDeepfakeClassifier(
                backbone=backbone,
                backbone_type=cfg.backbone_type,
                aggregation="mean",
            )
            ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
            # Checkpoint format: {'epoch':..., 'model_state': <state_dict with backbone. prefix>, ...}
            state = ckpt.get("model_state", ckpt)

            # Defensive: if checkpoint keys lack 'backbone.' prefix, remap them.
            # (All known checkpoints already have the prefix, but guard against old formats.)
            ckpt_keys = set(state.keys())
            has_backbone_prefix = any(k.startswith("backbone.") for k in ckpt_keys)
            if not has_backbone_prefix and len(ckpt_keys) > 0:
                state = {f"backbone.{k}": v for k, v in state.items()}
                logger.debug(
                    "Remapped %d keys: added 'backbone.' prefix for %s", len(state), model_id
                )

            # strict=False because the wrapper may have aggregation fields not in checkpoint.
            # Only the backbone weights matter — flag genuine backbone mismatches.
            incompat = model.load_state_dict(state, strict=False)
            missing_backbone = [k for k in incompat.missing_keys if k.startswith("backbone.")]
            if missing_backbone or incompat.unexpected_keys:
                logger.warning(
                    "State-dict mismatch for %s — missing backbone=%s  unexpected=%s",
                    model_id, missing_backbone, incompat.unexpected_keys,
                )
            else:
                logger.info(
                    "Loaded %s — all backbone keys matched (%d non-backbone fields skipped)",
                    model_id, len(incompat.missing_keys),
                )
            model.eval()
            cfg._model = model
            cfg.load_error = None
            return model
        except Exception as exc:
            msg = f"Failed to load: {exc}"
            logger.error("SKIP %s — %s", model_id, msg, exc_info=True)
            cfg.load_error = msg
            return None

    def load_all(self) -> Dict[str, nn.Module]:
        """Load every enabled model; skip failures."""
        loaded = {}
        for cfg in self.enabled_configs():
            m = self.load_model(cfg.model_id)
            if m is not None:
                loaded[cfg.model_id] = m
        return loaded

    # -- per-model preprocessing ---------------------------------------------

    def normalise(self, image: np.ndarray, model_id: str) -> torch.Tensor:
        """Resize + normalise an RGB uint8 HWC image for one specific model."""
        cfg = self._models[model_id]
        if cfg.color_order != "RGB":
            raise ValueError(f"Unsupported color order for {model_id}: {cfg.color_order}")

        interp_map = {
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
        }
        interp = interp_map.get(cfg.interpolation)
        if interp is None:
            raise ValueError(f"Unsupported interpolation for {model_id}: {cfg.interpolation}")

        sz = cfg.input_size
        img = cv2.resize(image, (sz, sz), interpolation=interp).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        mean = torch.tensor(cfg.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(cfg.std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def fake_probability(self, logits: torch.Tensor, model_id: str) -> float:
        """Convert raw model output to P(fake) respecting per-model config."""
        cfg = self._models[model_id]
        if cfg.output_units == 1:
            if logits.numel() != 1:
                raise ValueError(
                    f"Model {model_id} configured for 1 output but got shape {tuple(logits.shape)}"
                )
            raw = logits.reshape(-1)[0]
            if cfg.fake_index == 0:
                raw = -raw
            return float(torch.sigmoid(raw).item())
        # multi-class
        return float(
            torch.softmax(logits, dim=-1)[..., cfg.fake_index].reshape(-1)[0].item()
        )

    def is_fake(self, probability: float, model_id: str) -> bool:
        return probability >= self._models[model_id].threshold


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------
_registry: Optional[EnsembleRegistry] = None


def get_registry() -> EnsembleRegistry:
    global _registry
    if _registry is None:
        _registry = EnsembleRegistry()
    return _registry
