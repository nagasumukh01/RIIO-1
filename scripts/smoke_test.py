#!/usr/bin/env python3
"""
Stage-1 smoke test using SYNTHETIC data only.
Metrics are labelled "SYNTHETIC – NOT A RESEARCH RESULT".
Uses a tiny model to avoid OOM on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import SyntheticDeepfakeDataset, collate_synthetic
from src.models.factory import VideoDeepfakeClassifier, count_parameters
from src.training.trainer import Trainer
from src.utils.config import get_training_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("smoke_test")


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    logger.info("=" * 60)
    logger.info("STAGE-1 SMOKE TEST — SYNTHETIC DATA ONLY")
    logger.info("All metrics below are SYNTHETIC – NOT A RESEARCH RESULT")
    logger.info("=" * 60)

    train_ds = SyntheticDeepfakeDataset(num_videos=12, frames_per_video=4, image_size=64, split="train")
    val_ds = SyntheticDeepfakeDataset(num_videos=6, frames_per_video=4, image_size=64, split="val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_synthetic)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_synthetic)

    model = VideoDeepfakeClassifier(TinyBackbone(), backbone_type="cnn")
    logger.info("Parameters: %s", f"{count_parameters(model):,}")

    cfg = get_training_config()
    cfg["training"]["epochs"] = 2
    cfg["training"]["batch_size"] = 4
    cfg["training"]["early_stopping_patience"] = 5
    cfg["training"]["mixed_precision"] = False

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        experiment_id="smoke_synthetic_stage1",
        is_synthetic=True,
    )
    summary = trainer.fit()
    assert "best_val_auroc" in summary
    assert summary["is_synthetic"] is True
    assert "SYNTHETIC" in summary["label"]
    logger.info("SMOKE TEST PASSED (synthetic). best_val_auroc=%.4f", summary["best_val_auroc"])
    logger.info("Label: %s", summary["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
