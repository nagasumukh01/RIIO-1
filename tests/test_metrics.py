"""Unit tests for metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics, generalization_gap


def test_compute_metrics_basic():
    logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    m = compute_metrics(logits, labels)
    assert m["auroc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_generalization_gap():
    assert generalization_gap(0.95, 0.70) == pytest.approx(0.25)
    assert generalization_gap(0.80, 0.85) == pytest.approx(-0.05)
