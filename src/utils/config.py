"""Configuration loading utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_data_config() -> Dict[str, Any]:
    return load_yaml("configs/data.yaml")


def get_model_config() -> Dict[str, Any]:
    return load_yaml("configs/models.yaml")


def get_training_config() -> Dict[str, Any]:
    return load_yaml("configs/training.yaml")


def get_experiment_config() -> Dict[str, Any]:
    return load_yaml("configs/experiments.yaml")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
