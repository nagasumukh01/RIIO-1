"""
Configuration management for RobustDeep.
Loads and validates YAML configs with Pydantic models.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    face_size: int = 224
    frames_per_video: int = 16
    sampling_strategy: str = "uniform"
    face_detector: str = "retinaface"
    face_confidence_threshold: float = 0.9
    face_margin: float = 0.3
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    cache_dir: str = "artifacts/cache"
    use_cache: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> 'DataConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "EfficientNet-B0"
    type: str = "cnn"
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    pretrained_source: str = "imagenet"
    num_classes: int = 2
    dropout: float = 0.3
    input_size: int = 224
    patch_size: Optional[int] = None
    cnn_backbone: Optional[str] = None
    vit_backbone: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    optimizer: str = "adamw"
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    momentum: float = 0.9
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 0.000001
    epochs: int = 30
    batch_size: int = 16
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 7
    early_stopping_monitor: str = "val_auroc"
    early_stopping_mode: str = "max"
    early_stopping_min_delta: float = 0.001
    checkpoint_dir: str = "artifacts/checkpoints"
    save_best_only: bool = True
    save_every_n_epochs: int = 5
    loss_function: str = "cross_entropy"
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    class_weights: Any = "auto"
    balance_strategy: str = "weighted_sampler"
    log_interval: int = 10
    val_interval: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        # Flatten early_stopping sub-dict
        flat = {}
        for k, v in d.items():
            if k == 'early_stopping' and isinstance(v, dict):
                for sk, sv in v.items():
                    flat[f'early_stopping_{sk}'] = sv
            else:
                flat[k] = v
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    db_path: str = "artifacts/experiment_logs/experiments.db"
    results_dir: str = "artifacts/experiment_logs"
    figures_dir: str = "artifacts/figures"

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_all_configs(config_dir: Optional[str | Path] = None) -> dict:
    """Load all configuration files and return parsed configs."""
    if config_dir is None:
        config_dir = PROJECT_ROOT / "configs"
    config_dir = Path(config_dir)

    configs = {}

    data_path = config_dir / "data.yaml"
    if data_path.exists():
        raw = load_yaml(data_path)
        configs['data'] = DataConfig.from_dict(raw.get('data', {}))
        configs['datasets'] = raw.get('datasets', {})
        configs['augmentation'] = raw.get('augmentation', {})

    models_path = config_dir / "models.yaml"
    if models_path.exists():
        raw = load_yaml(models_path)
        configs['models'] = {}
        for name, mcfg in raw.get('models', {}).items():
            configs['models'][name] = ModelConfig.from_dict(mcfg)

    training_path = config_dir / "training.yaml"
    if training_path.exists():
        raw = load_yaml(training_path)
        configs['training'] = TrainingConfig.from_dict(raw.get('training', {}))

    experiments_path = config_dir / "experiments.yaml"
    if experiments_path.exists():
        raw = load_yaml(experiments_path)
        exp_data = raw.get('experiments', {})
        configs['experiments'] = ExperimentConfig.from_dict(exp_data)
        configs['cross_dataset_pairs'] = exp_data.get('cross_dataset_pairs', [])
        configs['augmentation_ablation'] = exp_data.get('augmentation_ablation', {})
        configs['loss_ablation'] = exp_data.get('loss_ablation', {})
        configs['backbone_ablation'] = exp_data.get('backbone_ablation', {})
        configs['robustness_tests'] = exp_data.get('robustness_tests', [])

    return configs


def get_device(device_str: str = "auto") -> str:
    """Determine the compute device."""
    try:
        import torch
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
    return device_str
