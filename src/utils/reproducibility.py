"""
Reproducibility utilities: seed setting, deterministic flags.
"""

import os
import random
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_hardware_info() -> dict:
    """Collect hardware/environment info for experiment logging."""
    import platform
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch_version': torch.__version__ if TORCH_AVAILABLE and torch is not None else "not_installed",
        'cuda_available': bool(torch.cuda.is_available()) if TORCH_AVAILABLE and torch is not None else False,
        'cuda_version': torch.version.cuda if (TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()) else None,
        'gpu_count': torch.cuda.device_count() if (TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()) else 0,
    }
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
        )
    return info
