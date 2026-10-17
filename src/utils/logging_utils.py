"""
Logging utilities for RobustDeep.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "robustdeep",
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    console: bool = True
) -> logging.Logger:
    """Set up a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"{name}_{timestamp}.log")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "robustdeep") -> logging.Logger:
    """Get existing logger or create a basic one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
