"""Configuration helpers: load config.yaml + .env, resolve paths, set seeds."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load secrets from .env (if present) once, at import time.
load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the YAML config. Override the location with KRISHIMITRA_CONFIG."""
    cfg_path = Path(path or os.getenv("KRISHIMITRA_CONFIG", PROJECT_ROOT / "configs" / "config.yaml"))
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Turn a project-relative path from the config into an absolute path."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """Make runs reproducible across random, numpy and torch (if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
