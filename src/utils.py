"""Utility helpers for configuration, logging, and artifact paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

RANDOM_SEED = 42


def ensure_directories() -> None:
    """Ensure expected project directories exist."""
    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure global logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def save_json(payload: Dict[str, Any], path: Path) -> None:
    """Save dictionary as JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON dictionary from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seeded_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Return reproducible NumPy random generator."""
    return np.random.default_rng(seed)
