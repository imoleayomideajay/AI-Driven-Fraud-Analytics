"""Utility helpers for the fraud analytics project."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib


SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"


def ensure_directories() -> None:
    """Create project output directories if they do not exist."""
    for path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Set up consistent logging format across modules."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def save_json(payload: Dict[str, Any], path: Path) -> None:
    """Save dict payload to JSON with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load json content from file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_artifact(obj: Any, path: Path) -> None:
    """Save Python object using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    """Load Python object from joblib."""
    return joblib.load(path)
