from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    """Configure concise application logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clip_power(values: pd.Series | np.ndarray, peak_power_mw: float, margin: float = 1.05):
    return np.clip(values, 0.0, peak_power_mw * margin)


def safe_divide(numerator, denominator, default: float = 0.0):
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    result = np.divide(
        num,
        den,
        out=np.full(num.shape, default, dtype=float),
        where=np.abs(den) > 1e-6,
    )
    return result
