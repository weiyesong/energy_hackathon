"""Split conformal calibration for probabilistic GHI intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_split_conformal_interval(
    validation: pd.DataFrame,
    lower_col: str = "GHI_P10_raw",
    upper_col: str = "GHI_P90_raw",
    target_col: str = "GHI_target",
    target_coverage: float = 0.80,
) -> float:
    """Fit split conformal interval expansion using validation data only."""
    required = [lower_col, upper_col, target_col]
    missing = [column for column in required if column not in validation.columns]
    if missing:
        raise KeyError(f"Missing conformal calibration columns: {', '.join(missing)}")

    y = pd.to_numeric(validation[target_col], errors="coerce")
    lower = pd.to_numeric(validation[lower_col], errors="coerce")
    upper = pd.to_numeric(validation[upper_col], errors="coerce")
    mask = y.notna() & lower.notna() & upper.notna()
    if not mask.any():
        return 0.0

    nonconformity = np.maximum.reduce(
        [
            (lower[mask] - y[mask]).to_numpy(dtype=float),
            (y[mask] - upper[mask]).to_numpy(dtype=float),
            np.zeros(mask.sum(), dtype=float),
        ]
    )
    return float(np.quantile(nonconformity, target_coverage))


def apply_conformal_interval(
    df: pd.DataFrame,
    expansion: float,
    lower_col: str = "GHI_P10_raw",
    upper_col: str = "GHI_P90_raw",
) -> pd.DataFrame:
    """Apply a fitted conformal interval expansion to raw quantile predictions."""
    calibrated = df.copy()
    calibrated["GHI_P10_calibrated"] = pd.to_numeric(calibrated[lower_col], errors="coerce") - float(expansion)
    calibrated["GHI_P90_calibrated"] = pd.to_numeric(calibrated[upper_col], errors="coerce") + float(expansion)
    return calibrated
