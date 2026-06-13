"""Persistence irradiance baselines for supervised SolarOps datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_persistence_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add naive and clear-sky-index persistence baselines with physical constraints."""
    required = ["ghi_issue", "target_GHI_clear"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required persistence columns: {', '.join(missing)}")

    out = df.copy()
    out["GHI_persistence_naive"] = pd.to_numeric(out["ghi_issue"], errors="coerce")
    issue_clear = _issue_clear_sky(out)
    k_c = out["GHI_persistence_naive"] / issue_clear.clip(lower=1.0)
    out["GHI_persistence_csi"] = k_c.clip(0.0, 1.2) * pd.to_numeric(out["target_GHI_clear"], errors="coerce").clip(lower=0.0)
    return apply_physical_constraints(out)


def apply_physical_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply non-negative and nighttime constraints to persistence predictions."""
    constrained = df.copy()
    columns = ["GHI_persistence_naive", "GHI_persistence_csi"]
    for column in columns:
        if column in constrained.columns:
            constrained[column] = pd.to_numeric(constrained[column], errors="coerce").clip(lower=0.0)
    if "target_solar_elevation" in constrained.columns:
        night = pd.to_numeric(constrained["target_solar_elevation"], errors="coerce").fillna(-90.0) <= 0.0
        constrained.loc[night, columns] = 0.0
    return constrained


def _issue_clear_sky(df: pd.DataFrame) -> pd.Series:
    """Return issue-time clear-sky GHI for clear-sky-index persistence."""
    if "issue_clear_sky_ghi" in df.columns:
        return pd.to_numeric(df["issue_clear_sky_ghi"], errors="coerce").fillna(1.0)
    if "target_GHI_clear" in df.columns:
        return pd.to_numeric(df["target_GHI_clear"], errors="coerce").fillna(1.0)
    return pd.Series(np.ones(len(df)), index=df.index)
