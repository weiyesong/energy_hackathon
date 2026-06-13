"""Physical irradiance constraints for measured and modeled radiation columns."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_IRRADIANCE_COLUMNS = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "ghi",
    "dni",
    "dhi",
    "clear_sky_ghi",
    "clear_sky_dni",
    "clear_sky_dhi",
]


def apply_nighttime_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Set irradiance columns to zero where solar elevation is at or below the horizon."""
    if "solar_elevation" not in df.columns:
        raise KeyError("apply_nighttime_zero requires a 'solar_elevation' column.")

    constrained = df.copy()
    columns = [column for column in DEFAULT_IRRADIANCE_COLUMNS if column in constrained.columns]
    night_mask = pd.to_numeric(constrained["solar_elevation"], errors="coerce") <= 0.0
    if columns:
        constrained.loc[night_mask, columns] = 0.0
    return constrained


def clip_negative_irradiance(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Clip selected irradiance columns so radiation values cannot be negative."""
    clipped = df.copy()
    for column in columns:
        if column in clipped.columns:
            clipped[column] = pd.to_numeric(clipped[column], errors="coerce").clip(lower=0.0)
    return clipped


def constrain_ghi_to_clear_sky(
    df: pd.DataFrame,
    ghi_col: str,
    ghi_clear_col: str,
    max_ratio: float = 1.2,
) -> pd.DataFrame:
    """Limit GHI to a configurable multiple of clear-sky GHI."""
    constrained = df.copy()
    if ghi_col not in constrained.columns:
        raise KeyError(f"GHI column not found: {ghi_col}")
    if ghi_clear_col not in constrained.columns:
        raise KeyError(f"Clear-sky GHI column not found: {ghi_clear_col}")

    ghi = pd.to_numeric(constrained[ghi_col], errors="coerce")
    clear = pd.to_numeric(constrained[ghi_clear_col], errors="coerce").clip(lower=0.0)
    upper_bound = max(float(max_ratio), 0.0) * clear
    constrained[ghi_col] = np.minimum(ghi, upper_bound)
    return constrained


def main() -> None:
    """Print the current role of this executable module."""
    print("Irradiance constraint utilities are importable; use clear_sky.py for file enrichment.")


if __name__ == "__main__":
    main()
