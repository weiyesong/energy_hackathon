"""Estimate DNI and DHI quantiles from predicted GHI quantiles."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pvlib


QUANTILE_GHI_COLUMNS = {
    "P10": "GHI_P10_calibrated",
    "P50": "GHI_P50",
    "P90": "GHI_P90_calibrated",
}


def decompose_ghi_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate DNI and DHI for each GHI quantile with pvlib decomposition when possible."""
    out = df.copy()
    times = pd.to_datetime(out["target_valid_time"], utc=True, errors="coerce")
    zenith = pd.to_numeric(out["target_solar_zenith"], errors="coerce").clip(0.0, 180.0)
    cos_zenith = pd.to_numeric(out["target_cos_zenith"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    for label, ghi_col in QUANTILE_GHI_COLUMNS.items():
        ghi = pd.to_numeric(out[ghi_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        dni_col = f"DNI_{label}_estimated"
        dhi_col = f"DHI_{label}_estimated"
        try:
            erbs = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith, datetime_or_doy=times)
            dni = pd.Series(erbs["dni"], index=out.index).fillna(0.0).clip(lower=0.0)
            dhi = pd.Series(erbs["dhi"], index=out.index).fillna(0.0).clip(lower=0.0)
        except Exception:
            dni, dhi = _fallback_decomposition(ghi, cos_zenith)

        reconstructed_dhi = (ghi - dni * cos_zenith).clip(lower=0.0)
        out[dni_col] = dni.clip(lower=0.0)
        out[dhi_col] = reconstructed_dhi.where(cos_zenith > 0.0, ghi).clip(lower=0.0)
        night = pd.to_numeric(out["target_solar_elevation"], errors="coerce").fillna(-90.0) <= 0.0
        out.loc[night, [dni_col, dhi_col]] = 0.0

    out["dni_dhi_estimation_method"] = "pvlib_erbs_estimated_from_ghi"
    return out


def _fallback_decomposition(ghi: pd.Series, cos_zenith: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Fallback diffuse/direct split when pvlib decomposition is unavailable."""
    diffuse_fraction = np.where(cos_zenith > 0.2, 0.35, 0.8)
    dhi = pd.Series(ghi * diffuse_fraction, index=ghi.index).clip(lower=0.0)
    dni = ((ghi - dhi) / cos_zenith.clip(lower=0.08)).clip(lower=0.0)
    return dni, dhi


def decomposition_columns() -> Iterable[str]:
    """Return estimated decomposition column names."""
    for label in QUANTILE_GHI_COLUMNS:
        yield f"DNI_{label}_estimated"
        yield f"DHI_{label}_estimated"
