"""Plane-of-array irradiance conversion for operational PV forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def compute_poa_quantiles(
    df: pd.DataFrame,
    surface_tilt: float = 35.0,
    surface_azimuth: float = 180.0,
    albedo: float = 0.2,
) -> pd.DataFrame:
    """Compute POA P10/P50/P90 using Perez when possible and isotropic fallback otherwise."""
    out = df.copy()
    times = pd.DatetimeIndex(pd.to_datetime(out["target_valid_time"], utc=True, errors="coerce"))
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    solar_zenith = pd.to_numeric(out["target_solar_zenith"], errors="coerce")
    solar_azimuth = pd.to_numeric(out["target_solar_azimuth"], errors="coerce")

    for label in ["P10", "P50", "P90"]:
        ghi = pd.to_numeric(out[_ghi_col(label)], errors="coerce").fillna(0.0).clip(lower=0.0)
        dni = pd.to_numeric(out[f"DNI_{label}_estimated"], errors="coerce").fillna(0.0).clip(lower=0.0)
        dhi = pd.to_numeric(out[f"DHI_{label}_estimated"], errors="coerce").fillna(0.0).clip(lower=0.0)
        try:
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=surface_tilt,
                surface_azimuth=surface_azimuth,
                solar_zenith=solar_zenith,
                solar_azimuth=solar_azimuth,
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                dni_extra=dni_extra,
                albedo=albedo,
                model="perez",
            )["poa_global"]
            method = "perez"
        except Exception:
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=surface_tilt,
                surface_azimuth=surface_azimuth,
                solar_zenith=solar_zenith,
                solar_azimuth=solar_azimuth,
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                albedo=albedo,
                model="isotropic",
            )["poa_global"]
            method = "isotropic"
        out[f"POA_{label}"] = pd.Series(poa, index=out.index).fillna(0.0).clip(lower=0.0)

    night = pd.to_numeric(out["target_solar_elevation"], errors="coerce").fillna(-90.0) <= 0.0
    out.loc[night, ["POA_P10", "POA_P50", "POA_P90"]] = 0.0
    ordered = np.sort(out[["POA_P10", "POA_P50", "POA_P90"]].to_numpy(dtype=float), axis=1)
    out[["POA_P10", "POA_P50", "POA_P90"]] = ordered
    out["poa_transposition_model"] = method
    return out


def _ghi_col(label: str) -> str:
    """Return the calibrated GHI column for a quantile label."""
    if label == "P10":
        return "GHI_P10_calibrated"
    if label == "P90":
        return "GHI_P90_calibrated"
    return "GHI_P50"


def main() -> None:
    """Print the current role of this executable module."""
    print("Use src/operations/decision_engine.py to compute operational POA forecasts.")


if __name__ == "__main__":
    main()
