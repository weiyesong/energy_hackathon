"""Atmospheric transmittance baseline for irradiance forecasting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BETA_OZONE_DEFAULT = 0.003


def compute_effective_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """Compute effective cloud cover on a 0..1 scale from Open-Meteo cloud columns."""
    index = df.index
    total = _percent_column(df, "cloud_cover", default=0.0)
    layer_columns = ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]

    if not all(column in df.columns for column in layer_columns):
        return total.rename("C_eff")

    low = _percent_column(df, "cloud_cover_low", default=np.nan)
    mid = _percent_column(df, "cloud_cover_mid", default=np.nan)
    high = _percent_column(df, "cloud_cover_high", default=np.nan)
    layered = 0.55 * low + 0.30 * mid + 0.15 * high
    fallback_mask = low.isna() | mid.isna() | high.isna()
    effective = layered.where(~fallback_mask, total)
    return effective.fillna(pd.Series(0.0, index=index)).clip(0.0, 1.0).rename("C_eff")


def compute_cloud_transmittance(df: pd.DataFrame) -> pd.Series:
    """Compute cloud transmittance from effective cloud cover."""
    effective_cloud = df["C_eff"] if "C_eff" in df.columns else compute_effective_cloud_cover(df)
    transmittance = 1.0 - 0.75 * np.power(pd.to_numeric(effective_cloud, errors="coerce").fillna(0.0), 1.5)
    return transmittance.clip(0.05, 1.0).rename("T_cloud")


def compute_aerosol_transmittance(df: pd.DataFrame, beta_aod: float = 0.15) -> pd.Series:
    """Compute aerosol transmittance from AOD_550 when available, otherwise return 1."""
    if "AOD_550" not in df.columns:
        return pd.Series(1.0, index=df.index, name="T_aerosol")

    aod = pd.to_numeric(df["AOD_550"], errors="coerce").fillna(0.0).clip(lower=0.0)
    air_mass = _air_mass(df)
    transmittance = np.exp(-float(beta_aod) * aod * air_mass)
    return pd.Series(transmittance, index=df.index, name="T_aerosol").clip(0.0, 1.0)


def compute_water_transmittance(df: pd.DataFrame, beta_w: float = 0.01) -> pd.Series:
    """Compute water-vapour transmittance, using relative humidity as the MVP fallback."""
    if "total_column_water_vapour" in df.columns:
        water_vapour = pd.to_numeric(df["total_column_water_vapour"], errors="coerce").fillna(0.0).clip(lower=0.0)
        transmittance = np.exp(-float(beta_w) * water_vapour)
        return pd.Series(transmittance, index=df.index, name="T_water").clip(0.0, 1.0)

    if "relative_humidity_2m" in df.columns:
        humidity = pd.to_numeric(df["relative_humidity_2m"], errors="coerce").fillna(0.0).clip(0.0, 100.0)
        transmittance = 1.0 - 0.0015 * humidity
        return transmittance.clip(0.85, 1.0).rename("T_water")

    return pd.Series(1.0, index=df.index, name="T_water")


def compute_ozone_transmittance(df: pd.DataFrame) -> pd.Series:
    """Compute ozone transmittance when ozone is available, otherwise return 1."""
    if "ozone" not in df.columns:
        return pd.Series(1.0, index=df.index, name="T_ozone")

    ozone = pd.to_numeric(df["ozone"], errors="coerce").fillna(0.0).clip(lower=0.0)
    transmittance = np.exp(-BETA_OZONE_DEFAULT * ozone)
    return pd.Series(transmittance, index=df.index, name="T_ozone").clip(0.0, 1.0)


def compute_physical_irradiance_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute GHI, DNI, and DHI physical baselines from clear-sky irradiance and transmittance."""
    _require_columns(df, ["clear_sky_ghi", "clear_sky_dni", "cos_zenith", "solar_elevation"])

    result = df.copy()
    result["C_eff"] = compute_effective_cloud_cover(result)
    result["T_cloud"] = compute_cloud_transmittance(result)
    result["T_aerosol"] = compute_aerosol_transmittance(result)
    result["T_water"] = compute_water_transmittance(result)
    result["T_ozone"] = compute_ozone_transmittance(result)

    ghi_clear = pd.to_numeric(result["clear_sky_ghi"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dni_clear = pd.to_numeric(result["clear_sky_dni"], errors="coerce").fillna(0.0).clip(lower=0.0)
    cos_zenith = pd.to_numeric(result["cos_zenith"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    result["GHI_phys"] = (
        ghi_clear
        * result["T_cloud"]
        * result["T_aerosol"]
        * result["T_water"]
        * result["T_ozone"]
    )
    result["DNI_phys"] = (
        dni_clear
        * np.power(result["T_cloud"], 1.2)
        * result["T_aerosol"]
        * result["T_water"]
    )
    result["DHI_phys"] = result["GHI_phys"] - result["DNI_phys"] * cos_zenith

    radiation_columns = ["GHI_phys", "DNI_phys", "DHI_phys"]
    result[radiation_columns] = result[radiation_columns].clip(lower=0.0)
    night_mask = pd.to_numeric(result["solar_elevation"], errors="coerce").fillna(-90.0) <= 0.0
    result.loc[night_mask, radiation_columns] = 0.0
    return result


def main() -> None:
    """Run the atmospheric transmittance baseline CLI."""
    parser = argparse.ArgumentParser(description="Compute atmospheric transmittance physical irradiance baseline.")
    parser.add_argument(
        "--input",
        default="data/processed/openmeteo_with_solar_features.parquet",
        help="Input parquet with Open-Meteo, solar geometry, and clear-sky features.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/openmeteo_with_physical_baseline.parquet",
        help="Output parquet path for physical baseline features.",
    )
    args = parser.parse_args()

    try:
        print(f"Reading solar feature data: {args.input}")
        df = pd.read_parquet(args.input)
        enriched = compute_physical_irradiance_baseline(df)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output, index=False)
        print(f"Saved {len(enriched)} rows to {output}")
        print("Atmospheric transmittance baseline complete.")
    except Exception as exc:
        print(f"Atmospheric transmittance baseline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _percent_column(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Return a percentage column converted to a clipped 0..1 fraction."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    values = pd.to_numeric(df[column], errors="coerce") / 100.0
    return values.clip(0.0, 1.0)


def _air_mass(df: pd.DataFrame) -> pd.Series:
    """Return finite relative air mass values for transmittance calculations."""
    if "relative_air_mass" not in df.columns:
        return pd.Series(1.0, index=df.index)
    return pd.to_numeric(df["relative_air_mass"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required input columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


if __name__ == "__main__":
    main()
