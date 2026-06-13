"""Solar geometry feature calculations using pvlib."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import pvlib


SOLAR_CONSTANT_WM2 = 1361.0


def compute_solar_geometry(
    times: Any,
    latitude: float,
    longitude: float,
    altitude: float,
    timezone: str,
) -> pd.DataFrame:
    """Compute solar position, cosine zenith, air mass, and extraterrestrial irradiance."""
    timestamps = _to_datetime_index(times, timezone)
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=timezone,
        altitude=altitude,
    )

    solar_position = location.get_solarposition(timestamps)
    solar_zenith = solar_position["zenith"].astype(float)
    solar_elevation = solar_position["elevation"].astype(float)
    solar_azimuth = solar_position["azimuth"].astype(float)
    cos_zenith = np.clip(np.cos(np.deg2rad(solar_zenith.to_numpy())), 0.0, 1.0)
    relative_air_mass = pvlib.atmosphere.get_relative_airmass(
        solar_position["apparent_zenith"].astype(float),
        model="kastenyoung1989",
    )
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)
    extraterrestrial = SOLAR_CONSTANT_WM2 * (1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "solar_zenith": solar_zenith.to_numpy(),
            "solar_elevation": solar_elevation.to_numpy(),
            "solar_azimuth": solar_azimuth.to_numpy(),
            "cos_zenith": cos_zenith,
            "relative_air_mass": np.asarray(relative_air_mass, dtype=float),
            "extraterrestrial_irradiance": extraterrestrial,
        }
    )


def main() -> None:
    """Run a tiny smoke calculation for the solar geometry module."""
    parser = argparse.ArgumentParser(description="Compute sample solar geometry for Munich.")
    parser.add_argument("--time", default="2025-06-21 12:00:00", help="Sample local timestamp to evaluate")
    args = parser.parse_args()

    result = compute_solar_geometry(
        [args.time],
        latitude=48.137,
        longitude=11.575,
        altitude=520,
        timezone="Europe/Berlin",
    )
    print(result.to_string(index=False))


def _to_datetime_index(times: Any, timezone: str) -> pd.DatetimeIndex:
    """Convert timestamp-like input into a timezone-aware DatetimeIndex."""
    try:
        parsed = pd.to_datetime(times, errors="raise")
        index = pd.DatetimeIndex(parsed)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(times, errors="coerce", utc=True)
        index = pd.DatetimeIndex(parsed)

    if index.isna().any():
        raise ValueError("One or more timestamps could not be parsed.")

    if index.tz is None:
        return index.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    return index.tz_convert(timezone)


if __name__ == "__main__":
    main()
