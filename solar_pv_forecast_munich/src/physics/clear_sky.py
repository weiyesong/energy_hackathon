"""Clear-sky irradiance calculations and Open-Meteo solar feature enrichment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pvlib
import yaml

try:
    from .irradiance_constraints import (
        DEFAULT_IRRADIANCE_COLUMNS,
        apply_nighttime_zero,
        clip_negative_irradiance,
        constrain_ghi_to_clear_sky,
    )
    from .solar_geometry import _to_datetime_index, compute_solar_geometry
except ImportError:
    from irradiance_constraints import (
        DEFAULT_IRRADIANCE_COLUMNS,
        apply_nighttime_zero,
        clip_negative_irradiance,
        constrain_ghi_to_clear_sky,
    )
    from solar_geometry import _to_datetime_index, compute_solar_geometry


def compute_clear_sky_irradiance(
    times: Any,
    latitude: float,
    longitude: float,
    altitude: float,
    timezone: str,
) -> pd.DataFrame:
    """Compute clear-sky GHI, DNI, and DHI using the pvlib Ineichen model."""
    timestamps = _to_datetime_index(times, timezone)
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=timezone,
        altitude=altitude,
    )
    clear_sky = location.get_clearsky(timestamps, model="ineichen")
    solar_position = location.get_solarposition(timestamps)

    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "solar_elevation": solar_position["elevation"].astype(float).to_numpy(),
            "clear_sky_ghi": clear_sky["ghi"].astype(float).to_numpy(),
            "clear_sky_dni": clear_sky["dni"].astype(float).to_numpy(),
            "clear_sky_dhi": clear_sky["dhi"].astype(float).to_numpy(),
        }
    )
    result = apply_nighttime_zero(result)
    result = clip_negative_irradiance(result, ["clear_sky_ghi", "clear_sky_dni", "clear_sky_dhi"])
    return result.drop(columns=["solar_elevation"])


def enrich_openmeteo_with_solar_features(
    input_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Add solar geometry and clear-sky irradiance features to an Open-Meteo CSV file."""
    location = config["location"]
    timezone = location["timezone"]
    print(f"Reading Open-Meteo input: {input_path}")
    weather = pd.read_csv(input_path)
    if "timestamp" not in weather.columns:
        raise KeyError("Input data must contain a 'timestamp' column.")

    timestamps = _to_datetime_index(weather["timestamp"], timezone)
    weather = weather.copy()
    weather["timestamp"] = timestamps

    print("Computing solar geometry features with pvlib.")
    geometry = compute_solar_geometry(
        timestamps,
        latitude=float(location["latitude"]),
        longitude=float(location["longitude"]),
        altitude=float(location["altitude"]),
        timezone=timezone,
    )

    print("Computing clear-sky irradiance with pvlib Ineichen model.")
    clear_sky = compute_clear_sky_irradiance(
        timestamps,
        latitude=float(location["latitude"]),
        longitude=float(location["longitude"]),
        altitude=float(location["altitude"]),
        timezone=timezone,
    )

    enriched = pd.concat(
        [
            weather.reset_index(drop=True),
            geometry.drop(columns=["timestamp"]).reset_index(drop=True),
            clear_sky.drop(columns=["timestamp"]).reset_index(drop=True),
        ],
        axis=1,
    )

    irradiance_columns = [column for column in DEFAULT_IRRADIANCE_COLUMNS if column in enriched.columns]
    enriched = apply_nighttime_zero(enriched)
    enriched = clip_negative_irradiance(enriched, irradiance_columns)
    if "shortwave_radiation" in enriched.columns and "clear_sky_ghi" in enriched.columns:
        enriched = constrain_ghi_to_clear_sky(enriched, "shortwave_radiation", "clear_sky_ghi", max_ratio=1.2)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    print(f"Saved {len(enriched)} enriched rows to {output}")
    return enriched


def main() -> None:
    """Run Open-Meteo solar geometry and clear-sky enrichment from the command line."""
    parser = argparse.ArgumentParser(description="Add pvlib solar geometry and clear-sky features to Open-Meteo data.")
    parser.add_argument("--config", default="config.yaml", help="Path to project config.yaml")
    parser.add_argument("--input", required=True, help="Input Open-Meteo CSV path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    try:
        config = _load_config(args.config)
        enrich_openmeteo_with_solar_features(args.input, args.output, config)
        print("Solar feature enrichment complete.")
    except Exception as exc:
        print(f"Solar feature enrichment failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML project configuration for the enrichment CLI."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


if __name__ == "__main__":
    main()
