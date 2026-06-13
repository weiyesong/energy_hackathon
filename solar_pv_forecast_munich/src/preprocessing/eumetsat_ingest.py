"""Manual-file ingestion utilities for EUMETSAT SSI NetCDF and GRIB products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EUMETSAT_EXTENSIONS = {".nc", ".nc4", ".netcdf", ".grib", ".grb", ".grib2", ".grb2"}
DEFAULT_EUMETSAT_ALIASES = {
    "satellite_ssi": ["ssi", "surface_solar_irradiance", "surface_downwelling_shortwave_flux"],
    "satellite_cloud_index": ["cloud_index", "effective_cloud_fraction"],
    "cloud_cover": ["cloud_fraction", "cloud_cover"],
}


def find_manual_eumetsat_files(directory: str | Path) -> list[Path]:
    """Find supported manual EUMETSAT NetCDF and GRIB files in a directory."""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EUMETSAT_EXTENSIONS)


def inspect_dataset_variables(path: str | Path) -> list[str]:
    """Return variable names from a manual EUMETSAT dataset without assuming any required variable exists."""
    dataset = load_eumetsat_file(path)
    return sorted(set(dataset.data_vars) | set(dataset.coords))


def load_eumetsat_file(path: str | Path) -> Any:
    """Load a NetCDF or GRIB file with xarray when optional dependencies are installed."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required to read EUMETSAT NetCDF/GRIB files.") from exc

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".grib", ".grb", ".grib2", ".grb2"}:
        return xr.open_dataset(file_path, engine="cfgrib")
    return xr.open_dataset(file_path)


def extract_munich_sites(dataset: Any, configured_sites: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract nearest dataset values for each configured Munich site."""
    lat_name = _first_existing_name(dataset, ["lat", "latitude", "y"])
    lon_name = _first_existing_name(dataset, ["lon", "longitude", "x"])
    frames: list[pd.DataFrame] = []

    for site in configured_sites:
        selected = dataset
        if lat_name and lon_name:
            selected = dataset.sel({lat_name: float(site["latitude"]), lon_name: float(site["longitude"])}, method="nearest")
        frame = selected.to_dataframe().reset_index()
        frame["site_id"] = site["id"]
        frame["latitude"] = float(site["latitude"])
        frame["longitude"] = float(site["longitude"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def standardize_eumetsat_data(df: pd.DataFrame, aliases: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Map extracted EUMETSAT fields into the canonical satellite output columns."""
    alias_map = aliases or DEFAULT_EUMETSAT_ALIASES
    standardized = pd.DataFrame(index=df.index)
    standardized["timestamp"] = _timestamp_series(df)
    standardized["site_id"] = df["site_id"]
    standardized["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    standardized["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    standardized["source_name"] = "eumetsat_ssi"
    standardized["source_type"] = "operational_satellite"
    standardized["is_satellite_derived"] = True
    standardized["satellite_ssi"] = _mapped_series(df, alias_map.get("satellite_ssi", []))
    standardized["satellite_cloud_index"] = _mapped_series(df, alias_map.get("satellite_cloud_index", []))
    standardized["cloud_cover"] = _mapped_series(df, alias_map.get("cloud_cover", []))
    standardized["quality_flag"] = "ok"
    standardized.loc[standardized["satellite_ssi"].isna(), "quality_flag"] = "missing_satellite_ssi"
    return standardized[
        [
            "timestamp",
            "site_id",
            "latitude",
            "longitude",
            "source_name",
            "source_type",
            "is_satellite_derived",
            "satellite_ssi",
            "satellite_cloud_index",
            "cloud_cover",
            "quality_flag",
        ]
    ].dropna(subset=["timestamp", "site_id"])


def _first_existing_name(dataset: Any, candidates: list[str]) -> str | None:
    """Return the first coordinate name present in a dataset."""
    names = set(dataset.coords) | set(dataset.dims)
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _timestamp_series(df: pd.DataFrame) -> pd.Series:
    """Extract a timezone-aware timestamp series from common EUMETSAT time columns."""
    for column in ["timestamp", "time", "valid_time"]:
        if column in df.columns:
            return pd.to_datetime(df[column], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    raise KeyError("EUMETSAT data does not contain timestamp, time, or valid_time.")


def _mapped_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    """Return the first available aliased variable as numeric values, otherwise NA."""
    for name in aliases:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(pd.NA, index=df.index)
