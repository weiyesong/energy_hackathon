"""Download and standardize NASA POWER hourly data for configured Munich sites."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_registry import register_data_source
from src.core.schema import ensure_canonical_columns


NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
TARGET_TIMEZONE = "Europe/Berlin"
SOURCE_NAME = "nasa_power"
SOURCE_TYPE_CANONICAL = "satellite_model_derived"
SOURCE_TYPE_REGISTRY = "satellite/model-derived historical solar baseline"

NASA_POWER_PARAMETER_MAP = {
    # NASA POWER API parameter names verified against the Hourly Point API examples and parameter dictionary.
    "ALLSKY_SFC_SW_DWN": "satellite_ssi",
    "CLRSKY_SFC_SW_DWN": "satellite_clear_sky_ssi",
    "T2M": "temperature_2m",
    "RH2M": "relative_humidity_2m",
    "WS10M": "wind_speed_10m",
    "PS": "surface_pressure",
    "PRECTOTCORR": "precipitation",
}

NASA_POWER_PARAMETERS = list(NASA_POWER_PARAMETER_MAP)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML project configuration from disk."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent)
    return config


def build_nasa_power_request(site: dict[str, Any], start_date: str, end_date: str) -> dict[str, Any]:
    """Build NASA POWER Hourly Point API request parameters for one site."""
    return {
        "parameters": ",".join(NASA_POWER_PARAMETERS),
        "community": "RE",
        "longitude": float(site["longitude"]),
        "latitude": float(site["latitude"]),
        "start": _format_power_date(start_date),
        "end": _format_power_date(end_date),
        "format": "JSON",
        "time-standard": "UTC",
    }


def download_nasa_power_site(site: dict[str, Any], start_date: str, end_date: str) -> dict[str, Any]:
    """Download raw NASA POWER JSON for one configured site, retrying unavailable parameters individually."""
    params = build_nasa_power_request(site, start_date, end_date)
    print(f"Downloading NASA POWER hourly data for {site['id']} ({start_date} to {end_date})")
    try:
        return _request_json_with_retry(params)
    except RuntimeError as exc:
        warnings.warn(f"Full NASA POWER request failed for {site['id']}; retrying per parameter. Reason: {exc}", stacklevel=2)

    merged_parameters: dict[str, dict[str, Any]] = {}
    last_error: str | None = None
    for parameter in NASA_POWER_PARAMETERS:
        try:
            payload = _request_json_with_retry(params | {"parameters": parameter})
            parameter_data = payload.get("properties", {}).get("parameter", {})
            if parameter in parameter_data:
                merged_parameters[parameter] = parameter_data[parameter]
                print(f"  available: {parameter}")
            else:
                warnings.warn(f"NASA POWER response did not include requested parameter '{parameter}'.", stacklevel=2)
        except RuntimeError as exc:
            last_error = str(exc)
            warnings.warn(f"Skipping unavailable NASA POWER parameter '{parameter}': {exc}", stacklevel=2)

    if not merged_parameters:
        raise RuntimeError(f"No NASA POWER parameters could be downloaded for {site['id']}: {last_error}")

    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(site["longitude"]), float(site["latitude"])]},
        "properties": {"parameter": merged_parameters},
    }


def parse_nasa_power_response(response_json: dict[str, Any], site: dict[str, Any]) -> pd.DataFrame:
    """Parse a NASA POWER JSON response into a timestamped dataframe with NASA parameter names."""
    parameters = response_json.get("properties", {}).get("parameter", {})
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"NASA POWER response for {site['id']} did not contain parameter time series.")

    missing = [parameter for parameter in NASA_POWER_PARAMETERS if parameter not in parameters]
    if missing:
        warnings.warn(f"NASA POWER response for {site['id']} missing parameters: {', '.join(missing)}", stacklevel=2)

    frames: list[pd.DataFrame] = []
    for parameter, values in parameters.items():
        if parameter not in NASA_POWER_PARAMETER_MAP:
            continue
        series = pd.Series(values, name=parameter, dtype="float64")
        frame = series.rename_axis("power_time").reset_index()
        frames.append(frame)

    if not frames:
        raise ValueError(f"NASA POWER response for {site['id']} had no requested parameter data.")

    parsed = frames[0]
    for frame in frames[1:]:
        parsed = parsed.merge(frame, on="power_time", how="outer")

    parsed["timestamp"] = pd.to_datetime(parsed["power_time"], format="%Y%m%d%H", utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE)
    parsed = parsed.drop(columns=["power_time"]).dropna(subset=["timestamp"])
    parsed["site_id"] = site["id"]
    parsed["latitude"] = float(site["latitude"])
    parsed["longitude"] = float(site["longitude"])
    return parsed.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def standardize_nasa_power_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map NASA POWER parameter columns into the SolarOps canonical adapter schema."""
    standardized = pd.DataFrame(index=df.index)
    standardized["timestamp"] = df["timestamp"]
    standardized["site_id"] = df["site_id"]
    standardized["latitude"] = df["latitude"]
    standardized["longitude"] = df["longitude"]
    standardized["source_name"] = SOURCE_NAME
    standardized["source_type"] = SOURCE_TYPE_CANONICAL
    standardized["is_satellite_derived"] = True

    for nasa_parameter, canonical_column in NASA_POWER_PARAMETER_MAP.items():
        standardized[canonical_column] = pd.to_numeric(df[nasa_parameter], errors="coerce") if nasa_parameter in df else pd.NA

    standardized["satellite_cloud_index"] = pd.NA
    standardized["quality_flag"] = "ok"
    missing_core = [
        column
        for column in ["satellite_ssi", "satellite_clear_sky_ssi"]
        if column not in standardized or standardized[column].isna().all()
    ]
    if missing_core:
        standardized["quality_flag"] = "missing_" + "_".join(missing_core)

    if "satellite_ssi" in standardized and "satellite_clear_sky_ssi" in standardized:
        denominator = pd.to_numeric(standardized["satellite_clear_sky_ssi"], errors="coerce").fillna(0.0).clip(lower=1.0)
        clear_sky_index = pd.to_numeric(standardized["satellite_ssi"], errors="coerce") / denominator
        standardized["satellite_clear_sky_index"] = clear_sky_index.clip(0.0, 1.2)
    else:
        standardized["satellite_clear_sky_index"] = pd.NA

    return ensure_canonical_columns(standardized)


def download_all_configured_sites(config: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Download or load cached NASA POWER data for all configured Munich sites."""
    project_root = Path(config.get("_project_root", "."))
    start_date = "2023-01-01"
    end_date = "2025-12-31"
    request_start_date = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).date().isoformat()
    raw_dir = project_root / "data/raw"
    processed_path = project_root / "data/processed/nasa_power_all_sites.parquet"
    registry_path = project_root / "data/processed/data_source_registry.json"

    site_frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for site in config.get("sites", []):
        raw_path = raw_dir / f"nasa_power_{site['id']}.csv"
        try:
            if raw_path.exists() and not force and _is_valid_cached_site_file(raw_path, start_date, end_date):
                print(f"Using cached NASA POWER file: {raw_path}")
                standardized = pd.read_csv(raw_path)
                standardized["timestamp"] = pd.to_datetime(standardized["timestamp"], utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE)
            else:
                payload = download_nasa_power_site(site, request_start_date, end_date)
                parsed = parse_nasa_power_response(payload, site)
                parsed = _filter_local_date_range(parsed, start_date, end_date)
                standardized = standardize_nasa_power_columns(parsed)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                standardized.to_csv(raw_path, index=False)
                print(f"Saved site NASA POWER data: {raw_path}")
            site_frames.append(standardized)
        except Exception as exc:
            message = f"{site.get('id', 'unknown')}: {exc}"
            errors.append(message)
            warnings.warn(f"NASA POWER site failed: {message}", stacklevel=2)

    if not site_frames:
        register_data_source(
            SOURCE_NAME,
            source_type=SOURCE_TYPE_REGISTRY,
            is_satellite_derived=True,
            download_status="unavailable",
            manual_action_required=False,
            error_message="; ".join(errors) if errors else "No configured sites produced data.",
            registry_path=registry_path,
        )
        raise RuntimeError("No NASA POWER site data available.")

    combined = pd.concat(site_frames, ignore_index=True).sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(processed_path, index=False)
    print(f"Saved combined NASA POWER data: {processed_path}")

    status = "available" if not errors else "partial_available"
    register_data_source(
        SOURCE_NAME,
        source_type=SOURCE_TYPE_REGISTRY,
        is_satellite_derived=True,
        date_range={"start": start_date, "end": end_date},
        temporal_resolution="hourly",
        file_path=str(processed_path.relative_to(project_root)),
        available_columns=list(combined.columns),
        download_status=status,
        manual_action_required=False,
        error_message="; ".join(errors) if errors else None,
        registry_path=registry_path,
    )
    return combined


def main() -> None:
    """Run the NASA POWER multi-site download workflow."""
    parser = argparse.ArgumentParser(description="Download NASA POWER hourly baseline data for configured Munich sites.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Redownload even when valid cached files already exist")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        combined = download_all_configured_sites(config, force=args.force)
        print(f"NASA POWER download complete: {len(combined)} rows")
    except Exception as exc:
        print(f"NASA POWER download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _request_json_with_retry(params: dict[str, Any]) -> dict[str, Any]:
    """Request NASA POWER JSON with timeout and exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(NASA_POWER_HOURLY_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {_response_error_message(response)}")
            payload = response.json()
            if not payload or payload.get("error"):
                raise RuntimeError(str(payload.get("messages") or payload.get("reason") or "empty/error response"))
            return payload
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NASA POWER request failed after {MAX_RETRIES} attempts: {last_error}")


def _response_error_message(response: requests.Response) -> str:
    """Extract a readable error message from a NASA POWER HTTP response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    return str(payload.get("messages") or payload.get("reason") or payload)


def _format_power_date(value: str) -> str:
    """Convert an ISO-like date into NASA POWER YYYYMMDD format."""
    return pd.Timestamp(value).strftime("%Y%m%d")


def _is_valid_cached_site_file(path: Path, start_date: str, end_date: str) -> bool:
    """Return whether a cached site CSV is non-empty and contains the required timestamp field."""
    try:
        cached = pd.read_csv(path, usecols=["timestamp", "site_id"])
    except Exception:
        return False
    if cached.empty:
        return False
    timestamps = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE)
    expected_start = pd.Timestamp(start_date, tz=TARGET_TIMEZONE)
    expected_end = pd.Timestamp(end_date, tz=TARGET_TIMEZONE) + pd.Timedelta(hours=23)
    return timestamps.min() == expected_start and timestamps.max() == expected_end


def _filter_local_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter a timestamped dataframe to an inclusive Europe/Berlin local-date range."""
    start = pd.Timestamp(start_date, tz=TARGET_TIMEZONE)
    end_exclusive = pd.Timestamp(end_date, tz=TARGET_TIMEZONE) + pd.Timedelta(days=1)
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end_exclusive)
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    main()
