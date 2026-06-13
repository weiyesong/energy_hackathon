"""Download Open-Meteo weather and radiation data for Munich."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 60

OPENMETEO_HOURLY_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "snow_depth",
    "is_day",
]


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


def download_openmeteo_historical(config: dict[str, Any]) -> pd.DataFrame:
    """Download hourly historical Open-Meteo data for the configured Munich location."""
    location = config["location"]
    data_period = config["data_period"]
    start_date = str(data_period["train_start"])
    end_date = str(data_period["validation_end"])
    request_start_date = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).date().isoformat()

    print(f"Downloading Open-Meteo historical data: {start_date} to {end_date}")
    params = _base_params(location) | {
        "start_date": request_start_date,
        "end_date": end_date,
        "hourly": ",".join(OPENMETEO_HOURLY_VARIABLES),
    }
    historical = _download_with_variable_fallback(HISTORICAL_URL, params, OPENMETEO_HOURLY_VARIABLES, location["timezone"])
    filtered = _filter_local_date_range(historical, start_date, end_date, location["timezone"])
    print(f"Kept {len(filtered)} historical rows on Europe/Berlin local dates {start_date} to {end_date}")
    return filtered


def download_openmeteo_forecast(config: dict[str, Any]) -> pd.DataFrame:
    """Download hourly Open-Meteo forecast data from now to at least 24 hours ahead."""
    location = config["location"]
    target_timezone = location["timezone"]

    print("Downloading Open-Meteo forecast data: current time to at least 24 hours ahead")
    params = _base_params(location) | {
        "forecast_days": 3,
        "hourly": ",".join(OPENMETEO_HOURLY_VARIABLES),
    }
    forecast = _download_with_variable_fallback(FORECAST_URL, params, OPENMETEO_HOURLY_VARIABLES, target_timezone)
    if forecast.empty:
        return forecast

    now = pd.Timestamp.now(tz=target_timezone).floor("h")
    end = now + pd.Timedelta(hours=24)
    mask = (forecast["timestamp"] >= now) & (forecast["timestamp"] <= end)
    filtered = forecast.loc[mask].reset_index(drop=True)

    if filtered.empty:
        warnings.warn("Forecast response was valid, but no rows remained after current-time filtering.", stacklevel=2)
        return forecast

    print(f"Kept {len(filtered)} forecast rows from {now.isoformat()} to {end.isoformat()}")
    return filtered


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe to CSV, creating parent directories when needed."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def main() -> None:
    """Run the Open-Meteo historical and forecast download workflow."""
    parser = argparse.ArgumentParser(description="Download Open-Meteo data for the Munich solar PV forecast project.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        project_root = Path(config["_project_root"])

        historical = download_openmeteo_historical(config)
        save_dataframe(historical, project_root / "data/raw/openmeteo_historical_munich.csv")

        forecast = download_openmeteo_forecast(config)
        save_dataframe(forecast, project_root / "data/raw/openmeteo_forecast_munich.csv")

        print("Open-Meteo download complete.")
    except Exception as exc:
        print(f"Open-Meteo download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _base_params(location: dict[str, Any]) -> dict[str, Any]:
    """Build common Open-Meteo request parameters for the configured location."""
    return {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": "UTC",
    }


def _download_with_variable_fallback(
    url: str,
    params: dict[str, Any],
    variables: list[str],
    target_timezone: str,
) -> pd.DataFrame:
    """Download Open-Meteo data, retrying by variable if the full request fails."""
    try:
        payload = _request_json(url, params)
        frame = _payload_to_dataframe(payload, target_timezone)
        _warn_about_missing_variables(frame, variables)
        return frame
    except RuntimeError as exc:
        warnings.warn(f"Full Open-Meteo request failed; retrying variable by variable. Reason: {exc}", stacklevel=2)

    frames: list[pd.DataFrame] = []
    available_variables: list[str] = []
    missing_variables: list[str] = []

    for variable in variables:
        variable_params = params | {"hourly": variable}
        try:
            payload = _request_json(url, variable_params)
            frame = _payload_to_dataframe(payload, target_timezone)
            frames.append(frame)
            available_variables.append(variable)
            print(f"  available: {variable}")
        except RuntimeError as exc:
            missing_variables.append(variable)
            warnings.warn(f"Skipping unavailable Open-Meteo variable '{variable}': {exc}", stacklevel=2)

    if not frames:
        raise RuntimeError("No requested Open-Meteo variables could be downloaded.")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    print(f"Downloaded {len(available_variables)} variables; missing {len(missing_variables)} variables.")
    return merged


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send one Open-Meteo request and return the decoded JSON payload."""
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise RuntimeError(f"request error: {exc}") from exc

    if response.status_code != 200:
        message = _response_error_message(response)
        raise RuntimeError(f"HTTP {response.status_code}: {message}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("response was not valid JSON") from exc

    if payload.get("error"):
        raise RuntimeError(str(payload.get("reason", "Open-Meteo returned an error response")))

    return payload


def _response_error_message(response: requests.Response) -> str:
    """Extract a readable error message from an Open-Meteo HTTP response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    return str(payload.get("reason") or payload.get("error") or payload)


def _payload_to_dataframe(payload: dict[str, Any], target_timezone: str) -> pd.DataFrame:
    """Convert an Open-Meteo hourly JSON payload into a timezone-aware dataframe."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("response did not contain hourly time-series data")

    frame = pd.DataFrame(hourly)
    frame = frame.rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(target_timezone)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def _warn_about_missing_variables(frame: pd.DataFrame, variables: list[str]) -> None:
    """Warn if requested Open-Meteo variables are absent from a returned dataframe."""
    missing = [variable for variable in variables if variable not in frame.columns]
    if missing:
        warnings.warn(f"Open-Meteo response did not include variables: {', '.join(missing)}", stacklevel=2)


def _filter_local_date_range(df: pd.DataFrame, start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    """Filter a timestamped dataframe to an inclusive local-date range."""
    start = pd.Timestamp(start_date, tz=timezone)
    end_exclusive = pd.Timestamp(end_date, tz=timezone) + pd.Timedelta(days=1)
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end_exclusive)
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    main()
