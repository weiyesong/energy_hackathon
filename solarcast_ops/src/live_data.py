from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.baseline_schedule import FREQ


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class LiveForecastResult:
    forecast: pd.DataFrame
    metadata: dict[str, Any]


def _now_utc_floor() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor(FREQ)


def _estimate_power_from_irradiance(df: pd.DataFrame, peak_power_mw: float, system_loss_percent: float) -> pd.Series:
    loss_factor = 1 - float(system_loss_percent) / 100.0
    power = float(peak_power_mw) * loss_factor * (df["global_irradiance_wm2"].clip(lower=0) / 1000.0)
    return power.clip(lower=0, upper=float(peak_power_mw) * 1.05)


def _parse_minutely_15(payload: dict[str, Any]) -> pd.DataFrame | None:
    block = payload.get("minutely_15")
    if not block or "time" not in block:
        return None
    data = pd.DataFrame(block)
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    out["global_irradiance_wm2"] = pd.to_numeric(data.get("shortwave_radiation"), errors="coerce")
    out["direct_irradiance_wm2"] = pd.to_numeric(data.get("direct_radiation"), errors="coerce")
    out["diffuse_irradiance_wm2"] = pd.to_numeric(data.get("diffuse_radiation"), errors="coerce")
    out["air_temperature_c"] = pd.to_numeric(data.get("temperature_2m"), errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(data.get("wind_speed_10m"), errors="coerce")
    return out.dropna(subset=["timestamp"])


def _parse_hourly(payload: dict[str, Any]) -> pd.DataFrame | None:
    block = payload.get("hourly")
    if not block or "time" not in block:
        return None
    data = pd.DataFrame(block)
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    out["global_irradiance_wm2"] = pd.to_numeric(data.get("shortwave_radiation"), errors="coerce")
    out["direct_irradiance_wm2"] = pd.to_numeric(data.get("direct_radiation"), errors="coerce")
    out["diffuse_irradiance_wm2"] = pd.to_numeric(data.get("diffuse_radiation"), errors="coerce")
    out["air_temperature_c"] = pd.to_numeric(data.get("temperature_2m"), errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(data.get("wind_speed_10m"), errors="coerce")
    out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if out.empty:
        return None
    full_index = pd.date_range(out.index.min(), out.index.max(), freq=FREQ, tz="UTC")
    out = out.reindex(full_index).interpolate(method="time").ffill().bfill()
    out.index.name = "timestamp"
    return out.reset_index()


def fetch_open_meteo_solar_forecast(config: dict[str, Any], start_time: Any | None = None) -> LiveForecastResult:
    """Fetch 0-24h 15-minute solar radiation forecast, falling back from 15-min to hourly."""
    site = config["site"]
    start = pd.Timestamp(start_time) if start_time is not None else _now_utc_floor()
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = start + pd.Timedelta(hours=24)
    base_params = {
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "timezone": "UTC",
        "forecast_days": 2,
    }
    minutely_params = {
        **base_params,
        "minutely_15": "shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m,wind_speed_10m",
    }
    hourly_params = {
        **base_params,
        "hourly": "shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m,wind_speed_10m",
    }

    errors: list[str] = []
    for label, params, parser in [
        ("open_meteo_minutely_15", minutely_params, _parse_minutely_15),
        ("open_meteo_hourly_interpolated_to_15min", hourly_params, _parse_hourly),
    ]:
        try:
            response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=int(config["data"].get("api_timeout_seconds", 30)))
            response.raise_for_status()
            parsed = parser(response.json())
            if parsed is None or parsed.empty:
                errors.append(f"{label}: empty response")
                continue
            parsed = parsed.sort_values("timestamp")
            parsed = parsed[(parsed["timestamp"] >= start) & (parsed["timestamp"] <= end)].copy()
            if parsed.empty:
                errors.append(f"{label}: no rows in requested 24h window")
                continue
            full_index = pd.date_range(start, end, freq=FREQ, tz="UTC")
            parsed = parsed.set_index("timestamp").reindex(full_index)
            for col in ["global_irradiance_wm2", "direct_irradiance_wm2", "diffuse_irradiance_wm2", "air_temperature_c", "wind_speed_ms"]:
                parsed[col] = pd.to_numeric(parsed[col], errors="coerce").interpolate(method="time").ffill().bfill()
            parsed.index.name = "timestamp"
            parsed = parsed.reset_index()
            parsed["forecast_power_mw"] = _estimate_power_from_irradiance(
                parsed,
                float(site["peak_power_mw"]),
                float(site["system_loss_percent"]),
            )
            parsed["horizon_hours"] = (parsed["timestamp"] - start).dt.total_seconds() / 3600
            parsed["data_source"] = label
            return LiveForecastResult(
                forecast=parsed,
                metadata={
                    "source": label,
                    "url": OPEN_METEO_FORECAST_URL,
                    "start_time": str(start),
                    "end_time": str(end),
                    "notes": "Open-Meteo solar radiation forecast converted to PV power with a simple capacity/loss factor.",
                },
            )
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise RuntimeError("Open-Meteo solar forecast unavailable; " + " | ".join(errors))


def forecast_from_schedule_baseline(schedule: pd.DataFrame, reason: str) -> LiveForecastResult:
    """Use schedule as a continuous forecast source when live irradiance is unavailable."""
    forecast = schedule[["timestamp", "horizon_hours", "scheduled_power_mw"]].copy()
    forecast["global_irradiance_wm2"] = np.nan
    forecast["direct_irradiance_wm2"] = np.nan
    forecast["diffuse_irradiance_wm2"] = np.nan
    forecast["air_temperature_c"] = np.nan
    forecast["wind_speed_ms"] = np.nan
    forecast["forecast_power_mw"] = forecast["scheduled_power_mw"]
    forecast["data_source"] = "computed_schedule_baseline"
    return LiveForecastResult(
        forecast=forecast,
        metadata={
            "source": "computed_schedule_baseline",
            "source_note": reason,
            "notes": "Forecast equals the selected schedule baseline.",
        },
    )


def prediction_interval_from_series(forecast: pd.Series, schedule: pd.Series, coverage_width_fraction: float = 0.12) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Simple operational interval for live mode until calibrated live residuals exist."""
    center = forecast.astype(float).clip(lower=0)
    width = np.maximum(center.abs() * coverage_width_fraction, schedule.astype(float).abs() * 0.05)
    p10 = (center - width).clip(lower=0)
    p90 = center + width
    return p10, center, p90
