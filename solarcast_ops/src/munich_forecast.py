from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pvlib

from src.config import get_paths
from src.data_download import download_openmeteo_forecast
from src.utils import write_json

LOGGER = logging.getLogger(__name__)

DEFAULT_HORIZONS_MINUTES = [15, 30, 60, 180, 360, 720, 1440]
FREQ = "15min"


@dataclass(frozen=True)
class MunichForecastResult:
    forecast: pd.DataFrame
    metadata: dict[str, Any]


def _pvlib_surface_azimuth(config: dict[str, Any]) -> float:
    # Project config uses PVGIS convention: 0 means south-facing.
    return (180.0 + float(config["site"].get("azimuth_deg", 0.0))) % 360.0


def _normalize_openmeteo_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    rename = {
        "shortwave_radiation": "ghi_input_wm2",
        "direct_radiation": "direct_horizontal_input_wm2",
        "diffuse_radiation": "dhi_input_wm2",
        "direct_normal_irradiance": "dni_input_wm2",
        "global_tilted_irradiance": "gti_input_wm2",
        "temperature_2m": "temperature_2m_c",
        "relative_humidity_2m": "relative_humidity_2m_pct",
        "wind_speed_10m": "wind_speed_10m_ms",
        "wind_direction_10m": "wind_direction_10m_deg",
    }
    data = data.rename(columns={k: v for k, v in rename.items() if k in data.columns})
    if "timestamp" not in data.columns:
        raise ValueError("Forecast frame must contain a timestamp column")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for col in data.columns:
        if col != "timestamp" and col != "source_resolution":
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _resample_to_15min(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = _normalize_openmeteo_columns(frame)
    data = data.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    index = pd.date_range(start, end, freq=FREQ, tz="UTC")
    data = data.reindex(data.index.union(index)).sort_index()
    numeric_cols = [c for c in data.columns if c != "source_resolution"]
    data[numeric_cols] = data[numeric_cols].interpolate(method="time").ffill().bfill()
    out = data.reindex(index).reset_index(names="timestamp")
    if "source_resolution" in data:
        out["source_resolution"] = data["source_resolution"].dropna().iloc[0] if data["source_resolution"].notna().any() else "interpolated"
    return out


def _solar_context(times: pd.DatetimeIndex, config: dict[str, Any]) -> pd.DataFrame:
    site = config["site"]
    location = pvlib.location.Location(
        float(site["latitude"]),
        float(site["longitude"]),
        tz="UTC",
        altitude=float(site.get("altitude_m", 520)),
    )
    solpos = location.get_solarposition(times)
    clearsky = location.get_clearsky(times, model=config.get("irradiance_model", {}).get("clear_sky_model", "ineichen"))
    out = pd.DataFrame(
        {
            "target_time": times,
            "solar_zenith_deg": solpos["zenith"].to_numpy(),
            "solar_elevation_deg": solpos["elevation"].to_numpy(),
            "solar_azimuth_deg": solpos["azimuth"].to_numpy(),
            "clear_sky_ghi_wm2": clearsky["ghi"].to_numpy(),
            "clear_sky_dni_wm2": clearsky["dni"].to_numpy(),
            "clear_sky_dhi_wm2": clearsky["dhi"].to_numpy(),
        }
    )
    out["cos_zenith"] = np.maximum(np.cos(np.deg2rad(out["solar_zenith_deg"].astype(float))), 0.0)
    out["air_mass"] = pvlib.atmosphere.get_relative_airmass(out["solar_zenith_deg"].astype(float)).replace([np.inf, -np.inf], np.nan)
    out["extraterrestrial_irradiance_wm2"] = pvlib.irradiance.get_extra_radiation(times).to_numpy()
    return out


def _cloud_transmittance(row: pd.Series) -> float:
    low = _to_float(row.get("cloud_cover_low", row.get("cloud_cover", 0.0)), 0.0) / 100.0
    mid = _to_float(row.get("cloud_cover_mid", row.get("cloud_cover", 0.0)), 0.0) / 100.0
    high = _to_float(row.get("cloud_cover_high", row.get("cloud_cover", 0.0)), 0.0) / 100.0
    total = _to_float(row.get("cloud_cover", 0.0), 0.0) / 100.0
    c_eff = np.clip(0.55 * low + 0.30 * mid + 0.15 * high, 0.0, 1.0)
    if c_eff <= 0 and total > 0:
        c_eff = np.clip(total, 0.0, 1.0)
    return float(np.clip(1.0 - 0.75 * c_eff**1.5, 0.05, 1.0))


def _atmospheric_transmittance(row: pd.Series) -> float:
    air_mass = max(_to_float(row.get("air_mass", 1.0), 1.0), 1.0)
    aod = max(_to_float(row.get("AOD_550", row.get("aod_550", 0.05)), 0.05), 0.0)
    water = max(_to_float(row.get("total_column_water_vapour", 0.0), 0.0), 0.0)
    humidity = max(_to_float(row.get("relative_humidity_2m_pct", 0.0), 0.0), 0.0) / 100.0
    aerosol = np.exp(-0.15 * aod * air_mass)
    water_vapour = np.exp(-0.01 * water) if water > 0 else 1.0 - 0.03 * humidity
    return float(np.clip(aerosol * water_vapour, 0.65, 1.05))


def _enforce_irradiance_consistency(ghi: float, dni: float, dhi: float, cos_zenith: float) -> tuple[float, float, float]:
    ghi = max(float(ghi), 0.0)
    dni = max(float(dni), 0.0)
    dhi = max(float(dhi), 0.0)
    if cos_zenith <= 0:
        return 0.0, 0.0, 0.0
    if dhi > ghi:
        dhi = ghi
    if dni * cos_zenith + dhi > ghi * 1.08:
        dni = max((ghi - dhi) / max(cos_zenith, 1e-6), 0.0)
    reconstructed = dni * cos_zenith + dhi
    if abs(reconstructed - ghi) > max(25.0, 0.12 * max(ghi, 1.0)):
        dhi = max(ghi - dni * cos_zenith, 0.0)
    return ghi, dni, dhi


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _poa_irradiance(row: pd.Series, config: dict[str, Any], ghi: float, dni: float, dhi: float) -> float:
    site = config["site"]
    snow_depth = _to_float(row.get("snow_depth", 0.0), 0.0)
    albedo = 0.7 if snow_depth > 0.02 else float(config.get("irradiance_model", {}).get("albedo", 0.2))
    dni_extra = _to_float(row.get("extraterrestrial_irradiance_wm2", 1361.0), 1361.0)
    airmass = _to_float(row.get("air_mass", 1.0), 1.0)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(site["tilt_deg"]),
        surface_azimuth=_pvlib_surface_azimuth(config),
        solar_zenith=float(row["solar_zenith_deg"]),
        solar_azimuth=float(row["solar_azimuth_deg"]),
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        model=str(config.get("irradiance_model", {}).get("poa_model", "perez")),
    )
    return float(max(poa["poa_global"], 0.0))


def _normalized_pv_power(poa_wm2: float, temperature_c: float, wind_speed_ms: float, config: dict[str, Any]) -> float:
    _ = wind_speed_ms  # Reserved for a wind-corrected thermal model.
    module_temperature = float(temperature_c) + (max(float(poa_wm2), 0.0) / 800.0) * (45.0 - 20.0)
    dc_kw_per_kwp = max(0.0, (poa_wm2 / 1000.0) * (1.0 - 0.004 * (module_temperature - 25.0)))
    loss_fraction = float(config["site"].get("system_loss_percent", 14.0)) / 100.0
    ac_kw_per_kwp = dc_kw_per_kwp * 0.96 * max(0.0, 1.0 - max(loss_fraction - 0.04, 0.0))
    return float(np.clip(ac_kw_per_kwp, 0.0, 1.0))


def _pv_state(power_kw_per_kwp: float) -> str:
    value = float(power_kw_per_kwp)
    if value < 0.1:
        return "Very low"
    if value < 0.3:
        return "Low"
    if value < 0.6:
        return "Medium"
    if value < 0.85:
        return "High"
    return "Very high"


def _main_limiting_factor(row: pd.Series, power: float, interval_width: float) -> str:
    if _to_float(row.get("solar_elevation_deg", 0.0), 0.0) <= 5:
        return "low sun angle"
    if _to_float(row.get("snow_depth", 0.0), 0.0) > 0.02:
        return "snow"
    cloud = _to_float(row.get("cloud_cover", 0.0), 0.0)
    if cloud >= 60:
        return "cloud"
    if interval_width > max(0.12, power * 0.45):
        return "uncertainty"
    humidity = _to_float(row.get("relative_humidity_2m_pct", 0.0), 0.0)
    if humidity >= 90:
        return "water vapour"
    temp = _to_float(row.get("temperature_2m_c", 20.0), 20.0)
    if temp >= 30:
        return "temperature"
    return "aerosol" if cloud < 30 else "cloud"


def _compute_row(row: pd.Series, horizon_minutes: int, config: dict[str, Any]) -> dict[str, Any]:
    night = _to_float(row["solar_elevation_deg"], 0.0) <= 0 or _to_float(row["cos_zenith"], 0.0) <= 0
    if night:
        ghi = dni = dhi = poa = power = 0.0
    else:
        clear_ghi = _to_float(row["clear_sky_ghi_wm2"], 0.0)
        clear_dni = _to_float(row["clear_sky_dni_wm2"], 0.0)
        clear_dhi = _to_float(row["clear_sky_dhi_wm2"], 0.0)
        transmittance = _cloud_transmittance(row) * _atmospheric_transmittance(row)
        ghi = _to_float(row.get("ghi_input_wm2", np.nan))
        if not np.isfinite(ghi):
            ghi = clear_ghi * transmittance
        dni = _to_float(row.get("dni_input_wm2", np.nan))
        direct_horizontal = _to_float(row.get("direct_horizontal_input_wm2", np.nan))
        if not np.isfinite(dni) and np.isfinite(direct_horizontal):
            dni = direct_horizontal / max(_to_float(row["cos_zenith"], 0.0), 0.08)
        if not np.isfinite(dni):
            dni = clear_dni * min(1.0, transmittance * 1.05)
        dhi = _to_float(row.get("dhi_input_wm2", np.nan))
        if not np.isfinite(dhi):
            dhi = max(ghi - dni * _to_float(row["cos_zenith"], 0.0), clear_dhi * (1.0 - min(transmittance, 1.0)) * 0.35)
        clear_upper = max(clear_ghi * 1.2, 0.0)
        ghi = min(max(ghi, 0.0), clear_upper)
        ghi, dni, dhi = _enforce_irradiance_consistency(ghi, dni, dhi, _to_float(row["cos_zenith"], 0.0))
        poa = _poa_irradiance(row, config, ghi, dni, dhi)
        power = _normalized_pv_power(
            poa,
            _to_float(row.get("temperature_2m_c", 20.0), 20.0),
            _to_float(row.get("wind_speed_10m_ms", 1.0), 1.0),
            config,
        )

    horizon_factor = np.sqrt(max(horizon_minutes, 15) / 60.0)
    cloud_factor = _to_float(row.get("cloud_cover", 30.0), 30.0) / 100.0
    interval_width = 0.035 + 0.10 * horizon_factor + 0.12 * cloud_factor
    p10_power = max(0.0, power * (1.0 - interval_width))
    p90_power = min(1.0, power * (1.0 + interval_width))
    duration_hours = horizon_minutes / 60.0
    limiting_factor = _main_limiting_factor(row, power, p90_power - p10_power)

    return {
        "horizon_minutes": int(horizon_minutes),
        "issue_time": row["issue_time"],
        "target_time": row["target_time"],
        "ghi_wm2": round(ghi, 3),
        "dni_wm2": round(dni, 3),
        "dhi_wm2": round(dhi, 3),
        "gti_poa_wm2": round(poa, 3),
        "normalized_pv_power_kw_per_kwp": round(power, 5),
        "pv_energy_kwh_per_kwp": round(power * duration_hours, 5),
        "p10_kw_per_kwp": round(p10_power, 5),
        "p50_kw_per_kwp": round(power, 5),
        "p90_kw_per_kwp": round(p90_power, 5),
        "pv_generation_state": _pv_state(power),
        "main_limiting_factor": limiting_factor,
        "diagnostic_explanation": (
            f"Main limiting factor: {limiting_factor}; solar elevation "
            f"{_to_float(row.get('solar_elevation_deg', 0.0), 0.0):.1f} deg, cloud cover "
            f"{_to_float(row.get('cloud_cover', 0.0), 0.0):.0f}%."
        ),
        "solar_elevation_deg": round(_to_float(row.get("solar_elevation_deg", 0.0), 0.0), 3),
        "cloud_cover_pct": round(_to_float(row.get("cloud_cover", np.nan)), 3) if "cloud_cover" in row else np.nan,
        "temperature_2m_c": round(_to_float(row.get("temperature_2m_c", np.nan)), 3),
        "wind_speed_10m_ms": round(_to_float(row.get("wind_speed_10m_ms", np.nan)), 3),
    }


def build_munich_operational_forecast(
    config: dict[str, Any],
    issue_time: Any | None = None,
    source_frame: pd.DataFrame | None = None,
) -> MunichForecastResult:
    """Build the required 15min-24h Munich irradiance and normalized PV forecast."""
    site = config["site"]
    issue = pd.Timestamp(issue_time) if issue_time is not None else pd.Timestamp.now(tz="UTC").floor(FREQ)
    issue = issue.tz_localize("UTC") if issue.tzinfo is None else issue.tz_convert("UTC")
    horizons = [int(h) for h in config.get("forecast", {}).get("horizons_minutes", DEFAULT_HORIZONS_MINUTES)]
    end = issue + pd.Timedelta(minutes=max(horizons))

    if source_frame is None:
        source_frame = download_openmeteo_forecast(float(site["latitude"]), float(site["longitude"]))
        source = "open_meteo_forecast"
    else:
        source = "provided_frame"
    weather = _resample_to_15min(source_frame, issue, end)
    target_times = pd.DatetimeIndex([issue + pd.Timedelta(minutes=h) for h in horizons])
    rows = weather.set_index("timestamp").reindex(target_times).reset_index(names="target_time")
    solar = _solar_context(target_times, config)
    rows = pd.concat([rows.reset_index(drop=True), solar.drop(columns=["target_time"]).reset_index(drop=True)], axis=1)
    rows["issue_time"] = issue

    forecast = pd.DataFrame([_compute_row(row, h, config) for (_, row), h in zip(rows.iterrows(), horizons)])
    metadata = {
        "source": source,
        "site": {
            "name": site["name"],
            "latitude": site["latitude"],
            "longitude": site["longitude"],
            "timezone": site.get("timezone", "Europe/Berlin"),
            "altitude_m": site.get("altitude_m", 520),
            "tilt_deg": site["tilt_deg"],
            "azimuth_deg_pvgis": site.get("azimuth_deg", 0),
        },
        "issue_time_utc": str(issue),
        "horizons_minutes": horizons,
        "model": "Open-Meteo MVP plus pvlib solar geometry, clear-sky constraints, POA, NOCT PV conversion, heuristic P10/P50/P90.",
    }
    return MunichForecastResult(forecast=forecast, metadata=metadata)


def save_munich_operational_forecast(config: dict[str, Any], issue_time: Any | None = None) -> MunichForecastResult:
    result = build_munich_operational_forecast(config, issue_time=issue_time)
    paths = get_paths()
    csv_path = paths.predictions_dir / "munich_operational_forecast.csv"
    json_path = paths.predictions_dir / "munich_operational_forecast.json"
    result.forecast.to_csv(csv_path, index=False)
    write_json(
        json_path,
        {
            "metadata": result.metadata,
            "forecasts": json.loads(result.forecast.to_json(orient="records", date_format="iso")),
        },
    )
    LOGGER.info("Saved Munich operational forecast to %s and %s", csv_path, json_path)
    return result
