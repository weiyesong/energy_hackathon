from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import get_paths
from src.synthetic_data import generate_synthetic_demo_data
from src.utils import utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

UNIFIED_COLUMNS = [
    "timestamp",
    "pv_power_mw",
    "global_irradiance_wm2",
    "direct_irradiance_wm2",
    "diffuse_irradiance_wm2",
    "air_temperature_c",
    "wind_speed_ms",
    "solar_elevation_deg",
    "data_source",
    "is_synthetic",
]


def _pvgis_params(config: dict[str, Any]) -> dict[str, Any]:
    site = config["site"]
    data = config["data"]
    return {
        "lat": site["latitude"],
        "lon": site["longitude"],
        "startyear": data["start_year"],
        "endyear": data["end_year"],
        "pvcalculation": 1,
        # PVGIS expects nominal PV power in kWp. Project config stores MWp.
        "peakpower": float(site["peak_power_mw"]) * 1000.0,
        "loss": site["system_loss_percent"],
        "angle": site["tilt_deg"],
        "aspect": site["azimuth_deg"],
        "components": 1,
        "outputformat": "json",
        "browser": 0,
        "raddatabase": data.get("pvgis_radiation_database", "PVGIS-SARAH3"),
    }


def _parse_pvgis_response(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("outputs", {}).get("hourly")
    if not hourly:
        raise ValueError("PVGIS response does not contain outputs.hourly")
    df = pd.DataFrame(hourly)
    if df.empty:
        raise ValueError("PVGIS hourly response is empty")

    time_col = "time" if "time" in df.columns else "timestamp"
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[time_col], format="%Y%m%d:%H%M", utc=True, errors="coerce")
    if out["timestamp"].isna().all():
        out["timestamp"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # PVGIS hourly field names can vary slightly by version/database.
    field_map = {
        "pv_power_mw": ["P"],
        "global_irradiance_wm2": ["G(i)", "G(h)", "GHI"],
        "direct_irradiance_wm2": ["Gb(i)", "Gb(n)", "B"],
        "diffuse_irradiance_wm2": ["Gd(i)", "Gd(h)", "D"],
        "air_temperature_c": ["T2m", "temp_air"],
        "wind_speed_ms": ["WS10m", "wind_speed"],
        "solar_elevation_deg": ["H_sun", "solar_elevation"],
    }
    for unified, candidates in field_map.items():
        source = next((c for c in candidates if c in df.columns), None)
        out[unified] = pd.to_numeric(df[source], errors="coerce") if source else pd.NA

    if out["global_irradiance_wm2"].isna().all() and {"Gb(i)", "Gd(i)"}.issubset(df.columns):
        reflected = pd.to_numeric(df["Gr(i)"], errors="coerce") if "Gr(i)" in df.columns else 0.0
        out["global_irradiance_wm2"] = (
            pd.to_numeric(df["Gb(i)"], errors="coerce")
            + pd.to_numeric(df["Gd(i)"], errors="coerce")
            + reflected
        )

    # PVGIS P is W for the requested system. Convert explicitly to MW.
    out["pv_power_mw"] = pd.to_numeric(out["pv_power_mw"], errors="coerce") / 1_000_000.0
    out["data_source"] = "PVGIS 5.3 SARAH-3 satellite-derived irradiance and modelled PV output"
    out["is_synthetic"] = False
    return out[UNIFIED_COLUMNS]


def _write_metadata(path: Path, config: dict[str, Any], source: str, params: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "source": source,
        "downloaded_at_utc": utc_now_iso(),
        "site": config["site"],
        "request_params": params,
        "notes": "PV output is public modelled PVGIS output, not real plant SCADA.",
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def download_or_load_data(config: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Fetch PVGIS hourly data, reuse cache, or create marked synthetic fallback."""
    paths = get_paths()
    processed_path = paths.processed_dir / "unified_hourly.csv"
    metadata_path = paths.processed_dir / "data_metadata.json"
    raw_path = paths.raw_dir / "pvgis_hourly_response.json"

    if config["data"].get("cache_enabled", True) and processed_path.exists() and not force:
        LOGGER.info("Using cached processed data: %s", processed_path)
        return pd.read_csv(processed_path)

    params = _pvgis_params(config)
    url = config["data"]["pvgis_base_url"]
    try:
        LOGGER.info("Requesting PVGIS data from %s", url)
        response = requests.get(url, params=params, timeout=int(config["data"].get("api_timeout_seconds", 30)))
        response.raise_for_status()
        payload = response.json()
        raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        df = _parse_pvgis_response(payload)
        _write_metadata(metadata_path, config, "PVGIS 5.3", params, {"raw_response_path": str(raw_path)})
        LOGGER.info("Using data source: PVGIS 5.3 SARAH-3")
    except Exception as exc:
        LOGGER.exception("PVGIS download failed: %s", exc)
        if processed_path.exists():
            LOGGER.info("Falling back to cached processed data: %s", processed_path)
            return pd.read_csv(processed_path)
        if not config["data"].get("allow_synthetic_fallback", True):
            raise
        df = generate_synthetic_demo_data(config)
        raw_path.write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")
        _write_metadata(
            metadata_path,
            config,
            "synthetic demo data",
            params,
            {"fallback_reason": str(exc), "warning": "Synthetic demo data is not real SCADA or measured satellite data."},
        )

    for column in UNIFIED_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[UNIFIED_COLUMNS]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    LOGGER.info("Saved unified hourly data to %s with %d rows", processed_path, len(df))
    return df
