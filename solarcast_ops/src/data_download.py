from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import numpy as np
import requests

from src.config import get_paths
from src.synthetic_data import generate_synthetic_demo_data
from src.utils import utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

OPENMETEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_SATELLITE_URL = "https://satellite-api.open-meteo.com/v1/archive"
PVGIS_SERIESCALC_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
DWD_CDC_BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/"

OPENMETEO_HOURLY_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
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

OPENMETEO_MINUTELY_15_VARIABLES = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "is_day",
]

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
    "irradiance_source",
    "pv_power_source",
    "satellite_archive_available",
    "is_synthetic",
]


def _timeout(timeout_seconds: int | None = None) -> int:
    return int(timeout_seconds or 30)


def _write_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved %s rows to %s", len(df), path)
    return df


def _parse_openmeteo_time_block(payload: dict[str, Any], block_name: str) -> pd.DataFrame:
    block = payload.get(block_name)
    if not block or "time" not in block:
        return pd.DataFrame()
    df = pd.DataFrame(block)
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True, errors="coerce")
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _request_openmeteo(
    url: str,
    lat: float,
    lon: float,
    variables: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    block: str = "hourly",
    extra_params: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        block: ",".join(variables),
    }
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date
    if extra_params:
        params.update(extra_params)
    response = requests.get(url, params=params, timeout=_timeout(timeout_seconds))
    response.raise_for_status()
    df = _parse_openmeteo_time_block(response.json(), block)
    if df.empty:
        raise ValueError(f"Open-Meteo response from {url} did not contain a non-empty {block} block")
    return df


def download_openmeteo_historical(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Download hourly Open-Meteo historical weather and radiation for Munich MVP training."""
    df = _request_openmeteo(
        OPENMETEO_HISTORICAL_URL,
        lat,
        lon,
        OPENMETEO_HOURLY_VARIABLES,
        start_date=start_date,
        end_date=end_date,
        block="hourly",
    )
    return _write_csv(df, get_paths().raw_dir / "openmeteo_historical_munich.csv")


def download_openmeteo_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Download current Open-Meteo forecast, preferring native 15-minute solar variables when available."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        "forecast_days": 2,
        "minutely_15": ",".join(OPENMETEO_MINUTELY_15_VARIABLES),
        "hourly": ",".join(OPENMETEO_HOURLY_VARIABLES),
        "tilt": 35,
        "azimuth": 0,
    }
    response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=_timeout())
    response.raise_for_status()
    payload = response.json()
    minutely = _parse_openmeteo_time_block(payload, "minutely_15")
    hourly = _parse_openmeteo_time_block(payload, "hourly")
    source_resolution = "hourly"
    if not minutely.empty:
        df = minutely
        source_resolution = "minutely_15_with_hourly_context"
        if not hourly.empty:
            hourly_context = hourly.set_index("timestamp").sort_index()
            target_index = pd.DatetimeIndex(df["timestamp"])
            hourly_context = hourly_context.reindex(hourly_context.index.union(target_index)).sort_index()
            numeric_cols = hourly_context.select_dtypes(include=["number"]).columns
            hourly_context[numeric_cols] = hourly_context[numeric_cols].interpolate(method="time").ffill().bfill()
            hourly_context = hourly_context.reindex(target_index).reset_index(drop=True)
            for col in hourly_context.columns:
                if col not in df.columns or df[col].isna().all():
                    df[col] = hourly_context[col].to_numpy()
    else:
        df = hourly
    if df.empty:
        raise ValueError("Open-Meteo forecast response did not contain minutely_15 or hourly data")
    df["source_resolution"] = source_resolution
    return _write_csv(df, get_paths().raw_dir / "openmeteo_forecast_munich.csv")


def download_openmeteo_satellite_radiation(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tilt: float = 35,
    azimuth: float = 0,
    timeout_seconds: int | None = None,
) -> pd.DataFrame:
    """Download Open-Meteo satellite radiation archive data for a site patch."""
    variables = [
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "shortwave_radiation_instant",
        "direct_radiation_instant",
        "diffuse_radiation_instant",
        "direct_normal_irradiance_instant",
        "global_tilted_irradiance_instant",
        "is_day",
    ]
    df = _request_openmeteo(
        OPENMETEO_SATELLITE_URL,
        lat,
        lon,
        variables,
        start_date=start_date,
        end_date=end_date,
        block="hourly",
        extra_params={"tilt": float(tilt), "azimuth": float(azimuth), "temporal_resolution": "native"},
        timeout_seconds=timeout_seconds,
    )
    return _write_csv(df, get_paths().raw_dir / "openmeteo_satellite_radiation_munich.csv")


def _standardize_openmeteo_satellite_radiation(df: pd.DataFrame) -> pd.DataFrame:
    """Map Open-Meteo satellite archive fields to the project irradiance schema."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    out = pd.DataFrame()
    out["satellite_hour"] = data["timestamp"].dt.floor("h")

    def first_available(candidates: list[str]) -> pd.Series:
        for candidate in candidates:
            if candidate in data:
                return pd.to_numeric(data[candidate], errors="coerce")
        return pd.Series(np.nan, index=data.index)

    out["satellite_global_irradiance_wm2"] = first_available(["shortwave_radiation", "shortwave_radiation_instant"])
    out["satellite_direct_irradiance_wm2"] = first_available(["direct_radiation", "direct_radiation_instant"])
    out["satellite_diffuse_irradiance_wm2"] = first_available(["diffuse_radiation", "diffuse_radiation_instant"])
    out["satellite_dni_wm2"] = first_available(["direct_normal_irradiance", "direct_normal_irradiance_instant"])
    out["satellite_gti_wm2"] = first_available(["global_tilted_irradiance", "global_tilted_irradiance_instant"])
    if "is_day" in data:
        out["satellite_is_day"] = pd.to_numeric(data["is_day"], errors="coerce")
    return out.dropna(subset=["satellite_hour"]).drop_duplicates("satellite_hour", keep="last")


def merge_openmeteo_satellite_archive(base: pd.DataFrame, satellite: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach station-level Open-Meteo satellite irradiance to the unified PV target frame."""
    if base.empty or satellite.empty:
        return base.copy(), {"matched_rows": 0, "coverage": 0.0, "used": False}

    out = base.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["_satellite_hour"] = out["timestamp"].dt.floor("h")
    sat = _standardize_openmeteo_satellite_radiation(satellite)
    merged = out.merge(sat, left_on="_satellite_hour", right_on="satellite_hour", how="left")

    replacements = {
        "global_irradiance_wm2": "satellite_global_irradiance_wm2",
        "direct_irradiance_wm2": "satellite_direct_irradiance_wm2",
        "diffuse_irradiance_wm2": "satellite_diffuse_irradiance_wm2",
    }
    available = merged[list(replacements.values())].notna().any(axis=1)
    for target, source in replacements.items():
        merged[target] = pd.to_numeric(merged[source], errors="coerce").combine_first(pd.to_numeric(merged[target], errors="coerce"))

    merged["satellite_archive_available"] = available
    merged["irradiance_source"] = np.where(
        available,
        "Open-Meteo satellite radiation archive",
        merged.get("irradiance_source", "PVGIS/SARAH-3 satellite-derived proxy"),
    )
    merged["pv_power_source"] = merged.get("pv_power_source", "PVGIS modelled PV output")
    merged["data_source"] = np.where(
        available,
        "Open-Meteo satellite radiation archive + PVGIS modelled PV output",
        merged["data_source"],
    )

    drop_cols = [
        "_satellite_hour",
        "satellite_hour",
        "satellite_global_irradiance_wm2",
        "satellite_direct_irradiance_wm2",
        "satellite_diffuse_irradiance_wm2",
        "satellite_dni_wm2",
        "satellite_gti_wm2",
        "satellite_is_day",
    ]
    merged = merged.drop(columns=[col for col in drop_cols if col in merged])
    coverage = float(available.mean()) if len(available) else 0.0
    return merged, {"matched_rows": int(available.sum()), "coverage": coverage, "used": bool(available.any())}


def download_pvgis_hourly(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    peakpower: float = 1,
    tilt: float = 35,
    azimuth: float = 0,
    loss: float = 14,
) -> pd.DataFrame:
    """Download PVGIS hourly radiation and 1 kWp PV baseline for Munich."""
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": int(start_year),
        "endyear": int(end_year),
        "pvcalculation": 1,
        "peakpower": float(peakpower),
        "loss": float(loss),
        "angle": float(tilt),
        "aspect": float(azimuth),
        "components": 1,
        "outputformat": "json",
        "browser": 0,
    }
    response = requests.get(PVGIS_SERIESCALC_URL, params=params, timeout=_timeout())
    response.raise_for_status()
    df = _parse_pvgis_response(response.json())
    # Also expose the requested normalized kW/kWp baseline for downstream sanity checks.
    df["pv_power_kw_per_kwp"] = df["pv_power_mw"] * 1000.0 / max(float(peakpower), 1e-9)
    return _write_csv(df, get_paths().raw_dir / "pvgis_munich_hourly.csv")


def _read_dwd_station_table(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=_timeout())
    response.raise_for_status()
    rows = []
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "station_id": parts[0].zfill(5),
                    "from_date": parts[1],
                    "to_date": parts[2],
                    "altitude_m": float(parts[3]),
                    "latitude": float(parts[4]),
                    "longitude": float(parts[5]),
                    "station_name": " ".join(parts[6:]),
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


def list_dwd_solar_stations() -> pd.DataFrame:
    """List DWD hourly solar stations from the public CDC metadata table."""
    url = urljoin(DWD_CDC_BASE_URL, "hourly/solar/ST_Stundenwerte_Beschreibung_Stationen.txt")
    stations = _read_dwd_station_table(url)
    if stations.empty:
        raise RuntimeError(
            "Could not parse DWD station metadata. Please manually download the DWD station radiation zip files "
            "from the DWD CDC hourly/solar and 10_minutes/solar directories, then put them into data/manual/dwd/."
        )
    return stations


def find_nearest_dwd_station(lat: float, lon: float) -> pd.Series:
    """Find the nearest available DWD hourly solar station to a target coordinate."""
    stations = list_dwd_solar_stations().copy()
    lat_rad = pd.Series(stations["latitude"]).astype(float).map(lambda v: v * 3.141592653589793 / 180.0)
    target_lat = float(lat) * 3.141592653589793 / 180.0
    dlat = (stations["latitude"].astype(float) - float(lat)) * 111.32
    dlon = (stations["longitude"].astype(float) - float(lon)) * 111.32 * pd.Series(np.cos((lat_rad + target_lat) / 2.0))
    stations["distance_km"] = (dlat.pow(2) + dlon.pow(2)).pow(0.5)
    preferred = ["Muenchen", "München", "Flughafen", "Hohenpeissenberg", "Hohenpeißenberg", "Augsburg"]
    stations["preferred_rank"] = stations["station_name"].map(
        lambda name: min([i for i, token in enumerate(preferred) if token.lower() in str(name).lower()] or [99])
    )
    return stations.sort_values(["preferred_rank", "distance_km"]).iloc[0]


def _download_dwd_zip_listing(dataset_path: str, station_id: str, output_name: str) -> pd.DataFrame:
    base = urljoin(DWD_CDC_BASE_URL, dataset_path)
    listing = requests.get(base, timeout=_timeout())
    listing.raise_for_status()
    station_key = str(station_id).zfill(5)
    matches = [
        token.split('"')[0]
        for token in listing.text.split('href="')[1:]
        if station_key in token and token.split('"')[0].endswith(".zip")
    ]
    if not matches:
        raise RuntimeError(
            "Please manually download the DWD station radiation zip files from the DWD CDC hourly/solar "
            "and 10_minutes/solar directories, then put them into data/manual/dwd/."
        )
    out = pd.DataFrame({"station_id": station_key, "source_url": [urljoin(base, matches[-1])]})
    return _write_csv(out, get_paths().raw_dir / output_name)


def download_dwd_hourly_solar(station_id: str) -> pd.DataFrame:
    return _download_dwd_zip_listing("hourly/solar/recent/", station_id, f"dwd_hourly_solar_station_{str(station_id).zfill(5)}.csv")


def download_dwd_10min_global_radiation(station_id: str) -> pd.DataFrame:
    return _download_dwd_zip_listing(
        "10_minutes/solar/recent/",
        station_id,
        f"dwd_10min_global_station_{str(station_id).zfill(5)}.csv",
    )


def download_dwd_10min_diffuse_radiation(station_id: str) -> pd.DataFrame:
    return _download_dwd_zip_listing(
        "10_minutes/solar/recent/",
        station_id,
        f"dwd_10min_diffuse_station_{str(station_id).zfill(5)}.csv",
    )


def download_era5_single_levels(lat_range: list[float], lon_range: list[float], start_date: str, end_date: str, variables: list[str]) -> Path:
    """Download ERA5 through cdsapi when configured, otherwise raise the manual-download instruction."""
    try:
        import cdsapi  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ERA5 cannot be downloaded automatically because the Copernicus CDS API key is missing. "
            "Please create a CDS account, accept the ERA5 licence, install cdsapi, configure ~/.cdsapirc, "
            "then rerun the script. Alternatively, manually download ERA5 single-level NetCDF files and place them in data/manual/era5/."
        ) from exc
    target = get_paths().raw_dir / "era5_munich_2021_2025.nc"
    client = cdsapi.Client()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year": [str(y) for y in range(start.year, end.year + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [max(lat_range), min(lon_range), min(lat_range), max(lon_range)],
            "format": "netcdf",
        },
        str(target),
    )
    return target


def preprocess_era5_to_hourly_features() -> pd.DataFrame:
    raise RuntimeError(
        "ERA5 preprocessing requires a local NetCDF file in data/raw/era5_munich_2021_2025.nc or data/manual/era5/. "
        "Radiation accumulations must be converted from J/m² to W/m² by dividing by the accumulation seconds."
    )


def download_cams_aerosol(lat_range: list[float], lon_range: list[float], start_date: str, end_date: str) -> Path:
    raise RuntimeError(
        "CAMS cannot be downloaded automatically because ADS/CDS API credentials are missing. "
        "Please manually download CAMS aerosol and atmospheric composition data for Munich and place the NetCDF files in data/manual/cams/."
    )


def preprocess_cams_to_hourly_features() -> pd.DataFrame:
    raise RuntimeError("CAMS preprocessing requires CAMS NetCDF files under data/raw/ or data/manual/cams/.")


def download_eumetsat_met_ssi() -> Path:
    raise RuntimeError(
        "EUMETSAT data cannot be downloaded automatically without account/API access. Please manually download Meteosat Surface "
        "Solar Irradiance and cloud mask products for the Munich region, ideally at 15-minute resolution, and place them under data/manual/eumetsat/."
    )


def download_eumetsat_cloud_mask() -> Path:
    return download_eumetsat_met_ssi()


def preprocess_satellite_to_munich_patch() -> pd.DataFrame:
    raise RuntimeError("Satellite patch preprocessing requires manually downloaded EUMETSAT files under data/manual/eumetsat/.")


def download_entsoe_solar_generation(start_date: str, end_date: str, api_key: str | None = None) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError(
            "ENTSO-E cannot be downloaded automatically because the API token is missing. Please register for an ENTSO-E "
            "Transparency Platform API token, or manually export German solar generation data and place it in data/manual/entsoe/."
        )
    raise NotImplementedError("ENTSO-E API client wiring is pending; provide exported CSV under data/manual/entsoe/ for now.")


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
    out["irradiance_source"] = "PVGIS/SARAH-3 satellite-derived proxy"
    out["pv_power_source"] = "PVGIS modelled PV output"
    out["satellite_archive_available"] = False
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
    """Fetch PVGIS target data plus satellite archive irradiance, reuse cache, or fall back explicitly."""
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
        metadata_extra: dict[str, Any] = {"raw_response_path": str(raw_path)}
        if config["data"].get("use_openmeteo_satellite_archive", True):
            try:
                times = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
                if times.empty:
                    raise ValueError("PVGIS frame has no valid timestamps for satellite archive alignment")
                start_date = times.min().date().isoformat()
                end_date = times.max().date().isoformat()
                satellite = download_openmeteo_satellite_radiation(
                    float(config["site"]["latitude"]),
                    float(config["site"]["longitude"]),
                    start_date,
                    end_date,
                    tilt=float(config["site"].get("tilt_deg", 35)),
                    azimuth=float(config["site"].get("azimuth_deg", 0)),
                    timeout_seconds=int(config["data"].get("api_timeout_seconds", 30)),
                )
                df, satellite_meta = merge_openmeteo_satellite_archive(df, satellite)
                metadata_extra["openmeteo_satellite_archive"] = {
                    "source": OPENMETEO_SATELLITE_URL,
                    "start_date": start_date,
                    "end_date": end_date,
                    **satellite_meta,
                }
                LOGGER.info(
                    "Attached Open-Meteo satellite archive irradiance: %s matched rows (%.1f%% coverage)",
                    satellite_meta["matched_rows"],
                    satellite_meta["coverage"] * 100,
                )
            except Exception as sat_exc:
                LOGGER.warning("Open-Meteo satellite archive unavailable; keeping PVGIS irradiance proxy: %s", sat_exc)
                metadata_extra["openmeteo_satellite_archive"] = {
                    "source": OPENMETEO_SATELLITE_URL,
                    "used": False,
                    "fallback_reason": str(sat_exc),
                }
        source_name = (
            "Open-Meteo satellite archive + PVGIS modelled PV"
            if bool(pd.Series(df.get("satellite_archive_available", False)).fillna(False).astype(bool).any())
            else "PVGIS 5.3"
        )
        _write_metadata(metadata_path, config, source_name, params, metadata_extra)
        LOGGER.info("Using data source: PVGIS 5.3 SARAH-3")
    except Exception as exc:
        LOGGER.exception("PVGIS download failed: %s", exc)
        if processed_path.exists():
            LOGGER.info("Falling back to cached processed data: %s", processed_path)
            return pd.read_csv(processed_path)
        if not config["data"].get("allow_synthetic_fallback", True):
            raise
        df = generate_synthetic_demo_data(config)
        df["irradiance_source"] = "synthetic demo irradiance"
        df["pv_power_source"] = "synthetic demo PV output"
        df["satellite_archive_available"] = False
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
