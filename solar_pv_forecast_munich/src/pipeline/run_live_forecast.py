"""End-to-end live forecast orchestrator for SolarOps."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import pvlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_download.openmeteo_download import download_openmeteo_forecast, load_config as load_openmeteo_config, save_dataframe
from src.models.train_hybrid_lightgbm import _apply_physical_constraints
from src.operations.decision_engine import build_operational_forecast, generate_operator_actions, save_operator_actions
from src.operations.site_ranking import rank_sites, save_site_ranking
from src.physics.clear_sky import compute_clear_sky_irradiance
from src.physics.poa_model import compute_poa_quantiles
from src.physics.pv_power_model import estimate_pv_power_quantiles
from src.physics.solar_geometry import compute_solar_geometry
from src.pipeline.source_health import inspect_source_health, write_source_status

TARGET_TIMEZONE = "Europe/Berlin"
DEFAULT_LIVE_OUTPUT_DIR = Path("outputs/live")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load project configuration with project-root metadata."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_project_root"] = str(path.parent)
    return config


def run_live_forecast(
    config_path: str | Path = "config.yaml",
    output_dir: str | Path = DEFAULT_LIVE_OUTPUT_DIR,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Run live forecast orchestration and write UI-ready outputs."""
    config = load_config(config_path)
    root = Path(config["_project_root"])
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=True)

    source_status = inspect_source_health(config, root / "data/processed/data_source_registry.json", root / "outputs/models")
    write_source_status(source_status, output / "source_status.json")

    weather = _fetch_or_load_openmeteo_forecast(config_path, root)
    issue_time = _issue_time(weather)
    current_state = build_live_current_state(config, weather, source_status, issue_time)
    with (output / "current_state.json").open("w", encoding="utf-8") as f:
        json.dump(current_state, f, indent=2, default=str)
    inference = build_live_inference_frame(config, weather, source_status, issue_time, root)
    probabilistic = predict_quantiles(inference, root / "outputs/models")
    operational = build_operational_forecast(probabilistic, config)
    operational = add_source_transparency(operational, source_status)
    operational.to_csv(output / "forecast.csv", index=False)
    operational.to_csv(root / "outputs/forecasts/live_operational_forecast.csv", index=False)

    actions = generate_operator_actions(operational)
    save_operator_actions(actions, output / "operator_actions.json")
    ranking, summary = rank_sites(operational, config)
    save_site_ranking(ranking, summary, output)

    print(f"Saved live UI forecast: {output / 'forecast.csv'} ({len(operational)} rows)")
    print(f"Saved live current state: {output / 'current_state.json'}")
    print(f"Saved live source status: {output / 'source_status.json'}")
    print(f"Saved live operator actions: {output / 'operator_actions.json'} ({len(actions)} actions)")
    print(f"Saved live site ranking: {output / 'site_ranking.csv'} ({len(ranking)} sites)")
    return operational, actions, ranking, source_status


def build_live_current_state(
    config: dict[str, Any],
    weather: pd.DataFrame,
    source_status: dict[str, Any],
    issue_time: pd.Timestamp,
) -> dict[str, Any]:
    """Estimate current PV state per configured site from near-real-time Open-Meteo conditions."""
    rows: list[dict[str, Any]] = []
    weather_row = _nearest_weather_row(weather, issue_time)
    for site in config.get("sites", []):
        geometry = _target_solar_features(pd.Series([issue_time]), site).iloc[0]
        cos_zenith = float(geometry["target_cos_zenith"])
        ghi = _current_ghi(weather_row, geometry)
        dni = _current_dni(weather_row, ghi, cos_zenith)
        dhi = max(ghi - dni * max(cos_zenith, 0.0), 0.0)
        frame = pd.DataFrame(
            [
                {
                    "target_valid_time": issue_time,
                    "target_solar_zenith": geometry["target_solar_zenith"],
                    "target_solar_elevation": geometry["target_solar_elevation"],
                    "target_solar_azimuth": geometry["target_solar_azimuth"],
                    "target_cos_zenith": cos_zenith,
                    "target_GHI_clear": geometry["target_GHI_clear"],
                    "target_DNI_clear": geometry["target_DNI_clear"],
                    "target_DHI_clear": geometry["target_DHI_clear"],
                    "target_temperature_2m_forecast_proxy": weather_row.get("temperature_2m", np.nan),
                    "horizon_minutes": 0,
                    "GHI_P10_calibrated": ghi,
                    "GHI_P50": ghi,
                    "GHI_P90_calibrated": ghi,
                    "DNI_P10_estimated": dni,
                    "DNI_P50_estimated": dni,
                    "DNI_P90_estimated": dni,
                    "DHI_P10_estimated": dhi,
                    "DHI_P50_estimated": dhi,
                    "DHI_P90_estimated": dhi,
                }
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poa = compute_poa_quantiles(
                frame,
                surface_tilt=float(config.get("pv_system", {}).get("surface_tilt", 35.0)),
                surface_azimuth=float(config.get("pv_system", {}).get("surface_azimuth", 180.0)),
                albedo=float(config.get("pv_system", {}).get("albedo_default", 0.2)),
            )
            pv = estimate_pv_power_quantiles(poa, config.get("pv_system", {}))
        rows.append(
            {
                "site_id": site["id"],
                "site_name": site["name"],
                "timestamp": issue_time.isoformat(),
                "current_ghi": float(ghi),
                "current_poa": float(pv.iloc[0]["POA_P50"]),
                "current_pv_output": float(pv.iloc[0]["PV_P50"]),
                "current_module_temperature": float(pv.iloc[0]["module_temperature"]),
                "current_cloud_cover": _as_float(weather_row.get("cloud_cover")),
                "current_source": "openmeteo_near_real_time_estimate",
                "current_output_basis": "near-real-time estimate from current Open-Meteo weather and irradiance, not plant telemetry",
                "satellite_data_available": bool(source_status["satellite_data_available"]),
            }
        )

    primary_id = _primary_site_id(config)
    selected = next((row for row in rows if row["site_id"] == primary_id), rows[0] if rows else None)
    return {
        "timestamp": issue_time.isoformat(),
        "primary_site_id": primary_id,
        "primary_site_name": _site_name(config, primary_id),
        "sites": rows,
        "selected_site": selected,
    }


def build_live_inference_frame(
    config: dict[str, Any],
    weather: pd.DataFrame,
    source_status: dict[str, Any],
    issue_time: pd.Timestamp,
    project_root: str | Path = ".",
) -> pd.DataFrame:
    """Construct leakage-safe live inference rows from current weather and stored training templates."""
    root = Path(project_root)
    templates = _load_issue_templates(root)
    rows: list[pd.DataFrame] = []
    horizons = _operational_horizons(config, source_status, root / "outputs/models")
    for site in config.get("sites", []):
        template = _template_for_site(templates, site["id"])
        for horizon in horizons:
            rows.append(_build_one_inference_row(template, site, weather, issue_time, horizon))
    if not rows:
        raise RuntimeError("No live inference rows could be constructed.")
    return pd.concat(rows, ignore_index=True)


def predict_quantiles(inference: pd.DataFrame, models_dir: str | Path) -> pd.DataFrame:
    """Load horizon-specific quantile models and predict GHI P10/P50/P90."""
    frames: list[pd.DataFrame] = []
    for horizon, horizon_df in inference.groupby("horizon_minutes", sort=True):
        if not bool(horizon_df["operationally_available"].iloc[0]):
            disabled = horizon_df.copy()
            for column in ["GHI_P10_raw", "GHI_P50", "GHI_P90_raw", "GHI_P10_calibrated", "GHI_P90_calibrated"]:
                disabled[column] = np.nan
            disabled["uncertainty_level"] = "Not operationally available"
            frames.append(disabled)
            continue

        payloads = {
            label: _load_quantile_payload(models_dir, int(horizon), label)
            for label in ["p10", "p50", "p90"]
        }
        encoder = payloads["p50"]["encoder"]
        x = encoder.transform(horizon_df)
        raw = np.vstack([payloads[label]["model"].predict(x) for label in ["p10", "p50", "p90"]]).T
        raw = np.sort(raw, axis=1)
        predicted = horizon_df.copy()
        predicted["GHI_P10_raw"] = _apply_physical_constraints(predicted, raw[:, 0])
        predicted["GHI_P50"] = _apply_physical_constraints(predicted, raw[:, 1])
        predicted["GHI_P90_raw"] = _apply_physical_constraints(predicted, raw[:, 2])
        ordered = np.sort(predicted[["GHI_P10_raw", "GHI_P50", "GHI_P90_raw"]].to_numpy(dtype=float), axis=1)
        predicted["GHI_P10_calibrated"] = ordered[:, 0]
        predicted["GHI_P50"] = ordered[:, 1]
        predicted["GHI_P90_calibrated"] = ordered[:, 2]
        predicted["uncertainty_level"] = _uncertainty_level(predicted)
        frames.append(predicted)
    return pd.concat(frames, ignore_index=True).sort_values(["site_id", "target_valid_time", "horizon_minutes"])


def add_source_transparency(df: pd.DataFrame, source_status: dict[str, Any]) -> pd.DataFrame:
    """Add UI-required data-source transparency fields to every output row."""
    out = df.copy()
    out["primary_satellite_source"] = source_status["primary_satellite_source"]
    out["weather_forecast_source"] = source_status["weather_forecast_source"]
    out["satellite_data_available"] = bool(source_status["satellite_data_available"])
    out["fallback_active"] = bool(source_status["fallback_active"])
    out["data_freshness_minutes"] = source_status["data_freshness_minutes"]
    out["data_quality_level"] = source_status["data_quality_level"]
    out["demo_mode"] = False
    out["display_mode"] = "Live Forecast"
    return out


def _fetch_or_load_openmeteo_forecast(config_path: str | Path, root: Path) -> pd.DataFrame:
    """Fetch current Open-Meteo forecast, falling back to the cached forecast file."""
    cached = root / "data/raw/openmeteo_forecast_munich.csv"
    try:
        om_config = load_openmeteo_config(config_path)
        forecast = download_openmeteo_forecast(om_config)
        if not forecast.empty:
            save_dataframe(forecast, cached)
            return forecast
    except Exception as exc:
        warnings.warn(f"Open-Meteo live fetch failed; using cached forecast if available. Reason: {exc}", stacklevel=2)
    if cached.exists():
        cached_forecast = pd.read_csv(cached)
        cached_forecast["timestamp"] = pd.to_datetime(cached_forecast["timestamp"], utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE)
        return cached_forecast
    raise RuntimeError("No live or cached Open-Meteo forecast is available.")


def _issue_time(weather: pd.DataFrame) -> pd.Timestamp:
    """Choose the live issue time from the weather forecast frame."""
    timestamps = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE).dropna()
    now = pd.Timestamp.now(tz=TARGET_TIMEZONE).floor("h")
    future = timestamps[timestamps >= now]
    return future.min() if not future.empty else timestamps.max()


def _operational_horizons(config: dict[str, Any], source_status: dict[str, Any], models_dir: Path) -> list[int]:
    """Return horizons to emit, including disabled high-frequency rows for transparency."""
    base = [int(h) for h in config.get("forecast", {}).get("valid_horizons_minutes", [60, 180, 360, 720, 1440])]
    high = [int(h) for h in config.get("forecast", {}).get("high_frequency_horizons_minutes", [15, 30])]
    available = []
    for horizon in high + base:
        models_exist = _quantile_models_exist(models_dir, horizon)
        high_status = source_status.get("high_frequency_horizons", {}).get(str(horizon), {})
        if horizon in high:
            available.append(horizon)
        elif models_exist:
            available.append(horizon)
    return list(dict.fromkeys(available))


def _build_one_inference_row(
    template: pd.Series,
    site: dict[str, Any],
    weather: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Build one live inference row for a site and horizon."""
    target_time = issue_time + pd.Timedelta(minutes=horizon)
    weather_row = _nearest_weather_row(weather, target_time)
    solar = _target_solar_features(pd.Series([target_time]), site).iloc[0]
    row = template.copy()
    row["site_id"] = site["id"]
    row["latitude"] = site["latitude"]
    row["longitude"] = site["longitude"]
    row["timestamp"] = issue_time
    row["target_timestamp"] = target_time
    row["target_valid_time"] = target_time
    row["horizon_minutes"] = int(horizon)
    row["target_source"] = "live_forecast"
    row["target_quality_level"] = "live_operational_inference"
    row["quality_flag"] = "live"
    for column, value in solar.items():
        row[column] = value

    for source, target in {
        "cloud_cover": "target_cloud_cover_forecast_proxy",
        "cloud_cover_low": "target_cloud_cover_low_forecast_proxy",
        "cloud_cover_mid": "target_cloud_cover_mid_forecast_proxy",
        "cloud_cover_high": "target_cloud_cover_high_forecast_proxy",
        "temperature_2m": "target_temperature_2m_forecast_proxy",
        "relative_humidity_2m": "target_relative_humidity_2m_forecast_proxy",
        "wind_speed_10m": "target_wind_speed_10m_forecast_proxy",
        "precipitation": "target_precipitation_forecast_proxy",
    }.items():
        row[target] = weather_row.get(source, np.nan)

    cloud = pd.to_numeric(row.get("target_cloud_cover_forecast_proxy", np.nan), errors="coerce")
    cloud_fraction = np.nan_to_num(cloud, nan=0.0) / 100.0
    t_cloud = np.clip(1.0 - 0.75 * cloud_fraction**1.5, 0.05, 1.0)
    row["GHI_phys_target"] = float(row["target_GHI_clear"]) * float(t_cloud)
    row["GHI_target"] = np.nan
    row["GHI_residual_target"] = np.nan
    row["operationally_available"] = _horizon_available(horizon)
    row["operational_status"] = "Operational" if row["operationally_available"] else "Not operationally available"
    row["operational_unavailable_reason"] = None if row["operationally_available"] else "High-frequency satellite input required"
    return pd.DataFrame([row])


def _horizon_available(horizon: int) -> bool:
    """Return whether a horizon has trained operational models in this MVP."""
    return horizon not in {15, 30}


def _nearest_weather_row(weather: pd.DataFrame, target_time: pd.Timestamp) -> pd.Series:
    """Select the nearest forecast weather row for a target timestamp."""
    frame = weather.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE)
    idx = (frame["timestamp"] - target_time).abs().idxmin()
    return frame.loc[idx]


def _current_ghi(weather_row: pd.Series, geometry: pd.Series) -> float:
    """Estimate current GHI from live weather inputs with a physical fallback."""
    shortwave = _as_float(weather_row.get("shortwave_radiation"))
    if shortwave is not None:
        return max(shortwave, 0.0)
    cloud_cover = _as_float(weather_row.get("cloud_cover")) or 0.0
    cloud_fraction = np.clip(cloud_cover / 100.0, 0.0, 1.0)
    t_cloud = np.clip(1.0 - 0.75 * cloud_fraction**1.5, 0.05, 1.0)
    return max(float(geometry["target_GHI_clear"]) * float(t_cloud), 0.0)


def _current_dni(weather_row: pd.Series, ghi: float, cos_zenith: float) -> float:
    """Estimate current DNI using direct normal irradiance when available or pvlib fallback."""
    dni = _as_float(weather_row.get("direct_normal_irradiance"))
    if dni is not None:
        return max(dni, 0.0)
    if cos_zenith <= 0.0 or ghi <= 0.0:
        return 0.0
    direct_horizontal = _as_float(weather_row.get("direct_radiation"))
    if direct_horizontal is not None:
        return max(direct_horizontal / max(cos_zenith, 0.08), 0.0)
    erbs = pvlib.irradiance.erbs(ghi=pd.Series([ghi]), zenith=pd.Series([np.degrees(np.arccos(np.clip(cos_zenith, 0.0, 1.0)))]), datetime_or_doy=pd.DatetimeIndex([pd.Timestamp.now(tz=TARGET_TIMEZONE)]))
    return max(float(erbs["dni"].iloc[0]), 0.0)


def _target_solar_features(times: pd.Series, site: dict[str, Any]) -> pd.DataFrame:
    """Compute target-time solar geometry and clear-sky irradiance for one site."""
    geometry = compute_solar_geometry(times, float(site["latitude"]), float(site["longitude"]), float(site.get("altitude", 520)), TARGET_TIMEZONE)
    clear = compute_clear_sky_irradiance(times, float(site["latitude"]), float(site["longitude"]), float(site.get("altitude", 520)), TARGET_TIMEZONE)
    return pd.DataFrame(
        {
            "target_solar_zenith": geometry["solar_zenith"],
            "target_solar_elevation": geometry["solar_elevation"],
            "target_solar_azimuth": geometry["solar_azimuth"],
            "target_cos_zenith": geometry["cos_zenith"],
            "target_air_mass": geometry["relative_air_mass"],
            "target_GHI_clear": clear["clear_sky_ghi"],
            "target_DNI_clear": clear["clear_sky_dni"],
            "target_DHI_clear": clear["clear_sky_dhi"],
        }
    )


def _load_issue_templates(root: Path) -> pd.DataFrame:
    """Load recent supervised rows to provide encoder-compatible issue-time features."""
    path = root / "data/processed/test_dataset.parquet"
    if path.exists():
        return pd.read_parquet(path).sort_values("timestamp").groupby("site_id", as_index=False).tail(1)
    path = root / "outputs/forecasts/test_probabilistic_predictions.csv"
    if path.exists():
        return pd.read_csv(path).sort_values("timestamp").groupby("site_id", as_index=False).tail(1)
    raise RuntimeError("No supervised template data is available for live inference.")


def _primary_site_id(config: dict[str, Any]) -> str:
    """Return the configured primary site id for cockpit presentation."""
    return str(config.get("product", {}).get("primary_site_id", config.get("sites", [{}])[0].get("id", "munich_centre")))


def _site_name(config: dict[str, Any], site_id: str) -> str:
    """Return a configured site name for a site id."""
    for site in config.get("sites", []):
        if site["id"] == site_id:
            return str(site["name"])
    return site_id


def _as_float(value: Any) -> float | None:
    """Convert a scalar to float when possible."""
    try:
        number = pd.to_numeric(value, errors="coerce")
    except Exception:
        return None
    return None if pd.isna(number) else float(number)


def _template_for_site(templates: pd.DataFrame, site_id: str) -> pd.Series:
    """Return a recent template row for a site, falling back to the first template."""
    matched = templates[templates["site_id"] == site_id]
    if not matched.empty:
        return matched.iloc[-1].copy()
    return templates.iloc[-1].copy()


def _load_quantile_payload(models_dir: str | Path, horizon: int, label: str) -> dict[str, Any]:
    """Load one quantile model payload."""
    path = Path(models_dir) / f"quantile_ghi_horizon_{horizon}_{label}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing quantile model for horizon {horizon}: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _quantile_models_exist(models_dir: str | Path, horizon: int) -> bool:
    """Return whether all quantile models exist for a horizon."""
    base = Path(models_dir)
    return all((base / f"quantile_ghi_horizon_{horizon}_{label}.pkl").exists() for label in ["p10", "p50", "p90"])


def _uncertainty_level(df: pd.DataFrame) -> pd.Series:
    """Classify uncertainty from relative calibrated interval width."""
    width = pd.to_numeric(df["GHI_P90_calibrated"], errors="coerce") - pd.to_numeric(df["GHI_P10_calibrated"], errors="coerce")
    relative = width / pd.to_numeric(df["GHI_P50"], errors="coerce").clip(lower=50.0)
    return pd.cut(relative, bins=[-np.inf, 0.5, 1.2, np.inf], labels=["Low", "Medium", "High"]).astype(str)


def main() -> None:
    """CLI entry point for live forecast orchestration."""
    parser = argparse.ArgumentParser(description="Run the SolarOps live forecast orchestrator.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_LIVE_OUTPUT_DIR))
    args = parser.parse_args()
    run_live_forecast(args.config, args.output_dir)


if __name__ == "__main__":
    main()
