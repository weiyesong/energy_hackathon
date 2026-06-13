"""Build leakage-safe supervised multi-horizon datasets for SolarOps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.persistence_baseline import add_persistence_baselines
from src.physics.clear_sky import compute_clear_sky_irradiance
from src.physics.solar_geometry import compute_solar_geometry


DEFAULT_HORIZONS_MINUTES = [60, 180, 360, 720, 1440]
HIGH_FREQUENCY_HORIZONS = [15, 30]
TARGET_TIMEZONE = "Europe/Berlin"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and attach the project root path."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["_project_root"] = str(path.parent)
    return config


def enabled_horizons_from_summary(summary_path: str | Path) -> list[int]:
    """Read enabled horizons from fusion summary, keeping 15/30 minutes disabled unless supported."""
    path = Path(summary_path)
    horizons = list(DEFAULT_HORIZONS_MINUTES)
    if not path.exists():
        return horizons
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    support = summary.get("horizon_support", {})
    if support.get("15_minutes", {}).get("supported", False):
        horizons.insert(0, 15)
    if support.get("30_minutes", {}).get("supported", False):
        insert_at = 1 if 15 in horizons else 0
        horizons.insert(insert_at, 30)
    return horizons


def build_supervised_dataset(
    fused: pd.DataFrame,
    config: dict[str, Any],
    horizons_minutes: list[int],
) -> pd.DataFrame:
    """Build a supervised dataset while separating issue-time, lagged, and target-time fields."""
    prepared = _prepare_fused_frame(fused)
    rows: list[pd.DataFrame] = []
    for site_id, site_frame in prepared.groupby("site_id", sort=False):
        site = _site_config(config, site_id)
        frequency = _infer_site_frequency(site_frame)
        issue = _build_issue_time_features(site_frame)
        for horizon in horizons_minutes:
            target = _build_target_frame(site_frame, horizon, frequency, site)
            merged = issue.merge(target, on=["site_id", "target_timestamp"], how="inner")
            if merged.empty:
                continue
            merged["horizon_minutes"] = int(horizon)
            rows.append(merged)

    if not rows:
        raise RuntimeError("No supervised rows could be created for the requested horizons.")

    supervised = pd.concat(rows, ignore_index=True).sort_values(["site_id", "timestamp", "horizon_minutes"])
    supervised = add_persistence_baselines(supervised)
    supervised["GHI_residual_target"] = supervised["GHI_target"] - supervised["GHI_phys_target"]
    return supervised.reset_index(drop=True)


def split_chronologically(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split supervised data chronologically, using fixed project dates or sensible fallback splits."""
    data = df.copy()
    data["timestamp"] = _to_berlin(data["timestamp"])
    train_start = pd.Timestamp("2023-01-01", tz=TARGET_TIMEZONE)
    train_end = pd.Timestamp("2024-12-31 23:59:59", tz=TARGET_TIMEZONE)
    validation_start = pd.Timestamp("2025-01-01", tz=TARGET_TIMEZONE)
    validation_end = pd.Timestamp("2025-06-30 23:59:59", tz=TARGET_TIMEZONE)
    test_start = pd.Timestamp("2025-07-01", tz=TARGET_TIMEZONE)
    test_end = pd.Timestamp("2025-12-31 23:59:59", tz=TARGET_TIMEZONE)

    fixed_train = data[(data["timestamp"] >= train_start) & (data["timestamp"] <= train_end)]
    fixed_validation = data[(data["timestamp"] >= validation_start) & (data["timestamp"] <= validation_end)]
    fixed_test = data[(data["timestamp"] >= test_start) & (data["timestamp"] <= test_end)]
    if not fixed_train.empty and not fixed_validation.empty and not fixed_test.empty:
        metadata = {
            "split_strategy": "fixed_chronological",
            "train": [str(train_start), str(train_end)],
            "validation": [str(validation_start), str(validation_end)],
            "test": [str(test_start), str(test_end)],
        }
        return fixed_train, fixed_validation, fixed_test, metadata

    ordered = data.sort_values("timestamp")
    n = len(ordered)
    train_end_idx = max(int(n * 0.6), 1)
    validation_end_idx = max(int(n * 0.8), train_end_idx + 1)
    train = ordered.iloc[:train_end_idx]
    validation = ordered.iloc[train_end_idx:validation_end_idx]
    test = ordered.iloc[validation_end_idx:]
    metadata = {
        "split_strategy": "fallback_60_20_20_chronological",
        "available_start": str(ordered["timestamp"].min()),
        "available_end": str(ordered["timestamp"].max()),
    }
    return train, validation, test, metadata


def main() -> None:
    """Create supervised datasets and persistence baseline predictions."""
    parser = argparse.ArgumentParser(description="Build supervised multi-horizon irradiance datasets.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", default="data/processed/fused_solar_dataset.parquet", help="Input fused dataset")
    parser.add_argument("--summary", default="data/processed/fusion_summary.json", help="Fusion summary JSON")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        project_root = Path(config["_project_root"])
        fused = pd.read_parquet(project_root / args.input)
        horizons = enabled_horizons_from_summary(project_root / args.summary)
        print(f"Enabled horizons: {horizons}")
        supervised = build_supervised_dataset(fused, config, horizons)
        train, validation, test, split_metadata = split_chronologically(supervised)

        processed = project_root / "data/processed"
        forecasts = project_root / "outputs/forecasts"
        processed.mkdir(parents=True, exist_ok=True)
        forecasts.mkdir(parents=True, exist_ok=True)
        supervised.to_parquet(processed / "supervised_multi_horizon.parquet", index=False)
        train.to_parquet(processed / "train_dataset.parquet", index=False)
        validation.to_parquet(processed / "validation_dataset.parquet", index=False)
        test.to_parquet(processed / "test_dataset.parquet", index=False)
        _baseline_prediction_columns(supervised).to_csv(forecasts / "persistence_baseline_predictions.csv", index=False)
        with (processed / "supervised_split_summary.json").open("w", encoding="utf-8") as file:
            json.dump(split_metadata, file, indent=2, sort_keys=True)
        print(f"Saved supervised rows: {len(supervised)}")
        print("Hindcast-development note: target-time weather columns ending in _forecast_proxy are historical proxies, not live operational forecasts.")
    except Exception as exc:
        print(f"Feature engineering failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _prepare_fused_frame(fused: pd.DataFrame) -> pd.DataFrame:
    """Normalize fused inputs and create source-specific helper columns."""
    frame = fused.copy()
    frame["timestamp"] = _to_berlin(frame["timestamp"])
    frame["ghi_current"] = _first_available(frame, ["ground_ghi_observed", "eumetsat_ssi_ssi", "openmeteo_ssi", "nasa_power_ssi", "best_available_satellite_ssi"])
    frame["issue_clear_sky_ghi"] = _first_available(frame, ["openmeteo_clear_sky_ghi", "nasa_power_clear_sky_ssi", "openmeteo_clear_sky_ssi"])
    frame["cloud_cover_current"] = _first_available(frame, ["openmeteo_cloud_cover", "eumetsat_ssi_cloud_cover", "nasa_power_cloud_cover"])
    return frame.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def _build_issue_time_features(site_frame: pd.DataFrame) -> pd.DataFrame:
    """Create issue-time and lagged features using only rows at or before issue time."""
    issue = site_frame[["site_id", "timestamp", "latitude", "longitude"]].copy()
    issue["target_timestamp"] = site_frame["timestamp"]
    issue["ghi_issue"] = site_frame["ghi_current"]
    issue["issue_clear_sky_ghi"] = site_frame["issue_clear_sky_ghi"]
    issue["satellite_ssi_issue"] = site_frame["best_available_satellite_ssi"]
    issue["satellite_clear_sky_index_issue"] = site_frame["satellite_clear_sky_index"]
    issue["cloud_cover_issue"] = site_frame["cloud_cover_current"]
    issue["quality_flag"] = "ok"

    csi = site_frame["ghi_current"] / site_frame["issue_clear_sky_ghi"].clip(lower=1.0)
    issue["ghi_lag_1"] = site_frame["ghi_current"].shift(1)
    issue["ghi_lag_2"] = site_frame["ghi_current"].shift(2)
    issue["ghi_lag_3"] = site_frame["ghi_current"].shift(3)
    issue["clear_sky_index_lag_1"] = csi.shift(1)
    issue["clear_sky_index_lag_2"] = csi.shift(2)
    issue["cloud_cover_lag_1"] = site_frame["cloud_cover_current"].shift(1)
    issue["cloud_cover_lag_3"] = site_frame["cloud_cover_current"].shift(3)
    issue["satellite_ssi_lag_1"] = site_frame["best_available_satellite_ssi"].shift(1)
    issue["satellite_ssi_trend"] = site_frame["best_available_satellite_ssi"] - site_frame["best_available_satellite_ssi"].shift(1)
    issue["cloud_cover_trend"] = site_frame["cloud_cover_current"] - site_frame["cloud_cover_current"].shift(1)

    issue["best_satellite_source"] = site_frame["best_satellite_source"]
    issue["satellite_data_available"] = site_frame["satellite_data_available"]
    issue["irradiance_source_std"] = site_frame["irradiance_source_std"]
    issue["number_of_available_irradiance_sources"] = site_frame["number_of_available_irradiance_sources"]
    return issue


def _build_target_frame(site_frame: pd.DataFrame, horizon_minutes: int, frequency: pd.Timedelta, site: dict[str, Any]) -> pd.DataFrame:
    """Create target labels, target-time deterministic features, and hindcast forecast proxies."""
    target = site_frame.copy()
    target["target_timestamp"] = target["timestamp"] - pd.Timedelta(minutes=horizon_minutes)
    issue_times = set(site_frame["timestamp"])
    target = target[target["target_timestamp"].isin(issue_times)].copy()
    if target.empty:
        return pd.DataFrame(columns=["site_id", "target_timestamp"])

    out = target[["site_id", "target_timestamp"]].copy()
    out["target_valid_time"] = target["timestamp"]
    out["GHI_target"] = _target_ghi(target)
    out["target_source"] = _target_source(target)
    out["target_quality_level"] = out["target_source"].map(_target_quality_level)

    deterministic = _target_solar_features(target["timestamp"], site)
    for column in deterministic.columns:
        if column != "timestamp":
            out[column] = deterministic[column].to_numpy()

    out["target_cloud_cover_forecast_proxy"] = target.get("openmeteo_cloud_cover", pd.Series(np.nan, index=target.index)).to_numpy()
    for column in ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]:
        out[f"target_{column}_forecast_proxy"] = target.get(f"openmeteo_{column}", pd.Series(np.nan, index=target.index)).to_numpy()

    cloud_proxy = pd.to_numeric(out["target_cloud_cover_forecast_proxy"], errors="coerce") / 100.0
    t_cloud = (1.0 - 0.75 * np.power(cloud_proxy.clip(0.0, 1.0), 1.5)).clip(0.05, 1.0)
    out["GHI_phys_target"] = out["target_GHI_clear"]
    has_cloud_proxy = out["target_cloud_cover_forecast_proxy"].notna()
    out.loc[has_cloud_proxy, "GHI_phys_target"] = out.loc[has_cloud_proxy, "target_GHI_clear"] * t_cloud.loc[has_cloud_proxy]
    return out


def _target_solar_features(times: pd.Series, site: dict[str, Any]) -> pd.DataFrame:
    """Compute deterministic target-time solar geometry and clear-sky features."""
    geometry = compute_solar_geometry(times, float(site["latitude"]), float(site["longitude"]), float(site.get("altitude", 520)), TARGET_TIMEZONE)
    clear = compute_clear_sky_irradiance(times, float(site["latitude"]), float(site["longitude"]), float(site.get("altitude", 520)), TARGET_TIMEZONE)
    return pd.DataFrame(
        {
            "timestamp": geometry["timestamp"],
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


def _target_ghi(target: pd.DataFrame) -> pd.Series:
    """Select the best available trustworthy target irradiance according to priority."""
    return _first_available(target, ["ground_ghi_observed", "eumetsat_ssi_ssi", "openmeteo_ssi", "nasa_power_ssi"])


def _target_source(target: pd.DataFrame) -> pd.Series:
    """Return target source names according to target priority."""
    source = pd.Series("missing", index=target.index, dtype="object")
    priorities = [
        ("ground_ghi_observed", "ground_observation"),
        ("eumetsat_ssi_ssi", "eumetsat_ssi"),
        ("openmeteo_ssi", "openmeteo_shortwave_proxy"),
        ("nasa_power_ssi", "nasa_power"),
    ]
    for column, name in priorities:
        if column in target.columns:
            mask = source.eq("missing") & target[column].notna()
            source.loc[mask] = name
    return source


def _target_quality_level(source: str) -> str:
    """Map target source names to coarse quality levels."""
    return {
        "ground_observation": "ground_observation",
        "eumetsat_ssi": "operational_satellite",
        "openmeteo_shortwave_proxy": "hindcast_forecast_proxy",
        "nasa_power": "satellite_model_historical_baseline",
    }.get(source, "missing")


def _infer_site_frequency(site_frame: pd.DataFrame) -> pd.Timedelta:
    """Infer actual dataset frequency from site timestamps."""
    diffs = site_frame["timestamp"].drop_duplicates().sort_values().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(hours=1)
    return diffs.median()


def _site_config(config: dict[str, Any], site_id: str) -> dict[str, Any]:
    """Return configured site metadata by site id."""
    for site in config.get("sites", []):
        if site["id"] == site_id:
            return site
    row = {"id": site_id, "latitude": np.nan, "longitude": np.nan, "altitude": config.get("location", {}).get("altitude", 520)}
    return row


def _first_available(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first non-null value across available columns."""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column in df.columns:
            result = result.combine_first(pd.to_numeric(df[column], errors="coerce"))
    return result


def _to_berlin(values: Any) -> pd.Series:
    """Parse timestamps and convert them to Europe/Berlin timezone."""
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE)


def _baseline_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select persistence baseline columns for CSV export."""
    columns = [
        "site_id",
        "timestamp",
        "target_valid_time",
        "horizon_minutes",
        "GHI_target",
        "target_source",
        "GHI_persistence_naive",
        "GHI_persistence_csi",
    ]
    return df[columns].copy()


if __name__ == "__main__":
    main()
