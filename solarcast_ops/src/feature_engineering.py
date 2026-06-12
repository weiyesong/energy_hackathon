from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pvlib

from src.config import get_paths
from src.utils import safe_divide

LOGGER = logging.getLogger(__name__)


BASE_FEATURES = [
    "pv_power_mw",
    "global_irradiance_wm2",
    "direct_irradiance_wm2",
    "diffuse_irradiance_wm2",
    "air_temperature_c",
    "wind_speed_ms",
    "solar_elevation_deg",
    "solar_zenith_deg",
    "solar_azimuth_deg",
]

HISTORY_ONLY_FEATURES = [
    "pv_power_mw",
    "solar_elevation_deg",
    "power_lag_1h",
    "power_lag_2h",
    "power_lag_3h",
    "power_rolling_mean_3h",
    "power_rolling_std_3h",
    "power_change_1h",
    "clear_sky_power_mw",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]

SATELLITE_FEATURES = HISTORY_ONLY_FEATURES + [
    "global_irradiance_wm2",
    "direct_irradiance_wm2",
    "diffuse_irradiance_wm2",
    "air_temperature_c",
    "wind_speed_ms",
    "irradiance_lag_1h",
    "irradiance_lag_2h",
    "irradiance_lag_3h",
    "temperature_lag_1h",
    "irradiance_rolling_mean_3h",
    "irradiance_rolling_std_3h",
    "irradiance_change_1h",
    "clear_sky_ghi_wm2",
    "clear_sky_index",
    "power_clear_sky_ratio",
    "diffuse_fraction",
    "beam_fraction",
]


def add_clear_sky_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    site = config["site"]
    data = df.copy()
    times = pd.DatetimeIndex(pd.to_datetime(data["timestamp"], utc=True))
    location = pvlib.location.Location(site["latitude"], site["longitude"], tz="UTC", altitude=520)
    clearsky = location.get_clearsky(times, model=config.get("irradiance_model", {}).get("clear_sky_model", "ineichen"))
    solpos = location.get_solarposition(times)
    data["clear_sky_ghi_wm2"] = clearsky["ghi"].to_numpy()
    data["clear_sky_dni_wm2"] = clearsky["dni"].to_numpy()
    data["clear_sky_dhi_wm2"] = clearsky["dhi"].to_numpy()
    data["solar_zenith_deg"] = solpos["zenith"].to_numpy()
    data["solar_azimuth_deg"] = solpos["azimuth"].to_numpy()
    loss_factor = 1 - float(site["system_loss_percent"]) / 100
    data["clear_sky_power_mw"] = (
        float(site["peak_power_mw"]) * loss_factor * (data["clear_sky_ghi_wm2"].clip(lower=0) / 1000.0)
    ).clip(0, float(site["peak_power_mw"]) * 1.05)
    data["clear_sky_index"] = safe_divide(data["global_irradiance_wm2"].to_numpy(), data["clear_sky_ghi_wm2"].to_numpy())
    data["clear_sky_index"] = data["clear_sky_index"].clip(0, float(config.get("irradiance_model", {}).get("clear_sky_index_max", 1.5)))
    data["power_clear_sky_ratio"] = safe_divide(data["pv_power_mw"].to_numpy(), data["clear_sky_power_mw"].to_numpy())
    data["power_clear_sky_ratio"] = data["power_clear_sky_ratio"].clip(0, 1.5)
    data["diffuse_fraction"] = safe_divide(data["diffuse_irradiance_wm2"].to_numpy(), data["global_irradiance_wm2"].to_numpy())
    data["diffuse_fraction"] = data["diffuse_fraction"].clip(
        float(config.get("irradiance_model", {}).get("diffuse_fraction_min", 0.0)),
        float(config.get("irradiance_model", {}).get("diffuse_fraction_max", 1.0)),
    )
    data["beam_fraction"] = safe_divide(data["direct_irradiance_wm2"].to_numpy(), data["global_irradiance_wm2"].to_numpy())
    data["beam_fraction"] = data["beam_fraction"].clip(0, 1.5)
    return data


def build_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create leakage-safe features and future targets."""
    paths = get_paths()
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    data = add_clear_sky_features(data, config)

    for lag in [1, 2, 3]:
        data[f"power_lag_{lag}h"] = data["pv_power_mw"].shift(lag)
        data[f"irradiance_lag_{lag}h"] = data["global_irradiance_wm2"].shift(lag)
    data["temperature_lag_1h"] = data["air_temperature_c"].shift(1)

    shifted_power = data["pv_power_mw"].shift(1)
    shifted_irr = data["global_irradiance_wm2"].shift(1)
    data["power_rolling_mean_3h"] = shifted_power.rolling(3, min_periods=2).mean()
    data["power_rolling_std_3h"] = shifted_power.rolling(3, min_periods=2).std().fillna(0)
    data["irradiance_rolling_mean_3h"] = shifted_irr.rolling(3, min_periods=2).mean()
    data["irradiance_rolling_std_3h"] = shifted_irr.rolling(3, min_periods=2).std().fillna(0)
    data["power_change_1h"] = data["pv_power_mw"] - data["pv_power_mw"].shift(1)
    data["irradiance_change_1h"] = data["global_irradiance_wm2"] - data["global_irradiance_wm2"].shift(1)

    ts = pd.DatetimeIndex(data["timestamp"])
    data["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    data["day_of_year_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    data["day_of_year_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    data["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    data["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

    for h in config["forecast"]["horizons_hours"]:
        data[f"target_h{h}"] = data["pv_power_mw"].shift(-int(h))
        data[f"clear_sky_power_h{h}"] = data["clear_sky_power_mw"].shift(-int(h))
        data[f"target_ghi_h{h}"] = data["global_irradiance_wm2"].shift(-int(h))
        data[f"target_dhi_h{h}"] = data["diffuse_irradiance_wm2"].shift(-int(h))
        data[f"target_dni_h{h}"] = data["direct_irradiance_wm2"].shift(-int(h))
        data[f"target_kstar_h{h}"] = data["clear_sky_index"].shift(-int(h))
        data[f"target_diffuse_fraction_h{h}"] = data["diffuse_fraction"].shift(-int(h))
        data[f"clear_sky_ghi_h{h}"] = data["clear_sky_ghi_wm2"].shift(-int(h))
        data[f"clear_sky_dni_h{h}"] = data["clear_sky_dni_wm2"].shift(-int(h))
        data[f"clear_sky_dhi_h{h}"] = data["clear_sky_dhi_wm2"].shift(-int(h))
        data[f"solar_zenith_h{h}"] = data["solar_zenith_deg"].shift(-int(h))
        data[f"solar_azimuth_h{h}"] = data["solar_azimuth_deg"].shift(-int(h))

    target_cols = []
    for h in config["forecast"]["horizons_hours"]:
        target_cols.extend(
            [
                f"target_h{h}",
                f"target_ghi_h{h}",
                f"target_kstar_h{h}",
                f"target_diffuse_fraction_h{h}",
                f"clear_sky_ghi_h{h}",
                f"solar_zenith_h{h}",
                f"solar_azimuth_h{h}",
            ]
        )
    required = sorted(set(SATELLITE_FEATURES + target_cols))
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required).reset_index(drop=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(paths.processed_dir / "features.csv", index=False)
    LOGGER.info("Feature engineering complete: %d rows, %d columns", len(data), len(data.columns))
    return data
