from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_paths
from src.utils import write_json

LOGGER = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Clean hourly solar data without using future observations."""
    paths = get_paths()
    peak = float(config["site"]["peak_power_mw"])
    daylight_threshold = float(config["forecast"]["daylight_solar_elevation_threshold"])
    report: dict[str, Any] = {"input_rows": int(len(df))}

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    invalid_timestamps = int(data["timestamp"].isna().sum())
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp")
    duplicates = int(data["timestamp"].duplicated().sum())
    data = data.drop_duplicates("timestamp", keep="first")

    full_index = pd.date_range(data["timestamp"].min(), data["timestamp"].max(), freq="h", tz="UTC")
    data = data.set_index("timestamp").reindex(full_index)
    data.index.name = "timestamp"
    inserted_missing = int(data["pv_power_mw"].isna().sum())
    data["is_missing"] = data["pv_power_mw"].isna()

    numeric_cols = [
        "pv_power_mw",
        "global_irradiance_wm2",
        "direct_irradiance_wm2",
        "diffuse_irradiance_wm2",
        "air_temperature_c",
        "wind_speed_ms",
        "solar_elevation_deg",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    negative_power = int((data["pv_power_mw"] < 0).sum())
    high_power = int((data["pv_power_mw"] > peak * 1.05).sum())
    negative_irradiance = int((data[["global_irradiance_wm2", "direct_irradiance_wm2", "diffuse_irradiance_wm2"]] < 0).sum().sum())

    data["pv_power_mw"] = data["pv_power_mw"].clip(lower=0, upper=peak * 1.05)
    for col in ["global_irradiance_wm2", "direct_irradiance_wm2", "diffuse_irradiance_wm2"]:
        data[col] = data[col].clip(lower=0)

    # Short gaps only: interpolate using past/current time direction, capped to 2 hours.
    for col in numeric_cols:
        data[col] = data[col].interpolate(method="time", limit=2, limit_direction="forward")

    data["data_source"] = data["data_source"].ffill().bfill()
    data["is_synthetic"] = data["is_synthetic"].ffill().bfill().fillna(False)
    data["data_quality_flag"] = np.where(data["is_missing"], "missing_or_interpolated", "ok")
    data["is_daylight"] = data["solar_elevation_deg"] >= daylight_threshold
    data.loc[~data["is_daylight"] & (data["solar_elevation_deg"].notna()), "pv_power_mw"] = data.loc[
        ~data["is_daylight"] & (data["solar_elevation_deg"].notna()), "pv_power_mw"
    ].clip(upper=0.02)

    out = data.reset_index()
    report.update(
        {
            "output_rows": int(len(out)),
            "invalid_timestamps": invalid_timestamps,
            "duplicate_timestamps": duplicates,
            "inserted_missing_hours": inserted_missing,
            "missing_rate": float(out["is_missing"].mean()),
            "negative_power_corrected": negative_power,
            "high_power_clipped": high_power,
            "negative_irradiance_corrected": negative_irradiance,
            "time_start": str(out["timestamp"].min()),
            "time_end": str(out["timestamp"].max()),
            "daylight_samples": int(out["is_daylight"].sum()),
            "night_samples": int((~out["is_daylight"]).sum()),
        }
    )
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.processed_dir / "clean_hourly.csv", index=False)
    write_json(paths.metrics_dir / "data_quality_report.json", report)
    LOGGER.info("Data cleaning complete: %s", report)
    return out
