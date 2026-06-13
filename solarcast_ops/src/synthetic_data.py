from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pvlib

from src.utils import clip_power

LOGGER = logging.getLogger(__name__)


def generate_synthetic_demo_data(config: dict[str, Any]) -> pd.DataFrame:
    """Generate deterministic hourly demo data with realistic solar patterns."""
    site = config["site"]
    data_cfg = config["data"]
    seed = int(config["project"]["random_seed"])
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(f"{data_cfg['start_year']}-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp(f"{data_cfg['end_year']}-12-31 23:00:00", tz="UTC")
    timestamps = pd.date_range(start, end, freq="h")

    location = pvlib.location.Location(
        latitude=site["latitude"],
        longitude=site["longitude"],
        tz="UTC",
        altitude=520,
    )
    clearsky = location.get_clearsky(timestamps, model="ineichen")
    solar_position = location.get_solarposition(timestamps)

    hour = timestamps.hour.to_numpy()
    day = timestamps.dayofyear.to_numpy()
    seasonal_cloud = 0.18 + 0.12 * np.cos(2 * np.pi * (day - 20) / 365.25)
    daily_cloud = rng.normal(0, 0.08, size=len(timestamps))
    cloud_factor = np.clip(0.78 - seasonal_cloud + daily_cloud, 0.18, 1.05)

    # Add coherent cloudy events so the demo has ramps and forecast failures.
    for center in pd.date_range(start + pd.Timedelta(days=20), end, freq="17D"):
        width = int(rng.integers(3, 9))
        center_idx = timestamps.get_indexer([center + pd.Timedelta(hours=int(rng.integers(8, 15)))], method="nearest")[0]
        lo = max(0, center_idx - width)
        hi = min(len(timestamps), center_idx + width)
        event = np.linspace(0.35, 0.05, hi - lo)
        cloud_factor[lo:hi] = np.minimum(cloud_factor[lo:hi], event)

    ghi = np.maximum(clearsky["ghi"].to_numpy() * cloud_factor + rng.normal(0, 18, len(timestamps)), 0)
    dni = np.maximum(clearsky["dni"].to_numpy() * np.clip(cloud_factor + 0.05, 0, 1.1), 0)
    dhi = np.maximum(clearsky["dhi"].to_numpy() * np.clip(1.25 - cloud_factor, 0.4, 1.8), 0)
    temp = 9 + 10 * np.sin(2 * np.pi * (day - 100) / 365.25) + 5 * np.sin(2 * np.pi * (hour - 7) / 24) + rng.normal(0, 2, len(timestamps))
    wind = np.clip(3.8 + 1.5 * rng.normal(size=len(timestamps)) + 0.6 * np.sin(2 * np.pi * day / 365.25), 0.2, 14)

    peak = float(site["peak_power_mw"])
    loss_factor = 1 - float(site["system_loss_percent"]) / 100
    temp_derate = 1 - 0.0035 * np.maximum(temp - 25, 0)
    pv_power = peak * loss_factor * (ghi / 1000.0) * temp_derate
    pv_power += rng.normal(0, 0.012, len(timestamps))
    pv_power = clip_power(pd.Series(pv_power), peak).to_numpy()
    pv_power[solar_position["elevation"].to_numpy() < 0] = 0.0

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pv_power_mw": pv_power,
            "global_irradiance_wm2": ghi,
            "direct_irradiance_wm2": dni,
            "diffuse_irradiance_wm2": dhi,
            "air_temperature_c": temp,
            "wind_speed_ms": wind,
            "solar_elevation_deg": solar_position["elevation"].to_numpy(),
            "data_source": "synthetic demo data",
            "is_synthetic": True,
        }
    )
    LOGGER.warning("Using generated demo data from %s to %s", df["timestamp"].min(), df["timestamp"].max())
    return df
