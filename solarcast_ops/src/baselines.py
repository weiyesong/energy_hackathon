from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_paths
from src.utils import clip_power, safe_divide

LOGGER = logging.getLogger(__name__)


def add_baseline_predictions(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create ordinary and clear-sky persistence forecasts."""
    paths = get_paths()
    data = df.copy()
    peak = float(config["site"]["peak_power_mw"])
    horizons = [int(h) for h in config["forecast"]["horizons_hours"]]

    cloud_factor = safe_divide(data["pv_power_mw"].to_numpy(), data["clear_sky_power_mw"].to_numpy())
    cloud_factor = np.clip(np.nan_to_num(cloud_factor, nan=0.0, posinf=0.0, neginf=0.0), 0, 1.3)

    for h in horizons:
        data[f"pred_persistence_h{h}"] = clip_power(data["pv_power_mw"], peak)
        csp = cloud_factor * data[f"clear_sky_power_h{h}"].fillna(0).to_numpy()
        csp = np.where(data["clear_sky_power_mw"].fillna(0).to_numpy() < 0.01, 0.0, csp)
        data[f"pred_clear_sky_persistence_h{h}"] = clip_power(csp, peak)

    cols = ["timestamp"] + [c for c in data.columns if c.startswith("pred_")]
    data[cols].to_csv(paths.predictions_dir / "baseline_predictions.csv", index=False)
    LOGGER.info("Saved baseline predictions for horizons: %s", horizons)
    return data
