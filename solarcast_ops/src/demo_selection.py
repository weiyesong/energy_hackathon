from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.config import get_paths
from src.utils import write_json

LOGGER = logging.getLogger(__name__)


def select_demo_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Select replay cases from test predictions only."""
    paths = get_paths()
    preds = pd.read_csv(paths.predictions_dir / "test_predictions_with_uncertainty.csv")
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
    peak = float(config["site"]["peak_power_mw"])
    daylight = preds[preds["is_daylight"].astype(bool)].copy()

    clear = daylight.sort_values(["clear_sky_index", "pv_power_mw"], ascending=False).head(48)
    clear_row = clear.iloc[len(clear) // 2] if len(clear) else daylight.iloc[0]

    variable = daylight.copy()
    variable["spread"] = (variable["pred_persistence_h1"] - variable["satellite_informed_h1"]).abs()
    variable_row = variable.sort_values("spread", ascending=False).iloc[0]

    ramp = daylight.copy()
    ramp["future_min"] = ramp[["target_h1", "target_h2", "target_h3"]].min(axis=1)
    ramp["drop_fraction"] = (ramp["pv_power_mw"] - ramp["future_min"]) / peak
    ramp_row = ramp.sort_values("drop_fraction", ascending=False).iloc[0]

    cases = [
        {
            "name": "Clear day",
            "timestamp": clear_row["timestamp"].isoformat(),
            "description": "High clear-sky index and stable production.",
        },
        {
            "name": "Variable-cloud day",
            "timestamp": variable_row["timestamp"].isoformat(),
            "description": "Large disagreement between persistence and satellite-informed model.",
        },
        {
            "name": "Large downward ramp event",
            "timestamp": ramp_row["timestamp"].isoformat(),
            "description": "Largest future 1-3h generation drop found in the test set.",
        },
    ]
    write_json(paths.demo_dir / "demo_cases.json", cases)
    LOGGER.info("Saved demo cases to %s", paths.demo_dir / "demo_cases.json")
    return cases
