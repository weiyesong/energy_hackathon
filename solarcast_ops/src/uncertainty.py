from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_paths
from src.utils import clip_power, write_json

LOGGER = logging.getLogger(__name__)


def build_prediction_intervals(config: dict[str, Any]) -> pd.DataFrame:
    """Create empirical P10/P50/P90 prediction intervals from validation residuals."""
    paths = get_paths()
    peak = float(config["site"]["peak_power_mw"])
    preds = pd.read_csv(paths.predictions_dir / "test_predictions.csv")
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
    summary = []

    for h in config["forecast"]["horizons_hours"]:
        val_path = paths.predictions_dir / f"validation_predictions_h{h}.csv"
        val = pd.read_csv(val_path)
        actual_col = f"actual_h{h}"
        pred_col = f"satellite_informed_h{h}"
        residual = val[actual_col] - val[pred_col]
        q10 = float(residual.quantile(0.1))
        q90 = float(residual.quantile(0.9))
        p50 = preds[pred_col]
        preds[f"forecast_p10_h{h}"] = clip_power(p50 + q10, peak)
        preds[f"forecast_p50_h{h}"] = clip_power(p50, peak)
        preds[f"forecast_p90_h{h}"] = clip_power(p50 + q90, peak)
        ordered = np.sort(preds[[f"forecast_p10_h{h}", f"forecast_p50_h{h}", f"forecast_p90_h{h}"]].to_numpy(), axis=1)
        preds[[f"forecast_p10_h{h}", f"forecast_p50_h{h}", f"forecast_p90_h{h}"]] = ordered
        preds[f"interval_width_h{h}"] = preds[f"forecast_p90_h{h}"] - preds[f"forecast_p10_h{h}"]
        preds[f"uncertainty_level_h{h}"] = pd.cut(
            preds[f"interval_width_h{h}"],
            bins=[-0.001, peak * 0.08, peak * 0.18, peak * 2],
            labels=["low", "medium", "high"],
        ).astype(str)
        actual = preds[f"target_h{h}"]
        coverage = ((actual >= preds[f"forecast_p10_h{h}"]) & (actual <= preds[f"forecast_p90_h{h}"])).mean()
        summary.append(
            {
                "horizon_h": h,
                "method": "validation residual empirical prediction interval",
                "empirical_coverage": float(coverage),
                "mean_interval_width_mw": float(preds[f"interval_width_h{h}"].mean()),
                "residual_q10_mw": q10,
                "residual_q90_mw": q90,
            }
        )

    preds.to_csv(paths.predictions_dir / "test_predictions_with_uncertainty.csv", index=False)
    write_json(paths.metrics_dir / "uncertainty_metrics.json", summary)
    LOGGER.info("Saved empirical prediction intervals")
    return preds
