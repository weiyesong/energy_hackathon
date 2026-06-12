from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_paths
from src.feature_engineering import SATELLITE_FEATURES
from src.train import load_model, split_time_series
from src.utils import write_json

LOGGER = logging.getLogger(__name__)


def classify_weather(df: pd.DataFrame) -> pd.Series:
    """Heuristic sky-condition labels, not strict meteorological classes."""
    csi = df["clear_sky_index"].fillna(0)
    variability = df["global_irradiance_wm2"].diff().abs().rolling(3, min_periods=1).mean().fillna(0)
    labels = pd.Series("variable_cloud", index=df.index)
    labels[(csi >= 0.72) & (variability < 140)] = "clear"
    labels[csi < 0.35] = "overcast"
    return labels


def metric_row(y_true: pd.Series, y_pred: pd.Series, peak: float, baseline_p: pd.Series, baseline_csp: pd.Series) -> dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "nmae": np.nan, "nrmse": np.nan, "r2": np.nan, "skill_vs_persistence": np.nan, "skill_vs_clear_sky_persistence": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae_p = mean_absolute_error(yt, baseline_p[mask])
    mae_csp = mean_absolute_error(yt, baseline_csp[mask])
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "nmae": float(mae / peak),
        "nrmse": float(rmse / peak),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "skill_vs_persistence": float(1 - mae / mae_p) if mae_p > 0 else np.nan,
        "skill_vs_clear_sky_persistence": float(1 - mae / mae_csp) if mae_csp > 0 else np.nan,
    }


def _plot_outputs(preds: pd.DataFrame, metrics: pd.DataFrame, config: dict[str, Any]) -> None:
    paths = get_paths()
    sample = preds.head(240)
    plt.figure(figsize=(12, 5))
    plt.plot(sample["timestamp"], sample["actual_h1"], label="Actual +1h", linewidth=1.8)
    plt.plot(sample["timestamp"], sample["pred_persistence_h1"], label="Persistence", alpha=0.8)
    plt.plot(sample["timestamp"], sample["pred_clear_sky_persistence_h1"], label="Clear-sky persistence", alpha=0.8)
    plt.plot(sample["timestamp"], sample["history_only_h1"], label="History-only ML", alpha=0.9)
    plt.plot(sample["timestamp"], sample["satellite_informed_h1"], label="Satellite-informed ML", alpha=0.9)
    plt.title("Forecast Comparison, Horizon +1h")
    plt.ylabel("Power (MW)")
    plt.xlabel("Replay Time (UTC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "forecast_comparison.png", dpi=160)
    plt.close()

    daylight = metrics[metrics["segment"] == "daylight"]
    plt.figure(figsize=(9, 5))
    for model in daylight["model"].unique():
        subset = daylight[daylight["model"] == model]
        plt.plot(subset["horizon_h"], subset["mae"], marker="o", label=model)
    plt.title("Daylight MAE by Horizon")
    plt.ylabel("MAE (MW)")
    plt.xlabel("Forecast Horizon (h)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "metrics_by_horizon.png", dpi=160)
    plt.close()

    weather = metrics[(metrics["segment"].isin(["clear", "variable_cloud", "overcast"])) & (metrics["model"] == "satellite_informed")]
    plt.figure(figsize=(9, 5))
    for condition in weather["segment"].unique():
        subset = weather[weather["segment"] == condition]
        plt.plot(subset["horizon_h"], subset["skill_vs_persistence"], marker="o", label=condition)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Satellite-Informed Forecast Skill by Weather Condition")
    plt.ylabel("Skill vs Persistence (MAE)")
    plt.xlabel("Forecast Horizon (h)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "skill_by_weather_condition.png", dpi=160)
    plt.close()

    for h in config["forecast"]["horizons_hours"]:
        model_obj = load_model(paths.models_dir / f"model_satellite_informed_h{h}.pkl")
        model = model_obj["model"]
        features = model_obj["features"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(features))
        top = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False).head(15)
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title(f"Feature Importance, Satellite-Informed +{h}h")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(paths.figures_dir / f"feature_importance_h{h}.png", dpi=160)
        plt.close()


def evaluate_models(features_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Evaluate baselines and ML forecasts on test data."""
    paths = get_paths()
    _, _, test_df, split_meta = split_time_series(features_df, config)
    base = pd.read_csv(paths.predictions_dir / "baseline_predictions.csv")
    ml = pd.read_csv(paths.predictions_dir / "ml_test_predictions.csv")
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    ml["timestamp"] = pd.to_datetime(ml["timestamp"], utc=True)
    test = test_df.copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
    preds = test.merge(base, on="timestamp", how="left").merge(ml, on="timestamp", how="left", suffixes=("", "_ml"))
    preds["weather_condition"] = classify_weather(preds)

    peak = float(config["site"]["peak_power_mw"])
    rows = []
    for h in config["forecast"]["horizons_hours"]:
        target = f"target_h{h}"
        model_cols = {
            "persistence": f"pred_persistence_h{h}",
            "clear_sky_persistence": f"pred_clear_sky_persistence_h{h}",
            "history_only": f"history_only_h{h}",
            "satellite_informed": f"satellite_informed_h{h}",
        }
        segments = {
            "all": preds.index == preds.index,
            "daylight": preds["is_daylight"].astype(bool),
            "clear": preds["weather_condition"] == "clear",
            "variable_cloud": preds["weather_condition"] == "variable_cloud",
            "overcast": preds["weather_condition"] == "overcast",
        }
        for segment, mask in segments.items():
            for model_name, pred_col in model_cols.items():
                row = metric_row(
                    preds.loc[mask, target],
                    preds.loc[mask, pred_col],
                    peak,
                    preds.loc[mask, f"pred_persistence_h{h}"],
                    preds.loc[mask, f"pred_clear_sky_persistence_h{h}"],
                )
                row.update({"horizon_h": h, "model": model_name, "segment": segment, "samples": int(mask.sum())})
                rows.append(row)

    metrics = pd.DataFrame(rows)
    context_cols = [
        "timestamp",
        "pv_power_mw",
        "global_irradiance_wm2",
        "direct_irradiance_wm2",
        "diffuse_irradiance_wm2",
        "air_temperature_c",
        "wind_speed_ms",
        "solar_elevation_deg",
        "clear_sky_index",
        "is_daylight",
        "data_source",
        "is_synthetic",
        "weather_condition",
    ]
    pred_cols = context_cols + [
        c for c in preds.columns if c.startswith(("target_h", "pred_", "history_only_h", "satellite_informed_h", "actual_h"))
    ]
    preds[pred_cols].to_csv(paths.predictions_dir / "test_predictions.csv", index=False)
    metrics.to_csv(paths.metrics_dir / "metrics.csv", index=False)
    write_json(paths.metrics_dir / "metrics.json", json.loads(metrics.to_json(orient="records")))
    write_json(paths.metrics_dir / "split_metadata.json", split_meta)
    _plot_outputs(preds, metrics, config)
    LOGGER.info("Saved metrics and figures to %s", paths.reports_dir)
    return metrics
