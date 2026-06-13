"""Benchmark evaluation for deterministic hybrid irradiance forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {
    "physical_baseline": "GHI_phys_target",
    "naive_persistence": "GHI_persistence_naive",
    "csi_persistence": "GHI_persistence_csi",
    "hybrid_satellite": "GHI_pred_hybrid",
    "hybrid_without_satellite": "GHI_pred_hybrid_without_satellite",
}


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, daylight_mask: pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, MBE, R2, and daylight nRMSE for one prediction series."""
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    mask = truth.notna() & pred.notna()
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "mbe": np.nan, "r2": np.nan, "daylight_nrmse": np.nan}

    errors = pred[mask] - truth[mask]
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mbe = float(errors.mean())
    denominator = float(np.sum(np.square(truth[mask] - truth[mask].mean())))
    r2 = float(1.0 - np.sum(np.square(errors)) / denominator) if denominator > 0 else np.nan

    daylight = mask & daylight_mask.fillna(False)
    if daylight.any():
        daylight_errors = pred[daylight] - truth[daylight]
        daylight_rmse = float(np.sqrt(np.mean(np.square(daylight_errors))))
        daylight_scale = float(truth[daylight].max() - truth[daylight].min())
        if daylight_scale <= 0:
            daylight_scale = float(truth[daylight].mean())
        daylight_nrmse = daylight_rmse / daylight_scale if daylight_scale > 0 else np.nan
    else:
        daylight_nrmse = np.nan

    return {"mae": mae, "rmse": rmse, "mbe": mbe, "r2": r2, "daylight_nrmse": float(daylight_nrmse)}


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all benchmark and hybrid prediction columns by horizon."""
    rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for horizon, group in predictions.groupby("horizon_minutes", sort=True):
        daylight = pd.to_numeric(group["target_solar_elevation"], errors="coerce") > 0.0
        horizon_metrics: dict[str, dict[str, float]] = {}
        for model_name, column in PREDICTION_COLUMNS.items():
            if column not in group.columns:
                continue
            metrics = compute_metrics(group["GHI_target"], group[column], daylight)
            horizon_metrics[model_name] = metrics
            rows.append({"horizon_minutes": int(horizon), "model": model_name, **metrics})

        hybrid_rmse = horizon_metrics.get("hybrid_satellite", {}).get("rmse", np.nan)
        csi_rmse = horizon_metrics.get("csi_persistence", {}).get("rmse", np.nan)
        skill = 1.0 - hybrid_rmse / csi_rmse if csi_rmse and np.isfinite(csi_rmse) and csi_rmse > 0 else np.nan
        skill_rows.append(
            {
                "horizon_minutes": int(horizon),
                "rmse_hybrid": hybrid_rmse,
                "rmse_csi_persistence": csi_rmse,
                "skill_score_vs_csi_persistence": float(skill) if np.isfinite(skill) else np.nan,
                "beats_persistence": bool(np.isfinite(skill) and skill > 0.0),
            }
        )

        rmse_without = horizon_metrics.get("hybrid_without_satellite", {}).get("rmse", np.nan)
        value_add = rmse_without - hybrid_rmse if np.isfinite(rmse_without) and np.isfinite(hybrid_rmse) else np.nan
        ablation_rows.append(
            {
                "horizon_minutes": int(horizon),
                "rmse_without_satellite": rmse_without,
                "rmse_with_satellite": hybrid_rmse,
                "satellite_value_add": float(value_add) if np.isfinite(value_add) else np.nan,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(skill_rows), pd.DataFrame(ablation_rows)


def save_evaluation_outputs(
    benchmark_metrics: pd.DataFrame,
    skill: pd.DataFrame,
    ablation: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save benchmark metric CSV outputs."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    benchmark_metrics.to_csv(path / "benchmark_metrics.csv", index=False)
    skill.to_csv(path / "forecast_skill_by_horizon.csv", index=False)
    ablation.to_csv(path / "satellite_ablation.csv", index=False)


def main() -> None:
    """Evaluate saved hybrid predictions from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate deterministic hybrid benchmark predictions.")
    parser.add_argument("--predictions", default="outputs/forecasts/test_hybrid_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/metrics")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    benchmark_metrics, skill, ablation = evaluate_predictions(predictions)
    save_evaluation_outputs(benchmark_metrics, skill, ablation, args.output_dir)
    print(f"Saved benchmark metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
