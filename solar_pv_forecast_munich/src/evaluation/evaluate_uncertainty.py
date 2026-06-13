"""Evaluate calibrated probabilistic GHI forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    """Compute mean pinball loss for one quantile."""
    y = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    mask = y.notna() & pred.notna()
    if not mask.any():
        return float("nan")
    error = y[mask] - pred[mask]
    return float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())


def interval_score(y_true: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float = 0.20) -> float:
    """Compute central interval score for an alpha-level prediction interval."""
    y = pd.to_numeric(y_true, errors="coerce")
    lo = pd.to_numeric(lower, errors="coerce")
    hi = pd.to_numeric(upper, errors="coerce")
    mask = y.notna() & lo.notna() & hi.notna()
    if not mask.any():
        return float("nan")
    width = hi[mask] - lo[mask]
    lower_penalty = (2.0 / alpha) * (lo[mask] - y[mask]).clip(lower=0.0)
    upper_penalty = (2.0 / alpha) * (y[mask] - hi[mask]).clip(lower=0.0)
    return float((width + lower_penalty + upper_penalty).mean())


def evaluate_uncertainty(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate calibrated interval quality by horizon and cloud regime."""
    metrics = []
    regimes = []
    for horizon, group in predictions.groupby("horizon_minutes", sort=True):
        metrics.append({"horizon_minutes": int(horizon), **_interval_metrics(group)})
        for regime, regime_group in group.assign(cloud_regime=group["target_cloud_cover_forecast_proxy"].map(_cloud_regime)).groupby("cloud_regime"):
            regimes.append({"horizon_minutes": int(horizon), "cloud_regime": regime, **_interval_metrics(regime_group)})
    return pd.DataFrame(metrics), pd.DataFrame(regimes)


def save_uncertainty_outputs(metrics: pd.DataFrame, regimes: pd.DataFrame, output_dir: str | Path) -> None:
    """Save uncertainty metric CSV files."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path / "uncertainty_metrics.csv", index=False)
    regimes.to_csv(path / "coverage_by_cloud_regime.csv", index=False)


def main() -> None:
    """Evaluate saved probabilistic predictions from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate calibrated probabilistic GHI forecasts.")
    parser.add_argument("--predictions", default="outputs/forecasts/test_probabilistic_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/metrics")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    metrics, regimes = evaluate_uncertainty(predictions)
    save_uncertainty_outputs(metrics, regimes, args.output_dir)
    print(f"Saved uncertainty metrics to {args.output_dir}")


def _interval_metrics(group: pd.DataFrame) -> dict[str, Any]:
    """Compute interval metrics for one grouped dataframe."""
    y = pd.to_numeric(group["GHI_target"], errors="coerce")
    p10 = pd.to_numeric(group["GHI_P10_calibrated"], errors="coerce")
    p50 = pd.to_numeric(group["GHI_P50"], errors="coerce")
    p90 = pd.to_numeric(group["GHI_P90_calibrated"], errors="coerce")
    mask = y.notna() & p10.notna() & p50.notna() & p90.notna()
    if not mask.any():
        return {
            "picp": np.nan,
            "mpiw": np.nan,
            "normalized_mpiw": np.nan,
            "pinball_loss_p10": np.nan,
            "pinball_loss_p50": np.nan,
            "pinball_loss_p90": np.nan,
            "interval_score": np.nan,
            "samples": 0,
        }
    width = (p90[mask] - p10[mask]).clip(lower=0.0)
    scale = max(float(y[mask].max() - y[mask].min()), float(y[mask].mean()), 1.0)
    return {
        "picp": float(((y[mask] >= p10[mask]) & (y[mask] <= p90[mask])).mean()),
        "mpiw": float(width.mean()),
        "normalized_mpiw": float(width.mean() / scale),
        "pinball_loss_p10": pinball_loss(y[mask], p10[mask], 0.10),
        "pinball_loss_p50": pinball_loss(y[mask], p50[mask], 0.50),
        "pinball_loss_p90": pinball_loss(y[mask], p90[mask], 0.90),
        "interval_score": interval_score(y[mask], p10[mask], p90[mask], alpha=0.20),
        "samples": int(mask.sum()),
    }


def _cloud_regime(value: Any) -> str:
    """Map cloud-cover percentage to a coarse cloud regime."""
    if pd.isna(value):
        return "unknown"
    cloud = float(value)
    if cloud < 30.0:
        return "clear"
    if cloud < 70.0:
        return "partly_cloudy"
    return "cloudy"


if __name__ == "__main__":
    main()
