"""Tests for probabilistic quantile forecasts and conformal calibration."""

from __future__ import annotations

import pandas as pd

from src.evaluation.evaluate_uncertainty import evaluate_uncertainty, pinball_loss
from src.models.conformal_calibration import apply_conformal_interval, fit_split_conformal_interval
from src.models.train_quantile_forecast import _constrain_calibrated_quantiles, _uncertainty_level


def test_split_conformal_uses_validation_residuals() -> None:
    """Conformal expansion should be fitted from validation interval misses."""
    validation = pd.DataFrame(
        {
            "GHI_target": [10.0, 20.0, 100.0],
            "GHI_P10_raw": [8.0, 22.0, 80.0],
            "GHI_P90_raw": [12.0, 30.0, 90.0],
        }
    )

    expansion = fit_split_conformal_interval(validation, target_coverage=0.80)
    calibrated = apply_conformal_interval(validation, expansion)

    assert expansion >= 2.0
    assert "GHI_P10_calibrated" in calibrated
    assert "GHI_P90_calibrated" in calibrated


def test_calibrated_quantile_constraints() -> None:
    """Calibrated quantiles should be monotonic, non-negative, capped, and zero at night."""
    df = pd.DataFrame(
        {
            "GHI_P10_calibrated": [90.0, 10.0, -5.0],
            "GHI_P50": [50.0, 40.0, 10.0],
            "GHI_P90_calibrated": [30.0, 200.0, 20.0],
            "target_solar_elevation": [20.0, 20.0, -1.0],
            "target_GHI_clear": [100.0, 100.0, 100.0],
        }
    )

    out = _constrain_calibrated_quantiles(df)

    assert (out["GHI_P10_calibrated"] <= out["GHI_P50"]).all()
    assert (out["GHI_P50"] <= out["GHI_P90_calibrated"]).all()
    assert out.loc[1, "GHI_P90_calibrated"] == 120.0
    assert out.loc[2, ["GHI_P10_calibrated", "GHI_P50", "GHI_P90_calibrated"]].tolist() == [0.0, 0.0, 0.0]


def test_uncertainty_metrics_and_levels() -> None:
    """Uncertainty evaluation should produce interval coverage and UI levels."""
    predictions = pd.DataFrame(
        {
            "horizon_minutes": [60, 60, 60],
            "GHI_target": [50.0, 100.0, 150.0],
            "GHI_P10_calibrated": [40.0, 90.0, 100.0],
            "GHI_P50": [55.0, 100.0, 140.0],
            "GHI_P90_calibrated": [70.0, 120.0, 200.0],
            "target_cloud_cover_forecast_proxy": [10.0, 50.0, 90.0],
        }
    )

    predictions["uncertainty_level"] = _uncertainty_level(predictions)
    metrics, regimes = evaluate_uncertainty(predictions)

    assert set(predictions["uncertainty_level"]).issubset({"Low", "Medium", "High"})
    assert metrics.loc[0, "picp"] == 1.0
    assert metrics.loc[0, "mpiw"] > 0
    assert pinball_loss(predictions["GHI_target"], predictions["GHI_P50"], 0.5) >= 0
    assert set(regimes["cloud_regime"]) == {"clear", "partly_cloudy", "cloudy"}
