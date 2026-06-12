from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline_schedule import (
    build_next_24h_schedule,
    predict_baseline_lgbm,
    predict_baseline_statistical,
    predict_baseline_statistical_with_metadata,
)


def _sample_power_frame() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", "2023-12-31 23:45", freq="15min", tz="UTC")
    minute_of_day = idx.hour * 60 + idx.minute
    solar_shape = np.maximum(0, np.sin(np.pi * (minute_of_day - 6 * 60) / (12 * 60)))
    seasonal_wave = np.asarray(np.sin(2 * np.pi * (idx.dayofyear - 80) / 365.25))
    seasonal = 0.65 + 0.35 * np.clip(seasonal_wave, 0, None)
    year_factor = 1 + (idx.year - 2020) * 0.02
    power = solar_shape * seasonal * year_factor
    return pd.DataFrame({"datetime": idx, "power_actual": power})


def test_statistical_baseline_outputs_15_minute_day() -> None:
    df = _sample_power_frame()
    pred = predict_baseline_statistical(df, "2023-06-15")
    assert len(pred) == 96
    assert {"timestamp", "p_base_mw", "alpha", "schedule_model_a_mw"}.issubset(pred.columns)
    assert pred["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert pred["schedule_model_a_mw"].notna().all()
    assert (pred["schedule_model_a_mw"] >= 0).all()


def test_statistical_baseline_handles_leap_day_target() -> None:
    df = _sample_power_frame()
    result = predict_baseline_statistical_with_metadata(df, "2024-02-29")
    assert len(result.schedule) == 96
    assert result.schedule["schedule_model_a_mw"].notna().all()
    assert "alpha_source" in result.metadata


def test_next_24h_schedule_has_97_points() -> None:
    df = _sample_power_frame()
    result = build_next_24h_schedule(df, "2023-06-15 10:15", schedule_model="Model A")
    assert len(result.schedule) == 97
    assert result.schedule["horizon_hours"].iloc[0] == 0
    assert result.schedule["horizon_hours"].iloc[-1] == 24
    assert result.schedule["scheduled_power_mw"].notna().all()


def test_lgbm_baseline_predicts_without_future_target_rows() -> None:
    df = _sample_power_frame()
    pred = predict_baseline_lgbm(df, "2023-06-15")
    assert len(pred) == 96
    assert pred["schedule_model_b_mw"].notna().all()
    assert (pred["schedule_model_b_mw"] >= 0).all()
