from __future__ import annotations

import pandas as pd

from app.dashboard import _decision_window_from_live, _decision_window_from_replay
from src.decision_engine import make_trading_decision


def test_live_decision_window_uses_delivery_duration_not_elapsed_horizon() -> None:
    live = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-13", periods=97, freq="15min", tz="UTC"),
            "horizon_hours": [i * 0.25 for i in range(97)],
            "scheduled_power_mw": [10.0] * 97,
            "forecast_p10_mw": [7.0] * 97,
            "forecast_p50_mw": [8.0] * 97,
            "forecast_p90_mw": [9.0] * 97,
        }
    )

    early = _decision_window_from_live(live, 1.0, 1.0, False, 0.0)
    late = _decision_window_from_live(live, 12.0, 1.0, False, 0.0)
    early_decision = make_trading_decision(early[0], early[1], early[2], early[3], early[4], 90, 70, 150, 40, 0, 1, "Balanced")
    late_decision = make_trading_decision(late[0], late[1], late[2], late[3], late[4], 90, 70, 150, 40, 0, 1, "Balanced")

    assert early[4] == 1.0
    assert late[4] == 1.0
    assert early_decision.expected_deviation_mwh == late_decision.expected_deviation_mwh == -2.0


def test_replay_decision_window_falls_back_when_target_forecast_missing() -> None:
    row = pd.Series(
        {
            "pv_power_mw": 5.0,
            "forecast_p10_h1": 4.0,
            "forecast_p50_h1": 5.0,
            "forecast_p90_h1": 6.0,
            "forecast_p10_h2": pd.NA,
            "forecast_p50_h2": pd.NA,
            "forecast_p90_h2": pd.NA,
            "forecast_p10_h3": 6.0,
            "forecast_p50_h3": 7.0,
            "forecast_p90_h3": 8.0,
        }
    )
    schedule = pd.DataFrame({"horizon_hours": [1.0, 2.0, 3.0], "scheduled_power_mw": [5.0, 5.5, 6.0]})

    scheduled, p10, p50, p90, duration = _decision_window_from_replay(row, schedule, 2.0, 1.0, [1, 2, 3], 5.0, False, 0.0)

    assert duration == 1.0
    assert scheduled == 5.5
    assert p10 <= p50 <= p90
    assert all(pd.notna(v) for v in [p10, p50, p90])
