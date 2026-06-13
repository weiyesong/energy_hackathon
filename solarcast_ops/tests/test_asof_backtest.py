from __future__ import annotations

import pandas as pd

from src.asof_backtest import run_asof_backtest
from src.config import load_config


def test_asof_backtest_blocks_future_labels() -> None:
    config = load_config()
    features = pd.read_csv("data/processed/features.csv")
    result = run_asof_backtest(
        features,
        config,
        horizons=[1, 2],
        asof_times=["2023-05-14 13:10:00+00:00"],
        persist_outputs=False,
    )
    assert not result.predictions.empty
    for row in result.predictions.itertuples(index=False):
        cutoff = pd.Timestamp(row.train_label_cutoff)
        asof = pd.Timestamp(row.asof_time)
        assert cutoff <= asof - pd.Timedelta(hours=int(row.horizon_h))
    assert {
        "cloud_opacity_proxy",
        "cloud_variability_proxy",
        "cloud_trend_proxy",
        "wind_advected_cloud_change_proxy",
        "cloud_ramp_risk_proxy",
        "wind_speed_ms",
        "pred_ghi_wm2",
    }.issubset(result.predictions.columns)
