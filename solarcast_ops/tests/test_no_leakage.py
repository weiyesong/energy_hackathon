from __future__ import annotations

import pandas as pd

from src.feature_engineering import build_features
from src.train import split_time_series


def _config():
    return {
        "project": {"random_seed": 42},
        "site": {"latitude": 48.14, "longitude": 11.58, "peak_power_mw": 1.0, "system_loss_percent": 14},
        "forecast": {
            "horizons_hours": [1, 2, 3],
            "daylight_solar_elevation_threshold": 3,
            "train_end": "2021-12-31 23:00:00",
            "validation_end": "2022-12-31 23:00:00",
            "test_end": "2023-12-31 23:00:00",
        },
    }


def test_lag_and_rolling_features_use_only_past(tmp_path, monkeypatch):
    import src.feature_engineering as fe

    class Paths:
        processed_dir = tmp_path

    monkeypatch.setattr(fe, "get_paths", lambda: Paths())
    ts = pd.date_range("2023-01-01", periods=12, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "pv_power_mw": range(12),
            "global_irradiance_wm2": [x * 10 for x in range(12)],
            "direct_irradiance_wm2": [x * 5 for x in range(12)],
            "diffuse_irradiance_wm2": [x * 2 for x in range(12)],
            "air_temperature_c": [5] * 12,
            "wind_speed_ms": [2] * 12,
            "solar_elevation_deg": [10] * 12,
            "is_daylight": [True] * 12,
            "data_source": ["x"] * 12,
            "is_synthetic": [False] * 12,
        }
    )
    out = build_features(df, _config())
    row = out.iloc[0]
    original = df.set_index("timestamp")
    t = row["timestamp"]
    assert row["power_lag_1h"] == original.loc[t - pd.Timedelta(hours=1), "pv_power_mw"]
    past_values = original.loc[t - pd.Timedelta(hours=3) : t - pd.Timedelta(hours=1), "pv_power_mw"]
    assert row["power_rolling_mean_3h"] == past_values.mean()


def test_time_split_order_has_no_overlap():
    ts = pd.date_range("2019-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "x": range(len(ts))})
    train, val, test, _ = split_time_series(df, _config())
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
