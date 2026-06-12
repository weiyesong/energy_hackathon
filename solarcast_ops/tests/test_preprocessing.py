from __future__ import annotations

import pandas as pd

from src.preprocessing import preprocess_data


def _config():
    return {
        "site": {"peak_power_mw": 1.0},
        "forecast": {"daylight_solar_elevation_threshold": 3},
    }


def test_preprocessing_sorts_deduplicates_and_clips_negative_power(tmp_path, monkeypatch):
    import src.preprocessing as pp

    class Paths:
        processed_dir = tmp_path
        metrics_dir = tmp_path

    monkeypatch.setattr(pp, "get_paths", lambda: Paths())
    df = pd.DataFrame(
        {
            "timestamp": ["2023-01-01 02:00Z", "2023-01-01 01:00Z", "2023-01-01 01:00Z"],
            "pv_power_mw": [0.5, -0.2, 0.4],
            "global_irradiance_wm2": [100, -5, 90],
            "direct_irradiance_wm2": [50, -2, 45],
            "diffuse_irradiance_wm2": [20, -1, 19],
            "air_temperature_c": [5, 4, 4],
            "wind_speed_ms": [2, 2, 2],
            "solar_elevation_deg": [10, 8, 8],
            "data_source": ["x", "x", "x"],
            "is_synthetic": [False, False, False],
        }
    )
    out = preprocess_data(df, _config())
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].duplicated().sum() == 0
    assert out["pv_power_mw"].min() >= 0
    assert out["global_irradiance_wm2"].min() >= 0


def test_power_unit_is_mw_not_w_after_preprocessing(tmp_path, monkeypatch):
    import src.preprocessing as pp

    class Paths:
        processed_dir = tmp_path
        metrics_dir = tmp_path

    monkeypatch.setattr(pp, "get_paths", lambda: Paths())
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=2, freq="h", tz="UTC"),
            "pv_power_mw": [0.2, 2.0],
            "global_irradiance_wm2": [0, 500],
            "direct_irradiance_wm2": [0, 300],
            "diffuse_irradiance_wm2": [0, 100],
            "air_temperature_c": [4, 5],
            "wind_speed_ms": [2, 2],
            "solar_elevation_deg": [0, 20],
            "data_source": ["x", "x"],
            "is_synthetic": [False, False],
        }
    )
    out = preprocess_data(df, _config())
    assert out["pv_power_mw"].max() <= 1.05
