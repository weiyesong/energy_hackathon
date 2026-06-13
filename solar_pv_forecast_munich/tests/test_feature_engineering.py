"""Tests for leakage-safe supervised feature engineering and persistence baselines."""

from __future__ import annotations

import pandas as pd

from src.models.persistence_baseline import add_persistence_baselines
from src.preprocessing.feature_engineering import build_supervised_dataset, enabled_horizons_from_summary, split_chronologically


def _config() -> dict:
    """Return a minimal site config for feature tests."""
    return {
        "location": {"altitude": 520},
        "sites": [{"id": "munich_centre", "latitude": 48.137, "longitude": 11.575, "altitude": 520}],
    }


def _fused_fixture() -> pd.DataFrame:
    """Return hourly fused data with deliberately changing future values."""
    times = pd.date_range("2025-01-01 00:00:00", periods=6, freq="h", tz="Europe/Berlin")
    return pd.DataFrame(
        {
            "site_id": ["munich_centre"] * len(times),
            "timestamp": times,
            "latitude": [48.137] * len(times),
            "longitude": [11.575] * len(times),
            "nasa_power_ssi": [0.0, 10.0, 20.0, 300.0, 400.0, 500.0],
            "nasa_power_clear_sky_ssi": [1.0, 100.0, 100.0, 600.0, 700.0, 800.0],
            "openmeteo_ssi": [0.0, 11.0, 22.0, 333.0, 444.0, 555.0],
            "openmeteo_clear_sky_ssi": [1.0, 100.0, 100.0, 600.0, 700.0, 800.0],
            "openmeteo_clear_sky_ghi": [1.0, 100.0, 100.0, 600.0, 700.0, 800.0],
            "openmeteo_cloud_cover": [90.0, 80.0, 70.0, 10.0, 20.0, 30.0],
            "best_available_satellite_ssi": [0.0, 10.0, 20.0, 300.0, 400.0, 500.0],
            "best_satellite_source": ["nasa_power"] * len(times),
            "satellite_data_available": [True] * len(times),
            "satellite_clear_sky_index": [0.0, 0.1, 0.2, 0.5, 0.57, 0.62],
            "irradiance_source_std": [0.0, 0.7, 1.4, 23.3, 31.1, 38.9],
            "number_of_available_irradiance_sources": [2] * len(times),
        }
    )


def test_supervised_features_do_not_leak_future_target_into_issue_features() -> None:
    """Issue and lagged features must use t or earlier, while targets use t+h."""
    supervised = build_supervised_dataset(_fused_fixture(), _config(), [60])
    row = supervised[supervised["timestamp"].eq(pd.Timestamp("2025-01-01 02:00:00", tz="Europe/Berlin"))].iloc[0]

    assert row["target_valid_time"] == pd.Timestamp("2025-01-01 03:00:00", tz="Europe/Berlin")
    assert row["ghi_issue"] == 22.0
    assert row["satellite_ssi_issue"] == 20.0
    assert row["ghi_lag_1"] == 11.0
    assert row["ghi_lag_2"] == 0.0
    assert row["GHI_target"] == 333.0
    assert row["target_source"] == "openmeteo_shortwave_proxy"
    assert row["target_cloud_cover_forecast_proxy"] == 10.0
    assert row["cloud_cover_issue"] == 70.0


def test_persistence_baselines_use_issue_time_values_and_target_clear_sky() -> None:
    """Naive and CSI persistence should be constrained physical baselines."""
    df = pd.DataFrame(
        {
            "ghi_issue": [50.0],
            "issue_clear_sky_ghi": [100.0],
            "target_GHI_clear": [400.0],
            "target_solar_elevation": [30.0],
        }
    )

    out = add_persistence_baselines(df)

    assert out.loc[0, "GHI_persistence_naive"] == 50.0
    assert out.loc[0, "GHI_persistence_csi"] == 200.0


def test_enabled_horizons_respect_fusion_summary_support(tmp_path) -> None:
    """15 and 30 minute horizons should be disabled unless source resolution supports them."""
    summary = tmp_path / "fusion_summary.json"
    summary.write_text('{"horizon_support":{"15_minutes":{"supported":false},"30_minutes":{"supported":true}}}', encoding="utf-8")

    assert enabled_horizons_from_summary(summary) == [30, 60, 180, 360, 720, 1440]


def test_short_available_range_uses_fallback_chronological_split() -> None:
    """Short data ranges should still produce chronological train/validation/test splits."""
    supervised = build_supervised_dataset(_fused_fixture(), _config(), [60])
    train, validation, test, metadata = split_chronologically(supervised)

    assert metadata["split_strategy"] == "fallback_60_20_20_chronological"
    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    assert train["timestamp"].max() <= validation["timestamp"].min()
    assert validation["timestamp"].max() <= test["timestamp"].min()
