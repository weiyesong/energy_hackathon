"""Tests for deterministic hybrid residual model helpers."""

from __future__ import annotations

import pandas as pd

from src.models.train_hybrid_lightgbm import _apply_physical_constraints, build_feature_encoder


def test_feature_encoder_excludes_targets_persistence_and_time_leakage() -> None:
    """Model features must not include targets, future target times, or persistence predictions."""
    df = pd.DataFrame(
        {
            "site_id": ["munich_centre"],
            "timestamp": [pd.Timestamp("2025-01-01", tz="Europe/Berlin")],
            "target_valid_time": [pd.Timestamp("2025-01-01 01:00", tz="Europe/Berlin")],
            "GHI_target": [100.0],
            "GHI_residual_target": [10.0],
            "target_source": ["nasa_power"],
            "GHI_persistence_naive": [90.0],
            "GHI_persistence_csi": [95.0],
            "ghi_issue": [80.0],
            "target_GHI_clear": [120.0],
            "target_cloud_cover_forecast_proxy": [20.0],
            "best_satellite_source": ["nasa_power"],
            "satellite_ssi_issue": [85.0],
            "horizon_minutes": [60],
            "quality_flag": ["ok"],
        }
    )

    encoder = build_feature_encoder(df, include_satellite_features=True)

    forbidden = {"timestamp", "target_valid_time", "GHI_target", "GHI_residual_target", "target_source", "GHI_persistence_naive", "GHI_persistence_csi"}
    assert forbidden.isdisjoint(encoder.feature_columns)
    assert "ghi_issue" in encoder.feature_columns
    assert "target_cloud_cover_forecast_proxy" in encoder.feature_columns
    assert "satellite_ssi_issue" in encoder.feature_columns


def test_without_satellite_encoder_removes_satellite_features() -> None:
    """Source ablation encoder should remove satellite-derived feature columns."""
    df = pd.DataFrame(
        {
            "site_id": ["munich_centre"],
            "GHI_residual_target": [10.0],
            "ghi_issue": [80.0],
            "satellite_ssi_issue": [85.0],
            "satellite_clear_sky_index_issue": [0.5],
            "irradiance_source_std": [3.0],
            "target_GHI_clear": [120.0],
            "horizon_minutes": [60],
            "best_satellite_source": ["nasa_power"],
        }
    )

    encoder = build_feature_encoder(df, include_satellite_features=False)

    assert "ghi_issue" in encoder.feature_columns
    assert "satellite_ssi_issue" not in encoder.feature_columns
    assert "satellite_clear_sky_index_issue" not in encoder.feature_columns
    assert "irradiance_source_std" not in encoder.feature_columns
    assert "best_satellite_source" not in encoder.feature_columns


def test_hybrid_prediction_physical_constraints() -> None:
    """Hybrid predictions should be non-negative, zero at night, and capped by clear sky."""
    df = pd.DataFrame(
        {
            "target_solar_elevation": [20.0, -1.0, 30.0],
            "target_GHI_clear": [100.0, 100.0, 100.0],
        }
    )

    constrained = _apply_physical_constraints(df, pd.Series([-5.0, 50.0, 200.0]))

    assert constrained.tolist() == [0.0, 0.0, 120.0]
