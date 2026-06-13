"""Tests for operational PV conversion, decisions, and site ranking."""

from __future__ import annotations

import pandas as pd

from src.operations.decision_engine import (
    add_limiting_factors,
    build_operational_forecast,
    generate_operator_actions,
)
from src.operations.site_ranking import rank_sites


def _sample_predictions() -> pd.DataFrame:
    """Create a compact probabilistic forecast fixture."""
    return pd.DataFrame(
        {
            "site_id": ["munich_centre", "munich_centre", "munich_north"],
            "latitude": [48.137, 48.137, 48.220],
            "longitude": [11.575, 11.575, 11.570],
            "target_valid_time": [
                "2025-07-01 10:00:00+02:00",
                "2025-07-01 11:00:00+02:00",
                "2025-07-01 10:00:00+02:00",
            ],
            "horizon_minutes": [60, 60, 60],
            "GHI_P10_calibrated": [300.0, 120.0, 260.0],
            "GHI_P50": [500.0, 180.0, 420.0],
            "GHI_P90_calibrated": [650.0, 280.0, 560.0],
            "target_solar_zenith": [40.0, 42.0, 41.0],
            "target_solar_elevation": [50.0, 48.0, 49.0],
            "target_solar_azimuth": [150.0, 165.0, 152.0],
            "target_cos_zenith": [0.766, 0.743, 0.755],
            "target_GHI_clear": [800.0, 760.0, 790.0],
            "target_temperature_2m_forecast_proxy": [24.0, 26.0, 23.0],
            "target_cloud_cover_forecast_proxy": [30.0, 82.0, 40.0],
            "target_cloud_cover_low_forecast_proxy": [20.0, 75.0, 30.0],
            "target_cloud_cover_mid_forecast_proxy": [15.0, 35.0, 20.0],
            "target_cloud_cover_high_forecast_proxy": [10.0, 25.0, 15.0],
            "cloud_cover_trend": [5.0, 35.0, 10.0],
            "irradiance_source_std": [20.0, 40.0, 18.0],
            "satellite_data_available": [True, True, True],
            "quality_flag": ["ok", "ok", "ok"],
            "uncertainty_level": ["Medium", "High", "Medium"],
        }
    )


def _config() -> dict:
    """Create a compact PV system config fixture."""
    return {
        "pv_system": {
            "surface_tilt": 35,
            "surface_azimuth": 180,
            "capacity_kwp": 1.0,
            "noct": 45,
            "temperature_coefficient": -0.004,
            "inverter_efficiency": 0.96,
            "other_losses": 0.10,
            "albedo_default": 0.2,
        }
    }


def test_operational_forecast_physical_columns() -> None:
    """Operational forecast should contain constrained DNI/DHI, POA, and PV columns."""
    forecast = build_operational_forecast(_sample_predictions(), _config())

    for col in ["DNI_P50_estimated", "DHI_P50_estimated", "POA_P50", "PV_P50", "PV_energy_P50"]:
        assert col in forecast.columns
        assert (forecast[col] >= 0).all()
    assert (forecast["PV_P10"] <= forecast["PV_P50"]).all()
    assert (forecast["PV_P50"] <= forecast["PV_P90"]).all()


def test_no_aerosol_label_without_aerosol_data() -> None:
    """Aerosol attenuation must not be claimed when aerosol fields are absent."""
    forecast = build_operational_forecast(_sample_predictions(), _config())
    labelled = add_limiting_factors(forecast)

    assert "aerosol attenuation" not in set(labelled["main_limiting_factor"])


def test_operator_actions_are_structured() -> None:
    """Generated actions should use the structured operational schema."""
    forecast = build_operational_forecast(_sample_predictions(), _config())
    actions = generate_operator_actions(forecast)

    assert actions
    assert {"action_type", "priority", "valid_from", "valid_until", "reason", "confidence"}.issubset(actions[0])


def test_site_ranking_outputs_scores_and_grades() -> None:
    """Site ranking should produce transparent scores and A-D grades."""
    forecast = build_operational_forecast(_sample_predictions(), _config())
    ranking, summary = rank_sites(forecast, _config())

    assert not ranking.empty
    assert {"site_score", "rank_grade", "expected_daily_energy"}.issubset(ranking.columns)
    assert set(ranking["rank_grade"]).issubset({"A", "B", "C", "D"})
    assert summary["number_of_sites"] == 2
