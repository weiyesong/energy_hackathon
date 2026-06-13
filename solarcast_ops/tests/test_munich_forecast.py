from __future__ import annotations

import pandas as pd

from src.munich_forecast import build_munich_operational_forecast


def _config() -> dict:
    return {
        "site": {
            "name": "Munich Demo Solar Plant",
            "latitude": 48.137,
            "longitude": 11.575,
            "timezone": "Europe/Berlin",
            "altitude_m": 520,
            "peak_power_mw": 1.0,
            "tilt_deg": 35,
            "azimuth_deg": 0,
            "system_loss_percent": 14,
        },
        "forecast": {"horizons_minutes": [15, 30, 60, 180, 360, 720, 1440]},
        "irradiance_model": {"clear_sky_model": "ineichen", "poa_model": "perez", "albedo": 0.2},
    }


def _source_frame(start: str, periods: int = 97) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": times,
            "shortwave_radiation": [650.0] * periods,
            "direct_radiation": [410.0] * periods,
            "diffuse_radiation": [160.0] * periods,
            "direct_normal_irradiance": [620.0] * periods,
            "cloud_cover": [25.0] * periods,
            "cloud_cover_low": [15.0] * periods,
            "cloud_cover_mid": [8.0] * periods,
            "cloud_cover_high": [10.0] * periods,
            "temperature_2m": [22.0] * periods,
            "relative_humidity_2m": [55.0] * periods,
            "wind_speed_10m": [3.0] * periods,
        }
    )


def test_munich_forecast_has_required_horizons_and_monotonic_intervals() -> None:
    result = build_munich_operational_forecast(
        _config(),
        issue_time="2026-06-21 09:00:00+00:00",
        source_frame=_source_frame("2026-06-21 09:00:00+00:00"),
    )

    out = result.forecast
    assert out["horizon_minutes"].tolist() == [15, 30, 60, 180, 360, 720, 1440]
    assert {
        "ghi_wm2",
        "dni_wm2",
        "dhi_wm2",
        "gti_poa_wm2",
        "normalized_pv_power_kw_per_kwp",
        "pv_energy_kwh_per_kwp",
        "p10_kw_per_kwp",
        "p50_kw_per_kwp",
        "p90_kw_per_kwp",
        "pv_generation_state",
        "main_limiting_factor",
    }.issubset(out.columns)
    assert (out["p10_kw_per_kwp"] <= out["p50_kw_per_kwp"]).all()
    assert (out["p50_kw_per_kwp"] <= out["p90_kw_per_kwp"]).all()
    assert out.loc[out["horizon_minutes"].eq(60), "normalized_pv_power_kw_per_kwp"].iloc[0] > 0


def test_munich_forecast_nighttime_constraint_sets_radiation_to_zero() -> None:
    result = build_munich_operational_forecast(
        _config(),
        issue_time="2026-12-21 22:00:00+00:00",
        source_frame=_source_frame("2026-12-21 22:00:00+00:00"),
    )

    first = result.forecast.iloc[0]
    assert first["solar_elevation_deg"] <= 0
    assert first["ghi_wm2"] == 0
    assert first["dni_wm2"] == 0
    assert first["dhi_wm2"] == 0
    assert first["gti_poa_wm2"] == 0
    assert first["normalized_pv_power_kw_per_kwp"] == 0
