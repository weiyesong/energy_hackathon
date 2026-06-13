from __future__ import annotations

import pandas as pd

from src.data_download import merge_openmeteo_satellite_archive


def test_openmeteo_satellite_archive_overrides_irradiance_by_hour() -> None:
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-06-01 10:10Z", "2023-06-01 11:10Z"]),
            "pv_power_mw": [0.4, 0.5],
            "global_irradiance_wm2": [400.0, 500.0],
            "direct_irradiance_wm2": [250.0, 300.0],
            "diffuse_irradiance_wm2": [100.0, 120.0],
            "air_temperature_c": [20.0, 21.0],
            "wind_speed_ms": [3.0, 4.0],
            "solar_elevation_deg": [45.0, 50.0],
            "data_source": ["PVGIS"] * 2,
            "irradiance_source": ["PVGIS/SARAH-3 satellite-derived irradiance"] * 2,
            "pv_power_source": ["PVGIS PV output"] * 2,
            "satellite_archive_available": [False, False],
            "is_synthetic": [False, False],
        }
    )
    satellite = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-06-01 10:00Z"]),
            "shortwave_radiation": [620.0],
            "direct_radiation": [410.0],
            "diffuse_radiation": [150.0],
        }
    )

    merged, meta = merge_openmeteo_satellite_archive(base, satellite)

    assert meta["used"] is True
    assert meta["matched_rows"] == 1
    assert merged.loc[0, "global_irradiance_wm2"] == 620.0
    assert merged.loc[0, "direct_irradiance_wm2"] == 410.0
    assert merged.loc[0, "diffuse_irradiance_wm2"] == 150.0
    assert merged.loc[0, "pv_power_mw"] == 0.4
    assert bool(merged.loc[0, "satellite_archive_available"])
    assert not bool(merged.loc[1, "satellite_archive_available"])
    assert merged.loc[1, "global_irradiance_wm2"] == 500.0
