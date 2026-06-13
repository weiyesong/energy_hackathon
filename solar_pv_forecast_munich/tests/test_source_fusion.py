"""Tests for satellite-first source fusion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocessing.source_fusion import build_fusion_summary, fuse_sources, load_available_sources, save_outputs


def _config() -> dict:
    """Return a minimal site configuration for fusion tests."""
    return {
        "sites": [
            {"id": "munich_centre", "latitude": 48.137, "longitude": 11.575},
            {"id": "munich_north", "latitude": 48.220, "longitude": 11.570},
        ]
    }


def test_source_priority_prefers_eumetsat_over_nasa_and_openmeteo() -> None:
    """Fusion should select EUMETSAT SSI before lower-priority irradiance sources."""
    timestamp = pd.Timestamp("2025-01-01 12:00:00", tz="Europe/Berlin")
    sources = {
        "eumetsat_ssi": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "site_id": ["munich_centre"],
                "latitude": [48.137],
                "longitude": [11.575],
                "ssi": [500.0],
                "cloud_index": [0.2],
            }
        ),
        "nasa_power": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "site_id": ["munich_centre"],
                "latitude": [48.137],
                "longitude": [11.575],
                "ssi": [450.0],
                "clear_sky_ssi": [600.0],
            }
        ),
        "openmeteo": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "site_id": ["munich_centre"],
                "latitude": [48.137],
                "longitude": [11.575],
                "ssi": [400.0],
                "clear_sky_ssi": [650.0],
            }
        ),
    }

    fused = fuse_sources(sources, _config())

    assert fused.loc[0, "best_available_satellite_ssi"] == 500.0
    assert fused.loc[0, "best_satellite_source"] == "eumetsat_ssi"
    assert fused.loc[0, "satellite_data_available"] is True or bool(fused.loc[0, "satellite_data_available"]) is True
    assert fused.loc[0, "number_of_available_irradiance_sources"] == 3
    assert fused.loc[0, "irradiance_source_range"] == 100.0


def test_missing_sources_fallback_to_openmeteo_and_summary_records_disabled_horizons(tmp_path: Path) -> None:
    """Fusion should work when only Open-Meteo fallback data exists."""
    project = tmp_path
    raw = project / "data/raw"
    processed = project / "data/processed"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01 12:00:00+01:00", "2025-01-01 13:00:00+01:00"],
            "shortwave_radiation": [300.0, 250.0],
            "clear_sky_ghi": [500.0, 500.0],
            "cloud_cover": [50.0, 60.0],
        }
    ).to_parquet(processed / "openmeteo_with_physical_baseline.parquet", index=False)

    sources = load_available_sources(project)
    fused = fuse_sources(sources, _config())
    summary = build_fusion_summary(fused, sources)
    save_outputs(fused, summary, project)

    assert list(sources) == ["openmeteo"]
    assert fused["best_satellite_source"].dropna().unique().tolist() == ["openmeteo"]
    assert fused["best_available_satellite_ssi"].tolist() == [300.0, 250.0]
    assert fused["satellite_clear_sky_index"].tolist() == [0.6, 0.5]
    assert "satellite_attenuation_proxy" in fused.columns
    assert summary["selected_primary_source"] == "openmeteo"
    assert summary["horizon_support"]["15_minutes"]["disabled"] is True
    assert summary["horizon_support"]["30_minutes"]["disabled"] is True

    with (project / "data/processed/fusion_summary.json").open("r", encoding="utf-8") as file:
        saved_summary = json.load(file)
    assert saved_summary["selected_primary_source"] == "openmeteo"
    assert (project / "data/processed/fused_solar_dataset.parquet").exists()
