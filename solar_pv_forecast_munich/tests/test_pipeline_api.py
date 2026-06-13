"""Tests for live/demo pipeline outputs and FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app
from src.pipeline.source_health import inspect_source_health


def test_source_health_marks_high_frequency_disabled() -> None:
    """15/30-minute horizons should be marked unavailable without matching models and source frequency."""
    config = {
        "forecast": {
            "high_frequency_horizons_minutes": [15, 30],
        }
    }
    status = inspect_source_health(config)

    assert status["high_frequency_horizons"]["15"]["status"] == "Not operationally available"
    assert status["high_frequency_horizons"]["15"]["reason"] == "High-frequency satellite input required"
    assert status["primary_satellite_source"] in {"eumetsat_ssi", "unavailable"}


def test_api_endpoints_return_expected_shapes() -> None:
    """The product API should expose the required UI endpoints."""
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert "primary_satellite_source" in overview.json()
    assert "evaluation_mode" in client.get("/api/benchmark").json()
    assert client.get("/api/forecast").status_code == 200
    assert client.get("/api/sites").status_code == 200
    assert client.get("/api/actions").status_code == 200
    assert client.get("/api/drivers").status_code == 200
    assert client.get("/api/data-sources").status_code == 200


def test_forecast_endpoint_contains_ui_fields() -> None:
    """Forecast endpoint should return the UI time-series field names."""
    client = TestClient(app)
    points = client.get("/api/forecast").json()

    assert points
    required = {
        "target_time",
        "horizon_minutes",
        "GHI_P10",
        "GHI_P50",
        "GHI_P90",
        "persistence_GHI",
        "POA_P50",
        "PV_P10",
        "PV_P50",
        "PV_P90",
        "cloud_cover",
        "solar_elevation",
        "main_limiting_factor",
        "uncertainty_level",
        "cloud_event",
    }
    assert required.issubset(points[0])
