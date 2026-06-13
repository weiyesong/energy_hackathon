"""Tests for SolarOps data source registry."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.data_registry import (
    SOURCE_TYPE_LABELS,
    get_available_sources,
    get_best_satellite_source,
    initialize_registry_from_config,
    register_data_source,
    update_data_source_status,
)


def test_registry_update_and_available_sources(tmp_path: Path) -> None:
    """Registry updates should persist source status and availability metadata."""
    registry_path = tmp_path / "registry.json"

    register_data_source(
        "openmeteo",
        date_range={"start": "2023-01-01", "end": "2025-12-31"},
        temporal_resolution="hourly",
        file_path="data/raw/openmeteo_historical_munich.csv",
        available_columns=["timestamp", "shortwave_radiation"],
        download_status="available",
        registry_path=registry_path,
    )
    update = update_data_source_status("openmeteo", "available", registry_path=registry_path)
    available = get_available_sources(registry_path=registry_path)

    assert update["download_status"] == "available"
    assert available[0]["source_name"] == "openmeteo"
    assert available[0]["source_type"] == SOURCE_TYPE_LABELS["openmeteo"]


def test_satellite_source_priority_skips_unavailable_sources(tmp_path: Path) -> None:
    """Best satellite source should follow priority and skip unavailable/manual sources."""
    registry_path = tmp_path / "registry.json"
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    register_data_source("eumetsat_ssi", download_status="manual_required", manual_action_required=True, registry_path=registry_path)
    register_data_source("nasa_power", download_status="available", registry_path=registry_path)
    register_data_source("openmeteo", download_status="available", registry_path=registry_path)

    best = get_best_satellite_source(config, registry_path=registry_path)

    assert best is not None
    assert best["source_name"] == "nasa_power"
    assert best["source_type"] == "satellite/model-derived historical solar baseline"


def test_unavailable_sources_are_handled_gracefully(tmp_path: Path) -> None:
    """Unavailable configured sources should not raise and should return no best satellite source."""
    registry_path = tmp_path / "registry.json"
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    initialize_registry_from_config(config, registry_path=registry_path)
    best = get_best_satellite_source(config, registry_path=registry_path)

    assert best is None
    available = get_available_sources(registry_path=registry_path)
    assert available == []
