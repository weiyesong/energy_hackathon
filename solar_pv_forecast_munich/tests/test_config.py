"""Tests for SolarOps configuration parsing."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_config_contains_satellite_first_product_settings() -> None:
    """Config should expose product positioning and satellite-first source priority."""
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    assert config["product"]["name"] == "SolarOps"
    assert config["product"]["region_name"] == "Munich"
    assert config["product"]["demo_mode"] is False
    assert config["data_sources"]["satellite_priority"] == ["eumetsat_ssi", "nasa_power", "openmeteo"]
    assert config["benchmark"]["use_clear_sky_persistence"] is True


def test_config_contains_site_comparison_set() -> None:
    """Config should define the Munich comparison sites required by the product."""
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    site_ids = {site["id"] for site in config["sites"]}

    assert site_ids == {
        "munich_centre",
        "munich_north",
        "munich_east",
        "munich_south",
        "munich_west",
    }
    assert config["forecast"]["valid_horizons_minutes"] == [60, 180, 360, 720, 1440]
    assert config["forecast"]["high_frequency_horizons_minutes"] == [15, 30]
