"""Tests for optional EUMETSAT SSI adapter and manual ingestion path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.data_download import eumetsat_download
from src.preprocessing.eumetsat_ingest import (
    extract_munich_sites,
    find_manual_eumetsat_files,
    standardize_eumetsat_data,
)


class FakeDataset:
    """Tiny xarray-like dataset for extraction tests."""

    coords = {"latitude": [48.0], "longitude": [11.0], "time": [pd.Timestamp("2025-01-01T00:00:00Z")]}
    dims = {"latitude": 1, "longitude": 1, "time": 1}
    data_vars = {"ssi": None, "cloud_index": None}

    def sel(self, selection: dict[str, float], method: str) -> "FakeDataset":
        """Return self for nearest-neighbour site extraction."""
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Return a dataframe shaped like xarray's tabular conversion output."""
        return pd.DataFrame(
            {
                "time": [pd.Timestamp("2025-01-01T00:00:00Z")],
                "ssi": [350.0],
                "cloud_index": [0.25],
                "cloud_fraction": [0.4],
            }
        )


def test_check_credentials_false_without_env_or_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Missing credentials should be detected without raising."""
    for name in ["EUMETSAT_CONSUMER_KEY", "EUMETSAT_CONSUMER_SECRET", "EUMETSAT_API_KEY", "EUMETSAT_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert eumetsat_download.check_eumetsat_credentials() is False
    with pytest.warns(UserWarning):
        assert eumetsat_download.list_available_products() == []


def test_find_manual_eumetsat_files_supports_netcdf_and_grib(tmp_path: Path) -> None:
    """Manual discovery should return only supported satellite file extensions."""
    (tmp_path / "a.nc").write_text("", encoding="utf-8")
    (tmp_path / "b.grib2").write_text("", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")

    files = find_manual_eumetsat_files(tmp_path)

    assert [path.name for path in files] == ["a.nc", "b.grib2"]


def test_extract_and_standardize_manual_eumetsat_data() -> None:
    """Manual extracted variables should map through configurable aliases."""
    sites = [{"id": "munich_centre", "latitude": 48.137, "longitude": 11.575}]
    extracted = extract_munich_sites(FakeDataset(), sites)
    standardized = standardize_eumetsat_data(
        extracted,
        aliases={
            "satellite_ssi": ["ssi"],
            "satellite_cloud_index": ["cloud_index"],
            "cloud_cover": ["cloud_fraction"],
        },
    )

    assert standardized.loc[0, "source_name"] == "eumetsat_ssi"
    assert standardized.loc[0, "source_type"] == "operational_satellite"
    assert bool(standardized.loc[0, "is_satellite_derived"]) is True
    assert standardized.loc[0, "satellite_ssi"] == 350.0
    assert standardized.loc[0, "satellite_cloud_index"] == 0.25
    assert standardized.loc[0, "cloud_cover"] == 0.4
    assert standardized.loc[0, "quality_flag"] == "ok"


def test_standardize_does_not_invent_missing_satellite_ssi() -> None:
    """Missing EUMETSAT SSI variables should remain NA and be flagged."""
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "site_id": ["munich_centre"],
            "latitude": [48.137],
            "longitude": [11.575],
            "cloud_fraction": [0.5],
        }
    )

    standardized = standardize_eumetsat_data(df)

    assert pd.isna(standardized.loc[0, "satellite_ssi"])
    assert standardized.loc[0, "quality_flag"] == "missing_satellite_ssi"


def test_main_without_credentials_or_files_updates_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unavailable EUMETSAT should print manual instructions and update registry without blocking."""
    for name in ["EUMETSAT_CONSUMER_KEY", "EUMETSAT_CONSUMER_SECRET", "EUMETSAT_API_KEY", "EUMETSAT_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = {
        "data_sources": {
            "eumetsat": {"enabled": True, "manual_directory": "data/manual/eumetsat"},
            "eumetsat_variable_aliases": {
                "satellite_ssi": ["ssi"],
                "satellite_cloud_index": ["cloud_index"],
                "cloud_cover": ["cloud_fraction"],
            },
        },
        "sites": [{"id": "munich_centre", "latitude": 48.137, "longitude": 11.575}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "data/manual/eumetsat").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["eumetsat_download.py", "--config", str(config_path)])

    eumetsat_download.main()

    output = capsys.readouterr().out
    assert "EUMETSAT SSI is not currently available." in output
    with (tmp_path / "data/processed/data_source_registry.json").open("r", encoding="utf-8") as file:
        registry = json.load(file)
    assert registry["eumetsat_ssi"]["download_status"] == "unavailable"
    assert registry["eumetsat_ssi"]["manual_action_required"] is True
    assert registry["eumetsat_ssi"]["error_message"] == "credentials missing; manual download required"
