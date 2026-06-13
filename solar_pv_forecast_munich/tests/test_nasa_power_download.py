"""Tests for the NASA POWER data adapter using mocked API responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd
import pytest

from src.data_download import nasa_power_download as nasa


class FakeResponse:
    """Small fake requests response for NASA POWER tests."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        """Create a fake response with JSON payload and status code."""
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        """Return the fake JSON payload."""
        return self._payload


def _site() -> dict[str, Any]:
    """Return a minimal configured site for tests."""
    return {
        "id": "munich_centre",
        "name": "Munich Centre",
        "latitude": 48.137,
        "longitude": 11.575,
        "altitude": 520,
    }


def _payload(parameters: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Build a minimal NASA POWER payload."""
    return {"properties": {"parameter": parameters}}


def test_build_nasa_power_request_uses_verified_hourly_parameters() -> None:
    """NASA POWER requests should use verified Hourly Point API parameter names."""
    request = nasa.build_nasa_power_request(_site(), "2023-01-01", "2025-12-31")

    assert request["parameters"] == "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M,PS,PRECTOTCORR"
    assert request["community"] == "RE"
    assert request["time-standard"] == "UTC"
    assert request["start"] == "20230101"
    assert request["end"] == "20251231"


def test_parse_and_standardize_nasa_power_response() -> None:
    """NASA POWER payloads should map into canonical SolarOps columns."""
    response = _payload(
        {
            "ALLSKY_SFC_SW_DWN": {"2023010100": 100.0, "2023010101": 50.0},
            "CLRSKY_SFC_SW_DWN": {"2023010100": 200.0, "2023010101": 0.0},
            "T2M": {"2023010100": 3.0, "2023010101": 2.0},
            "RH2M": {"2023010100": 80.0, "2023010101": 82.0},
            "WS10M": {"2023010100": 4.0, "2023010101": 5.0},
            "PS": {"2023010100": 96.0, "2023010101": 95.0},
            "PRECTOTCORR": {"2023010100": 0.2, "2023010101": 0.0},
        }
    )

    parsed = nasa.parse_nasa_power_response(response, _site())
    standardized = nasa.standardize_nasa_power_columns(parsed)

    assert standardized["source_name"].eq("nasa_power").all()
    assert standardized["source_type"].eq("satellite_model_derived").all()
    assert standardized["is_satellite_derived"].eq(True).all()
    assert standardized.loc[0, "satellite_ssi"] == 100.0
    assert standardized.loc[0, "satellite_clear_sky_ssi"] == 200.0
    assert standardized.loc[0, "satellite_clear_sky_index"] == 0.5
    assert pd.isna(standardized.loc[0, "satellite_cloud_index"])
    assert str(standardized.loc[0, "timestamp"]).endswith("+01:00")


def test_download_site_retries_by_parameter_when_full_request_fails(monkeypatch: Any) -> None:
    """Unavailable parameters should warn and continue through per-parameter fallback."""
    calls: list[str] = []

    def fake_get(url: str, params: dict[str, Any], timeout: int) -> FakeResponse:
        """Return mocked NASA POWER responses, failing the full request and one parameter."""
        calls.append(str(params["parameters"]))
        if "," in params["parameters"]:
            return FakeResponse({"messages": "bad parameter"}, status_code=422)
        parameter = params["parameters"]
        if parameter == "PRECTOTCORR":
            return FakeResponse({"messages": "unavailable"}, status_code=422)
        return FakeResponse(_payload({parameter: {"2023010100": 1.0}}))

    monkeypatch.setattr(nasa.requests, "get", fake_get)
    monkeypatch.setattr(nasa.time, "sleep", lambda seconds: None)

    with pytest.warns(UserWarning):
        response = nasa.download_nasa_power_site(_site(), "2023-01-01", "2023-01-01")
    parameters = response["properties"]["parameter"]

    assert "ALLSKY_SFC_SW_DWN" in parameters
    assert "PRECTOTCORR" not in parameters
    assert calls[0].startswith("ALLSKY_SFC_SW_DWN,")


def test_download_all_configured_sites_uses_cache_and_updates_registry(tmp_path: Path) -> None:
    """Existing valid site files should be reused unless force is requested."""
    project = tmp_path
    raw_dir = project / "data/raw"
    raw_dir.mkdir(parents=True)
    cached = pd.DataFrame(
        {
            "timestamp": ["2023-01-01 00:00:00+01:00", "2025-12-31 23:00:00+01:00"],
            "latitude": [48.137, 48.137],
            "longitude": [11.575, 11.575],
            "site_id": ["munich_centre", "munich_centre"],
            "source_name": ["nasa_power", "nasa_power"],
            "source_type": ["satellite_model_derived", "satellite_model_derived"],
            "is_satellite_derived": [True, True],
            "satellite_ssi": [100.0, 50.0],
            "satellite_clear_sky_ssi": [200.0, 100.0],
            "satellite_clear_sky_index": [0.5, 0.5],
            "satellite_cloud_index": [pd.NA, pd.NA],
            "quality_flag": ["ok", "ok"],
        }
    )
    cached.to_csv(raw_dir / "nasa_power_munich_centre.csv", index=False)
    config = {
        "_project_root": str(project),
        "sites": [_site()],
    }

    combined = nasa.download_all_configured_sites(config, force=False)
    with (project / "data/processed/data_source_registry.json").open("r", encoding="utf-8") as file:
        registry = json.load(file)

    assert len(combined) == 2
    assert (project / "data/processed/nasa_power_all_sites.parquet").exists()
    assert registry["nasa_power"]["download_status"] == "available"
    assert registry["nasa_power"]["source_type"] == "satellite/model-derived historical solar baseline"
