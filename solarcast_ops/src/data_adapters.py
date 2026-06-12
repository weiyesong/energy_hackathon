from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class AdapterStatus:
    name: str
    available: bool
    source_issue_time_column: str | None
    source_latency_minutes: float | None
    quality_flags: list[str]


class ForecastDataAdapter(Protocol):
    """Contract for forecast-issue-time data adapters."""

    name: str

    def status(self) -> AdapterStatus:
        """Return availability and quality metadata for the current adapter."""

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach adapter fields without using data after forecast issue time."""


class PVGISSatelliteProxyAdapter:
    """Hourly PVGIS irradiance adapter used as a satellite-derived proxy in the MVP."""

    name = "pvgis_hourly_proxy"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name=self.name,
            available=True,
            source_issue_time_column="timestamp",
            source_latency_minutes=0.0,
            quality_flags=["satellite_irradiance_proxy", "not_raw_meteosat_imagery"],
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["satellite_available"] = out["global_irradiance_wm2"].notna()
        out["satellite_latency_minutes"] = 0.0
        return out


class GroundProxyAdapter:
    """PVGIS modelled PV output adapter standing in for local plant telemetry."""

    name = "pvgis_modelled_pv_proxy"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name=self.name,
            available=True,
            source_issue_time_column="timestamp",
            source_latency_minutes=0.0,
            quality_flags=["ground_power_proxy", "not_real_scada"],
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["ground_available"] = out["pv_power_mw"].notna()
        out["ground_latency_minutes"] = 0.0
        return out


class UnavailableNWPAdapter:
    """Explicit placeholder for future real NWP forecasts."""

    name = "unavailable_demo_adapter"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name=self.name,
            available=False,
            source_issue_time_column=None,
            source_latency_minutes=None,
            quality_flags=["nwp_missing", "raw_nwp_baseline_unavailable"],
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["nwp_available"] = False
        out["nwp_latency_minutes"] = pd.NA
        return out


def attach_demo_adapters(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[AdapterStatus]]:
    """Attach MVP adapter metadata and missing-input masks."""
    adapters: list[ForecastDataAdapter] = [
        PVGISSatelliteProxyAdapter(),
        GroundProxyAdapter(),
        UnavailableNWPAdapter(),
    ]
    out = frame.copy()
    statuses = []
    for adapter in adapters:
        out = adapter.attach(out)
        statuses.append(adapter.status())
    out["missing_satellite"] = ~out["satellite_available"].astype(bool)
    out["missing_ground"] = ~out["ground_available"].astype(bool)
    out["missing_nwp"] = ~out["nwp_available"].astype(bool)
    return out, statuses
