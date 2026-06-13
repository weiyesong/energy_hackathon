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


class SatelliteArchiveAdapter:
    """Station-level satellite-derived irradiance adapter for the MVP."""

    name = "satellite_irradiance_archive"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name=self.name,
            available=True,
            source_issue_time_column="timestamp",
            source_latency_minutes=0.0,
            quality_flags=[
                "satellite_irradiance",
                "openmeteo_satellite_archive",
                "pvgis_sarah3_satellite",
            ],
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        archive_available = (
            out["satellite_archive_available"].fillna(False).astype(bool)
            if "satellite_archive_available" in out
            else pd.Series(False, index=out.index)
        )
        out["satellite_available"] = archive_available | out["global_irradiance_wm2"].notna()
        out["satellite_archive_available"] = archive_available
        out["satellite_latency_minutes"] = 0.0
        return out


class GroundProxyAdapter:
    """PV output adapter for the forecast target series."""

    name = "pv_power_reference"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name=self.name,
            available=True,
            source_issue_time_column="timestamp",
            source_latency_minutes=0.0,
            quality_flags=["pv_power_reference"],
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["ground_available"] = out["pv_power_mw"].notna()
        out["ground_latency_minutes"] = 0.0
        return out


class UnavailableNWPAdapter:
    """Weather forecast context adapter."""

    name = "weather_context"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name=self.name,
            available=False,
            source_issue_time_column=None,
            source_latency_minutes=None,
            quality_flags=["weather_context_optional"],
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["nwp_available"] = False
        out["nwp_latency_minutes"] = pd.NA
        return out


def attach_demo_adapters(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[AdapterStatus]]:
    """Attach MVP adapter metadata and missing-input masks."""
    adapters: list[ForecastDataAdapter] = [
        SatelliteArchiveAdapter(),
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
