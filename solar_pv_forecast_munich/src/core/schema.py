"""Canonical data schema for SolarOps data adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "site_id",
    "source_name",
    "source_type",
    "is_satellite_derived",
    "satellite_ssi",
    "satellite_clear_sky_ssi",
    "satellite_clear_sky_index",
    "satellite_cloud_index",
    "ghi_observed",
    "dni_observed",
    "dhi_observed",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "snow_depth",
    "quality_flag",
]


REQUIRED_CANONICAL_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "site_id",
    "source_name",
    "source_type",
    "is_satellite_derived",
]


@dataclass(frozen=True)
class SchemaValidationResult:
    """Result of validating a dataframe against the canonical schema."""

    is_valid: bool
    missing_required_columns: list[str]
    unknown_columns: list[str]


def validate_canonical_schema(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_CANONICAL_COLUMNS,
) -> SchemaValidationResult:
    """Validate that a dataframe contains required canonical identity columns."""
    required = list(required_columns)
    missing = [column for column in required if column not in df.columns]
    unknown = [column for column in df.columns if column not in CANONICAL_COLUMNS]
    return SchemaValidationResult(
        is_valid=not missing,
        missing_required_columns=missing,
        unknown_columns=unknown,
    )


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with all canonical columns present, filling missing optional fields with NaN."""
    canonical = df.copy()
    for column in CANONICAL_COLUMNS:
        if column not in canonical.columns:
            canonical[column] = pd.NA
    return canonical[CANONICAL_COLUMNS]
