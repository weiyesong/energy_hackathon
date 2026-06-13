"""Tests for canonical SolarOps schema utilities."""

from __future__ import annotations

import pandas as pd

from src.core.schema import CANONICAL_COLUMNS, ensure_canonical_columns, validate_canonical_schema


def test_schema_validation_requires_identity_columns() -> None:
    """Schema validation should flag missing identity columns without inventing validity."""
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1), "latitude": [48.137]})

    result = validate_canonical_schema(df)

    assert result.is_valid is False
    assert "site_id" in result.missing_required_columns
    assert result.unknown_columns == []


def test_ensure_canonical_columns_fills_missing_values_with_na() -> None:
    """Missing canonical source fields should be represented as NA."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=1),
            "latitude": [48.137],
            "longitude": [11.575],
            "site_id": ["munich_centre"],
            "source_name": ["openmeteo"],
            "source_type": ["weather forecast and fallback irradiance source"],
            "is_satellite_derived": [False],
        }
    )

    canonical = ensure_canonical_columns(df)

    assert list(canonical.columns) == CANONICAL_COLUMNS
    assert pd.isna(canonical.loc[0, "satellite_ssi"])
    assert pd.isna(canonical.loc[0, "quality_flag"])
