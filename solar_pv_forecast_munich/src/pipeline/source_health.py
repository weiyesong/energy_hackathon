"""Data-source health and horizon support checks for SolarOps pipelines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ROLES = {
    "eumetsat_ssi": "primary operational satellite-derived irradiance source",
    "nasa_power": "satellite/model-derived historical solar baseline",
    "openmeteo": "weather forecast and fallback irradiance source",
}


def load_registry(registry_path: str | Path = "data/processed/data_source_registry.json") -> dict[str, dict[str, Any]]:
    """Load data-source registry JSON, returning an empty registry when unavailable."""
    path = Path(registry_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def inspect_source_health(
    config: dict[str, Any],
    registry_path: str | Path = "data/processed/data_source_registry.json",
    models_dir: str | Path = "outputs/models",
) -> dict[str, Any]:
    """Inspect source availability, freshness, fallback status, and horizon support."""
    registry = load_registry(registry_path)
    eumetsat = registry.get("eumetsat_ssi", {})
    openmeteo = registry.get("openmeteo", {})
    nasa = registry.get("nasa_power", {})
    eumetsat_live = _is_available(eumetsat) and not eumetsat.get("manual_action_required", False)
    openmeteo_available = _is_available(openmeteo)
    nasa_available = _is_available(nasa)
    primary_satellite_source = "eumetsat_ssi" if eumetsat_live else "unavailable"
    fallback_active = not eumetsat_live

    source_resolution = _resolution_minutes(eumetsat.get("temporal_resolution")) if eumetsat_live else None
    if source_resolution is None and openmeteo_available:
        source_resolution = _resolution_minutes(openmeteo.get("temporal_resolution"))

    high_frequency = {}
    for horizon in config.get("forecast", {}).get("high_frequency_horizons_minutes", [15, 30]):
        model_exists = _quantile_models_exist(models_dir, int(horizon))
        source_supports = source_resolution is not None and source_resolution <= int(horizon)
        operational = bool(model_exists and source_supports)
        high_frequency[str(horizon)] = {
            "horizon_minutes": int(horizon),
            "operational": operational,
            "status": "Operational" if operational else "Not operationally available",
            "reason": None if operational else "High-frequency satellite input required",
            "trained_models_exist": model_exists,
            "source_resolution_minutes": source_resolution,
        }

    rows = []
    for source_name, entry in registry.items():
        rows.append(
            {
                "source_name": source_name,
                "source_role": SOURCE_ROLES.get(source_name, entry.get("source_type", "unknown")),
                "status": entry.get("download_status", "unknown"),
                "last_update": entry.get("last_update_time"),
                "temporal_resolution": entry.get("temporal_resolution"),
                "coverage": entry.get("date_range", {}),
                "manual_action_required": bool(entry.get("manual_action_required", False)),
                "fallback_active": fallback_active and source_name != "eumetsat_ssi",
                "error_message": entry.get("error_message"),
            }
        )

    return {
        "primary_satellite_source": primary_satellite_source,
        "weather_forecast_source": "openmeteo" if openmeteo_available else "openmeteo_unavailable",
        "satellite_data_available": bool(eumetsat_live),
        "fallback_active": bool(fallback_active),
        "nasa_power_context_available": bool(nasa_available),
        "data_freshness_minutes": _freshness_minutes(openmeteo.get("last_update_time")),
        "data_quality_level": _quality_level(eumetsat_live, openmeteo_available, nasa_available),
        "source_resolution_minutes": source_resolution,
        "high_frequency_horizons": high_frequency,
        "sources": rows,
        "inspected_at": datetime.now(timezone.utc).isoformat(),
    }


def write_source_status(status: dict[str, Any], path: str | Path) -> None:
    """Write source health status to JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)


def _is_available(entry: dict[str, Any]) -> bool:
    """Return whether a registry entry is currently available."""
    return entry.get("download_status") == "available" and not entry.get("error_message")


def _freshness_minutes(last_update: str | None) -> float | None:
    """Return minutes since last registry update."""
    if not last_update:
        return None
    try:
        updated = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds() / 60.0, 0.0)
    except ValueError:
        return None


def _quality_level(eumetsat_live: bool, openmeteo_available: bool, nasa_available: bool) -> str:
    """Return a coarse data-quality label for UI display."""
    if eumetsat_live and openmeteo_available:
        return "high"
    if openmeteo_available and nasa_available:
        return "fallback"
    if openmeteo_available:
        return "weather_only"
    return "degraded"


def _resolution_minutes(resolution: str | None) -> int | None:
    """Convert common temporal-resolution labels to minutes."""
    if not resolution:
        return None
    text = str(resolution).lower()
    if "15" in text:
        return 15
    if "30" in text:
        return 30
    if "hour" in text or "60" in text:
        return 60
    return None


def _quantile_models_exist(models_dir: str | Path, horizon: int) -> bool:
    """Return whether all three quantile models exist for a horizon."""
    base = Path(models_dir)
    return all((base / f"quantile_ghi_horizon_{horizon}_{label}.pkl").exists() for label in ["p10", "p50", "p90"])
