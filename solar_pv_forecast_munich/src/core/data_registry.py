"""JSON-backed registry for SolarOps data-source availability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH = Path("data/processed/data_source_registry.json")


SOURCE_TYPE_LABELS = {
    "eumetsat_ssi": "primary operational satellite-derived irradiance source",
    "nasa_power": "satellite/model-derived historical solar baseline",
    "openmeteo": "weather forecast and fallback irradiance source",
}


DEFAULT_SOURCE_METADATA = {
    "eumetsat_ssi": {
        "source_name": "eumetsat_ssi",
        "source_type": SOURCE_TYPE_LABELS["eumetsat_ssi"],
        "is_satellite_derived": True,
        "manual_action_required": True,
    },
    "nasa_power": {
        "source_name": "nasa_power",
        "source_type": SOURCE_TYPE_LABELS["nasa_power"],
        "is_satellite_derived": True,
        "manual_action_required": False,
    },
    "openmeteo": {
        "source_name": "openmeteo",
        "source_type": SOURCE_TYPE_LABELS["openmeteo"],
        "is_satellite_derived": False,
        "manual_action_required": False,
    },
}


def register_data_source(
    source_name: str,
    source_type: str | None = None,
    is_satellite_derived: bool | None = None,
    date_range: dict[str, str] | None = None,
    temporal_resolution: str | None = None,
    file_path: str | None = None,
    available_columns: list[str] | None = None,
    download_status: str = "registered",
    manual_action_required: bool | None = None,
    error_message: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Register or replace a data source in the JSON registry."""
    registry = _read_registry(registry_path)
    defaults = DEFAULT_SOURCE_METADATA.get(source_name, {})
    entry = {
        "source_name": source_name,
        "source_type": source_type or defaults.get("source_type", "unknown source"),
        "is_satellite_derived": (
            bool(is_satellite_derived)
            if is_satellite_derived is not None
            else bool(defaults.get("is_satellite_derived", False))
        ),
        "date_range": date_range or {},
        "temporal_resolution": temporal_resolution,
        "file_path": file_path,
        "available_columns": available_columns or [],
        "last_update_time": _utc_now_iso(),
        "download_status": download_status,
        "manual_action_required": (
            bool(manual_action_required)
            if manual_action_required is not None
            else bool(defaults.get("manual_action_required", False))
        ),
        "error_message": error_message,
    }
    registry[source_name] = entry
    write_registry_json(registry, registry_path)
    return entry


def update_data_source_status(
    source_name: str,
    download_status: str,
    error_message: str | None = None,
    manual_action_required: bool | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Update download status fields for an existing or newly registered source."""
    registry = _read_registry(registry_path)
    if source_name not in registry:
        register_data_source(source_name, registry_path=registry_path)
        registry = _read_registry(registry_path)

    entry = registry[source_name]
    entry["download_status"] = download_status
    entry["error_message"] = error_message
    entry["last_update_time"] = _utc_now_iso()
    if manual_action_required is not None:
        entry["manual_action_required"] = bool(manual_action_required)
    registry[source_name] = entry
    write_registry_json(registry, registry_path)
    return entry


def get_available_sources(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Return registered sources that are marked available and have no error message."""
    registry = _read_registry(registry_path)
    return [
        entry
        for entry in registry.values()
        if entry.get("download_status") == "available" and not entry.get("error_message")
    ]


def get_best_satellite_source(
    config: dict[str, Any] | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any] | None:
    """Return the best available satellite-derived source according to configured priority."""
    registry = _read_registry(registry_path)
    priority = _satellite_priority(config)
    for source_name in priority:
        entry = registry.get(source_name)
        if not entry:
            continue
        if entry.get("download_status") != "available":
            continue
        if entry.get("manual_action_required"):
            continue
        if entry.get("error_message"):
            continue
        if entry.get("is_satellite_derived"):
            return entry
    return None


def write_registry_json(
    registry: dict[str, dict[str, Any]],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Write registry state to JSON."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, sort_keys=True)


def initialize_registry_from_config(
    config: dict[str, Any],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, dict[str, Any]]:
    """Create registry entries for configured sources without downloading data."""
    sources = config.get("data_sources", {})
    if sources.get("eumetsat", {}).get("enabled", False):
        register_data_source(
            "eumetsat_ssi",
            download_status="manual_required",
            manual_action_required=True,
            file_path=sources.get("eumetsat", {}).get("manual_directory"),
            registry_path=registry_path,
        )
    if sources.get("nasa_power", {}).get("enabled", False):
        register_data_source("nasa_power", download_status="unavailable", registry_path=registry_path)
    if sources.get("openmeteo", {}).get("enabled", False):
        register_data_source("openmeteo", download_status="unavailable", registry_path=registry_path)
    return _read_registry(registry_path)


def _read_registry(registry_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read registry JSON, returning an empty registry when it does not exist."""
    path = Path(registry_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _satellite_priority(config: dict[str, Any] | None) -> list[str]:
    """Read satellite source priority from config with a safe default."""
    if not config:
        return ["eumetsat_ssi", "nasa_power", "openmeteo"]
    return list(config.get("data_sources", {}).get("satellite_priority", ["eumetsat_ssi", "nasa_power", "openmeteo"]))


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
