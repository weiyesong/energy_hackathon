"""Fuse available irradiance sources into one satellite-first SolarOps dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TARGET_TIMEZONE = "Europe/Berlin"
DEFAULT_ALIGNMENT_TOLERANCE = "35min"
SOURCE_PRIORITY = ["eumetsat_ssi", "nasa_power", "openmeteo"]
FUSED_OUTPUT_PATH = Path("data/processed/fused_solar_dataset.parquet")
SUMMARY_OUTPUT_PATH = Path("data/processed/fusion_summary.json")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load project configuration and attach the project root path."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["_project_root"] = str(path.parent)
    return config


def load_available_sources(project_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load source files that exist, skipping missing optional sources."""
    root = Path(project_root)
    sources: dict[str, pd.DataFrame] = {}

    eumetsat_path = root / "data/processed/eumetsat_ssi_all_sites.parquet"
    if eumetsat_path.exists():
        sources["eumetsat_ssi"] = _load_eumetsat(eumetsat_path)

    nasa_path = root / "data/processed/nasa_power_all_sites.parquet"
    if nasa_path.exists():
        sources["nasa_power"] = _load_nasa_power(nasa_path)

    physical_path = root / "data/processed/openmeteo_with_physical_baseline.parquet"
    raw_openmeteo_path = root / "data/raw/openmeteo_historical_munich.csv"
    if physical_path.exists():
        sources["openmeteo"] = _load_openmeteo_physical(physical_path)
    elif raw_openmeteo_path.exists():
        sources["openmeteo"] = _load_openmeteo_raw(raw_openmeteo_path)

    return sources


def fuse_sources(
    sources: dict[str, pd.DataFrame],
    config: dict[str, Any],
    tolerance: str | pd.Timedelta = DEFAULT_ALIGNMENT_TOLERANCE,
) -> pd.DataFrame:
    """Fuse available irradiance sources by site and nearest timestamp."""
    if not sources:
        raise RuntimeError("No source files are available for fusion.")

    tolerance_delta = pd.Timedelta(tolerance)
    normalized = {name: _normalize_source_frame(name, frame) for name, frame in sources.items() if not frame.empty}
    if not normalized:
        raise RuntimeError("Source files exist but contain no usable rows.")

    base = _build_base_grid(normalized, config)
    fused = base.copy()

    for source_name, frame in normalized.items():
        fused = _merge_source_asof(fused, frame, source_name, tolerance_delta)

    irradiance_columns = [f"{name}_ssi" for name in SOURCE_PRIORITY if f"{name}_ssi" in fused.columns]
    fused["number_of_available_irradiance_sources"] = fused[irradiance_columns].notna().sum(axis=1)
    if irradiance_columns:
        fused["irradiance_source_mean"] = fused[irradiance_columns].mean(axis=1, skipna=True)
        fused["irradiance_source_std"] = fused[irradiance_columns].std(axis=1, skipna=True).fillna(0.0)
        fused["irradiance_source_range"] = fused[irradiance_columns].max(axis=1, skipna=True) - fused[irradiance_columns].min(axis=1, skipna=True)
    else:
        fused["irradiance_source_mean"] = np.nan
        fused["irradiance_source_std"] = np.nan
        fused["irradiance_source_range"] = np.nan

    fused["best_available_satellite_ssi"] = np.nan
    fused["best_satellite_source"] = pd.NA
    for source_name in SOURCE_PRIORITY:
        column = f"{source_name}_ssi"
        if column not in fused.columns:
            continue
        missing = fused["best_available_satellite_ssi"].isna() & fused[column].notna()
        fused.loc[missing, "best_available_satellite_ssi"] = fused.loc[missing, column]
        fused.loc[missing, "best_satellite_source"] = source_name

    fused["satellite_data_available"] = fused["best_available_satellite_ssi"].notna()
    ghi_clear = _best_clear_sky_column(fused)
    fused["satellite_clear_sky_index"] = (fused["best_available_satellite_ssi"] / ghi_clear.clip(lower=1.0)).clip(0.0, 1.2)
    fused.loc[fused["best_available_satellite_ssi"].isna(), "satellite_clear_sky_index"] = np.nan
    fused["satellite_attenuation_proxy"] = 1.0 - fused["satellite_clear_sky_index"]
    return fused.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def build_fusion_summary(
    fused: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    tolerance: str | pd.Timedelta = DEFAULT_ALIGNMENT_TOLERANCE,
) -> dict[str, Any]:
    """Build a JSON-serializable summary of source coverage, selection, and horizon support."""
    total_rows = max(len(fused), 1)
    source_coverage = {
        source_name: float(fused.get(f"{source_name}_ssi", pd.Series(index=fused.index, dtype="float64")).notna().mean() * 100.0)
        for source_name in SOURCE_PRIORITY
    }
    missing_data_percentage = float((~fused["satellite_data_available"]).mean() * 100.0)
    selected_primary_source = _selected_primary_source(fused)
    source_resolution_minutes = {
        source_name: _infer_resolution_minutes(frame)
        for source_name, frame in sources.items()
        if not frame.empty
    }
    selected_resolution = source_resolution_minutes.get(selected_primary_source) if selected_primary_source else None

    return {
        "source_coverage_percentage": source_coverage,
        "missing_data_percentage": missing_data_percentage,
        "selected_primary_source": selected_primary_source,
        "temporal_resolution": {
            "selected_primary_source_minutes": selected_resolution,
            "by_source_minutes": source_resolution_minutes,
            "alignment_tolerance_minutes": pd.Timedelta(tolerance).total_seconds() / 60.0,
        },
        "number_of_sites": int(fused["site_id"].nunique()) if "site_id" in fused else 0,
        "horizon_support": {
            "15_minutes": {
                "supported": bool(selected_resolution is not None and selected_resolution <= 15.0),
                "disabled": bool(selected_resolution is None or selected_resolution > 15.0),
            },
            "30_minutes": {
                "supported": bool(selected_resolution is not None and selected_resolution <= 30.0),
                "disabled": bool(selected_resolution is None or selected_resolution > 30.0),
            },
        },
        "rows": int(total_rows),
    }


def save_outputs(fused: pd.DataFrame, summary: dict[str, Any], project_root: str | Path) -> None:
    """Save fused parquet and JSON summary outputs."""
    root = Path(project_root)
    fused_path = root / FUSED_OUTPUT_PATH
    summary_path = root / SUMMARY_OUTPUT_PATH
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(fused_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    print(f"Saved fused dataset: {fused_path}")
    print(f"Saved fusion summary: {summary_path}")


def main() -> None:
    """Run source fusion from the command line."""
    parser = argparse.ArgumentParser(description="Fuse EUMETSAT, NASA POWER, and Open-Meteo irradiance sources.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--tolerance", default=None, help="Nearest timestamp alignment tolerance, e.g. 35min")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        tolerance = args.tolerance or f"{config.get('source_fusion', {}).get('alignment_tolerance_minutes', 35)}min"
        sources = load_available_sources(config["_project_root"])
        print(f"Loaded sources: {', '.join(sources) if sources else 'none'}")
        fused = fuse_sources(sources, config, tolerance=tolerance)
        summary = build_fusion_summary(fused, sources, tolerance=tolerance)
        save_outputs(fused, summary, config["_project_root"])
        print("Source fusion complete.")
    except Exception as exc:
        print(f"Source fusion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _load_eumetsat(path: Path) -> pd.DataFrame:
    """Load standardized EUMETSAT SSI parquet data."""
    frame = pd.read_parquet(path)
    return frame.rename(columns={"satellite_ssi": "ssi", "satellite_cloud_index": "cloud_index"})


def _load_nasa_power(path: Path) -> pd.DataFrame:
    """Load standardized NASA POWER parquet data."""
    frame = pd.read_parquet(path)
    return frame.rename(columns={"satellite_ssi": "ssi", "satellite_clear_sky_ssi": "clear_sky_ssi"})


def _load_openmeteo_physical(path: Path) -> pd.DataFrame:
    """Load Open-Meteo physical-baseline data as an irradiance fallback for Munich centre."""
    frame = pd.read_parquet(path).copy()
    frame["site_id"] = "munich_centre"
    frame["latitude"] = 48.137
    frame["longitude"] = 11.575
    frame["ssi"] = frame["shortwave_radiation"] if "shortwave_radiation" in frame else frame.get("GHI_phys")
    if "clear_sky_ghi" in frame:
        frame["clear_sky_ssi"] = frame["clear_sky_ghi"]
    return frame


def _load_openmeteo_raw(path: Path) -> pd.DataFrame:
    """Load raw Open-Meteo data as a Munich centre irradiance fallback."""
    frame = pd.read_csv(path).copy()
    frame["site_id"] = "munich_centre"
    frame["latitude"] = 48.137
    frame["longitude"] = 11.575
    frame["ssi"] = frame["shortwave_radiation"]
    return frame


def _normalize_source_frame(source_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp, site, and irradiance columns for one source."""
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce").dt.tz_convert(TARGET_TIMEZONE)
    normalized = normalized.dropna(subset=["timestamp", "site_id"]).sort_values(["site_id", "timestamp"])
    normalized = normalized.drop_duplicates(subset=["site_id", "timestamp"])
    keep = ["timestamp", "site_id", "latitude", "longitude", "ssi", "clear_sky_ssi", "cloud_index", "cloud_cover", "clear_sky_ghi"]
    available = [column for column in keep if column in normalized.columns]
    normalized = normalized[available].copy()
    rename = {
        "ssi": f"{source_name}_ssi",
        "clear_sky_ssi": f"{source_name}_clear_sky_ssi",
        "cloud_index": f"{source_name}_cloud_index",
        "cloud_cover": f"{source_name}_cloud_cover",
        "clear_sky_ghi": "openmeteo_clear_sky_ghi",
    }
    return normalized.rename(columns=rename)


def _build_base_grid(sources: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    """Build the fusion grid from all available source site/timestamp pairs."""
    pieces = []
    site_lookup = {site["id"]: site for site in config.get("sites", [])}
    for frame in sources.values():
        pieces.append(frame[["site_id", "timestamp"]].copy())
    base = pd.concat(pieces, ignore_index=True).drop_duplicates().sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    base["latitude"] = base["site_id"].map(lambda site_id: site_lookup.get(site_id, {}).get("latitude", np.nan))
    base["longitude"] = base["site_id"].map(lambda site_id: site_lookup.get(site_id, {}).get("longitude", np.nan))
    return base


def _merge_source_asof(base: pd.DataFrame, source: pd.DataFrame, source_name: str, tolerance: pd.Timedelta) -> pd.DataFrame:
    """Merge one source onto the base grid by nearest timestamp within tolerance and site."""
    merged_groups = []
    source_payload_columns = [column for column in source.columns if column not in {"site_id", "timestamp", "latitude", "longitude"}]
    for site_id, base_group in base.groupby("site_id", sort=False):
        source_group = source[source["site_id"] == site_id].sort_values("timestamp")
        if source_group.empty:
            empty = base_group.copy()
            for column in source_payload_columns:
                empty[column] = np.nan
            merged_groups.append(empty)
            continue
        merged = pd.merge_asof(
            base_group.sort_values("timestamp"),
            source_group[["timestamp", *source_payload_columns]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=tolerance,
        )
        merged_groups.append(merged)
    out = pd.concat(merged_groups, ignore_index=True).sort_values(["site_id", "timestamp"])
    return out.reset_index(drop=True)


def _best_clear_sky_column(fused: pd.DataFrame) -> pd.Series:
    """Return the best available clear-sky irradiance column for clear-sky-index derivation."""
    for column in ["openmeteo_clear_sky_ghi", "eumetsat_ssi_clear_sky_ssi", "nasa_power_clear_sky_ssi", "openmeteo_clear_sky_ssi"]:
        if column in fused.columns:
            series = pd.to_numeric(fused[column], errors="coerce")
            if series.notna().any():
                return series
    return pd.Series(1.0, index=fused.index)


def _selected_primary_source(fused: pd.DataFrame) -> str | None:
    """Return the highest-priority source with any selected irradiance."""
    for source_name in SOURCE_PRIORITY:
        if (fused["best_satellite_source"] == source_name).any():
            return source_name
    return None


def _infer_resolution_minutes(frame: pd.DataFrame) -> float | None:
    """Infer median temporal resolution in minutes from a source dataframe."""
    if frame.empty:
        return None
    times = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna().sort_values()
    if len(times) < 2:
        return None
    diffs = times.drop_duplicates().diff().dropna()
    if diffs.empty:
        return None
    return float(diffs.median().total_seconds() / 60.0)


if __name__ == "__main__":
    main()
