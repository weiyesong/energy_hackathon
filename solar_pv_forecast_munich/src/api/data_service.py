"""Data loading and response shaping for the SolarOps API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SolarOpsDataService:
    """Load live or demo SolarOps outputs for API endpoints."""

    def __init__(self, project_root: str | Path = PROJECT_ROOT) -> None:
        """Initialize the service with a project root."""
        self.project_root = Path(project_root)

    def data_mode(self) -> str:
        """Return live when live outputs exist, otherwise demo."""
        forced = os.getenv("SOLAROPS_DATA_MODE", "").strip().lower()
        if forced in {"demo", "live"}:
            return forced
        live = self.project_root / "outputs/live/forecast.csv"
        return "live" if live.exists() and live.stat().st_size > 0 else "demo"

    def demo_mode(self) -> bool:
        """Return whether the API is serving demo snapshot data."""
        return self.data_mode() == "demo"

    def load_forecast(self) -> pd.DataFrame:
        """Load the active forecast dataframe."""
        if self.data_mode() == "live":
            path = self.project_root / "outputs/live/forecast.csv"
        else:
            path = self.project_root / "outputs/demo/demo_forecast.csv"
        if not path.exists():
            fallback = self.project_root / "outputs/forecasts/operational_forecast.csv"
            if not fallback.exists():
                return pd.DataFrame()
            frame = pd.read_csv(fallback)
            frame["demo_mode"] = True
            frame["display_mode"] = "Demo Snapshot"
            return frame
        return pd.read_csv(path)

    def load_actions(self) -> list[dict[str, Any]]:
        """Load active operator actions."""
        path = self.project_root / ("outputs/live/operator_actions.json" if self.data_mode() == "live" else "outputs/demo/demo_operator_actions.json")
        if not path.exists():
            fallback = self.project_root / "outputs/operations/operator_actions.json"
            path = fallback
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("actions", []) if isinstance(payload, dict) else []

    def load_sites(self) -> pd.DataFrame:
        """Load active site ranking."""
        path = self.project_root / ("outputs/live/site_ranking.csv" if self.data_mode() == "live" else "outputs/demo/demo_site_ranking.csv")
        if not path.exists():
            path = self.project_root / "outputs/operations/site_ranking.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def load_source_status(self) -> dict[str, Any]:
        """Load active source status."""
        path = self.project_root / ("outputs/live/source_status.json" if self.data_mode() == "live" else "outputs/demo/demo_source_status.json")
        if not path.exists():
            path = self.project_root / "data/processed/data_source_registry.json"
        if not path.exists():
            return {"sources": []}
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "sources" not in payload and isinstance(payload, dict):
            payload = {"sources": [_registry_entry_to_source(name, entry) for name, entry in payload.items()]}
        return payload

    def overview(self) -> dict[str, Any]:
        """Build dashboard overview fields."""
        forecast = self.load_forecast()
        actions = self.load_actions()
        sources = self.load_source_status()
        if forecast.empty:
            return {
                "region": "Munich",
                "current_pv_output": 0.0,
                "next_peak_time": None,
                "next_peak_power": 0.0,
                "expected_daily_energy": 0.0,
                "forecast_risk": "unknown",
                "forecast_skill_vs_persistence": None,
                "primary_satellite_source": sources.get("primary_satellite_source", "unavailable"),
                "satellite_data_available": bool(sources.get("satellite_data_available", False)),
                "next_expected_pv_drop": None,
                "recommended_action": actions[0] if actions else None,
                "demo_mode": self.demo_mode(),
            }

        times = pd.to_datetime(forecast["target_valid_time"], utc=True, errors="coerce")
        ordered = forecast.assign(_time=times).sort_values("_time")
        first = ordered.iloc[0]
        peak_idx = pd.to_numeric(ordered["PV_P50"], errors="coerce").idxmax()
        peak = ordered.loc[peak_idx]
        expected_daily_energy = float(pd.to_numeric(ordered["PV_energy_P50"], errors="coerce").fillna(0.0).sum())
        return {
            "region": "Munich",
            "current_pv_output": _float(first.get("PV_P50")),
            "next_peak_time": _string_time(peak.get("target_valid_time")),
            "next_peak_power": _float(peak.get("PV_P50")),
            "expected_daily_energy": expected_daily_energy,
            "forecast_risk": _forecast_risk(ordered),
            "forecast_skill_vs_persistence": self._forecast_skill(),
            "primary_satellite_source": str(first.get("primary_satellite_source", sources.get("primary_satellite_source", "unavailable"))),
            "satellite_data_available": bool(first.get("satellite_data_available", sources.get("satellite_data_available", False))),
            "next_expected_pv_drop": _next_drop(ordered),
            "recommended_action": actions[0] if actions else None,
            "demo_mode": bool(first.get("demo_mode", self.demo_mode())),
        }

    def forecast_points(self, site_id: str | None = None) -> list[dict[str, Any]]:
        """Return forecast rows shaped for the UI."""
        forecast = self.load_forecast()
        if site_id is not None and not forecast.empty:
            forecast = forecast[forecast["site_id"] == site_id]
        if forecast.empty:
            return []
        return [_forecast_point(row) for _, row in forecast.sort_values(["target_valid_time", "site_id", "horizon_minutes"]).iterrows()]

    def benchmark(self) -> dict[str, Any]:
        """Return offline benchmark metrics."""
        metrics_path = self.project_root / "outputs/metrics/benchmark_metrics.csv"
        skill_path = self.project_root / "outputs/metrics/forecast_skill_by_horizon.csv"
        ablation_path = self.project_root / "outputs/metrics/satellite_ablation.csv"
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        skill = pd.read_csv(skill_path) if skill_path.exists() else pd.DataFrame()
        ablation = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
        model_rmse = _metric_value(metrics, "hybrid_satellite", "RMSE")
        persistence_rmse = _metric_value(metrics, "csi_persistence", "RMSE")
        return {
            "model_RMSE": model_rmse,
            "persistence_RMSE": persistence_rmse,
            "skill_score": _mean_column(skill, "skill_score_vs_csi_persistence") or _mean_column(skill, "skill_score"),
            "satellite_value_add": _mean_column(ablation, "satellite_value_add"),
            "evaluation_mode": "offline hindcast",
            "evaluation_period": _evaluation_period(),
        }

    def drivers(self) -> list[dict[str, Any]]:
        """Return limiting-factor distribution for UI driver panels."""
        forecast = self.load_forecast()
        if forecast.empty or "main_limiting_factor" not in forecast:
            return []
        counts = forecast["main_limiting_factor"].fillna("unknown").value_counts()
        total = max(float(counts.sum()), 1.0)
        return [{"driver": str(name), "count": int(count), "share": float(count / total)} for name, count in counts.items()]

    def data_sources(self) -> list[dict[str, Any]]:
        """Return source transparency records."""
        status = self.load_source_status()
        return status.get("sources", [])

    def _forecast_skill(self) -> float | None:
        """Return mean forecast skill from saved metrics."""
        path = self.project_root / "outputs/metrics/forecast_skill_by_horizon.csv"
        if not path.exists():
            return None
        skill = pd.read_csv(path)
        return _mean_column(skill, "skill_score_vs_csi_persistence") or _mean_column(skill, "skill_score")


def _forecast_point(row: pd.Series) -> dict[str, Any]:
    """Convert one forecast row into API response fields."""
    cloud_cover = _float(row.get("target_cloud_cover_forecast_proxy", row.get("cloud_cover_issue")))
    limiting = row.get("main_limiting_factor")
    return {
        "target_time": _string_time(row.get("target_valid_time")),
        "horizon_minutes": _int(row.get("horizon_minutes")),
        "GHI_P10": _float(row.get("GHI_P10_calibrated")),
        "GHI_P50": _float(row.get("GHI_P50")),
        "GHI_P90": _float(row.get("GHI_P90_calibrated")),
        "persistence_GHI": _float(row.get("GHI_persistence_csi", row.get("GHI_persistence_naive"))),
        "POA_P50": _float(row.get("POA_P50")),
        "PV_P10": _float(row.get("PV_P10")),
        "PV_P50": _float(row.get("PV_P50")),
        "PV_P90": _float(row.get("PV_P90")),
        "cloud_cover": cloud_cover,
        "solar_elevation": _float(row.get("target_solar_elevation")),
        "main_limiting_factor": None if pd.isna(limiting) else str(limiting),
        "uncertainty_level": None if pd.isna(row.get("uncertainty_level")) else str(row.get("uncertainty_level")),
        "cloud_event": str(limiting) in {"low cloud", "mid/high cloud", "high cloud variability"},
    }


def _registry_entry_to_source(source_name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw registry entry to API source fields."""
    return {
        "source_name": source_name,
        "source_role": entry.get("source_type", "unknown"),
        "status": entry.get("download_status", "unknown"),
        "last_update": entry.get("last_update_time"),
        "temporal_resolution": entry.get("temporal_resolution"),
        "coverage": entry.get("date_range", {}),
        "manual_action_required": bool(entry.get("manual_action_required", False)),
        "fallback_active": source_name != "eumetsat_ssi",
    }


def _metric_value(metrics: pd.DataFrame, model_name: str, metric_name: str) -> float | None:
    """Read a metric value from benchmark metrics with flexible column names."""
    if metrics.empty:
        return None
    frame = metrics.copy()
    model_col = "model" if "model" in frame.columns else "forecast_type" if "forecast_type" in frame.columns else None
    metric_col = "metric" if "metric" in frame.columns else None
    if model_col and metric_col and "value" in frame.columns:
        matched = frame[(frame[model_col].astype(str) == model_name) & (frame[metric_col].astype(str) == metric_name)]
        if not matched.empty:
            return _float(matched.iloc[0]["value"])
    if model_col and "rmse" in frame.columns:
        matched = frame[frame[model_col].astype(str) == model_name]
        if not matched.empty:
            return _float(pd.to_numeric(matched["rmse"], errors="coerce").mean())
    rmse_cols = [c for c in frame.columns if c.lower() == "rmse"]
    if rmse_cols:
        return _float(frame[rmse_cols[0]].mean())
    return None


def _mean_column(df: pd.DataFrame, column: str) -> float | None:
    """Return mean of a column when available."""
    if df.empty or column not in df.columns:
        return None
    return _float(pd.to_numeric(df[column], errors="coerce").mean())


def _forecast_risk(df: pd.DataFrame) -> str:
    """Classify forecast risk from uncertainty and source transparency."""
    if "uncertainty_level" in df and (df["uncertainty_level"].astype(str) == "High").mean() > 0.2:
        return "high"
    if "fallback_active" in df and bool(df["fallback_active"].iloc[0]):
        return "medium"
    return "low"


def _next_drop(df: pd.DataFrame) -> dict[str, Any] | None:
    """Find the next substantial PV drop in the active forecast."""
    ordered = df.sort_values("target_valid_time").copy()
    pv = pd.to_numeric(ordered["PV_P50"], errors="coerce").fillna(0.0)
    drop = pv.pct_change().fillna(0.0)
    candidates = ordered[drop <= -0.30]
    if candidates.empty:
        return None
    row = candidates.iloc[0]
    return {"time": _string_time(row.get("target_valid_time")), "drop_fraction": _float(abs(drop.loc[row.name]))}


def _evaluation_period() -> str | None:
    """Return the current offline evaluation period description."""
    return "2025-07-01 to 2025-12-31"


def _float(value: Any) -> float | None:
    """Convert values to JSON-safe floats."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _int(value: Any) -> int | None:
    """Convert values to JSON-safe ints."""
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_time(value: Any) -> str | None:
    """Return a string timestamp or None."""
    if value is None or pd.isna(value):
        return None
    return str(value)
