"""Prepare a reproducible demo snapshot from existing processed SolarOps outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operations.decision_engine import generate_operator_actions
from src.operations.site_ranking import rank_sites
from src.pipeline.source_health import inspect_source_health, write_source_status

DEMO_OUTPUT_DIR = Path("outputs/demo")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load project configuration with project-root metadata."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_project_root"] = str(path.parent)
    return config


def prepare_demo_snapshot(
    config_path: str | Path = "config.yaml",
    input_forecast: str | Path = "outputs/forecasts/operational_forecast.csv",
    output_dir: str | Path = DEMO_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Create deterministic demo files from existing real processed forecast outputs."""
    config = load_config(config_path)
    root = Path(config["_project_root"])
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(root / input_forecast)
    demo = _select_demo_window(forecast)
    demo = _mark_demo(demo)
    demo.to_csv(output / "demo_forecast.csv", index=False)

    actions = generate_operator_actions(demo)
    for action in actions:
        action["display_mode"] = "Demo Snapshot"
        action["demo_mode"] = True
        action["basis"] = "demo snapshot from cached processed data, not a live forecast"
    with (output / "demo_operator_actions.json").open("w", encoding="utf-8") as f:
        json.dump({"actions": actions, "count": len(actions), "demo_mode": True, "display_mode": "Demo Snapshot"}, f, indent=2, default=str)

    ranking, summary = rank_sites(demo, config)
    ranking.to_csv(output / "demo_site_ranking.csv", index=False)
    summary["demo_mode"] = True
    summary["display_mode"] = "Demo Snapshot"
    with (output / "demo_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "demo_mode": True,
                "display_mode": "Demo Snapshot",
                "source": str(input_forecast),
                "rows": int(len(demo)),
                "sites": sorted(demo["site_id"].dropna().astype(str).unique().tolist()),
                "start_time": str(pd.to_datetime(demo["target_valid_time"], utc=True, errors="coerce").min()),
                "end_time": str(pd.to_datetime(demo["target_valid_time"], utc=True, errors="coerce").max()),
                "site_ranking_summary": summary,
                "note": "Demo Snapshot uses cached real processed data and must not be described as live.",
            },
            f,
            indent=2,
            default=str,
        )

    source_status = inspect_source_health(config, root / "data/processed/data_source_registry.json", root / "outputs/models")
    source_status["demo_mode"] = True
    source_status["display_mode"] = "Demo Snapshot"
    source_status["note"] = "Demo source status describes cached data used for reproducible presentation, not a live source check."
    write_source_status(source_status, output / "demo_source_status.json")

    print(f"Saved demo forecast: {output / 'demo_forecast.csv'} ({len(demo)} rows)")
    print(f"Saved demo snapshot metadata: {output / 'demo_snapshot.json'}")
    print(f"Saved demo source status: {output / 'demo_source_status.json'}")
    return demo, ranking, actions, source_status


def _select_demo_window(forecast: pd.DataFrame) -> pd.DataFrame:
    """Select a deterministic 24-hour-ish daylight-rich demo window from cached forecast data."""
    data = forecast.copy()
    data["target_valid_time"] = pd.to_datetime(data["target_valid_time"], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    data = data.dropna(subset=["target_valid_time"]).sort_values(["target_valid_time", "site_id", "horizon_minutes"])
    daily_energy = data.groupby(data["target_valid_time"].dt.date)["PV_energy_P50"].sum().sort_values(ascending=False)
    selected_date = daily_energy.index[0] if not daily_energy.empty else data["target_valid_time"].dt.date.iloc[0]
    window = data[data["target_valid_time"].dt.date == selected_date].copy()
    if window.empty:
        window = data.head(500).copy()
    return window.reset_index(drop=True)


def _mark_demo(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit demo metadata to every demo row."""
    out = df.copy()
    out["demo_mode"] = True
    out["display_mode"] = "Demo Snapshot"
    out["fallback_active"] = out.get("fallback_active", True)
    out["data_quality_level"] = out.get("data_quality_level", "demo_cached")
    out["primary_satellite_source"] = out.get("primary_satellite_source", out.get("best_satellite_source", "cached_demo"))
    out["weather_forecast_source"] = out.get("weather_forecast_source", "cached_demo")
    out["data_freshness_minutes"] = out.get("data_freshness_minutes", None)
    return out


def main() -> None:
    """CLI entry point for demo snapshot preparation."""
    parser = argparse.ArgumentParser(description="Prepare a reproducible SolarOps demo snapshot.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input-forecast", default="outputs/forecasts/operational_forecast.csv")
    parser.add_argument("--output-dir", default=str(DEMO_OUTPUT_DIR))
    args = parser.parse_args()
    prepare_demo_snapshot(args.config, args.input_forecast, args.output_dir)


if __name__ == "__main__":
    main()
