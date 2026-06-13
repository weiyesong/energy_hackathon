"""Operational forecast generation and recommendation rules for SolarOps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operations.site_ranking import rank_sites, save_site_ranking
from src.physics.irradiance_decomposition import decompose_ghi_quantiles
from src.physics.poa_model import compute_poa_quantiles
from src.physics.pv_power_model import estimate_pv_power_quantiles


def load_config(config_path: str | Path) -> dict:
    """Load the project YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_operational_forecast(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert probabilistic GHI predictions into operational PV forecasts."""
    pv_system = config.get("pv_system", {})
    forecast = decompose_ghi_quantiles(predictions)
    forecast = compute_poa_quantiles(
        forecast,
        surface_tilt=float(pv_system.get("surface_tilt", 35.0)),
        surface_azimuth=float(pv_system.get("surface_azimuth", 180.0)),
        albedo=float(pv_system.get("albedo_default", 0.2)),
    )
    forecast = estimate_pv_power_quantiles(forecast, pv_system)
    forecast = add_limiting_factors(forecast)
    return forecast


def add_limiting_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic main limiting factor labels and JSON evidence."""
    out = df.copy()
    labels: list[str] = []
    evidence: list[str] = []
    for _, row in out.iterrows():
        label, details = determine_main_limiting_factor(row)
        labels.append(label)
        evidence.append(json.dumps(details, sort_keys=True))
    out["main_limiting_factor"] = labels
    out["limiting_factor_evidence"] = evidence
    return out


def determine_main_limiting_factor(row: pd.Series) -> tuple[str, dict[str, Any]]:
    """Return a limiting-factor label and supporting evidence for one forecast row."""
    solar_elevation = _num(row, "target_solar_elevation")
    low_cloud = _num(row, "target_cloud_cover_low_forecast_proxy")
    mid_cloud = _num(row, "target_cloud_cover_mid_forecast_proxy")
    high_cloud = _num(row, "target_cloud_cover_high_forecast_proxy")
    cloud_cover = _num(row, "target_cloud_cover_forecast_proxy", _num(row, "cloud_cover_issue"))
    cloud_trend = abs(_num(row, "cloud_cover_trend"))
    module_temperature = _num(row, "module_temperature")
    source_std = _num(row, "irradiance_source_std")
    ghi_p50 = max(_num(row, "GHI_P50"), 1.0)
    snow_depth = _num(row, "snow_depth")

    if solar_elevation <= 10.0:
        return "low sun angle", {"target_solar_elevation": round(solar_elevation, 2)}
    if snow_depth > 0.02:
        return "snow", {"snow_depth_m": round(snow_depth, 3)}
    if _has_aerosol_signal(row):
        return "aerosol attenuation", {"aerosol_data_present": True, "T_aerosol": _num(row, "T_aerosol", np.nan)}
    if module_temperature >= 60.0:
        return "high module temperature", {"module_temperature": round(module_temperature, 1)}
    if source_std >= max(75.0, 0.25 * ghi_p50):
        return "source disagreement", {"irradiance_source_std": round(source_std, 1), "GHI_P50": round(ghi_p50, 1)}
    if cloud_trend >= 30.0:
        return "high cloud variability", {"cloud_cover_trend": round(_num(row, "cloud_cover_trend"), 1)}
    if low_cloud >= 60.0 or (np.isnan(low_cloud) and cloud_cover >= 75.0):
        return "low cloud", {"low_cloud_percent": _safe_round(low_cloud), "cloud_cover_percent": _safe_round(cloud_cover)}
    if mid_cloud >= 55.0 or high_cloud >= 55.0 or cloud_cover >= 60.0:
        return "mid/high cloud", {
            "mid_cloud_percent": _safe_round(mid_cloud),
            "high_cloud_percent": _safe_round(high_cloud),
            "cloud_cover_percent": _safe_round(cloud_cover),
        }
    return "normal conditions", {"cloud_cover_percent": _safe_round(cloud_cover), "PV_state_index": _safe_round(_num(row, "PV_state_index"))}


def generate_operator_actions(df: pd.DataFrame, max_actions_per_site: int = 5) -> list[dict[str, Any]]:
    """Generate structured operational suggestions from the PV forecast."""
    if df.empty:
        return []

    actions: list[dict[str, Any]] = []
    working = df.copy()
    if "operational_status" in working.columns:
        operational = working["operational_status"].astype(str).eq("Operational")
        if operational.any():
            working = working[operational].copy()
    working["target_valid_time"] = _to_berlin_time(working["target_valid_time"])
    horizon = int(pd.to_numeric(working["horizon_minutes"], errors="coerce").dropna().min())
    horizon_df = working[working["horizon_minutes"] == horizon].copy()
    if horizon_df.empty:
        horizon_df = working.copy()

    for site_id, site_df in horizon_df.groupby("site_id", dropna=False):
        site_actions = _site_actions(site_df.sort_values("target_valid_time"), str(site_id), horizon)
        if not site_actions:
            site_actions = [_no_intervention_action(site_df, str(site_id), horizon)]
        actions.extend(site_actions[:max_actions_per_site])
    return actions


def save_operator_actions(actions: list[dict[str, Any]], path: str | Path) -> None:
    """Save operational suggestions as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"actions": actions, "count": len(actions)}, f, indent=2, default=str)


def run_operational_pipeline(
    config_path: str | Path = "config.yaml",
    input_path: str | Path = "outputs/forecasts/test_probabilistic_predictions.csv",
    forecast_output: str | Path = "outputs/forecasts/operational_forecast.csv",
    operations_dir: str | Path = "outputs/operations",
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, dict]:
    """Run the full operational-product pipeline and save all requested outputs."""
    config = load_config(config_path)
    predictions = pd.read_csv(input_path)
    forecast = build_operational_forecast(predictions, config)

    forecast_path = Path(forecast_output)
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(forecast_path, index=False)

    actions = generate_operator_actions(forecast)
    operations_path = Path(operations_dir)
    operations_path.mkdir(parents=True, exist_ok=True)
    save_operator_actions(actions, operations_path / "operator_actions.json")

    ranking, summary = rank_sites(forecast, config)
    save_site_ranking(ranking, summary, operations_path)
    return forecast, actions, ranking, summary


def _site_actions(site_df: pd.DataFrame, site_id: str, horizon: int) -> list[dict[str, Any]]:
    """Generate action suggestions for a single site."""
    actions: list[dict[str, Any]] = []
    pv = pd.to_numeric(site_df["PV_P50"], errors="coerce").fillna(0.0)
    capacity_reference = max(float(pv.max()), 0.1)
    uncertainty_width = (pd.to_numeric(site_df["PV_P90"], errors="coerce") - pd.to_numeric(site_df["PV_P10"], errors="coerce")).clip(lower=0.0)
    relative_uncertainty = uncertainty_width / pv.clip(lower=0.05)

    drops = pv.pct_change().fillna(0.0)
    drop_candidates = site_df[(drops <= -0.30) & (pv.shift(1).fillna(0.0) >= 0.25 * capacity_reference)]
    for _, row in drop_candidates.head(2).iterrows():
        previous_power = float(pv.shift(1).loc[row.name])
        current_power = _num(row, "PV_P50")
        reduction = 100.0 * max(previous_power - current_power, 0.0) / max(previous_power, 0.01)
        actions.append(
            _action(
                "battery_charge",
                "high",
                row,
                horizon,
                f"Charge battery before expected PV drop of {reduction:.0f}%",
                _confidence(row),
                site_id,
            )
        )

    uncertain = site_df[relative_uncertainty >= 0.8]
    for _, row in uncertain.head(1).iterrows():
        actions.append(
            _action(
                "monitor_forecast",
                "medium",
                row,
                horizon,
                "Monitor forecast because uncertainty is high",
                "medium",
                site_id,
            )
        )

    preserve = site_df[(relative_uncertainty >= 0.6) & (pv <= 0.35 * capacity_reference)]
    for _, row in preserve.head(1).iterrows():
        actions.append(
            _action(
                "preserve_battery",
                "medium",
                row,
                horizon,
                "Preserve battery during uncertain low-PV period",
                "medium",
                site_id,
            )
        )

    high_solar = site_df[(pv >= 0.70 * capacity_reference) & (relative_uncertainty <= 0.45)]
    for _, row in high_solar.head(1).iterrows():
        actions.append(
            _action(
                "shift_flexible_load",
                "medium",
                row,
                horizon,
                "Shift flexible load into high-solar window",
                _confidence(row),
                site_id,
            )
        )

    constrained = site_df[
        site_df["main_limiting_factor"].isin(["low cloud", "mid/high cloud", "source disagreement", "high cloud variability"])
        & (pv <= 0.50 * capacity_reference)
    ]
    for _, row in constrained.head(1).iterrows():
        actions.append(
            _action(
                "reduce_feed_in_commitment",
                "medium",
                row,
                horizon,
                f"Reduce expected feed-in commitment because {row['main_limiting_factor']} is limiting PV output",
                _confidence(row),
                site_id,
            )
        )
    return _dedupe_actions(actions)


def _action(
    action_type: str,
    priority: str,
    row: pd.Series,
    horizon: int,
    reason: str,
    confidence: str,
    site_id: str,
) -> dict[str, Any]:
    """Create a structured operator-action record."""
    start = pd.to_datetime(row["target_valid_time"], errors="coerce")
    end = start + pd.Timedelta(minutes=horizon) if pd.notna(start) else pd.NaT
    return {
        "site_id": site_id,
        "action_type": action_type,
        "priority": priority,
        "valid_from": None if pd.isna(start) else start.isoformat(),
        "valid_until": None if pd.isna(end) else end.isoformat(),
        "reason": reason,
        "confidence": confidence,
        "basis": "operational suggestion from probabilistic PV forecast",
    }


def _no_intervention_action(site_df: pd.DataFrame, site_id: str, horizon: int) -> dict[str, Any]:
    """Create a default no-intervention suggestion for a site."""
    row = site_df.iloc[0] if not site_df.empty else pd.Series({"target_valid_time": pd.Timestamp.utcnow()})
    return _action(
        "no_intervention",
        "low",
        row,
        horizon,
        "No intervention required under current forecast conditions",
        "medium",
        site_id,
    )


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate actions for the same site, type, and start time."""
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[dict[str, Any]] = []
    for action in actions:
        key = (action["site_id"], action["action_type"], action["valid_from"])
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return unique


def _confidence(row: pd.Series) -> str:
    """Map uncertainty level and data availability to a qualitative confidence label."""
    if str(row.get("uncertainty_level", "")).lower() == "high":
        return "low"
    if bool(row.get("satellite_data_available", False)):
        return "medium"
    return "low"


def _to_berlin_time(values: pd.Series) -> pd.Series:
    """Parse mixed-offset timestamps and convert them to Europe/Berlin."""
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")


def _has_aerosol_signal(row: pd.Series) -> bool:
    """Return true only when real aerosol fields exist and indicate attenuation."""
    if "AOD_550" in row.index and pd.notna(row.get("AOD_550")):
        return _num(row, "AOD_550") > 0.2
    if "aerosol_optical_depth" in row.index and pd.notna(row.get("aerosol_optical_depth")):
        return _num(row, "aerosol_optical_depth") > 0.2
    if "T_aerosol" in row.index and pd.notna(row.get("T_aerosol")):
        return _num(row, "T_aerosol", 1.0) < 0.92
    return False


def _num(row: pd.Series, key: str, default: float = np.nan) -> float:
    """Read a numeric value from a row with a default."""
    try:
        value = pd.to_numeric(row.get(key, default), errors="coerce")
    except Exception:
        return default
    return float(value) if pd.notna(value) else default


def _safe_round(value: float) -> float | None:
    """Round finite values and return None for missing values."""
    return None if pd.isna(value) else round(float(value), 2)


def main() -> None:
    """CLI entry point for the operational-product layer."""
    parser = argparse.ArgumentParser(description="Generate SolarOps operational PV forecasts and recommendations.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default="outputs/forecasts/test_probabilistic_predictions.csv")
    parser.add_argument("--forecast-output", default="outputs/forecasts/operational_forecast.csv")
    parser.add_argument("--operations-dir", default="outputs/operations")
    args = parser.parse_args()

    forecast, actions, ranking, _ = run_operational_pipeline(
        config_path=args.config,
        input_path=args.input,
        forecast_output=args.forecast_output,
        operations_dir=args.operations_dir,
    )
    print(f"Saved operational forecast: {args.forecast_output} ({len(forecast)} rows)")
    print(f"Saved operator actions: {Path(args.operations_dir) / 'operator_actions.json'} ({len(actions)} actions)")
    print(f"Saved site ranking: {Path(args.operations_dir) / 'site_ranking.csv'} ({len(ranking)} sites)")


if __name__ == "__main__":
    main()
