"""Transparent site ranking for operational SolarOps forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def rank_sites(df: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Calculate operational site metrics, a transparent score, and A-D ranks."""
    if df.empty:
        empty = pd.DataFrame(columns=_ranking_columns())
        return empty, {"number_of_sites": 0, "ranking_method": _score_formula()}

    working = df.copy()
    if "operational_status" in working.columns:
        operational = working["operational_status"].astype(str).eq("Operational")
        if operational.any():
            working = working[operational].copy()
    working["target_valid_time"] = pd.to_datetime(working["target_valid_time"], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    selected_horizon = _select_ranking_horizon(working)
    horizon_df = working[working["horizon_minutes"] == selected_horizon].copy()
    if horizon_df.empty:
        horizon_df = working.copy()

    rows = []
    for site_id, site_df in horizon_df.groupby("site_id", dropna=False):
        site_df = site_df.sort_values("target_valid_time")
        daily_energy = _expected_daily_energy(site_df)
        peak_power = pd.to_numeric(site_df.get("PV_P50", 0.0), errors="coerce").max()
        uncertainty_width = _mean_uncertainty_width(site_df)
        cloud_risk = _cloud_risk(site_df)
        volatility = _forecast_volatility(site_df)
        data_quality = _data_quality(site_df)
        rows.append(
            {
                "site_id": site_id,
                "expected_daily_energy": daily_energy,
                "peak_PV_P50": float(peak_power) if pd.notna(peak_power) else 0.0,
                "mean_uncertainty_width": uncertainty_width,
                "cloud_risk": cloud_risk,
                "forecast_volatility": volatility,
                "data_quality": data_quality,
                "ranking_horizon_minutes": int(selected_horizon),
            }
        )

    ranking = pd.DataFrame(rows)
    ranking["normalized_energy"] = _normalize_positive(ranking["expected_daily_energy"])
    ranking["confidence_score"] = 1.0 - _normalize_positive(ranking["mean_uncertainty_width"])
    ranking["data_quality_score"] = ranking["data_quality"].clip(0.0, 1.0)
    ranking["normalized_cloud_risk"] = ranking["cloud_risk"].clip(0.0, 1.0)
    ranking["normalized_volatility"] = _normalize_positive(ranking["forecast_volatility"])
    ranking["site_score"] = (
        0.45 * ranking["normalized_energy"]
        + 0.20 * ranking["confidence_score"]
        + 0.15 * ranking["data_quality_score"]
        - 0.10 * ranking["normalized_cloud_risk"]
        - 0.10 * ranking["normalized_volatility"]
    ).clip(lower=0.0)
    ranking = ranking.sort_values(["site_score", "expected_daily_energy"], ascending=False).reset_index(drop=True)
    ranking["rank_position"] = np.arange(1, len(ranking) + 1)
    ranking["rank_grade"] = ranking["site_score"].map(_grade_score)

    summary = {
        "number_of_sites": int(ranking["site_id"].nunique()),
        "ranking_horizon_minutes": int(selected_horizon),
        "top_site": None if ranking.empty else str(ranking.iloc[0]["site_id"]),
        "ranking_method": _score_formula(),
        "notes": [
            "expected_daily_energy is computed from the shortest available operational horizon to avoid mixing horizons",
            "site_score is transparent and heuristic; it supports operations review rather than guaranteed financial outcomes",
        ],
    }
    return ranking[_ranking_columns()], summary


def save_site_ranking(ranking: pd.DataFrame, summary: dict, output_dir: str | Path) -> None:
    """Save site ranking CSV and JSON summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_path / "site_ranking.csv", index=False)
    with (output_path / "site_ranking_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def _select_ranking_horizon(df: pd.DataFrame) -> int:
    """Choose the shortest available horizon for site ranking."""
    horizons = pd.to_numeric(df.get("horizon_minutes"), errors="coerce").dropna()
    return int(horizons.min()) if not horizons.empty else 60


def _expected_daily_energy(df: pd.DataFrame) -> float:
    """Estimate mean daily P50 energy in kWh for the selected horizon."""
    power = pd.to_numeric(df.get("PV_P50", 0.0), errors="coerce").fillna(0.0)
    minutes = pd.to_numeric(df.get("horizon_minutes", 60), errors="coerce").fillna(60.0)
    energy = power * minutes / 60.0
    dates = df["target_valid_time"].dt.date
    daily = energy.groupby(dates).sum()
    return float(daily.mean()) if not daily.empty else 0.0


def _mean_uncertainty_width(df: pd.DataFrame) -> float:
    """Return mean PV P90-P10 interval width in kW."""
    width = pd.to_numeric(df.get("PV_P90", 0.0), errors="coerce") - pd.to_numeric(df.get("PV_P10", 0.0), errors="coerce")
    return float(width.clip(lower=0.0).mean()) if len(width) else 0.0


def _cloud_risk(df: pd.DataFrame) -> float:
    """Estimate mean cloud risk as a normalized 0-1 value."""
    cloud = pd.to_numeric(
        df.get("target_cloud_cover_forecast_proxy", df.get("cloud_cover_issue", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    return float((cloud / 100.0).clip(0.0, 1.0).mean())


def _forecast_volatility(df: pd.DataFrame) -> float:
    """Estimate forecast volatility from absolute sequential PV changes."""
    pv = pd.to_numeric(df.get("PV_P50", 0.0), errors="coerce").fillna(0.0)
    return float(pv.diff().abs().fillna(0.0).mean())


def _data_quality(df: pd.DataFrame) -> float:
    """Estimate data quality from satellite availability and missing PV values."""
    satellite = df.get("satellite_data_available", pd.Series(False, index=df.index))
    satellite_score = pd.Series(satellite, index=df.index).fillna(False).astype(bool).mean()
    pv_complete = 1.0 - pd.to_numeric(df.get("PV_P50", np.nan), errors="coerce").isna().mean()
    quality_flag = df.get("quality_flag", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    known_quality = (quality_flag.str.lower() != "missing").mean()
    return float(np.mean([satellite_score, pv_complete, known_quality]))


def _normalize_positive(series: pd.Series) -> pd.Series:
    """Min-max normalize a positive metric and handle constant values gracefully."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = values.min()
    maximum = values.max()
    if not np.isfinite(maximum - minimum) or maximum == minimum:
        return pd.Series(1.0 if maximum > 0 else 0.0, index=series.index)
    return ((values - minimum) / (maximum - minimum)).clip(0.0, 1.0)


def _grade_score(score: float) -> str:
    """Map a site score to an A-D operational rank."""
    if score >= 0.75:
        return "A"
    if score >= 0.55:
        return "B"
    if score >= 0.35:
        return "C"
    return "D"


def _score_formula() -> str:
    """Return the transparent site-score formula."""
    return "0.45*normalized_energy + 0.20*confidence_score + 0.15*data_quality_score - 0.10*cloud_risk - 0.10*volatility"


def _ranking_columns() -> list[str]:
    """Return the public ranking column order."""
    return [
        "rank_position",
        "rank_grade",
        "site_id",
        "site_score",
        "expected_daily_energy",
        "peak_PV_P50",
        "mean_uncertainty_width",
        "cloud_risk",
        "forecast_volatility",
        "data_quality",
        "normalized_energy",
        "confidence_score",
        "data_quality_score",
        "normalized_cloud_risk",
        "normalized_volatility",
        "ranking_horizon_minutes",
    ]
