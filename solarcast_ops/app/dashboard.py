from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components import info_box, kpi, warning_box
from src.baseline_schedule import build_next_24h_schedule
from src.config import get_paths, load_config
from src.decision_engine import assess_grid_risk, make_trading_decision
from src.live_data import fallback_forecast_from_schedule, fetch_open_meteo_solar_forecast, prediction_interval_from_series
from src.utils import read_json

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@st.cache_data
def load_assets() -> tuple[
    dict,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    list[dict],
    dict,
]:
    config = load_config()
    paths = get_paths()
    preds_path = paths.predictions_dir / "test_predictions_with_uncertainty.csv"
    metrics_path = paths.metrics_dir / "metrics.csv"
    history_path = paths.processed_dir / "clean_hourly.csv"
    irradiance_path = paths.predictions_dir / "irradiance_test_predictions.csv"
    irradiance_metrics_path = paths.metrics_dir / "irradiance_metrics.csv"
    asof_path = paths.predictions_dir / "asof_backtest_predictions.csv"
    asof_metrics_path = paths.metrics_dir / "asof_backtest_metrics.csv"
    demo_path = paths.demo_dir / "demo_cases.json"
    meta_path = paths.processed_dir / "data_metadata.json"
    preds = pd.read_csv(preds_path) if preds_path.exists() else None
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    history = pd.read_csv(history_path) if history_path.exists() else None
    irradiance = pd.read_csv(irradiance_path) if irradiance_path.exists() else None
    irradiance_metrics = pd.read_csv(irradiance_metrics_path) if irradiance_metrics_path.exists() else None
    asof = pd.read_csv(asof_path) if asof_path.exists() else None
    asof_metrics = pd.read_csv(asof_metrics_path) if asof_metrics_path.exists() else None
    demos = read_json(demo_path) if demo_path.exists() else []
    meta = read_json(meta_path) if meta_path.exists() else {}
    if preds is not None:
        preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
    if history is not None:
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    if irradiance is not None:
        irradiance["timestamp"] = pd.to_datetime(irradiance["timestamp"], utc=True)
    if asof is not None:
        for col in ["asof_time", "visible_data_cutoff", "train_label_cutoff", "valid_time"]:
            if col in asof:
                asof[col] = pd.to_datetime(asof[col], utc=True)
    return config, preds, metrics, history, irradiance, irradiance_metrics, asof, asof_metrics, demos, meta


@st.cache_data(ttl=900)
def load_live_24h_assets(
    config: dict[str, Any],
    history: pd.DataFrame,
    start_time_iso: str,
    schedule_model: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], bool]:
    schedule_result = build_next_24h_schedule(history, pd.Timestamp(start_time_iso), schedule_model=schedule_model)
    try:
        live_result = fetch_open_meteo_solar_forecast(config, pd.Timestamp(start_time_iso))
        used_fallback = False
    except Exception as exc:
        live_result = fallback_forecast_from_schedule(schedule_result.schedule, str(exc))
        used_fallback = True
    live = live_result.forecast.merge(
        schedule_result.schedule[["timestamp", "scheduled_power_mw", "schedule_model_a_mw", "schedule_model_b_mw"]],
        on="timestamp",
        how="left",
    )
    live["scheduled_power_mw"] = live["scheduled_power_mw"].interpolate(limit_direction="both").ffill().bfill()
    p10, p50, p90 = prediction_interval_from_series(live["forecast_power_mw"], live["scheduled_power_mw"])
    live["forecast_p10_mw"] = p10
    live["forecast_p50_mw"] = p50
    live["forecast_p90_mw"] = p90
    return live, schedule_result.metadata, live_result.metadata, used_fallback


def _fmt(value: float | int | None, unit: str = "", digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f} {unit}".strip()


def _fmt_eur(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"EUR {float(value):,.{digits}f}"


def _pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _row_at(preds: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
    idx = (preds["timestamp"] - timestamp).abs().idxmin()
    return preds.loc[idx]


def _metric_value(metrics: pd.DataFrame | None, horizon: int, model: str, segment: str, column: str) -> float | None:
    if metrics is None:
        return None
    subset = metrics[
        (metrics["horizon_h"] == horizon)
        & (metrics["model"] == model)
        & (metrics["segment"] == segment)
    ]
    if subset.empty:
        return None
    return float(subset.iloc[0][column])


def _weather_label(value: Any) -> str:
    labels = {
        "clear": "Clear sky",
        "variable_cloud": "Variable cloud",
        "overcast": "Overcast",
    }
    return labels.get(str(value), str(value) if value is not None else "Unknown")


def _decision_headline(action: str, trade_mwh: float, avoided_cost_eur: float, grid_level: str, horizon: int) -> str:
    if action == "HOLD":
        return (
            f"{grid_level} ramp risk in the next {horizon}h. Hold position; "
            f"forecast deviation is below the trading threshold."
        )
    return (
        f"{grid_level} ramp risk in the next {horizon}h. "
        f"{action} {_fmt(trade_mwh, 'MWh')} now to avoid about {_fmt_eur(avoided_cost_eur)} on this event."
    )


def _portfolio_preview(
    row: pd.Series,
    horizon: int,
    duration: float,
    buy_price: float,
    sell_price: float,
    shortage_price: float,
    surplus_price: float,
    min_trade: float,
    confidence: float,
    risk_mode: str,
    ramp_threshold_fraction: float,
) -> pd.DataFrame:
    assets = [
        {"site": "Munich North", "capacity_mw": 1.0, "schedule_fraction": 0.65, "signal": 1.00},
        {"site": "Augsburg West", "capacity_mw": 3.4, "schedule_fraction": 0.58, "signal": 0.82},
        {"site": "Nuremberg South", "capacity_mw": 7.5, "schedule_fraction": 0.61, "signal": 1.08},
        {"site": "Regensburg East", "capacity_mw": 12.0, "schedule_fraction": 0.52, "signal": 0.72},
        {"site": "Stuttgart PPA", "capacity_mw": 18.0, "schedule_fraction": 0.49, "signal": 0.92},
        {"site": "Bavaria Aggregate", "capacity_mw": 45.0, "schedule_fraction": 0.55, "signal": 0.78},
    ]
    rows = []
    signal_capacity = float(row.get("_signal_capacity_mw", 1.0))
    base_current = float(row["pv_power_mw"]) / max(signal_capacity, 1e-6)
    for asset in assets:
        capacity = float(asset["capacity_mw"])
        signal = float(asset["signal"])
        current = min(capacity, max(0.0, base_current * capacity * signal))
        p10 = min(capacity, max(0.0, float(row[f"forecast_p10_h{horizon}"]) / max(signal_capacity, 1e-6) * capacity * signal))
        p50 = min(capacity, max(0.0, float(row[f"forecast_p50_h{horizon}"]) / max(signal_capacity, 1e-6) * capacity * signal))
        p90 = min(capacity, max(0.0, float(row[f"forecast_p90_h{horizon}"]) / max(signal_capacity, 1e-6) * capacity * signal))
        p10, p50, p90 = sorted([p10, p50, p90])
        schedule = capacity * float(asset["schedule_fraction"])
        trading = make_trading_decision(
            schedule,
            p10,
            p50,
            p90,
            duration,
            buy_price,
            sell_price,
            shortage_price,
            surplus_price,
            min_trade,
            confidence,
            risk_mode,
        )
        grid = assess_grid_risk(current, p10, p50, p90, capacity, horizon, ramp_threshold_fraction)
        signed_trade = trading.recommended_trade_mwh if trading.action == "BUY" else -trading.recommended_trade_mwh if trading.action == "SELL" else 0.0
        quality = "Good"
        if grid.ramp_risk_level == "High" or trading.action != "HOLD":
            quality = "Bad / action needed"
        elif grid.ramp_risk_level == "Medium" or abs(trading.expected_deviation_mwh) >= min_trade * 0.5:
            quality = "Watch"
        rows.append(
            {
                "Site": asset["site"],
                "Capacity": _fmt(capacity, "MW", 1),
                "Status": quality,
                "Risk": grid.ramp_risk_level,
                "Action": trading.action,
                "Trade": _fmt(trading.recommended_trade_mwh, "MWh"),
                "Current": _fmt(current, "MW"),
                "Forecast": _fmt(p50, "MW"),
                "Schedule": _fmt(schedule, "MW"),
                "Signed trade MWh": signed_trade,
                "Expected imbalance": _fmt(trading.expected_deviation_mwh, "MWh"),
                "Avoided cost": _fmt_eur(max(0.0, trading.estimated_avoided_cost_eur)),
                "Avoided cost EUR": max(0.0, trading.estimated_avoided_cost_eur),
                "Sort score": (
                    (2 if quality == "Bad / action needed" else 1 if quality == "Watch" else 0) * 1_000_000
                    + max(0.0, trading.estimated_avoided_cost_eur)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Sort score", ascending=False).reset_index(drop=True)


def _forecast_figure(
    preds: pd.DataFrame,
    row: pd.Series,
    horizon: int,
    scheduled_power: float,
    action: str,
    trade_mwh: float,
    scale: float,
) -> go.Figure:
    ts = row["timestamp"]
    window = preds[(preds["timestamp"] >= ts - pd.Timedelta(hours=24)) & (preds["timestamp"] <= ts + pd.Timedelta(hours=6))]
    fig = go.Figure()
    hist = window[window["timestamp"] <= ts]
    future = window[window["timestamp"] > ts]
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["pv_power_mw"] * scale, mode="lines", name="Historical actual power"))
    fig.add_trace(
        go.Scatter(
            x=future["timestamp"],
            y=future["pv_power_mw"] * scale,
            mode="lines",
            line=dict(dash="dash"),
            name="Observed after selected replay time",
        )
    )
    target_time = ts + pd.Timedelta(hours=horizon)
    for col, name in [
        (f"pred_persistence_h{horizon}", "Persistence"),
        (f"pred_clear_sky_persistence_h{horizon}", "Clear-sky persistence"),
        (f"history_only_h{horizon}", "History-only ML"),
        (f"satellite_informed_h{horizon}", "Satellite-informed ML"),
    ]:
        fig.add_trace(go.Scatter(x=[ts, target_time], y=[row["pv_power_mw"], row[col]], mode="lines+markers", name=name))
    fig.add_trace(
        go.Scatter(
            x=[target_time, target_time],
            y=[row[f"forecast_p10_h{horizon}"], row[f"forecast_p90_h{horizon}"]],
            mode="lines",
            line=dict(width=8, color="rgba(31, 119, 180, 0.35)"),
            name="P10-P90 prediction interval",
        )
    )
    actual_col = f"target_h{horizon}"
    if actual_col in row and not pd.isna(row[actual_col]):
        fig.add_trace(
            go.Scatter(
                x=[target_time],
                y=[row[actual_col]],
                mode="markers",
                marker=dict(color="#111827", size=12, symbol="diamond"),
                name="Observed target",
            )
        )
    fig.add_hline(
        y=scheduled_power,
        line=dict(color="#7c3aed", width=1, dash="dot"),
        annotation_text="Schedule",
        annotation_position="top left",
    )
    fig.add_annotation(
        x=target_time,
        y=row[f"satellite_informed_h{horizon}"],
        text=f"{action} {_fmt(trade_mwh, 'MWh')}",
        showarrow=True,
        arrowhead=2,
        bgcolor="#ffffff",
        bordercolor="#d1d5db",
    )
    fig.update_layout(
        title=f"Historical Replay Forecast (+{horizon}h)",
        yaxis_title="Power (MW)",
        xaxis_title="UTC time",
        legend_orientation="h",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def _live_forecast_figure(live: pd.DataFrame, selected_horizon: float, scheduled_power: float, action: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=live["horizon_hours"],
            y=live["forecast_p90_mw"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=live["horizon_hours"],
            y=live["forecast_p10_mw"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.18)",
            line=dict(width=0),
            name="P10-P90 operational interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=live["horizon_hours"],
            y=live["forecast_p50_mw"],
            mode="lines",
            line=dict(color="#1d4ed8", width=3),
            name="Live irradiance forecast",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=live["horizon_hours"],
            y=live["scheduled_power_mw"],
            mode="lines",
            line=dict(color="#7c3aed", width=2, dash="dot"),
            name="Computed schedule",
        )
    )
    selected = live.iloc[(live["horizon_hours"] - selected_horizon).abs().idxmin()]
    fig.add_trace(
        go.Scatter(
            x=[float(selected["horizon_hours"])],
            y=[float(selected["forecast_p50_mw"])],
            mode="markers+text",
            marker=dict(size=13, color="#b91c1c"),
            text=[action],
            textposition="top center",
            name="Decision point",
        )
    )
    fig.add_hline(y=scheduled_power, line=dict(color="#111827", width=1, dash="dash"), annotation_text="Decision schedule")
    fig.update_layout(
        title="Live 0-24h Solar Power Forecast",
        xaxis_title="Hours from now",
        yaxis_title="Power (MW)",
        hovermode="x unified",
        legend_orientation="h",
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def _cloud_proxy_map_figure(site: dict[str, Any], row: pd.Series, live: pd.DataFrame | None, horizon_hours: float) -> go.Figure:
    """Map-like situation view using irradiance-derived cloud proxies available to the app."""
    lat = float(site["latitude"])
    lon = float(site["longitude"])
    if live is not None and not live.empty:
        selected = live.iloc[(live["horizon_hours"] - horizon_hours).abs().idxmin()]
        ghi = float(selected.get("global_irradiance_wm2", 0.0) or 0.0)
        direct = float(selected.get("direct_irradiance_wm2", 0.0) or 0.0)
        diffuse = float(selected.get("diffuse_irradiance_wm2", 0.0) or 0.0)
        wind = float(selected.get("wind_speed_ms", 0.0) or 0.0)
    else:
        ghi = float(row.get("global_irradiance_wm2", 0.0) or 0.0)
        direct = 0.0
        diffuse = 0.0
        wind = float(row.get("wind_speed_ms", 0.0) or 0.0)
    cloud_opacity = float(np.clip(1.0 - ghi / 900.0, 0.05, 0.95))
    diffuse_share = float(np.clip(diffuse / max(ghi, 1.0), 0.05, 0.95)) if diffuse else cloud_opacity
    drift = min(max(wind, 1.0), 18.0) * horizon_hours / 650.0
    centers = [
        (lat + 0.04 + drift * 0.2, lon - 0.10 + drift, "cloud mass"),
        (lat - 0.03 + drift * 0.1, lon + 0.02 + drift * 0.8, "aerosol / haze proxy"),
        (lat + 0.08, lon + 0.13 + drift * 1.1, "thin cloud edge"),
    ]
    sizes = [
        95 + 95 * cloud_opacity,
        60 + 90 * diffuse_share,
        45 + 70 * abs(direct - diffuse) / max(ghi + 1.0, 1.0),
    ]
    opacities = [0.18 + 0.45 * cloud_opacity, 0.15 + 0.35 * diffuse_share, 0.18 + 0.25 * cloud_opacity]
    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=[c[0] for c in centers],
            lon=[c[1] for c in centers],
            mode="markers",
            marker=dict(
                size=sizes,
                color=["#6b7280", "#9ca3af", "#cbd5e1"],
                opacity=opacities,
            ),
            text=[c[2] for c in centers],
            hovertemplate="%{text}<extra></extra>",
            name="Cloud proxy",
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode="markers+text",
            marker=dict(size=16, color="#f97316"),
            text=["PV plant"],
            textposition="top center",
            hovertemplate="Target PV plant<br>Lat %{lat:.2f}, Lon %{lon:.2f}<extra></extra>",
            name="PV plant",
        )
    )
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=7.2),
        margin=dict(l=0, r=0, t=36, b=0),
        height=470,
        title=f"Satellite-Derived Cloud Proxy, +{horizon_hours:g}h",
        legend=dict(orientation="h", y=0.01),
    )
    return fig


def _brain_curve_figure(
    live: pd.DataFrame | None,
    row: pd.Series,
    scheduled_power: float,
    p50: float,
    decision_time: pd.Timestamp,
    horizon_hours: float,
) -> go.Figure:
    fig = go.Figure()
    if live is not None and not live.empty:
        plot = live.copy()
        x = plot["timestamp"]
        forecast = plot["forecast_p50_mw"]
        schedule = plot["scheduled_power_mw"]
        fig.add_trace(go.Scatter(x=x, y=schedule, mode="lines", line=dict(color="#6b7280", width=2, dash="dash"), name="Committed baseline schedule"))
        fig.add_trace(go.Scatter(x=x, y=forecast, mode="lines", line=dict(color="#f97316", width=3), name="Satellite AI forecast"))
        fig.add_trace(
            go.Scatter(
                x=[plot["timestamp"].iloc[0]],
                y=[float(row["pv_power_mw"])],
                mode="markers",
                marker=dict(color="#2563eb", size=12),
                name="Current monitored output",
            )
        )
        for i in range(len(plot) - 1):
            y0 = float(min(schedule.iloc[i], forecast.iloc[i]))
            y1 = float(max(schedule.iloc[i], forecast.iloc[i]))
            if abs(y1 - y0) < 1e-6:
                continue
            color = "rgba(220, 38, 38, 0.16)" if forecast.iloc[i] < schedule.iloc[i] else "rgba(22, 163, 74, 0.14)"
            fig.add_shape(type="rect", x0=plot["timestamp"].iloc[i], x1=plot["timestamp"].iloc[i + 1], y0=y0, y1=y1, fillcolor=color, line_width=0)
    else:
        valid_time = decision_time + pd.Timedelta(hours=horizon_hours)
        fig.add_trace(go.Scatter(x=[decision_time, valid_time], y=[row["pv_power_mw"], row["pv_power_mw"]], mode="lines+markers", line=dict(color="#2563eb", width=3), name="Actual to issue time"))
        fig.add_trace(go.Scatter(x=[decision_time, valid_time], y=[scheduled_power, scheduled_power], mode="lines", line=dict(color="#6b7280", width=2, dash="dash"), name="Committed baseline schedule"))
        fig.add_trace(go.Scatter(x=[decision_time, valid_time], y=[row["pv_power_mw"], p50], mode="lines+markers", line=dict(color="#f97316", width=3), name="Satellite AI forecast"))
    fig.update_layout(
        title="Baseline vs Satellite-Corrected Power Forecast",
        yaxis_title="Power (MW)",
        xaxis_title="Delivery time",
        hovermode="x unified",
        legend_orientation="h",
        margin=dict(l=20, r=20, t=58, b=20),
        height=305,
    )
    return fig


def _horizon_label(hours: float) -> str:
    if hours <= 0.25:
        return "15 minutes"
    if hours < 1:
        return f"{int(round(hours * 60))} minutes"
    if abs(hours - 1.0) < 1e-9:
        return "60 minutes"
    return f"{hours:g} hours"


def _forecast_horizon_options(config: dict[str, Any]) -> list[tuple[int, str]]:
    """Return configured discrete forecast horizons as minute values and display labels."""
    minutes = config.get("forecast", {}).get("horizons_minutes", [15, 30, 60, 180, 360, 720, 1440])
    options = sorted({int(value) for value in minutes if int(value) > 0})
    return [(value, _horizon_label(value / 60.0)) for value in options]


def _interpolate_horizon_value(
    row: pd.Series,
    stem: str,
    selected_horizon: float,
    available_horizons: list[int],
    current_value: float | None = None,
) -> float:
    xs: list[float] = []
    ys: list[float] = []
    if current_value is not None and not pd.isna(current_value):
        xs.append(0.0)
        ys.append(float(current_value))
    for h in sorted(available_horizons):
        col = f"{stem}_h{h}"
        if col in row and not pd.isna(row[col]):
            xs.append(float(h))
            ys.append(float(row[col]))
    if not ys:
        return float("nan")
    order = np.argsort(xs)
    ordered_x = np.asarray(xs, dtype=float)[order]
    ordered_y = np.asarray(ys, dtype=float)[order]
    return float(np.interp(float(selected_horizon), ordered_x, ordered_y))


def _finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if np.isfinite(number) else float(fallback)


def _target_delivery_window(frame: pd.DataFrame, selected_horizon: float, delivery_duration: float) -> pd.DataFrame:
    """Rows representing the selected delivery product, not the whole 0..horizon path."""
    start_horizon = max(float(selected_horizon), 0.0)
    duration = max(float(delivery_duration), 0.25)
    end_horizon = min(start_horizon + duration, float(frame["horizon_hours"].max()))
    if end_horizon <= start_horizon:
        end_horizon = start_horizon
    window = frame[(frame["horizon_hours"] >= start_horizon) & (frame["horizon_hours"] < end_horizon)].copy()
    if window.empty:
        window = frame.iloc[[(frame["horizon_hours"] - start_horizon).abs().idxmin()]].copy()
    return window


def _mean_finite(series: pd.Series, fallback: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else float(fallback)


def _decision_window_from_live(
    live: pd.DataFrame,
    selected_horizon: float,
    delivery_duration: float,
    override_schedule: bool,
    override_power: float,
) -> tuple[float, float, float, float, float]:
    selected = live.iloc[(live["horizon_hours"] - max(float(selected_horizon), 0.0)).abs().idxmin()]
    fallback_forecast = _finite_or(selected.get("forecast_p50_mw"), 0.0)
    fallback_schedule = _finite_or(selected.get("scheduled_power_mw"), fallback_forecast)
    window = _target_delivery_window(live, selected_horizon, delivery_duration)
    duration_hours = max(float(delivery_duration), 0.25)
    scheduled = pd.Series(float(override_power), index=window.index) if override_schedule else window["scheduled_power_mw"].astype(float)
    return (
        _mean_finite(scheduled, fallback_schedule),
        _mean_finite(window["forecast_p10_mw"], fallback_forecast),
        _mean_finite(window["forecast_p50_mw"], fallback_forecast),
        _mean_finite(window["forecast_p90_mw"], fallback_forecast),
        duration_hours,
    )


def _decision_window_from_replay(
    row: pd.Series,
    schedule: pd.DataFrame,
    selected_horizon: float,
    delivery_duration: float,
    available_horizons: list[int],
    current_power: float,
    override_schedule: bool,
    override_power: float,
) -> tuple[float, float, float, float, float]:
    start_horizon = max(float(selected_horizon), 0.0)
    duration_hours = max(float(delivery_duration), 0.25)
    end_horizon = start_horizon + duration_hours
    grid = np.arange(start_horizon, end_horizon, 0.25)
    if grid.size == 0:
        grid = np.asarray([start_horizon])
    if override_schedule:
        scheduled_power = float(override_power)
    else:
        schedule_window = schedule[(schedule["horizon_hours"] >= start_horizon) & (schedule["horizon_hours"] < end_horizon)]
        if schedule_window.empty:
            schedule_window = schedule.iloc[[(schedule["horizon_hours"] - start_horizon).abs().idxmin()]]
        scheduled_power = _mean_finite(schedule_window["scheduled_power_mw"], current_power)
    p10_values = [_interpolate_horizon_value(row, "forecast_p10", h, available_horizons, current_power) for h in grid]
    p50_values = [_interpolate_horizon_value(row, "forecast_p50", h, available_horizons, current_power) for h in grid]
    p90_values = [_interpolate_horizon_value(row, "forecast_p90", h, available_horizons, current_power) for h in grid]
    return (
        scheduled_power,
        _mean_finite(pd.Series(p10_values), current_power),
        _mean_finite(pd.Series(p50_values), current_power),
        _mean_finite(pd.Series(p90_values), current_power),
        duration_hours,
    )


def main() -> None:
    st.set_page_config(page_title="SolarCast Ops", layout="wide")
    st.markdown(
        """
        <style>
        .operator-alert {
            border-left: 6px solid #b91c1c;
            background: #fff7ed;
            padding: 1.1rem 1.25rem;
            margin: 0.5rem 0 1.2rem 0;
        }
        .operator-alert .eyebrow {
            color: #9a3412;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .operator-alert .headline {
            color: #111827;
            font-size: 1.45rem;
            font-weight: 750;
            line-height: 1.25;
        }
        .operator-alert .subcopy {
            color: #4b5563;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }
        .proof-strip {
            border: 1px solid #e5e7eb;
            background: #f8fafc;
            padding: 0.85rem 1rem;
            margin: 0.2rem 0 1rem 0;
        }
        .proof-strip strong {
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("SolarCast Ops")
    st.caption("Satellite-informed solar nowcasting for intraday trading and grid balancing.")
    config, preds, metrics, history, irradiance, irradiance_metrics, asof, asof_metrics, demos, meta = load_assets()
    paths = get_paths()

    if preds is None and history is None:
        st.error("Pipeline outputs are missing. Run `python run_pipeline.py --step all` first.")
        return
    if history is None and preds is not None:
        history = preds.rename(columns={"pv_power_mw": "power_actual"})[["timestamp", "power_actual"]].copy()

    site = config["site"]
    market = config["market"]
    decision_cfg = config["decision"]
    data_source = meta.get("source", "unknown")
    is_synthetic = (
        bool(preds.get("is_synthetic", pd.Series([False])).astype(str).str.lower().eq("true").any())
        if preds is not None and "is_synthetic" in preds
        else "synthetic" in data_source.lower()
    )

    st.sidebar.header("Site information")
    peak_power = float(site["peak_power_mw"])
    st.sidebar.caption(f"{site['name']} unit signal | {peak_power} MWp | {site['latitude']}, {site['longitude']}")
    portfolio_capacity_mw = st.sidebar.number_input(
        "Portfolio capacity (MWp)",
        min_value=peak_power,
        max_value=50_000.0,
        value=1_000.0,
        step=50.0,
    )
    portfolio_scale = float(portfolio_capacity_mw) / max(peak_power, 1e-6)
    operating_mode = st.sidebar.radio("Operating mode", ["Live 24h forecast", "Historical replay"], horizontal=False)
    schedule_model = st.sidebar.selectbox("Computed schedule model", ["Model A", "Model B", "Blend"], index=0)
    horizon_options = _forecast_horizon_options(config)
    horizon_labels = {minutes: label for minutes, label in horizon_options}
    default_horizon_minutes = 60 if 60 in horizon_labels else horizon_options[0][0]
    decision_horizon_minutes = int(
        st.sidebar.selectbox(
            "Forecast horizon",
            options=[minutes for minutes, _ in horizon_options],
            index=[minutes for minutes, _ in horizon_options].index(default_horizon_minutes),
            format_func=lambda minutes: horizon_labels[int(minutes)],
        )
    )
    decision_horizon = decision_horizon_minutes / 60.0
    duration = float(st.sidebar.number_input("Delivery duration (hours)", min_value=0.25, max_value=4.0, value=1.0, step=0.25))
    buy_price = float(market["default_intraday_buy_price_eur_mwh"])
    sell_price = float(market["default_intraday_sell_price_eur_mwh"])
    shortage_price = float(market["default_shortage_imbalance_price_eur_mwh"])
    surplus_price = float(market["default_surplus_imbalance_price_eur_mwh"])
    risk_mode = "Balanced"
    min_trade = float(decision_cfg["minimum_trade_threshold_mwh"]) * portfolio_scale
    override_schedule = False
    override_power = float(portfolio_capacity_mw) * 0.65
    with st.sidebar.expander("Market and operator inputs"):
        st.caption("Imbalance energy is computed for the selected delivery window, not accumulated from now.")
        buy_price = st.number_input("Buy price (EUR/MWh)", min_value=0.0, value=buy_price)
        sell_price = st.number_input("Sell price (EUR/MWh)", min_value=0.0, value=sell_price)
        shortage_price = st.number_input("Shortage imbalance price (EUR/MWh)", min_value=0.0, value=shortage_price)
        surplus_price = st.number_input("Surplus imbalance price (EUR/MWh)", min_value=0.0, value=surplus_price)
        risk_mode = st.selectbox("Risk mode", ["Balanced", "Conservative", "Aggressive"])
        min_trade = st.number_input(
            "Minimum trade threshold (MWh)",
            min_value=0.0,
            value=min_trade,
            step=max(0.25, portfolio_scale * 0.01),
        )
        override_schedule = st.checkbox("Override computed schedule")
        override_power = st.number_input(
            "Operator schedule override (MW)",
            min_value=0.0,
            max_value=float(portfolio_capacity_mw) * 1.5,
            value=override_power,
            step=max(1.0, float(portfolio_capacity_mw) * 0.01),
            disabled=not override_schedule,
        )

    is_live_mode = operating_mode == "Live 24h forecast"
    live = None
    schedule_meta: dict[str, Any] = {}
    live_meta: dict[str, Any] = {}
    used_live_fallback = False
    replay_ts: pd.Timestamp | None = None
    display_horizon = decision_horizon
    horizon = min(config["forecast"]["horizons_hours"], key=lambda h: abs(float(h) - max(decision_horizon, 1.0)))

    if is_live_mode:
        start_time = pd.Timestamp.now(tz="UTC").floor("15min")
        live, schedule_meta, live_meta, used_live_fallback = load_live_24h_assets(config, history, start_time.isoformat(), schedule_model)
        for col in [
            "forecast_power_mw",
            "forecast_p10_mw",
            "forecast_p50_mw",
            "forecast_p90_mw",
            "scheduled_power_mw",
            "schedule_model_a_mw",
            "schedule_model_b_mw",
        ]:
            if col in live:
                live[col] = live[col] * portfolio_scale
                live[col] = pd.to_numeric(live[col], errors="coerce").interpolate(limit_direction="both").ffill().bfill()
        selected_live = live.iloc[(live["horizon_hours"] - decision_horizon).abs().idxmin()]
        point_p50 = _finite_or(selected_live.get("forecast_p50_mw"), 0.0)
        point_scheduled_power = float(override_power) if override_schedule else _finite_or(selected_live.get("scheduled_power_mw"), point_p50)
        point_p10 = _finite_or(selected_live.get("forecast_p10_mw"), point_p50)
        point_p90 = _finite_or(selected_live.get("forecast_p90_mw"), point_p50)
        brain_scheduled_power = point_scheduled_power
        brain_p50 = point_p50
        scheduled_power, p10, p50, p90, duration = _decision_window_from_live(
            live,
            decision_horizon,
            duration,
            override_schedule,
            override_power,
        )
        grid_p10, grid_p50, grid_p90 = point_p10, point_p50, point_p90
        current_power = _finite_or(live.iloc[0].get("forecast_p50_mw"), point_p50)
        row = pd.Series(
            {
                "timestamp": start_time,
                "pv_power_mw": current_power,
                "global_irradiance_wm2": selected_live.get("global_irradiance_wm2"),
                "clear_sky_index": pd.NA,
                "weather_condition": "live_forecast",
                "data_source": live_meta.get("source", "live forecast"),
                f"pred_persistence_h{horizon}": current_power,
                f"satellite_informed_h{horizon}": point_p50,
                f"target_h{horizon}": pd.NA,
                f"forecast_p10_h{horizon}": point_p10,
                f"forecast_p50_h{horizon}": point_p50,
                f"forecast_p90_h{horizon}": point_p90,
                "_signal_capacity_mw": portfolio_capacity_mw,
            }
        )
        data_source = live_meta.get("source", data_source)
        if used_live_fallback:
            warning_box("Live irradiance could not be fetched. Forecast is falling back to the computed schedule baseline.")
    else:
        if preds is None:
            st.error("Historical replay outputs are missing. Run `python run_pipeline.py --step all` first.")
            return
        case_names = [d["name"] for d in demos] or ["Manual"]
        default_idx = case_names.index("Large downward ramp event") if "Large downward ramp event" in case_names else 0
        case = st.sidebar.selectbox("Replay/backtest case", case_names, index=default_idx)
        default_ts = pd.Timestamp(demos[default_idx]["timestamp"]) if demos else preds["timestamp"].iloc[len(preds) // 2]
        if demos and case in case_names:
            default_ts = pd.Timestamp(next(d["timestamp"] for d in demos if d["name"] == case))
        selected = st.sidebar.selectbox(
            "Historical replay timestamp",
            preds["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC").tolist(),
            index=int((preds["timestamp"] - default_ts).abs().idxmin()),
        )
        replay_ts = pd.Timestamp(selected.replace(" UTC", ""), tz="UTC")
        row = _row_at(preds, replay_ts).copy()
        scale_cols = ["pv_power_mw"]
        for h in config["forecast"]["horizons_hours"]:
            scale_cols.extend(
                [
                    f"pred_persistence_h{h}",
                    f"pred_clear_sky_persistence_h{h}",
                    f"history_only_h{h}",
                    f"satellite_informed_h{h}",
                    f"target_h{h}",
                    f"forecast_p10_h{h}",
                    f"forecast_p50_h{h}",
                    f"forecast_p90_h{h}",
                ]
            )
        for col in scale_cols:
            if col in row and not pd.isna(row[col]):
                row[col] = float(row[col]) * portfolio_scale
        row["_signal_capacity_mw"] = portfolio_capacity_mw
        point_p10 = _interpolate_horizon_value(row, "forecast_p10", decision_horizon, config["forecast"]["horizons_hours"], float(row["pv_power_mw"]))
        point_p50 = _interpolate_horizon_value(row, "forecast_p50", decision_horizon, config["forecast"]["horizons_hours"], float(row["pv_power_mw"]))
        point_p90 = _interpolate_horizon_value(row, "forecast_p90", decision_horizon, config["forecast"]["horizons_hours"], float(row["pv_power_mw"]))
        schedule_result = build_next_24h_schedule(history, replay_ts, schedule_model=schedule_model)
        schedule_row = schedule_result.schedule.iloc[(schedule_result.schedule["horizon_hours"] - decision_horizon).abs().idxmin()]
        point_scheduled_power = float(override_power) if override_schedule else float(schedule_row["scheduled_power_mw"]) * portfolio_scale
        brain_scheduled_power = point_scheduled_power
        brain_p50 = point_p50
        current_power = float(row["pv_power_mw"])
        scheduled_power, p10, p50, p90, duration = _decision_window_from_replay(
            row,
            schedule_result.schedule.assign(
                scheduled_power_mw=schedule_result.schedule["scheduled_power_mw"] * portfolio_scale
            ),
            decision_horizon,
            duration,
            config["forecast"]["horizons_hours"],
            current_power,
            override_schedule,
            override_power,
        )
        grid_p10, grid_p50, grid_p90 = point_p10, point_p50, point_p90
        if not bool(row.get("is_daylight", False)):
            warning_box("Selected replay time is night or low sun. Forecast and trading actions may be naturally near zero.")
        future_cols = [f"target_h{h}" for h in config["forecast"]["horizons_hours"]]
        if row[future_cols].isna().any():
            warning_box("Selected replay time does not have enough future observed values for full historical verification.")
        schedule_meta = schedule_result.metadata

    if scheduled_power > float(portfolio_capacity_mw):
        warning_box("Scheduled power exceeds nominal portfolio capacity. This is allowed for operator override but unusual.")

    if not (p10 <= p50 <= p90):
        warning_box("Prediction interval was out of order and has been sorted for display.")
        p10, p50, p90 = sorted([p10, p50, p90])
    trading = make_trading_decision(
        scheduled_power, p10, p50, p90, duration, buy_price, sell_price, shortage_price, surplus_price, min_trade, 1.0, risk_mode
    )
    grid = assess_grid_risk(
        current_power,
        grid_p10,
        grid_p50,
        grid_p90,
        float(portfolio_capacity_mw),
        display_horizon,
        float(decision_cfg["ramp_threshold_fraction"]),
    )
    reserve = grid.recommended_upward_reserve_mw or grid.recommended_downward_flexibility_mw
    daylight_skill = _metric_value(metrics, horizon, "satellite_informed", "daylight", "skill_vs_persistence")
    clear_sky_skill = _metric_value(metrics, horizon, "satellite_informed", "daylight", "skill_vs_clear_sky_persistence")
    satellite_mae = _metric_value(metrics, horizon, "satellite_informed", "daylight", "mae")
    history_mae = _metric_value(metrics, horizon, "history_only", "daylight", "mae")
    satellite_edge = (history_mae - satellite_mae) / history_mae if history_mae and satellite_mae else None
    portfolio = _portfolio_preview(
        row,
        horizon,
        duration,
        buy_price,
        sell_price,
        shortage_price,
        surplus_price,
        min_trade,
        1.0,
        risk_mode,
        float(decision_cfg["ramp_threshold_fraction"]),
    )
    portfolio_net_trade = float(portfolio["Signed trade MWh"].sum())
    portfolio_avoided = float(portfolio["Avoided cost EUR"].sum())
    portfolio_action = "BUY" if portfolio_net_trade > 0 else "SELL" if portfolio_net_trade < 0 else "HOLD"
    portfolio_trade_label = _fmt(abs(portfolio_net_trade), "MWh")
    portfolio_bad = int((portfolio["Status"] == "Bad / action needed").sum())
    portfolio_watch = int((portfolio["Status"] == "Watch").sum())
    skill_label = "Backtest skill proxy" if is_live_mode else "Skill vs Persistence"

    if is_synthetic:
        warning_box("Current run uses synthetic demo data fallback. It is not measured satellite data or real plant SCADA.")

    st.markdown("### Solar Operations Command Center")
    st.caption("From satellite-derived weather signal to generation forecast to market action.")
    proof_cols = st.columns(5)
    proof_cols[0].metric("Action", trading.action)
    proof_cols[1].metric("Trade size", _fmt(trading.recommended_trade_mwh, "MWh"))
    proof_cols[2].metric(skill_label, _pct(daylight_skill), f"+{horizon}h daylight")
    proof_cols[3].metric("Skill vs clear-sky persistence", _pct(clear_sky_skill))
    proof_cols[4].metric("Sites needing action", str(portfolio_bad), f"{portfolio_watch} watch")
    st.markdown(
        f"""
        <div class="proof-strip">
            <strong>Persistence benchmark:</strong> at +{horizon}h daylight, the satellite-informed model reports {_pct(daylight_skill)}
            MAE skill versus ordinary persistence and {_pct(clear_sky_skill)} versus clear-sky persistence.
            The current decision compares a {_fmt(p50, "MW")} forecast against a {_fmt(scheduled_power, "MW")} delivery schedule.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.43, 0.57], gap="large")
    with left:
        st.markdown("#### 1. Situation Awareness")
        st.plotly_chart(_cloud_proxy_map_figure(site, row, live, decision_horizon), width="stretch")
        map_status = pd.DataFrame(
            [
                {"Signal": "Remote sensing layer", "Value": str(row.get("data_source", data_source))},
                {"Signal": "Satellite-derived irradiance", "Value": _fmt(row.get("global_irradiance_wm2"), "W/m²", 1)},
                {"Signal": "Cloud proxy state", "Value": _weather_label(row.get("weather_condition"))},
                {"Signal": "Target portfolio", "Value": _fmt(portfolio_capacity_mw, "MWp", 0)},
            ]
        )
        st.dataframe(map_status, hide_index=True, width="stretch")
        st.caption(
            "The map visualizes cloud thickness and movement proxies derived from satellite irradiance, diffuse/direct radiation, and wind context. "
            "It is not raw Google Earth imagery."
        )

    with right:
        st.markdown("#### 2. Forecast Brain")
        st.plotly_chart(_brain_curve_figure(live, row, brain_scheduled_power, brain_p50, pd.Timestamp(row["timestamp"]), decision_horizon), width="stretch")

        st.markdown("#### 3. Trading Decision")
        status = "Predicted generation is close to schedule."
        if trading.action == "BUY":
            status = f"Generation shortfall expected for the +{_horizon_label(decision_horizon)} delivery window."
        elif trading.action == "SELL":
            status = f"Generation surplus expected for the +{_horizon_label(decision_horizon)} delivery window."
        action_text = "Hold position"
        if trading.action in {"BUY", "SELL"}:
            action_text = f"Place simulated limit {trading.action} order for {_fmt(trading.recommended_trade_mwh, 'MWh')}."
        st.markdown(
            f"""
            <div class="operator-alert">
                <div class="eyebrow">Operational output</div>
                <div class="headline">{status}</div>
                <div class="subcopy">{action_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Energy and cost values below are for a {_horizon_label(duration)} delivery product starting at +{_horizon_label(decision_horizon)}.")
        d1, d2, d3 = st.columns(3)
        d1.metric("Expected imbalance", _fmt(trading.expected_deviation_mwh, "MWh"))
        d2.metric("No-action exposure", _fmt_eur(trading.estimated_cost_without_action_eur))
        d3.metric("Avoided cost", _fmt_eur(max(0.0, trading.estimated_avoided_cost_eur)))
        d4, d5, d6 = st.columns(3)
        d4.metric("Action", trading.action)
        d5.metric("Trade size", _fmt(trading.recommended_trade_mwh, "MWh"))
        d6.metric("Reserve / flex", _fmt(reserve, "MW"))
        if st.button("Simulate one-click order", type="primary", disabled=trading.action == "HOLD"):
            st.success(f"Simulated {trading.action} order prepared for {_fmt(trading.recommended_trade_mwh, 'MWh')}.")

    st.markdown("#### 4. Good vs Bad Sites")
    site_display = portfolio.drop(columns=["Signed trade MWh", "Avoided cost EUR", "Sort score"])
    st.dataframe(site_display, hide_index=True, width="stretch")
    st.caption(
        "Good/watch/bad status is derived from the selected satellite-informed signal, ramp risk, schedule deviation, and simulated avoided cost. "
        "Rows are sorted by operational urgency."
    )

    with st.expander("Validation evidence: strict as-of backtest", expanded=True):
        if asof is None or asof_metrics is None:
            warning_box("As-of backtest output is missing. Run `python3 run_pipeline.py --step asof`.")
        else:
            daylight_asof_metrics = asof_metrics[asof_metrics.get("segment", "all").eq("daylight")] if "segment" in asof_metrics else asof_metrics
            best_source = daylight_asof_metrics if not daylight_asof_metrics.empty else asof_metrics
            best = best_source.sort_values("skill_vs_persistence", ascending=False).iloc[0]
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Replay cases", str(asof["case_name"].nunique()))
            v2.metric("Horizons tested", str(asof["horizon_h"].nunique()))
            v3.metric("Best daylight skill", _pct(best["skill_vs_persistence"]), f"+{int(best['horizon_h'])}h")
            v4.metric("Leakage policy", "No future labels")
            st.markdown(
                "For horizon h, training labels are allowed only when `sample_timestamp + h <= issue_time`. "
                "The model uses satellite-derived irradiance, cloud opacity/trend proxies, diffuse/direct radiation, wind, temperature, and history available at issue time."
            )
            metric_display = asof_metrics.copy()
            metric_display["skill_vs_persistence"] = metric_display["skill_vs_persistence"].map(_pct)
            metric_display["skill_vs_clear_sky_persistence"] = metric_display["skill_vs_clear_sky_persistence"].map(_pct)
            metric_display["mae_mw"] = metric_display["mae_mw"].map(lambda v: _fmt(v, "MW"))
            metric_display["rmse_mw"] = metric_display["rmse_mw"].map(lambda v: _fmt(v, "MW"))
            metric_display["ghi_mae_wm2"] = metric_display["ghi_mae_wm2"].map(lambda v: _fmt(v, "W/m²", 1))
            st.dataframe(metric_display, hide_index=True, width="stretch")

    with st.expander("Data lineage and limitations", expanded=True):
        st.markdown(
            f"""
            **Historical remote-sensing source:** Open-Meteo satellite archive irradiance when available, with PVGIS/SARAH-3 as fallback.

            **Live forecast source:** {live_meta.get("source", "not loaded in replay mode")}.

            **What is real:** public satellite-derived GHI/direct/diffuse irradiance, forecast solar radiation API values, historical weather variables, model backtest actuals from PVGIS output.

            **What is derived:** cloud opacity/trend proxies, wind-advected cloud-change proxy, baseline schedule, satellite-corrected forecast, imbalance and trading recommendation.

            **What remains simulated:** raw satellite cloud imagery, real plant SCADA, live order book execution, and the one-click trade button.
            """
        )
    st.divider()

    tabs = st.tabs(
        [
            "Operation Overview",
            "Forecast",
            "Trading Decision",
            "Portfolio Preview",
            "Grid Risk",
            "Irradiance Model",
            "As-of Backtest",
            "Model Evaluation",
            "Data and Limitations",
        ]
    )

    with tabs[0]:
        headline = _decision_headline(
            trading.action,
            trading.recommended_trade_mwh,
            max(0.0, trading.estimated_avoided_cost_eur),
            grid.ramp_risk_level,
            display_horizon,
        )
        st.markdown(
            f"""
            <div class="operator-alert">
                <div class="eyebrow">Operator alert</div>
                <div class="headline">{headline}</div>
                <div class="subcopy">
                    Satellite-derived irradiance is translated into a plant-level forecast, an imbalance estimate, and a concrete trading action.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c = st.columns(4)
        with c[0]:
            kpi("Action", trading.action)
            kpi("Trade Now", _fmt(trading.recommended_trade_mwh, "MWh"))
        with c[1]:
            kpi("Avoided Cost", _fmt_eur(max(0.0, trading.estimated_avoided_cost_eur)))
            kpi("Portfolio Capacity", _fmt(portfolio_capacity_mw, "MWp", 0))
        with c[2]:
            kpi(skill_label, _pct(daylight_skill))
            kpi("Satellite Edge", _pct(satellite_edge))
        with c[3]:
            kpi("Ramp Risk", grid.ramp_risk_level)
            kpi("Reserve/Flex", _fmt(reserve, "MW"))

        st.markdown(
            f"""
            <div class="proof-strip">
                <strong>What changed:</strong> current output is {_fmt(row["pv_power_mw"], "MW")}, but the +{display_horizon:g}h forecast is {_fmt(p50, "MW")} against a schedule of {_fmt(scheduled_power, "MW")}. 
                The expected imbalance is {_fmt(trading.expected_deviation_mwh, "MWh")}; persistence would assume {_fmt(row[f"pred_persistence_h{horizon}"], "MW")}.
            </div>
            """,
            unsafe_allow_html=True,
        )
        signal = pd.DataFrame(
            [
                {"Signal": "Satellite irradiance", "Value": _fmt(row.get("global_irradiance_wm2"), "W/m²", 1)},
                {"Signal": "Clear-sky index", "Value": _fmt(row.get("clear_sky_index"), "", 2)},
                {"Signal": "Sky regime", "Value": _weather_label(row.get("weather_condition"))},
                {"Signal": "Prediction interval", "Value": f"{_fmt(p10, 'MW')} to {_fmt(p90, 'MW')}"},
                {"Signal": "Data source", "Value": str(row.get("data_source", data_source))},
                {"Signal": "Schedule source", "Value": "Operator override" if override_schedule else schedule_meta.get("selected_schedule_model", schedule_model)},
                {"Signal": "Decision time", "Value": str(row["timestamp"])},
            ]
        )
        st.subheader("Satellite Signal to Action")
        st.dataframe(signal, hide_index=True, width="stretch")

    with tabs[1]:
        if is_live_mode and live is not None:
            st.plotly_chart(_live_forecast_figure(live, decision_horizon, scheduled_power, trading.action), width="stretch")
        else:
            st.plotly_chart(
                _forecast_figure(preds, row, horizon, scheduled_power, trading.action, trading.recommended_trade_mwh, portfolio_scale),
                width="stretch",
            )
        c = st.columns(4)
        c[0].metric("Persistence forecast", _fmt(row[f"pred_persistence_h{horizon}"], "MW"))
        c[1].metric("Forecast power", _fmt(row[f"satellite_informed_h{horizon}"], "MW"))
        c[2].metric("Observed target", _fmt(row.get(f"target_h{horizon}"), "MW"))
        c[3].metric("P10-P90 interval", f"{_fmt(p10, 'MW')} - {_fmt(p90, 'MW')}")
        if is_live_mode:
            st.caption("Live mode uses Open-Meteo solar radiation when available; otherwise it explicitly falls back to the computed schedule.")
        else:
            st.caption("Future actual power is shown only for historical replay verification and is not used by the forecast.")

    with tabs[2]:
        st.subheader("Trading Decision")
        c = st.columns(3)
        c[0].metric("Scheduled energy", _fmt(scheduled_power * duration, "MWh"))
        c[1].metric("Forecast energy", _fmt(p50 * duration, "MWh"))
        c[2].metric("Expected imbalance", _fmt(trading.expected_deviation_mwh, "MWh"))
        c = st.columns(4)
        c[0].metric("Action", trading.action)
        c[1].metric("Trade size", _fmt(trading.recommended_trade_mwh, "MWh"))
        c[2].metric("Avoided cost", _fmt_eur(max(0.0, trading.estimated_avoided_cost_eur)))
        c[3].metric("Remaining imbalance", _fmt(trading.remaining_imbalance_mwh, "MWh"))
        cost_table = pd.DataFrame(
            [
                {"Scenario": "No action", "Estimated cost": _fmt_eur(trading.estimated_cost_without_action_eur), "Meaning": "Settle the full forecast imbalance."},
                {"Scenario": "Recommended action", "Estimated cost": _fmt_eur(trading.estimated_cost_with_action_eur), "Meaning": f"{trading.action} {_fmt(trading.recommended_trade_mwh, 'MWh')} before delivery."},
                {"Scenario": "Avoided cost", "Estimated cost": _fmt_eur(trading.estimated_avoided_cost_eur), "Meaning": "Difference between no action and recommended action."},
            ]
        )
        st.dataframe(cost_table, hide_index=True, width="stretch")
        info_box(f"{trading.rationale} The calculation is applied directly to the selected {_fmt(portfolio_capacity_mw, 'MWp', 0)} portfolio.")
        st.caption("Costs may be negative when the simplified calculation represents net revenue. This is not a complete real market settlement model.")

    with tabs[3]:
        st.subheader("Portfolio-Scale Preview")
        c = st.columns(4)
        c[0].metric("Assets monitored", str(len(portfolio)))
        c[1].metric("Net action", f"{portfolio_action} {portfolio_trade_label}")
        c[2].metric("Portfolio avoided cost", _fmt_eur(portfolio_avoided))
        c[3].metric("High-risk assets", str(int((portfolio["Risk"] == "High").sum())))
        st.dataframe(
            portfolio.drop(columns=["Signed trade MWh", "Avoided cost EUR", "Sort score"]),
            hide_index=True,
            width="stretch",
        )
        st.caption("Portfolio rows are a scale-out product preview derived from the selected decision signal; the Munich site remains the measured/modelled demo case.")

    with tabs[4]:
        c = st.columns(4)
        c[0].metric("Ramp direction", grid.ramp_direction)
        c[1].metric("Ramp magnitude", _fmt(grid.ramp_magnitude_mw, "MW"))
        c[2].metric("Ramp ratio", _pct(grid.ramp_ratio))
        c[3].metric("Reserve/Flex", _fmt(reserve, "MW"))
        info_box(grid.explanation)
        risk_table = pd.DataFrame(
            [
                {"Input": "P10 forecast", "Value": _fmt(p10, "MW")},
                {"Input": "P50 forecast", "Value": _fmt(p50, "MW")},
                {"Input": "P90 forecast", "Value": _fmt(p90, "MW")},
                {"Input": "Uncertainty level", "Value": str(row.get(f"uncertainty_level_h{horizon}", "n/a"))},
                {"Input": "Expected time to event", "Value": _fmt(display_horizon, "h")},
            ]
        )
        st.dataframe(risk_table, hide_index=True, width="stretch")

    with tabs[5]:
        st.subheader("Physical Irradiance Forecast Layer")
        if is_live_mode and live is not None:
            c = st.columns(4)
            c[0].metric("Source", str(live_meta.get("source", "unknown")))
            c[1].metric("GHI at decision", _fmt(row.get("global_irradiance_wm2"), "W/m²", 1))
            c[2].metric("Live fallback", "Yes" if used_live_fallback else "No")
            c[3].metric("Resolution", "15 min")
            live_irr = live[["timestamp", "horizon_hours", "global_irradiance_wm2", "direct_irradiance_wm2", "diffuse_irradiance_wm2", "forecast_power_mw"]].copy()
            live_irr = live_irr.rename(
                columns={
                    "timestamp": "Valid time",
                    "horizon_hours": "Horizon h",
                    "global_irradiance_wm2": "GHI W/m2",
                    "direct_irradiance_wm2": "Direct W/m2",
                    "diffuse_irradiance_wm2": "Diffuse W/m2",
                    "forecast_power_mw": "PV forecast MW",
                }
            )
            st.dataframe(live_irr.head(97), hide_index=True, width="stretch")
            st.caption(
                "Live mode uses Open-Meteo solar radiation forecasts where available. Hourly responses are interpolated to 15-minute resolution; "
                "if the API is unavailable, the app explicitly falls back to the computed schedule baseline."
            )
        elif irradiance is None:
            warning_box("Irradiance probabilistic model output is missing. Run `python3 run_pipeline.py --step irradiance` or `make all`.")
        else:
            if replay_ts is None:
                warning_box("Select Historical replay mode to inspect stored irradiance replay predictions.")
                st.stop()
            irr_row = _row_at(irradiance, replay_ts)
            q_cols = {
                "P05": f"irradiance_ghi_q05_h{horizon}",
                "P10": f"irradiance_ghi_q10_h{horizon}",
                "P50": f"irradiance_ghi_q50_h{horizon}",
                "P90": f"irradiance_ghi_q90_h{horizon}",
                "P95": f"irradiance_ghi_q95_h{horizon}",
            }
            c = st.columns(4)
            c[0].metric("GHI P50", _fmt(irr_row[q_cols["P50"]], "W/m²", 1))
            c[1].metric("k* P50", _fmt(irr_row[f"irradiance_kstar_q50_h{horizon}"], "", 2))
            c[2].metric("Diffuse fraction P50", _fmt(irr_row[f"irradiance_fd_q50_h{horizon}"], "", 2))
            c[3].metric("POA P50", _fmt(irr_row[f"irradiance_poa_p50_h{horizon}"], "W/m²", 1))

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=list(q_cols.keys()),
                    y=[irr_row[col] for col in q_cols.values()],
                    name="GHI quantiles",
                    marker_color="#2563eb",
                )
            )
            fig.add_hline(
                y=float(irr_row.get(f"target_ghi_h{horizon}", 0.0)),
                line=dict(color="#111827", dash="dash"),
                annotation_text="Observed future GHI",
                annotation_position="top left",
            )
            fig.update_layout(title=f"GHI probabilistic forecast (+{horizon}h)", yaxis_title="W/m²", margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, width="stretch")

            physical = pd.DataFrame(
                [
                    {"Output": "GHI P50", "Value": _fmt(irr_row[f"irradiance_ghi_q50_h{horizon}"], "W/m²", 1)},
                    {"Output": "DHI P50", "Value": _fmt(irr_row[f"irradiance_dhi_p50_h{horizon}"], "W/m²", 1)},
                    {"Output": "DNI P50", "Value": _fmt(irr_row[f"irradiance_dni_p50_h{horizon}"], "W/m²", 1)},
                    {"Output": "POA P50", "Value": _fmt(irr_row[f"irradiance_poa_p50_h{horizon}"], "W/m²", 1)},
                    {"Output": "Valid time", "Value": str(irr_row[f"valid_time_h{horizon}"])},
                ]
            )
            st.dataframe(physical, hide_index=True, width="stretch")

            weights = json.loads(irr_row[f"expert_weights_h{horizon}"])
            weight_df = pd.DataFrame([{"Expert": key, "Weight": _pct(value)} for key, value in weights.items()])
            st.markdown("**Expert gate weights**")
            st.dataframe(weight_df, hide_index=True, width="stretch")
            st.markdown("**Quality flags**")
            st.write(", ".join(json.loads(irr_row[f"quality_flags_h{horizon}"])))
            st.caption(
                "This layer predicts clear-sky index and diffuse fraction, then reconstructs GHI/DHI/DNI/POA with pvlib. "
                "In this MVP, Open-Meteo satellite archive irradiance is used when available, with PVGIS/SARAH-3 as fallback; raw Meteosat imagery, optical flow, and real NWP adapters are explicit future hooks."
            )
            if irradiance_metrics is not None:
                st.dataframe(irradiance_metrics, hide_index=True, width="stretch")

    with tabs[6]:
        st.subheader("Strict As-of Backtest")
        if asof is None or asof_metrics is None:
            warning_box("As-of backtest output is missing. Run `python3 run_pipeline.py --step asof`.")
        else:
            c = st.columns(4)
            daylight_asof_metrics = asof_metrics[asof_metrics.get("segment", "all").eq("daylight")] if "segment" in asof_metrics else asof_metrics
            best_source = daylight_asof_metrics if not daylight_asof_metrics.empty else asof_metrics
            best = best_source.sort_values("skill_vs_persistence", ascending=False).iloc[0]
            c[0].metric("Replay cases", str(asof["case_name"].nunique()))
            c[1].metric("Horizons tested", str(asof["horizon_h"].nunique()))
            c[2].metric("Best skill vs persistence", _pct(best["skill_vs_persistence"]), f"+{int(best['horizon_h'])}h")
            c[3].metric("Metric segment", str(best.get("segment", "all")))
            st.markdown(
                """
                <div class="proof-strip">
                    <strong>As-of rule:</strong> for horizon h, a training label is allowed only when
                    sample_timestamp + h <= issue_time. The forecast row can use satellite-derived irradiance,
                    cloud opacity/trend proxies, wind-advected cloud-change proxies, temperature, and history available at the issue time.
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_display = asof_metrics.copy()
            metric_display["skill_vs_persistence"] = metric_display["skill_vs_persistence"].map(_pct)
            metric_display["skill_vs_clear_sky_persistence"] = metric_display["skill_vs_clear_sky_persistence"].map(_pct)
            metric_display["mae_mw"] = metric_display["mae_mw"].map(lambda v: _fmt(v, "MW"))
            metric_display["rmse_mw"] = metric_display["rmse_mw"].map(lambda v: _fmt(v, "MW"))
            metric_display["ghi_mae_wm2"] = metric_display["ghi_mae_wm2"].map(lambda v: _fmt(v, "W/m²", 1))
            st.dataframe(metric_display, hide_index=True, width="stretch")

            case_names = asof["case_name"].drop_duplicates().tolist()
            selected_case = st.selectbox("As-of replay case", case_names)
            case_rows = asof[asof["case_name"] == selected_case].copy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=case_rows["valid_time"], y=case_rows["actual_power_mw"], mode="lines+markers", name="Actual future power"))
            fig.add_trace(go.Scatter(x=case_rows["valid_time"], y=case_rows["pred_power_mw"], mode="lines+markers", name="As-of model forecast"))
            fig.add_trace(go.Scatter(x=case_rows["valid_time"], y=case_rows["pred_persistence_mw"], mode="lines+markers", name="Persistence baseline"))
            fig.update_layout(title=f"{selected_case}: forecast made at {case_rows['asof_time'].iloc[0]}", yaxis_title="MW per MWp unit", legend_orientation="h")
            st.plotly_chart(fig, width="stretch")
            detail_cols = [
                "case_name",
                "asof_time",
                "visible_data_cutoff",
                "train_label_cutoff",
                "horizon_h",
                "valid_time",
                "pred_power_mw",
                "actual_power_mw",
                "pred_ghi_wm2",
                "actual_ghi_wm2",
                "cloud_opacity_proxy",
                "cloud_variability_proxy",
                "cloud_trend_proxy",
                "wind_advected_cloud_change_proxy",
                "cloud_ramp_risk_proxy",
                "diffuse_fraction",
                "beam_fraction",
                "wind_speed_ms",
                "valid_is_daylight",
                "weather_condition",
            ]
            st.dataframe(case_rows[detail_cols], hide_index=True, width="stretch")

    with tabs[7]:
        if metrics is None:
            warning_box("Metrics file is missing.")
        else:
            best_daylight = metrics[(metrics["segment"] == "daylight") & (metrics["model"] == "satellite_informed")].sort_values(
                "skill_vs_persistence",
                ascending=False,
            )
            best_skill = float(best_daylight.iloc[0]["skill_vs_persistence"]) if not best_daylight.empty else None
            best_horizon = int(best_daylight.iloc[0]["horizon_h"]) if not best_daylight.empty else horizon
            c = st.columns(4)
            c[0].metric("Current horizon skill", _pct(daylight_skill))
            c[1].metric("Best daylight skill", _pct(best_skill), f"+{best_horizon}h")
            c[2].metric("Skill vs clear-sky", _pct(clear_sky_skill))
            c[3].metric("Satellite edge vs history-only", _pct(satellite_edge))
            st.markdown(
                f"""
                <div class="proof-strip">
                    <strong>Proof it beats persistence:</strong> on daylight test samples, the satellite-informed model reduces MAE by {_pct(daylight_skill)} versus ordinary persistence at +{horizon}h. 
                    The strongest daylight horizon reaches {_pct(best_skill)} skill versus persistence.
                </div>
                """,
                unsafe_allow_html=True,
            )
            daylight = metrics[metrics["segment"] == "daylight"]
            eval_table = daylight[["horizon_h", "model", "mae", "rmse", "skill_vs_persistence", "skill_vs_clear_sky_persistence"]].copy()
            eval_table["skill_vs_persistence"] = eval_table["skill_vs_persistence"].map(_pct)
            eval_table["skill_vs_clear_sky_persistence"] = eval_table["skill_vs_clear_sky_persistence"].map(_pct)
            eval_table["mae"] = eval_table["mae"].map(lambda v: _fmt(v, "MW"))
            eval_table["rmse"] = eval_table["rmse"].map(lambda v: _fmt(v, "MW"))
            st.dataframe(eval_table, hide_index=True, width="stretch")
            for fig_name in ["metrics_by_horizon.png", "skill_by_weather_condition.png", f"feature_importance_h{horizon}.png"]:
                fig_path = paths.figures_dir / fig_name
                if fig_path.exists():
                    st.image(str(fig_path))
            best = preds.iloc[(preds[f"target_h{horizon}"] - preds[f"satellite_informed_h{horizon}"]).abs().idxmin()]
            worst = preds.iloc[(preds[f"target_h{horizon}"] - preds[f"satellite_informed_h{horizon}"]).abs().idxmax()]
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Replay case": "Closest forecast", "UTC time": str(best["timestamp"])},
                        {"Replay case": "Largest miss", "UTC time": str(worst["timestamp"])},
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    with tabs[8]:
        history_min = history["timestamp"].min() if history is not None else "n/a"
        history_max = history["timestamp"].max() if history is not None else "n/a"
        replay_min = preds["timestamp"].min() if preds is not None else "n/a"
        replay_max = preds["timestamp"].max() if preds is not None else "n/a"
        st.markdown(
            f"""
            **Data source:** {data_source}

            **Historical model range:** {replay_min} to {replay_max}

            **Schedule history range:** {history_min} to {history_max}

            **Live forecast source:** {live_meta.get("source", "not loaded in replay mode")}

            **What is productized in this demo**

            - Historical training pulls Open-Meteo satellite archive irradiance when available and aligns it to the Munich hourly PV target frame.
            - Live mode pulls Open-Meteo solar radiation forecasts when available and interpolates hourly data to 15-minute resolution if needed.
            - Model A builds an explainable multiplicative seasonal schedule from historical power.
            - Model B builds a LightGBM schedule baseline from historical same-time and recent trend features.
            - Operator schedule override has the highest priority when operators have a better internal schedule.
            - Forecasts are evaluated against ordinary and clear-sky persistence baselines.
            - The decision layer turns forecast deviation into BUY, SELL, HOLD, reserve, and cost exposure.

            **Important limitations**

            - Current public historical PV output may be PVGIS physical model output, not real SCADA.
            - Live power forecast converts solar radiation to PV output with a simple capacity/loss factor until a calibrated live PV model is connected.
            - Model A/B schedules are baselines, not contractual market nominations.
            - PV output may be public physical model output from PVGIS, not real SCADA.
            - Trading module is a simplified decision-support simulation.
            - No live order book, network constraints, or automated trading are included.
            - Raw satellite imagery is not processed directly in this MVP; the implemented adapter is station/patch aggregate irradiance.

            **Production path**

            - Replace PVGIS modelled output with SCADA/market schedules.
            - Calibrate live PV conversion against inverter/SCADA data.
            - Add raw satellite imagery, optical-flow cloud motion, and NWP ensemble adapters.
            - Run the same decision engine across a portfolio and connect it to trader approval workflows.
            """
        )
        dq = paths.metrics_dir / "data_quality_report.json"
        if dq.exists():
            with st.expander("Data quality report"):
                st.json(json.loads(dq.read_text()))


if __name__ == "__main__":
    main()
