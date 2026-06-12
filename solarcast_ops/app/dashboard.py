from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

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
        rows.append(
            {
                "Site": asset["site"],
                "Capacity": _fmt(capacity, "MW", 1),
                "Risk": grid.ramp_risk_level,
                "Action": trading.action,
                "Trade": _fmt(trading.recommended_trade_mwh, "MWh"),
                "Signed trade MWh": signed_trade,
                "Expected imbalance": _fmt(trading.expected_deviation_mwh, "MWh"),
                "Avoided cost": _fmt_eur(max(0.0, trading.estimated_avoided_cost_eur)),
                "Avoided cost EUR": max(0.0, trading.estimated_avoided_cost_eur),
            }
        )
    return pd.DataFrame(rows)


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
    decision_horizon = float(st.sidebar.slider("Decision horizon (hours)", min_value=0.0, max_value=24.0, value=12.0, step=0.25))
    duration = st.sidebar.number_input("Delivery duration (hours)", min_value=0.25, max_value=24.0, value=1.0, step=0.25)
    buy_price = st.sidebar.number_input("Buy price (EUR/MWh)", min_value=0.0, value=float(market["default_intraday_buy_price_eur_mwh"]))
    sell_price = st.sidebar.number_input("Sell price (EUR/MWh)", min_value=0.0, value=float(market["default_intraday_sell_price_eur_mwh"]))
    shortage_price = st.sidebar.number_input("Shortage imbalance price (EUR/MWh)", min_value=0.0, value=float(market["default_shortage_imbalance_price_eur_mwh"]))
    surplus_price = st.sidebar.number_input("Surplus imbalance price (EUR/MWh)", min_value=0.0, value=float(market["default_surplus_imbalance_price_eur_mwh"]))
    risk_mode = st.sidebar.selectbox("Risk mode", ["Balanced", "Conservative", "Aggressive"])
    min_trade = st.sidebar.number_input(
        "Minimum trade threshold (MWh)",
        min_value=0.0,
        value=float(decision_cfg["minimum_trade_threshold_mwh"]) * portfolio_scale,
        step=max(0.25, portfolio_scale * 0.01),
    )
    override_schedule = st.sidebar.checkbox("Override computed schedule")
    override_power = st.sidebar.number_input(
        "Operator schedule override (MW)",
        min_value=0.0,
        max_value=float(portfolio_capacity_mw) * 1.5,
        value=float(portfolio_capacity_mw) * 0.65,
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
        selected_live = live.iloc[(live["horizon_hours"] - decision_horizon).abs().idxmin()]
        scheduled_power = float(override_power) if override_schedule else float(selected_live["scheduled_power_mw"])
        p10 = float(selected_live["forecast_p10_mw"])
        p50 = float(selected_live["forecast_p50_mw"])
        p90 = float(selected_live["forecast_p90_mw"])
        current_power = float(live.iloc[0]["forecast_p50_mw"])
        row = pd.Series(
            {
                "timestamp": selected_live["timestamp"],
                "pv_power_mw": current_power,
                "global_irradiance_wm2": selected_live.get("global_irradiance_wm2"),
                "clear_sky_index": pd.NA,
                "weather_condition": "live_forecast",
                "data_source": live_meta.get("source", "live forecast"),
                f"pred_persistence_h{horizon}": current_power,
                f"satellite_informed_h{horizon}": p50,
                f"target_h{horizon}": pd.NA,
                f"forecast_p10_h{horizon}": p10,
                f"forecast_p50_h{horizon}": p50,
                f"forecast_p90_h{horizon}": p90,
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
        for col in [
            "pv_power_mw",
            f"pred_persistence_h{horizon}",
            f"pred_clear_sky_persistence_h{horizon}",
            f"history_only_h{horizon}",
            f"satellite_informed_h{horizon}",
            f"target_h{horizon}",
            f"forecast_p10_h{horizon}",
            f"forecast_p50_h{horizon}",
            f"forecast_p90_h{horizon}",
        ]:
            if col in row and not pd.isna(row[col]):
                row[col] = float(row[col]) * portfolio_scale
        row["_signal_capacity_mw"] = portfolio_capacity_mw
        p10, p50, p90 = [float(row[f"forecast_p{x}_h{horizon}"]) for x in [10, 50, 90]]
        schedule_result = build_next_24h_schedule(history, replay_ts, schedule_model=schedule_model)
        schedule_row = schedule_result.schedule.iloc[(schedule_result.schedule["horizon_hours"] - decision_horizon).abs().idxmin()]
        scheduled_power = float(override_power) if override_schedule else float(schedule_row["scheduled_power_mw"]) * portfolio_scale
        current_power = float(row["pv_power_mw"])
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
        p10,
        p50,
        p90,
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
    skill_label = "Backtest skill proxy" if is_live_mode else "Skill vs Persistence"

    if is_synthetic:
        warning_box("Current run uses synthetic demo data fallback. It is not measured satellite data or real plant SCADA.")

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
            portfolio.drop(columns=["Signed trade MWh", "Avoided cost EUR"]),
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
                "In this MVP, PVGIS hourly irradiance is used as a satellite-derived proxy; raw Meteosat imagery, optical flow, and real NWP adapters are explicit future hooks."
            )
            if irradiance_metrics is not None:
                st.dataframe(irradiance_metrics, hide_index=True, width="stretch")

    with tabs[6]:
        st.subheader("Strict As-of Backtest")
        if asof is None or asof_metrics is None:
            warning_box("As-of backtest output is missing. Run `python3 run_pipeline.py --step asof`.")
        else:
            c = st.columns(4)
            best = asof_metrics.sort_values("skill_vs_persistence", ascending=False).iloc[0]
            c[0].metric("Replay cases", str(asof["case_name"].nunique()))
            c[1].metric("Horizons tested", str(asof["horizon_h"].nunique()))
            c[2].metric("Best skill vs persistence", _pct(best["skill_vs_persistence"]), f"+{int(best['horizon_h'])}h")
            c[3].metric("Policy", "No future labels")
            st.markdown(
                """
                <div class="proof-strip">
                    <strong>As-of rule:</strong> for horizon h, a training label is allowed only when
                    sample_timestamp + h <= issue_time. The forecast row can use satellite-derived irradiance,
                    cloud proxies, wind, temperature, and history available at the issue time.
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
                "diffuse_fraction",
                "beam_fraction",
                "wind_speed_ms",
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
            - Raw satellite imagery is not processed directly in this MVP.

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
