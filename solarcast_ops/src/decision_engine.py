from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingDecision:
    action: str
    expected_deviation_mwh: float
    recommended_trade_mwh: float
    remaining_imbalance_mwh: float
    estimated_cost_without_action_eur: float
    estimated_cost_with_action_eur: float
    estimated_avoided_cost_eur: float
    rationale: str


@dataclass(frozen=True)
class GridRisk:
    ramp_direction: str
    ramp_magnitude_mw: float
    ramp_ratio: float
    ramp_risk_level: str
    expected_time_to_event_hours: float
    recommended_upward_reserve_mw: float
    recommended_downward_flexibility_mw: float
    explanation: str


def _imbalance_cost(deviation_mwh: float, shortage_price: float, surplus_price: float) -> float:
    if deviation_mwh < 0:
        return abs(deviation_mwh) * shortage_price
    return -abs(deviation_mwh) * surplus_price


def make_trading_decision(
    scheduled_power_mw: float,
    forecast_p10_mw: float,
    forecast_p50_mw: float,
    forecast_p90_mw: float,
    delivery_duration_hours: float,
    intraday_buy_price_eur_mwh: float,
    intraday_sell_price_eur_mwh: float,
    shortage_imbalance_price_eur_mwh: float,
    surplus_imbalance_price_eur_mwh: float,
    minimum_trade_threshold_mwh: float,
    confidence_factor: float,
    risk_mode: str,
) -> TradingDecision:
    """Simplified decision-support simulation for intraday balancing."""
    risk = risk_mode.lower()
    confidence = max(0.0, min(float(confidence_factor), 1.0))
    expected_deviation = (forecast_p50_mw - scheduled_power_mw) * delivery_duration_hours

    if abs(expected_deviation) < minimum_trade_threshold_mwh:
        action = "HOLD"
        trade = 0.0
    elif expected_deviation < 0:
        action = "BUY"
        conservative_dev = max(0.0, (scheduled_power_mw - forecast_p90_mw) * delivery_duration_hours)
        p50_dev = abs(expected_deviation)
        if risk == "conservative":
            trade = min(p50_dev, conservative_dev) * confidence
        elif risk == "aggressive":
            trade = p50_dev
        else:
            trade = p50_dev * confidence
    else:
        action = "SELL"
        conservative_dev = max(0.0, (forecast_p10_mw - scheduled_power_mw) * delivery_duration_hours)
        p50_dev = abs(expected_deviation)
        if risk == "conservative":
            trade = min(p50_dev, conservative_dev) * confidence
        elif risk == "aggressive":
            trade = p50_dev
        else:
            trade = p50_dev * confidence

    trade = max(0.0, min(trade, abs(expected_deviation)))
    signed_trade = -trade if action == "BUY" else trade if action == "SELL" else 0.0
    remaining = expected_deviation - signed_trade
    cost_without = _imbalance_cost(expected_deviation, shortage_imbalance_price_eur_mwh, surplus_imbalance_price_eur_mwh)
    intraday_cashflow = trade * intraday_buy_price_eur_mwh if action == "BUY" else -trade * intraday_sell_price_eur_mwh
    cost_with = intraday_cashflow + _imbalance_cost(remaining, shortage_imbalance_price_eur_mwh, surplus_imbalance_price_eur_mwh)
    avoided = cost_without - cost_with
    rationale = (
        f"{action}: expected deviation is {expected_deviation:.3f} MWh over {delivery_duration_hours:.1f}h. "
        f"Risk mode {risk_mode} recommends trading {trade:.3f} MWh in this simplified decision-support simulation."
    )
    return TradingDecision(action, expected_deviation, trade, remaining, cost_without, cost_with, avoided, rationale)


def assess_grid_risk(
    current_power_mw: float,
    forecast_p10_mw: float,
    forecast_p50_mw: float,
    forecast_p90_mw: float,
    peak_power_mw: float,
    expected_time_to_event_hours: float,
    ramp_threshold_fraction: float,
) -> GridRisk:
    """Assess ramp direction, risk level, and reserve suggestion."""
    change = forecast_p50_mw - current_power_mw
    ratio = abs(change) / max(peak_power_mw, 1e-6)
    if abs(change) < peak_power_mw * 0.02:
        direction = "STABLE"
    else:
        direction = "UP" if change > 0 else "DOWN"
    if ratio < 0.10:
        level = "Low"
    elif ratio < ramp_threshold_fraction:
        level = "Medium"
    else:
        level = "High"
    upward = max(0.0, current_power_mw - forecast_p10_mw) if direction == "DOWN" else 0.0
    downward = max(0.0, forecast_p90_mw - current_power_mw) if direction == "UP" else 0.0
    if direction == "DOWN":
        explanation = (
            f"{level} downward ramp risk: generation may fall by approximately {abs(change):.2f} MW within "
            f"{expected_time_to_event_hours:.0f} hour(s). Consider reserving up to {upward:.2f} MW of upward balancing capacity."
        )
    elif direction == "UP":
        explanation = (
            f"{level} upward ramp risk: generation may rise by approximately {abs(change):.2f} MW within "
            f"{expected_time_to_event_hours:.0f} hour(s). Consider reserving up to {downward:.2f} MW of downward flexibility."
        )
    else:
        explanation = "Low ramp risk: forecast generation is close to the current output."
    return GridRisk(direction, abs(change), ratio, level, expected_time_to_event_hours, upward, downward, explanation)
