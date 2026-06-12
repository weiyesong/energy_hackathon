from __future__ import annotations

from src.decision_engine import assess_grid_risk, make_trading_decision


def test_shortage_returns_buy_and_nonnegative_trade():
    d = make_trading_decision(0.8, 0.4, 0.5, 0.6, 1, 90, 70, 150, 40, 0.03, 0.7, "Balanced")
    assert d.action == "BUY"
    assert d.recommended_trade_mwh >= 0


def test_surplus_returns_sell():
    d = make_trading_decision(0.4, 0.5, 0.7, 0.9, 1, 90, 70, 150, 40, 0.03, 0.7, "Balanced")
    assert d.action == "SELL"


def test_small_deviation_returns_hold():
    d = make_trading_decision(0.5, 0.45, 0.51, 0.55, 1, 90, 70, 150, 40, 0.03, 0.7, "Balanced")
    assert d.action == "HOLD"
    assert d.recommended_trade_mwh == 0


def test_cost_calculation_for_buy_reduces_shortage():
    d = make_trading_decision(1.0, 0.5, 0.6, 0.7, 1, 90, 70, 150, 40, 0.03, 1.0, "Aggressive")
    assert d.estimated_cost_without_action_eur == 60
    assert d.estimated_cost_with_action_eur == 36
    assert d.estimated_avoided_cost_eur == 24


def test_ramp_risk_levels():
    low = assess_grid_risk(0.5, 0.48, 0.45, 0.5, 1.0, 1, 0.2)
    high = assess_grid_risk(0.8, 0.3, 0.5, 0.7, 1.0, 1, 0.2)
    assert low.ramp_risk_level == "Low"
    assert high.ramp_risk_level == "High"
    assert high.ramp_direction == "DOWN"
