"""PV power conversion for operational probabilistic forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_pv_power_quantiles(df: pd.DataFrame, pv_system: dict) -> pd.DataFrame:
    """Estimate AC PV P10/P50/P90, energy, module temperature, and clear-sky PV power."""
    out = df.copy()
    noct = float(pv_system.get("noct", 45.0))
    gamma = float(pv_system.get("temperature_coefficient", -0.004))
    capacity_kwp = float(pv_system.get("capacity_kwp", 1.0))
    inverter_efficiency = float(pv_system.get("inverter_efficiency", 0.96))
    other_losses = float(pv_system.get("other_losses", 0.10))
    inverter_limit = float(pv_system.get("inverter_limit_kw", capacity_kwp))
    air_temperature = pd.to_numeric(out.get("target_temperature_2m_forecast_proxy", 20.0), errors="coerce").fillna(20.0)

    out["module_temperature"] = air_temperature + pd.to_numeric(out["POA_P50"], errors="coerce").fillna(0.0) / 800.0 * (noct - 20.0)
    for label in ["P10", "P50", "P90"]:
        out[f"PV_{label}"] = _pv_ac_from_poa(
            pd.to_numeric(out[f"POA_{label}"], errors="coerce").fillna(0.0),
            out["module_temperature"],
            capacity_kwp,
            gamma,
            inverter_efficiency,
            other_losses,
            inverter_limit,
        )

    ordered = np.sort(out[["PV_P10", "PV_P50", "PV_P90"]].to_numpy(dtype=float), axis=1)
    out[["PV_P10", "PV_P50", "PV_P90"]] = ordered
    out["PV_energy_P50"] = out["PV_P50"] * pd.to_numeric(out["horizon_minutes"], errors="coerce").fillna(60.0) / 60.0
    clear_poa = pd.to_numeric(out["target_GHI_clear"], errors="coerce").fillna(0.0)
    out["clear_sky_PV_power"] = _pv_ac_from_poa(clear_poa, air_temperature + clear_poa / 800.0 * (noct - 20.0), capacity_kwp, gamma, inverter_efficiency, other_losses, inverter_limit)
    out["PV_state_index"] = (out["PV_P50"] / out["clear_sky_PV_power"].clip(lower=0.01)).clip(0.0, 1.5)
    return out


def _pv_ac_from_poa(
    poa: pd.Series,
    module_temperature: pd.Series,
    capacity_kwp: float,
    gamma: float,
    inverter_efficiency: float,
    other_losses: float,
    inverter_limit: float,
) -> pd.Series:
    """Convert POA irradiance to clipped AC power in kW."""
    p_dc = capacity_kwp * poa.clip(lower=0.0) / 1000.0 * (1.0 + gamma * (module_temperature - 25.0))
    p_ac = p_dc * inverter_efficiency * (1.0 - other_losses)
    return p_ac.clip(lower=0.0, upper=inverter_limit)


def main() -> None:
    """Print the current role of this executable module."""
    print("Use src/operations/decision_engine.py to compute operational PV power forecasts.")


if __name__ == "__main__":
    main()
