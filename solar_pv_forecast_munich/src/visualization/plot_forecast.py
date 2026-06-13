"""Forecast plotting placeholder."""

from __future__ import annotations

from typing import Any


def plot_irradiance_and_power_forecast(forecast_data: Any, output_path: str | None = None) -> Any:
    """Plot irradiance and PV power forecasts for the configured horizons."""
    raise NotImplementedError("Forecast plotting will be implemented after forecast outputs exist.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Forecast plotting placeholder. No figure is created in this skeleton step.")


if __name__ == "__main__":
    main()
