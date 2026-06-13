"""Irradiance forecast evaluation placeholder."""

from __future__ import annotations

from typing import Any


def evaluate_irradiance_forecast(observed: Any, predicted: Any) -> dict:
    """Evaluate GHI, DNI, DHI, and POA irradiance forecasts."""
    raise NotImplementedError("Irradiance evaluation will be implemented after forecasts exist.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Irradiance evaluation placeholder. No evaluation is run in this skeleton step.")


if __name__ == "__main__":
    main()
