"""PV power forecast evaluation placeholder."""

from __future__ import annotations

from typing import Any


def evaluate_pv_forecast(observed_power: Any, predicted_power: Any) -> dict:
    """Evaluate PV power forecasts against observed or reference PV production."""
    raise NotImplementedError("PV evaluation will be implemented after PV forecasts exist.")


def main() -> None:
    """Print the current role of this executable module."""
    print("PV evaluation placeholder. No evaluation is run in this skeleton step.")


if __name__ == "__main__":
    main()
