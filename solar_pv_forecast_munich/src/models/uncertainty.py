"""Forecast uncertainty estimation placeholder."""

from __future__ import annotations

from typing import Any


def build_prediction_intervals(point_forecast: Any, quantile_forecasts: dict) -> Any:
    """Build uncertainty intervals from point and quantile forecasts."""
    raise NotImplementedError("Uncertainty intervals will be implemented after quantile models exist.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Uncertainty placeholder. No intervals are generated in this skeleton step.")


if __name__ == "__main__":
    main()
