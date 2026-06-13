"""Forecast uncertainty plotting placeholder."""

from __future__ import annotations

from typing import Any


def plot_prediction_intervals(interval_data: Any, output_path: str | None = None) -> Any:
    """Plot point forecasts with prediction intervals."""
    raise NotImplementedError("Uncertainty plotting will be implemented after intervals exist.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Uncertainty plotting placeholder. No figure is created in this skeleton step.")


if __name__ == "__main__":
    main()
