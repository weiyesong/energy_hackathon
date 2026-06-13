"""Shared evaluation metrics placeholder."""

from __future__ import annotations

from typing import Any


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict:
    """Compute regression metrics for irradiance and PV power forecasts."""
    raise NotImplementedError("Metrics will be implemented once forecast outputs are defined.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Metrics placeholder. No evaluation is run in this skeleton step.")


if __name__ == "__main__":
    main()
