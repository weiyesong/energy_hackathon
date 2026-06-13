"""Multi-horizon forecast generation placeholder."""

from __future__ import annotations

from typing import Any


def predict_for_horizons(features: Any, models: dict, horizons_minutes: list[int]) -> Any:
    """Generate forecasts for all configured forecast horizons."""
    raise NotImplementedError("Multi-horizon prediction will be implemented after models are trained.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Multi-horizon prediction placeholder. No forecast is generated in this skeleton step.")


if __name__ == "__main__":
    main()
