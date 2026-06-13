"""LightGBM residual model training placeholder."""

from __future__ import annotations

from typing import Any


def train_lightgbm_residual_model(features: Any, target: Any, config: dict) -> Any:
    """Train a LightGBM residual model for irradiance correction in a later step."""
    raise NotImplementedError("Model training is intentionally excluded from this skeleton step.")


def main() -> None:
    """Print the current role of this executable module."""
    print("LightGBM training placeholder. No model is trained in this skeleton step.")


if __name__ == "__main__":
    main()
