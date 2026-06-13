"""Quantile model training placeholder for uncertainty estimation."""

from __future__ import annotations

from typing import Any


def train_quantile_models(features: Any, target: Any, config: dict) -> dict:
    """Train quantile models for forecast uncertainty intervals in a later step."""
    raise NotImplementedError("Quantile training is intentionally excluded from this skeleton step.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Quantile model training placeholder. No model is trained in this skeleton step.")


if __name__ == "__main__":
    main()
