"""ENTSO-E advanced grid and market data source placeholder."""

from __future__ import annotations


def describe_entsoe_source() -> str:
    """Return a short description of the planned ENTSO-E integration."""
    return "ENTSO-E is an advanced grid data source and is intentionally excluded from the MVP."


def download_entsoe_data(config: dict) -> None:
    """Download ENTSO-E data in a future advanced data-source implementation."""
    raise NotImplementedError(describe_entsoe_source())


def main() -> None:
    """Print the current role of this executable module."""
    print(describe_entsoe_source())


if __name__ == "__main__":
    main()
