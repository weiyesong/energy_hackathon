"""Unit conversion utilities for meteorological and PV variables."""

from __future__ import annotations

from typing import Any


def convert_weather_units(data: Any) -> Any:
    """Convert raw weather variables into the project's standard units."""
    raise NotImplementedError("Unit conversion rules will be added with the MVP data schemas.")


def main() -> None:
    """Print the current role of this executable module."""
    print("Unit conversion placeholder. No preprocessing is run in this skeleton step.")


if __name__ == "__main__":
    main()
