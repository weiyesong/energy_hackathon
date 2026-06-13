"""PV operating-state plotting placeholder."""

from __future__ import annotations

from typing import Any


def plot_pv_operating_state(pv_data: Any, output_path: str | None = None) -> Any:
    """Plot PV operating state, expected generation, and limiting factors."""
    raise NotImplementedError("PV state plotting will be implemented after PV outputs exist.")


def main() -> None:
    """Print the current role of this executable module."""
    print("PV state plotting placeholder. No figure is created in this skeleton step.")


if __name__ == "__main__":
    main()
