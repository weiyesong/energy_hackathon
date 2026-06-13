"""CAMS advanced aerosol and atmospheric composition data source placeholder."""

from __future__ import annotations


def describe_cams_source() -> str:
    """Return a short description of the planned CAMS integration."""
    return "CAMS is an advanced aerosol and atmospheric source and is intentionally excluded from the MVP."


def download_cams_data(config: dict) -> None:
    """Download CAMS data in a future advanced data-source implementation."""
    raise NotImplementedError(describe_cams_source())


def main() -> None:
    """Print the current role of this executable module."""
    print(describe_cams_source())


if __name__ == "__main__":
    main()
