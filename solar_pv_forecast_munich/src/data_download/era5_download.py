"""ERA5 advanced data source placeholder."""

from __future__ import annotations


def describe_era5_source() -> str:
    """Return a short description of the planned ERA5 integration."""
    return "ERA5 is an advanced reanalysis data source and is intentionally excluded from the MVP."


def download_era5_data(config: dict) -> None:
    """Download ERA5 data in a future advanced data-source implementation."""
    raise NotImplementedError(describe_era5_source())


def main() -> None:
    """Print the current role of this executable module."""
    print(describe_era5_source())


if __name__ == "__main__":
    main()
