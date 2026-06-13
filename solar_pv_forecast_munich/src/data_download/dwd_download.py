"""DWD advanced data source placeholder."""

from __future__ import annotations


def describe_dwd_source() -> str:
    """Return a short description of the planned DWD integration."""
    return "DWD is an advanced data source and is intentionally excluded from the MVP."


def download_dwd_data(config: dict) -> None:
    """Download DWD data in a future advanced data-source implementation."""
    raise NotImplementedError(describe_dwd_source())


def main() -> None:
    """Print the current role of this executable module."""
    print(describe_dwd_source())


if __name__ == "__main__":
    main()
