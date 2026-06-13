"""PVGIS MVP reference data download placeholder."""

from __future__ import annotations


def build_pvgis_request(config: dict) -> dict:
    """Build the PVGIS request parameters from the project config."""
    return {
        "lat": config["location"]["latitude"],
        "lon": config["location"]["longitude"],
        "peakpower": config["pv_system"]["capacity_kwp"],
        "angle": config["pv_system"]["surface_tilt"],
        "aspect": config["pv_system"]["surface_azimuth"],
    }


def download_pvgis_data(config: dict) -> None:
    """Download PVGIS reference data in a later implementation if it stays simple."""
    raise NotImplementedError("PVGIS download logic will be implemented after the Open-Meteo MVP.")


def main() -> None:
    """Print the current role of this executable module."""
    print("PVGIS optional MVP downloader placeholder. No data is downloaded in this skeleton step.")


if __name__ == "__main__":
    main()
