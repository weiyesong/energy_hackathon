from __future__ import annotations

import importlib

import pandas as pd
import pvlib


REQUIRED_MODULES = [
    "cartopy",
    "dask",
    "geopandas",
    "h5py",
    "lightgbm",
    "netCDF4",
    "numpy",
    "odc.stac",
    "planetary_computer",
    "pyproj",
    "pyresample",
    "rasterio",
    "rioxarray",
    "satpy",
    "shapely",
    "sklearn",
    "xarray",
    "zarr",
]


def main() -> None:
    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - smoke-test diagnostics
            missing.append(f"{module}: {exc}")

    if missing:
        raise SystemExit("Missing or broken modules:\n" + "\n".join(missing))

    times = pd.date_range("2026-06-12 10:00", periods=3, freq="h", tz="UTC")
    location = pvlib.location.Location(latitude=52.52, longitude=13.405, tz="UTC", altitude=34)
    clearsky = location.get_clearsky(times)

    print("Environment OK")
    print(clearsky[["ghi", "dni", "dhi"]].round(2).to_string())


if __name__ == "__main__":
    main()
