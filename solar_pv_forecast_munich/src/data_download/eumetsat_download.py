"""Optional EUMETSAT SSI adapter for the primary satellite layer."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_registry import register_data_source
from src.preprocessing.eumetsat_ingest import (
    DEFAULT_EUMETSAT_ALIASES,
    extract_munich_sites,
    find_manual_eumetsat_files,
    inspect_dataset_variables,
    load_eumetsat_file,
    standardize_eumetsat_data,
)


SOURCE_NAME = "eumetsat_ssi"
SOURCE_TYPE_REGISTRY = "primary operational satellite-derived irradiance source"
OUTPUT_PATH = Path("data/processed/eumetsat_ssi_all_sites.parquet")
MANUAL_INSTRUCTION = """EUMETSAT SSI is not currently available.

To enable the primary satellite layer:
1. obtain EUMETSAT Data Store access;
2. download Meteosat surface solar irradiance products covering Munich;
3. prefer 15-minute temporal resolution;
4. place NetCDF or GRIB files in data/manual/eumetsat/;
5. rerun this script."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load project YAML configuration for the EUMETSAT adapter."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["_project_root"] = str(path.parent)
    return config


def check_eumetsat_credentials() -> bool:
    """Return whether EUMETSAT credentials appear to be configured without exposing secrets."""
    env_has_pair = bool(os.getenv("EUMETSAT_CONSUMER_KEY") and os.getenv("EUMETSAT_CONSUMER_SECRET"))
    env_has_token = bool(os.getenv("EUMETSAT_API_KEY") or os.getenv("EUMETSAT_TOKEN"))
    config_files = [
        Path.home() / ".eumdac" / "credentials",
        Path.home() / ".eumetsat" / "credentials",
    ]
    file_has_credentials = any(path.exists() and path.stat().st_size > 0 for path in config_files)
    return env_has_pair or env_has_token or file_has_credentials


def list_available_products() -> list[dict[str, Any]]:
    """List EUMETSAT products when credentials and optional client libraries are available."""
    if not check_eumetsat_credentials():
        warnings.warn("EUMETSAT credentials are not configured; switching to manual-file mode.", stacklevel=2)
        return []
    try:
        import eumdac  # type: ignore  # noqa: F401
    except ImportError:
        warnings.warn("EUMETSAT credentials exist, but the optional eumdac client is not installed.", stacklevel=2)
        return []
    warnings.warn("Automatic EUMETSAT product discovery is not configured for this environment; use manual-file mode.", stacklevel=2)
    return []


def download_eumetsat_ssi(start_time: str, end_time: str, area: dict[str, float]) -> list[Path]:
    """Attempt automatic EUMETSAT SSI download, returning downloaded file paths when supported."""
    if not check_eumetsat_credentials():
        raise RuntimeError("EUMETSAT credentials missing; manual download required.")
    products = list_available_products()
    if not products:
        raise RuntimeError("EUMETSAT SSI automatic product discovery unavailable; manual download required.")
    raise RuntimeError("Automatic EUMETSAT SSI download is not implemented without a configured Data Store client.")


def download_eumetsat_cloud_product(*args: Any, **kwargs: Any) -> list[Path]:
    """Attempt automatic EUMETSAT cloud-product download, returning paths when supported."""
    if not check_eumetsat_credentials():
        raise RuntimeError("EUMETSAT credentials missing; manual download required.")
    products = list_available_products()
    if not products:
        raise RuntimeError("EUMETSAT cloud-product discovery unavailable; manual download required.")
    raise RuntimeError("Automatic EUMETSAT cloud-product download is not implemented without a configured Data Store client.")


def main() -> None:
    """Run optional EUMETSAT automatic discovery or manual-file ingestion."""
    parser = argparse.ArgumentParser(description="Optional EUMETSAT SSI adapter for SolarOps.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config["_project_root"])
    registry_path = project_root / "data/processed/data_source_registry.json"
    manual_directory = project_root / config["data_sources"]["eumetsat"]["manual_directory"]
    aliases = config.get("data_sources", {}).get("eumetsat_variable_aliases", DEFAULT_EUMETSAT_ALIASES)

    if check_eumetsat_credentials():
        try:
            list_available_products()
        except Exception as exc:
            warnings.warn(f"EUMETSAT automatic mode failed; switching to manual-file mode. Reason: {exc}", stacklevel=2)

    files = find_manual_eumetsat_files(manual_directory)
    if not files:
        print(MANUAL_INSTRUCTION)
        register_data_source(
            SOURCE_NAME,
            source_type=SOURCE_TYPE_REGISTRY,
            is_satellite_derived=True,
            download_status="unavailable",
            manual_action_required=True,
            error_message="credentials missing; manual download required",
            file_path=str(manual_directory.relative_to(project_root)),
            registry_path=registry_path,
        )
        return

    frames: list[pd.DataFrame] = []
    for path in files:
        print(f"Reading manual EUMETSAT file: {path}")
        print(f"Available variables: {', '.join(inspect_dataset_variables(path))}")
        dataset = load_eumetsat_file(path)
        extracted = extract_munich_sites(dataset, config["sites"])
        frames.append(standardize_eumetsat_data(extracted, aliases=aliases))

    combined = pd.concat(frames, ignore_index=True).sort_values(["site_id", "timestamp"]).drop_duplicates(["site_id", "timestamp"])
    output_path = project_root / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"Saved EUMETSAT SSI data: {output_path}")
    register_data_source(
        SOURCE_NAME,
        source_type=SOURCE_TYPE_REGISTRY,
        is_satellite_derived=True,
        date_range={"start": str(combined["timestamp"].min()), "end": str(combined["timestamp"].max())},
        temporal_resolution="manual_file",
        file_path=str(output_path.relative_to(project_root)),
        available_columns=list(combined.columns),
        download_status="available",
        manual_action_required=False,
        error_message=None,
        registry_path=registry_path,
    )


if __name__ == "__main__":
    main()
