from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = ROOT_DIR
    raw_dir: Path = ROOT_DIR / "data" / "raw"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    demo_dir: Path = ROOT_DIR / "data" / "demo"
    models_dir: Path = ROOT_DIR / "models"
    reports_dir: Path = ROOT_DIR / "reports"
    figures_dir: Path = ROOT_DIR / "reports" / "figures"
    metrics_dir: Path = ROOT_DIR / "reports" / "metrics"
    predictions_dir: Path = ROOT_DIR / "reports" / "predictions"

    def ensure(self) -> None:
        for path in self.__dict__.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load project YAML configuration."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths() -> ProjectPaths:
    paths = ProjectPaths()
    paths.ensure()
    return paths
