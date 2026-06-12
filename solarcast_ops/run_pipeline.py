from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.baselines import add_baseline_predictions
from src.asof_backtest import run_asof_backtest
from src.config import get_paths, load_config
from src.data_download import download_or_load_data
from src.demo_selection import select_demo_cases
from src.evaluate import evaluate_models
from src.feature_engineering import build_features
from src.irradiance_model import train_irradiance_model
from src.preprocessing import preprocess_data
from src.train import train_models
from src.uncertainty import build_prediction_intervals
from src.utils import setup_logging


LOGGER = logging.getLogger(__name__)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def run(step: str, force: bool = False) -> None:
    setup_logging()
    config = load_config()
    paths = get_paths()

    if step in {"all", "download"}:
        download_or_load_data(config, force=force)

    if step in {"all", "preprocess"}:
        raw = download_or_load_data(config, force=False)
        preprocess_data(raw, config)

    if step in {"all", "features"}:
        if force or not (paths.processed_dir / "clean_hourly.csv").exists():
            raw = download_or_load_data(config, force=False)
            clean = preprocess_data(raw, config)
        else:
            clean = _read(paths.processed_dir / "clean_hourly.csv")
        features = build_features(clean, config)
        add_baseline_predictions(features, config)

    if step in {"all", "baselines"}:
        features = _read(paths.processed_dir / "features.csv")
        add_baseline_predictions(features, config)

    if step in {"all", "train"}:
        features = _read(paths.processed_dir / "features.csv")
        train_models(features, config, force=force)

    if step in {"all", "irradiance"}:
        features = _read(paths.processed_dir / "features.csv")
        train_irradiance_model(features, config)

    if step in {"all", "evaluate"}:
        features = _read(paths.processed_dir / "features.csv")
        if not (paths.predictions_dir / "baseline_predictions.csv").exists():
            add_baseline_predictions(features, config)
        if not (paths.predictions_dir / "ml_test_predictions.csv").exists():
            train_models(features, config, force=force)
        evaluate_models(features, config)
        build_prediction_intervals(config)

    if step in {"all", "demo"}:
        if not (paths.predictions_dir / "test_predictions_with_uncertainty.csv").exists():
            features = _read(paths.processed_dir / "features.csv")
            evaluate_models(features, config)
            build_prediction_intervals(config)
        select_demo_cases(config)

    if step in {"all", "asof"}:
        features = _read(paths.processed_dir / "features.csv")
        run_asof_backtest(features, config)

    LOGGER.info("Pipeline step complete: %s", step)


def main() -> None:
    parser = argparse.ArgumentParser(description="SolarCast Ops pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "download", "preprocess", "features", "baselines", "train", "irradiance", "evaluate", "demo", "asof"],
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="Force refresh where supported")
    args = parser.parse_args()
    run(args.step, force=args.force)


if __name__ == "__main__":
    main()
