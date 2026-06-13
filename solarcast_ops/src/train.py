from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.config import get_paths
from src.feature_engineering import HISTORY_ONLY_FEATURES, SATELLITE_FEATURES
from src.utils import clip_power, set_random_seed, write_json

LOGGER = logging.getLogger(__name__)

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - environment dependent
    LGBMRegressor = None


def split_time_series(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    train_end = pd.Timestamp(config["forecast"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(config["forecast"]["validation_end"], tz="UTC")
    test_end = pd.Timestamp(config["forecast"]["test_end"], tz="UTC")
    train = data[data["timestamp"] <= train_end]
    val = data[(data["timestamp"] > train_end) & (data["timestamp"] <= val_end)]
    test = data[(data["timestamp"] > val_end) & (data["timestamp"] <= test_end)]
    method = "configured_calendar"
    if min(len(train), len(val), len(test)) < 100:
        n = len(data)
        train = data.iloc[: int(n * 0.6)]
        val = data.iloc[int(n * 0.6) : int(n * 0.8)]
        test = data.iloc[int(n * 0.8) :]
        method = "chronological_60_20_20_time_order"
    metadata = {
        "split_method": method,
        "train_start": str(train["timestamp"].min()),
        "train_end": str(train["timestamp"].max()),
        "validation_start": str(val["timestamp"].min()),
        "validation_end": str(val["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
    }
    return train, val, test, metadata


def _make_model(seed: int, max_iter: int = 220):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=max_iter,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.05, random_state=seed, l2_regularization=0.01)


def _fit_one(train: pd.DataFrame, val: pd.DataFrame, features: list[str], target: str, seed: int):
    best_model = None
    best_mae = float("inf")
    for max_iter in [120, 220]:
        model = _make_model(seed, max_iter=max_iter)
        model.fit(train[features], train[target])
        pred = model.predict(val[features])
        mae = mean_absolute_error(val[target], pred)
        if mae < best_mae:
            best_mae = mae
            best_model = model
    return best_model, best_mae


def train_models(df: pd.DataFrame, config: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Train history-only and satellite-informed models for each horizon."""
    paths = get_paths()
    set_random_seed(int(config["project"]["random_seed"]))
    train_df, val_df, test_df, split_meta = split_time_series(df, config)
    horizons = [int(h) for h in config["forecast"]["horizons_hours"]]
    peak = float(config["site"]["peak_power_mw"])
    prediction_frames = []
    metadata: dict[str, Any] = {"split": split_meta, "models": {}}

    for h in horizons:
        target = f"target_h{h}"
        frame = test_df[["timestamp", "pv_power_mw", "is_daylight", "clear_sky_index", target]].copy()
        frame = frame.rename(columns={target: f"actual_h{h}"})
        val_frame = val_df[["timestamp", target]].copy().rename(columns={target: f"actual_h{h}"})
        for model_name, features in {
            "history_only": HISTORY_ONLY_FEATURES,
            "satellite_informed": SATELLITE_FEATURES,
        }.items():
            model_path = paths.models_dir / f"model_{model_name}_h{h}.pkl"
            model, val_mae = _fit_one(train_df, val_df, features, target, int(config["project"]["random_seed"]) + h)
            joblib.dump({"model": model, "features": features, "target": target, "model_name": model_name}, model_path)
            val_pred = clip_power(model.predict(val_df[features]), peak)
            test_pred = clip_power(model.predict(test_df[features]), peak)
            val_frame[f"{model_name}_h{h}"] = val_pred
            frame[f"{model_name}_h{h}"] = test_pred
            metadata["models"][f"{model_name}_h{h}"] = {
                "path": str(model_path),
                "features": features,
                "validation_mae_mw": float(val_mae),
            }
            LOGGER.info("Saved %s h%s model to %s", model_name, h, model_path)
        val_frame.to_csv(paths.predictions_dir / f"validation_predictions_h{h}.csv", index=False)
        prediction_frames.append(frame)

    merged = prediction_frames[0]
    for frame in prediction_frames[1:]:
        keep_cols = [c for c in frame.columns if c not in {"pv_power_mw", "is_daylight", "clear_sky_index"}]
        merged = merged.merge(frame[keep_cols], on="timestamp", how="inner")
    merged.to_csv(paths.predictions_dir / "ml_test_predictions.csv", index=False)
    write_json(paths.models_dir / "training_metadata.json", metadata)
    return merged


def load_model(path: Path) -> dict[str, Any]:
    return joblib.load(path)
