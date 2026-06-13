"""Train calibrated probabilistic GHI quantile forecasts."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_uncertainty import evaluate_uncertainty, save_uncertainty_outputs
from src.models.conformal_calibration import apply_conformal_interval, fit_split_conformal_interval
from src.models.train_hybrid_lightgbm import FeatureEncoder, _apply_physical_constraints, _lightgbm_module, build_feature_encoder


QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
MAX_FALLBACK_TRAIN_ROWS_PER_HORIZON = 50_000
_QUANTILE_FALLBACK_WARNING_PRINTED = False


def train_quantile_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_root: str | Path,
) -> pd.DataFrame:
    """Train horizon-specific quantile models, calibrate on validation, and predict test intervals."""
    output = Path(output_root)
    (output / "outputs/models").mkdir(parents=True, exist_ok=True)
    (output / "outputs/forecasts").mkdir(parents=True, exist_ok=True)
    (output / "outputs/metrics").mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    for horizon in sorted(train["horizon_minutes"].dropna().unique()):
        train_h = _clean(train[train["horizon_minutes"] == horizon])
        validation_h = _clean(validation[validation["horizon_minutes"] == horizon])
        test_h = test[test["horizon_minutes"] == horizon].copy()
        if train_h.empty or validation_h.empty or test_h.empty:
            continue

        encoder = build_feature_encoder(train_h, include_satellite_features=True)
        models: dict[str, Any] = {}
        backends: dict[str, str] = {}
        for label, quantile in QUANTILES.items():
            model, backend = _fit_quantile_model(train_h, encoder, quantile)
            models[label] = model
            backends[label] = backend
            with (output / f"outputs/models/quantile_ghi_horizon_{int(horizon)}_{label}.pkl").open("wb") as file:
                pickle.dump({"model": model, "encoder": encoder, "backend": backend, "horizon_minutes": int(horizon), "quantile": quantile}, file)

        validation_pred = _predict_quantiles(validation_h, encoder, models)
        expansion = fit_split_conformal_interval(validation_pred)
        test_pred = _predict_quantiles(test_h, encoder, models)
        test_pred = apply_conformal_interval(test_pred, expansion)
        test_pred = _constrain_calibrated_quantiles(test_pred)
        test_pred["conformal_expansion"] = expansion
        test_pred["uncertainty_level"] = _uncertainty_level(test_pred)
        prediction_frames.append(test_pred)

    if not prediction_frames:
        raise RuntimeError("No quantile models were trained.")

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["site_id", "timestamp", "horizon_minutes"])
    predictions.to_csv(output / "outputs/forecasts/test_probabilistic_predictions.csv", index=False)
    metrics, regimes = evaluate_uncertainty(predictions)
    save_uncertainty_outputs(metrics, regimes, output / "outputs/metrics")
    return predictions


def main() -> None:
    """Run probabilistic quantile training and conformal calibration."""
    parser = argparse.ArgumentParser(description="Train calibrated probabilistic GHI quantile forecasts.")
    parser.add_argument("--train", default="data/processed/train_dataset.parquet")
    parser.add_argument("--validation", default="data/processed/validation_dataset.parquet")
    parser.add_argument("--test", default="data/processed/test_dataset.parquet")
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()

    root = Path(args.output_root)
    train = pd.read_parquet(root / args.train)
    validation = pd.read_parquet(root / args.validation)
    test = pd.read_parquet(root / args.test)
    predictions = train_quantile_models(train, validation, test, root)
    print(f"Saved calibrated probabilistic predictions: {len(predictions)} rows")


def _fit_quantile_model(train: pd.DataFrame, encoder: FeatureEncoder, quantile: float) -> tuple[Any, str]:
    """Fit one quantile model, preferring LightGBM quantile regression with sklearn fallback."""
    x_train = encoder.transform(train)
    y_train = pd.to_numeric(train["GHI_target"], errors="coerce").fillna(0.0)
    lgb, error = _lightgbm_module()
    if lgb is not None:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=260,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model, "lightgbm_quantile"

    global _QUANTILE_FALLBACK_WARNING_PRINTED
    if not _QUANTILE_FALLBACK_WARNING_PRINTED:
        print(f"LightGBM quantile unavailable, using sklearn HistGradientBoostingRegressor fallback: {error}")
        _QUANTILE_FALLBACK_WARNING_PRINTED = True
    x_fit, y_fit = _fallback_training_sample(x_train, y_train)
    loss = "quantile" if quantile != 0.50 else "squared_error"
    kwargs = {
        "loss": loss,
        "max_iter": 80,
        "learning_rate": 0.07,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.05,
        "random_state": 42,
    }
    if loss == "quantile":
        kwargs["quantile"] = quantile
    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(x_fit, y_fit)
    return model, "sklearn_hist_gradient_boosting_quantile_fallback"


def _predict_quantiles(df: pd.DataFrame, encoder: FeatureEncoder, models: dict[str, Any]) -> pd.DataFrame:
    """Predict raw quantiles and apply physical monotonicity constraints."""
    predicted = df.copy()
    x = encoder.transform(df)
    raw = np.vstack([models[label].predict(x) for label in ["p10", "p50", "p90"]]).T
    raw = np.sort(raw, axis=1)
    predicted["GHI_P10_raw"] = raw[:, 0]
    predicted["GHI_P50"] = raw[:, 1]
    predicted["GHI_P90_raw"] = raw[:, 2]
    predicted["GHI_P10_raw"] = _apply_physical_constraints(predicted, predicted["GHI_P10_raw"])
    predicted["GHI_P50"] = _apply_physical_constraints(predicted, predicted["GHI_P50"])
    predicted["GHI_P90_raw"] = _apply_physical_constraints(predicted, predicted["GHI_P90_raw"])
    sorted_after_constraints = np.sort(predicted[["GHI_P10_raw", "GHI_P50", "GHI_P90_raw"]].to_numpy(dtype=float), axis=1)
    predicted["GHI_P10_raw"] = sorted_after_constraints[:, 0]
    predicted["GHI_P50"] = sorted_after_constraints[:, 1]
    predicted["GHI_P90_raw"] = sorted_after_constraints[:, 2]
    return predicted


def _uncertainty_level(df: pd.DataFrame) -> pd.Series:
    """Classify uncertainty from calibrated relative interval width."""
    width = pd.to_numeric(df["GHI_P90_calibrated"], errors="coerce") - pd.to_numeric(df["GHI_P10_calibrated"], errors="coerce")
    relative_width = width / pd.to_numeric(df["GHI_P50"], errors="coerce").clip(lower=50.0)
    return pd.cut(relative_width, bins=[-np.inf, 0.5, 1.2, np.inf], labels=["Low", "Medium", "High"]).astype(str)


def _constrain_calibrated_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physical constraints and row-wise monotonicity to calibrated quantiles."""
    constrained = df.copy()
    constrained["GHI_P10_calibrated"] = _apply_physical_constraints(constrained, constrained["GHI_P10_calibrated"])
    constrained["GHI_P50"] = _apply_physical_constraints(constrained, constrained["GHI_P50"])
    constrained["GHI_P90_calibrated"] = _apply_physical_constraints(constrained, constrained["GHI_P90_calibrated"])
    ordered = np.sort(constrained[["GHI_P10_calibrated", "GHI_P50", "GHI_P90_calibrated"]].to_numpy(dtype=float), axis=1)
    constrained["GHI_P10_calibrated"] = ordered[:, 0]
    constrained["GHI_P50"] = ordered[:, 1]
    constrained["GHI_P90_calibrated"] = ordered[:, 2]
    return constrained


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without target values for quantile fitting."""
    return df[df["GHI_target"].notna()].copy()


def _fallback_training_sample(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Cap fallback training rows to keep non-LightGBM quantile training tractable."""
    if len(x_train) <= MAX_FALLBACK_TRAIN_ROWS_PER_HORIZON:
        return x_train, y_train
    indices = np.linspace(0, len(x_train) - 1, MAX_FALLBACK_TRAIN_ROWS_PER_HORIZON, dtype=int)
    return x_train.iloc[indices].reset_index(drop=True), y_train.iloc[indices].reset_index(drop=True)


if __name__ == "__main__":
    main()
