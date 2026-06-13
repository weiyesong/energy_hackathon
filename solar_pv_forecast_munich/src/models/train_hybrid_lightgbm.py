"""Train deterministic hybrid residual irradiance models by forecast horizon."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_benchmarks import evaluate_predictions, save_evaluation_outputs


_LIGHTGBM_IMPORT_ERROR: str | None = None
_LIGHTGBM_MODULE: Any | None = None
_FALLBACK_WARNING_PRINTED = False

FORBIDDEN_FEATURE_COLUMNS = {
    "issue_time",
    "target_time",
    "target_valid_time",
    "target_timestamp",
    "GHI_target",
    "GHI_residual_target",
    "target_source",
    "target_quality_level",
    "GHI_persistence_naive",
    "GHI_persistence_csi",
    "GHI_pred_hybrid",
    "GHI_pred_hybrid_without_satellite",
}

SATELLITE_FEATURE_COLUMNS = {
    "satellite_ssi_issue",
    "satellite_clear_sky_index_issue",
    "satellite_ssi_lag_1",
    "satellite_ssi_trend",
    "satellite_data_available",
    "irradiance_source_std",
    "number_of_available_irradiance_sources",
    "best_satellite_source",
}

CATEGORICAL_COLUMNS = ["site_id", "best_satellite_source", "quality_flag"]


@dataclass
class FeatureEncoder:
    """Small deterministic feature encoder shared by training and inference."""

    feature_columns: list[str]
    categorical_maps: dict[str, dict[str, int]]
    include_satellite_features: bool
    fill_values: dict[str, float]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe into numeric model features with fitted mappings."""
        features = pd.DataFrame(index=df.index)
        for column in self.feature_columns:
            if column in CATEGORICAL_COLUMNS:
                mapping = self.categorical_maps.get(column, {})
                values = df[column].astype("string").map(mapping).fillna(-1).astype(float) if column in df else -1.0
                features[column] = values
            else:
                features[column] = pd.to_numeric(df[column], errors="coerce") if column in df else np.nan
                features[column] = features[column].fillna(self.fill_values.get(column, 0.0))
        return features


def build_feature_encoder(df: pd.DataFrame, include_satellite_features: bool = True) -> FeatureEncoder:
    """Fit a reusable feature encoder while excluding leakage-prone columns."""
    feature_columns: list[str] = []
    for column in df.columns:
        if column in FORBIDDEN_FEATURE_COLUMNS:
            continue
        if column.startswith("GHI_persistence"):
            continue
        if column in {"timestamp"}:
            continue
        if not include_satellite_features and column in SATELLITE_FEATURE_COLUMNS:
            continue
        if column.endswith("_forecast_proxy") or column.startswith("target_") or column in _safe_issue_columns():
            feature_columns.append(column)
        elif column in CATEGORICAL_COLUMNS or column in {"horizon_minutes", "GHI_phys_target"}:
            feature_columns.append(column)

    feature_columns = list(dict.fromkeys(feature_columns))
    categorical_maps = {
        column: {value: idx for idx, value in enumerate(sorted(df[column].dropna().astype(str).unique()))}
        for column in feature_columns
        if column in CATEGORICAL_COLUMNS and column in df
    }
    fill_values: dict[str, float] = {}
    for column in feature_columns:
        if column in CATEGORICAL_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce") if column in df else pd.Series(dtype="float64")
        fill_values[column] = float(values.median()) if values.notna().any() else 0.0
    return FeatureEncoder(feature_columns, categorical_maps, include_satellite_features, fill_values)


def train_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_root: str | Path,
) -> pd.DataFrame:
    """Train horizon-specific residual models and return test predictions."""
    output = Path(output_root)
    (output / "outputs/models").mkdir(parents=True, exist_ok=True)
    (output / "outputs/forecasts").mkdir(parents=True, exist_ok=True)
    (output / "outputs/metrics").mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    for horizon in sorted(train["horizon_minutes"].dropna().unique()):
        train_h = _clean_training_frame(train[train["horizon_minutes"] == horizon])
        val_h = _clean_training_frame(validation[validation["horizon_minutes"] == horizon])
        test_h = test[test["horizon_minutes"] == horizon].copy()
        if train_h.empty or test_h.empty:
            continue

        model, encoder, backend = _fit_one_model(train_h, include_satellite_features=True)
        model_no_sat, encoder_no_sat, backend_no_sat = _fit_one_model(train_h, include_satellite_features=False)

        pred_residual = model.predict(encoder.transform(test_h))
        pred_residual_no_sat = model_no_sat.predict(encoder_no_sat.transform(test_h))
        test_h["GHI_pred_hybrid"] = _apply_physical_constraints(test_h, test_h["GHI_phys_target"] + pred_residual)
        test_h["GHI_pred_hybrid_without_satellite"] = _apply_physical_constraints(test_h, test_h["GHI_phys_target"] + pred_residual_no_sat)
        prediction_frames.append(test_h)

        payload = {
            "model": model,
            "encoder": encoder,
            "backend": backend,
            "horizon_minutes": int(horizon),
            "model_philosophy": "GHI_pred = GHI_phys_target + residual_model(features)",
        }
        with (output / f"outputs/models/hybrid_ghi_horizon_{int(horizon)}.pkl").open("wb") as file:
            pickle.dump(payload, file)

        importance = _feature_importance(model, encoder, train_h, val_h)
        importance["backend"] = backend
        importance.to_csv(output / f"outputs/metrics/feature_importance_horizon_{int(horizon)}.csv", index=False)

        with (output / f"outputs/models/hybrid_ghi_horizon_{int(horizon)}_without_satellite.pkl").open("wb") as file:
            pickle.dump({"model": model_no_sat, "encoder": encoder_no_sat, "backend": backend_no_sat, "horizon_minutes": int(horizon)}, file)

    if not prediction_frames:
        raise RuntimeError("No hybrid models were trained.")
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["site_id", "timestamp", "horizon_minutes"])
    predictions.to_csv(output / "outputs/forecasts/test_hybrid_predictions.csv", index=False)
    metrics, skill, ablation = evaluate_predictions(predictions)
    save_evaluation_outputs(metrics, skill, ablation, output / "outputs/metrics")
    return predictions


def main() -> None:
    """Run deterministic hybrid residual training and benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Train deterministic hybrid LightGBM-style residual models.")
    parser.add_argument("--train", default="data/processed/train_dataset.parquet")
    parser.add_argument("--validation", default="data/processed/validation_dataset.parquet")
    parser.add_argument("--test", default="data/processed/test_dataset.parquet")
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()

    root = Path(args.output_root)
    train = pd.read_parquet(root / args.train)
    validation = pd.read_parquet(root / args.validation)
    test = pd.read_parquet(root / args.test)
    predictions = train_models(train, validation, test, root)
    print(f"Saved hybrid test predictions: {len(predictions)} rows")
    print("Offline hindcast evaluation complete. Do not present these metrics as live operational performance.")


def _fit_one_model(df: pd.DataFrame, include_satellite_features: bool) -> tuple[Any, FeatureEncoder, str]:
    """Fit one residual model, preferring LightGBM and falling back if its runtime is unavailable."""
    encoder = build_feature_encoder(df, include_satellite_features=include_satellite_features)
    x_train = encoder.transform(df)
    y_train = pd.to_numeric(df["GHI_residual_target"], errors="coerce").fillna(0.0)
    lgb, import_error = _lightgbm_module()
    if lgb is not None:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model, encoder, "lightgbm"

    global _FALLBACK_WARNING_PRINTED
    if not _FALLBACK_WARNING_PRINTED:
        print(f"LightGBM unavailable, using sklearn HistGradientBoostingRegressor fallback: {import_error}")
        _FALLBACK_WARNING_PRINTED = True
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, l2_regularization=0.05, random_state=42)
    model.fit(x_train, y_train)
    return model, encoder, "sklearn_hist_gradient_boosting_fallback"


def _clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without target residuals for supervised fitting."""
    return df[df["GHI_residual_target"].notna() & df["GHI_phys_target"].notna() & df["GHI_target"].notna()].copy()


def _apply_physical_constraints(df: pd.DataFrame, prediction: pd.Series | np.ndarray) -> pd.Series:
    """Apply non-negative, nighttime, and clear-sky upper-bound constraints to GHI predictions."""
    pred = pd.Series(prediction, index=df.index, dtype="float64").clip(lower=0.0)
    night = pd.to_numeric(df["target_solar_elevation"], errors="coerce").fillna(-90.0) <= 0.0
    pred.loc[night] = 0.0
    upper = 1.2 * pd.to_numeric(df["target_GHI_clear"], errors="coerce").fillna(0.0).clip(lower=0.0)
    pred = np.minimum(pred, upper)
    return pd.Series(pred, index=df.index)


def _feature_importance(model: Any, encoder: FeatureEncoder, train_h: pd.DataFrame, validation_h: pd.DataFrame) -> pd.DataFrame:
    """Return feature importance from LightGBM when available, otherwise permutation-style validation importance."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": encoder.feature_columns, "importance": values}).sort_values("importance", ascending=False)

    if validation_h.empty:
        validation_h = train_h.tail(min(len(train_h), 2000))
    sample = validation_h.head(min(len(validation_h), 2000)).copy()
    x = encoder.transform(sample)
    y = pd.to_numeric(sample["GHI_residual_target"], errors="coerce").fillna(0.0)
    base_pred = model.predict(x)
    base_rmse = float(np.sqrt(np.mean(np.square(base_pred - y))))
    rows = []
    for column in encoder.feature_columns:
        shuffled = x.copy()
        shuffled[column] = shuffled[column].sample(frac=1.0, random_state=42).to_numpy()
        rmse = float(np.sqrt(np.mean(np.square(model.predict(shuffled) - y))))
        rows.append({"feature": column, "importance": rmse - base_rmse})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def _safe_issue_columns() -> set[str]:
    """Return issue-time and lagged columns safe for model features."""
    return {
        "ghi_issue",
        "issue_clear_sky_ghi",
        "satellite_ssi_issue",
        "satellite_clear_sky_index_issue",
        "cloud_cover_issue",
        "ghi_lag_1",
        "ghi_lag_2",
        "ghi_lag_3",
        "clear_sky_index_lag_1",
        "clear_sky_index_lag_2",
        "cloud_cover_lag_1",
        "cloud_cover_lag_3",
        "satellite_ssi_lag_1",
        "satellite_ssi_trend",
        "cloud_cover_trend",
        "satellite_data_available",
        "irradiance_source_std",
        "number_of_available_irradiance_sources",
        "latitude",
        "longitude",
    }


def _lightgbm_module() -> tuple[Any | None, str | None]:
    """Import LightGBM once and cache any runtime import error."""
    global _LIGHTGBM_IMPORT_ERROR, _LIGHTGBM_MODULE
    if _LIGHTGBM_MODULE is not None:
        return _LIGHTGBM_MODULE, None
    if _LIGHTGBM_IMPORT_ERROR is not None:
        return None, _LIGHTGBM_IMPORT_ERROR
    try:
        import lightgbm as lgb
    except Exception as exc:
        _LIGHTGBM_IMPORT_ERROR = str(exc)
        return None, _LIGHTGBM_IMPORT_ERROR
    _LIGHTGBM_MODULE = lgb
    return _LIGHTGBM_MODULE, None


if __name__ == "__main__":
    main()
