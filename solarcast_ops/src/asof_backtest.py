from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import get_paths
from src.evaluate import classify_weather
from src.utils import clip_power, safe_divide, write_json

LOGGER = logging.getLogger(__name__)

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - environment dependent
    LGBMRegressor = None


DEFAULT_ASOF_HORIZONS_HOURS = [1, 2, 3, 6, 12, 24]

ASOF_FEATURES = [
    "pv_power_mw",
    "global_irradiance_wm2",
    "direct_irradiance_wm2",
    "diffuse_irradiance_wm2",
    "air_temperature_c",
    "wind_speed_ms",
    "solar_elevation_deg",
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "clear_sky_ghi_wm2",
    "clear_sky_dni_wm2",
    "clear_sky_dhi_wm2",
    "clear_sky_power_mw",
    "clear_sky_index",
    "power_clear_sky_ratio",
    "diffuse_fraction",
    "beam_fraction",
    "cloud_opacity_proxy",
    "cloud_variability_proxy",
    "cloud_trend_proxy",
    "wind_advected_cloud_change_proxy",
    "cloud_ramp_risk_proxy",
    "irradiance_lag_1h",
    "irradiance_lag_2h",
    "irradiance_lag_3h",
    "irradiance_rolling_mean_3h",
    "irradiance_rolling_std_3h",
    "irradiance_change_1h",
    "power_lag_1h",
    "power_lag_2h",
    "power_lag_3h",
    "power_rolling_mean_3h",
    "power_rolling_std_3h",
    "power_change_1h",
    "temperature_lag_1h",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]


@dataclass(frozen=True)
class AsofResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict[str, Any]


def _make_model(seed: int, n_estimators: int = 180):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=0.05, random_state=seed, l2_regularization=0.01)


def _prepare_asof_frame(features_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    data = features_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    data["weather_condition"] = classify_weather(data)
    data["cloud_opacity_proxy"] = (1.0 - pd.to_numeric(data["clear_sky_index"], errors="coerce")).clip(0, 1.5)
    data["cloud_variability_proxy"] = pd.to_numeric(data["irradiance_rolling_std_3h"], errors="coerce").fillna(0)
    data["cloud_trend_proxy"] = -pd.to_numeric(data["clear_sky_index"], errors="coerce").diff().fillna(0)
    data["wind_advected_cloud_change_proxy"] = (
        pd.to_numeric(data["wind_speed_ms"], errors="coerce").fillna(0)
        * pd.to_numeric(data["irradiance_change_1h"], errors="coerce").fillna(0).abs()
    )
    data["cloud_ramp_risk_proxy"] = (
        data["cloud_opacity_proxy"].fillna(0) * data["cloud_variability_proxy"].fillna(0)
        + 0.01 * data["wind_advected_cloud_change_proxy"].fillna(0)
    )

    indexed = data.set_index("timestamp")
    for h in horizons:
        valid_time = data["timestamp"] + pd.to_timedelta(h, unit="h")
        data[f"valid_time_h{h}"] = valid_time
        for source_col, target_col in [
            ("pv_power_mw", f"target_power_h{h}"),
            ("global_irradiance_wm2", f"target_ghi_h{h}"),
            ("clear_sky_index", f"target_kstar_h{h}"),
            ("diffuse_fraction", f"target_diffuse_fraction_h{h}"),
            ("clear_sky_power_mw", f"clear_sky_power_h{h}"),
        ]:
            if target_col in data.columns and data[target_col].notna().any():
                continue
            data[target_col] = indexed[source_col].reindex(pd.DatetimeIndex(valid_time)).to_numpy()

    for col in ASOF_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def _select_replay_times(data: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> list[dict[str, Any]]:
    candidates = data.copy()
    if start is not None:
        candidates = candidates[candidates["timestamp"] >= start]
    if end is not None:
        candidates = candidates[candidates["timestamp"] <= end]
    daylight = candidates[candidates["is_daylight"].astype(bool)].dropna(subset=["target_power_h3"]).copy()
    if daylight.empty:
        return []
    daylight["ramp_3h"] = daylight["target_power_h3"] - daylight["pv_power_mw"]
    daylight["irradiance_variability"] = daylight["global_irradiance_wm2"].diff().abs().rolling(3, min_periods=1).mean()
    daylight["date"] = daylight["timestamp"].dt.date

    cases: list[dict[str, Any]] = []
    picks = {
        "Large downward ramp": daylight.nsmallest(1, "ramp_3h"),
        "Large upward ramp": daylight.nlargest(1, "ramp_3h"),
        "Variable-cloud shock": daylight.nlargest(1, "irradiance_variability"),
    }
    for name, frame in picks.items():
        if not frame.empty:
            row = frame.iloc[0]
            cases.append({"name": name, "timestamp": row["timestamp"], "reason": f"Auto-selected by {name.lower()} criterion"})

    daily = daylight.groupby("date").agg(
        daylight=("timestamp", "count"),
        weather_clear=("weather_condition", lambda s: float((s == "clear").mean())),
        weather_over=("weather_condition", lambda s: float((s == "overcast").mean())),
        irr_std=("global_irradiance_wm2", "std"),
        mean_power=("pv_power_mw", "mean"),
    ).reset_index()
    clear_days = daily[daily["daylight"] >= 6].sort_values(["weather_clear", "irr_std"], ascending=[False, True])
    overcast_days = daily[daily["daylight"] >= 6].sort_values(["weather_over", "mean_power"], ascending=[False, True])
    for name, days in [("Stable clear day", clear_days), ("Overcast low-output day", overcast_days)]:
        if not days.empty:
            day = days.iloc[0]["date"]
            day_rows = daylight[daylight["date"] == day]
            noonish = day_rows.iloc[(day_rows["timestamp"].dt.hour - 12).abs().argmin()]
            cases.append({"name": name, "timestamp": noonish["timestamp"], "reason": f"Auto-selected representative {name.lower()}"})

    seen: set[pd.Timestamp] = set()
    unique_cases = []
    for case in cases:
        ts = pd.Timestamp(case["timestamp"])
        if ts not in seen:
            seen.add(ts)
            case["timestamp"] = ts
            unique_cases.append(case)
    return unique_cases


def _fit_target(train: pd.DataFrame, feature_cols: list[str], target_col: str, seed: int):
    frame = train.dropna(subset=feature_cols + [target_col]).copy()
    if len(frame) < 500:
        raise ValueError(f"Not enough as-of training rows for {target_col}: {len(frame)}")
    model = _make_model(seed)
    model.fit(frame[feature_cols], frame[target_col])
    return model, len(frame)


def _clear_sky_persistence(row: pd.Series, h: int, peak: float) -> float:
    current_clear = float(row.get("clear_sky_power_mw", 0.0) or 0.0)
    future_clear = float(row.get(f"clear_sky_power_h{h}", 0.0) or 0.0)
    if current_clear < 1e-6:
        return 0.0
    cloud_factor = safe_divide(np.array([float(row["pv_power_mw"])]), np.array([current_clear]))[0]
    cloud_factor = float(np.clip(np.nan_to_num(cloud_factor, nan=0.0, posinf=0.0, neginf=0.0), 0, 1.3))
    return float(clip_power(np.array([cloud_factor * future_clear]), peak)[0])


def run_asof_backtest(
    features_df: pd.DataFrame,
    config: dict[str, Any],
    horizons: list[int] | None = None,
    asof_times: list[Any] | None = None,
    persist_outputs: bool = True,
) -> AsofResult:
    """Run strict as-of replay: train labels must be observable by each issue time."""
    paths = get_paths()
    horizons = [int(h) for h in (horizons or DEFAULT_ASOF_HORIZONS_HOURS)]
    data = _prepare_asof_frame(features_df, horizons)
    peak = float(config["site"]["peak_power_mw"])
    feature_cols = [col for col in ASOF_FEATURES if col in data.columns]

    if asof_times is None:
        selection_start = pd.Timestamp(config["forecast"]["validation_end"], tz="UTC") + pd.Timedelta(hours=1)
        selection_end = pd.Timestamp(config["forecast"]["test_end"], tz="UTC")
        cases = _select_replay_times(data, start=selection_start, end=selection_end)
    else:
        cases = [{"name": "Manual as-of replay", "timestamp": pd.Timestamp(ts), "reason": "User supplied"} for ts in asof_times]
    if not cases:
        raise ValueError("No as-of replay cases could be selected")

    rows = []
    model_cache: dict[tuple[pd.Timestamp, int, str], tuple[Any, int]] = {}
    for case in cases:
        asof_time = pd.Timestamp(case["timestamp"])
        if asof_time.tzinfo is None:
            asof_time = asof_time.tz_localize("UTC")
        else:
            asof_time = asof_time.tz_convert("UTC")
        current_candidates = data[data["timestamp"] <= asof_time]
        if current_candidates.empty:
            continue
        current = current_candidates.iloc[-1]
        actual_asof = pd.Timestamp(current["timestamp"])

        for h in horizons:
            valid_time = actual_asof + pd.to_timedelta(h, unit="h")
            if pd.isna(current.get(f"target_power_h{h}")):
                continue
            train_cutoff = actual_asof - pd.to_timedelta(h, unit="h")
            train = data[data["timestamp"] <= train_cutoff].copy()
            key_power = (actual_asof, h, "power")
            key_ghi = (actual_asof, h, "ghi")
            try:
                if key_power not in model_cache:
                    model_cache[key_power] = _fit_target(train, feature_cols, f"target_power_h{h}", int(config["project"]["random_seed"]) + h)
                if key_ghi not in model_cache:
                    model_cache[key_ghi] = _fit_target(train, feature_cols, f"target_ghi_h{h}", int(config["project"]["random_seed"]) + 1000 + h)
            except ValueError as exc:
                LOGGER.warning("Skipping as-of case %s h%s: %s", actual_asof, h, exc)
                continue

            power_model, train_rows = model_cache[key_power]
            ghi_model, ghi_train_rows = model_cache[key_ghi]
            x = pd.DataFrame([current[feature_cols].astype(float).to_dict()])
            pred_power = float(clip_power(power_model.predict(x), peak)[0])
            pred_ghi = float(max(0.0, ghi_model.predict(x)[0]))
            persistence = float(clip_power(np.array([current["pv_power_mw"]]), peak)[0])
            csp = _clear_sky_persistence(current, h, peak)
            actual_power = float(current[f"target_power_h{h}"])
            actual_ghi = float(current[f"target_ghi_h{h}"])
            valid_is_daylight = bool(actual_ghi > 20 or actual_power > 0.02)
            rows.append(
                {
                    "case_name": case["name"],
                    "case_reason": case["reason"],
                    "asof_time": actual_asof,
                    "visible_data_cutoff": actual_asof,
                    "train_label_cutoff": train_cutoff,
                    "horizon_h": h,
                    "valid_time": valid_time,
                    "current_power_mw": float(current["pv_power_mw"]),
                    "actual_power_mw": actual_power,
                    "pred_power_mw": pred_power,
                    "pred_persistence_mw": persistence,
                    "pred_clear_sky_persistence_mw": csp,
                    "absolute_error_mw": abs(actual_power - pred_power),
                    "persistence_absolute_error_mw": abs(actual_power - persistence),
                    "clear_sky_persistence_absolute_error_mw": abs(actual_power - csp),
                    "actual_ghi_wm2": actual_ghi,
                    "pred_ghi_wm2": pred_ghi,
                    "ghi_absolute_error_wm2": abs(actual_ghi - pred_ghi),
                    "valid_is_daylight": valid_is_daylight,
                    "current_ghi_wm2": float(current["global_irradiance_wm2"]),
                    "current_clear_sky_index": float(current["clear_sky_index"]),
                    "cloud_opacity_proxy": float(current["cloud_opacity_proxy"]),
                    "cloud_variability_proxy": float(current["cloud_variability_proxy"]),
                    "cloud_trend_proxy": float(current["cloud_trend_proxy"]),
                    "wind_advected_cloud_change_proxy": float(current["wind_advected_cloud_change_proxy"]),
                    "cloud_ramp_risk_proxy": float(current["cloud_ramp_risk_proxy"]),
                    "diffuse_fraction": float(current["diffuse_fraction"]),
                    "beam_fraction": float(current["beam_fraction"]),
                    "wind_speed_ms": float(current["wind_speed_ms"]),
                    "air_temperature_c": float(current["air_temperature_c"]),
                    "weather_condition": current.get("weather_condition", "unknown"),
                    "train_rows_power": train_rows,
                    "train_rows_ghi": ghi_train_rows,
                    "feature_family": "satellite_irradiance_cloud_wind_history",
                    "data_policy": "strict_asof_no_labels_after_issue_time",
                }
            )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("As-of backtest produced no predictions")
    metric_rows = []
    for (h, segment), subset in pd.concat(
        [
            predictions.assign(segment="all"),
            predictions[predictions["valid_is_daylight"].astype(bool)].assign(segment="daylight"),
        ],
        ignore_index=True,
    ).groupby(["horizon_h", "segment"]):
        if subset.empty:
            continue
        mae = mean_absolute_error(subset["actual_power_mw"], subset["pred_power_mw"])
        rmse = float(np.sqrt(mean_squared_error(subset["actual_power_mw"], subset["pred_power_mw"])))
        mae_p = mean_absolute_error(subset["actual_power_mw"], subset["pred_persistence_mw"])
        mae_csp = mean_absolute_error(subset["actual_power_mw"], subset["pred_clear_sky_persistence_mw"])
        ghi_mae = mean_absolute_error(subset["actual_ghi_wm2"], subset["pred_ghi_wm2"])
        metric_rows.append(
            {
                "horizon_h": int(h),
                "segment": segment,
                "cases": int(len(subset)),
                "mae_mw": float(mae),
                "rmse_mw": rmse,
                "skill_vs_persistence": float(1 - mae / mae_p) if mae_p > 0 else np.nan,
                "skill_vs_clear_sky_persistence": float(1 - mae / mae_csp) if mae_csp > 0 else np.nan,
                "ghi_mae_wm2": float(ghi_mae),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["segment", "horizon_h"])
    metadata = {
        "method": "strict_asof_backtest",
        "policy": "For horizon h, training labels must satisfy sample_timestamp + h <= asof_time.",
        "selection_window": {
            "start": str(pd.Timestamp(config["forecast"]["validation_end"], tz="UTC") + pd.Timedelta(hours=1)),
            "end": str(pd.Timestamp(config["forecast"]["test_end"], tz="UTC")),
        },
        "horizons_hours": horizons,
        "features": feature_cols,
        "cases": [
            {"name": case["name"], "timestamp": str(pd.Timestamp(case["timestamp"])), "reason": case["reason"]}
            for case in cases
        ],
        "remote_sensing_proxies": [
            "Open-Meteo satellite archive GHI/direct/diffuse radiation when available",
            "PVGIS/SARAH-3 satellite-derived irradiance fallback",
            "clear_sky_index as cloud opacity proxy",
            "diffuse_fraction and beam_fraction as cloud/sky-condition proxies",
            "irradiance rolling variability and ramp as cloud-motion proxy",
            "cloud trend and wind-advected irradiance-change proxies",
            "wind_speed_ms and air_temperature_c as meteorological context",
        ],
    }

    if persist_outputs:
        predictions.to_csv(paths.predictions_dir / "asof_backtest_predictions.csv", index=False)
        metrics.to_csv(paths.metrics_dir / "asof_backtest_metrics.csv", index=False)
        joblib.dump({"metadata": metadata}, paths.models_dir / "asof_backtest_metadata.pkl")
        write_json(paths.metrics_dir / "asof_backtest_metadata.json", metadata)
    return AsofResult(predictions=predictions, metrics=metrics, metadata=metadata)
