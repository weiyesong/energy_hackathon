from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - exercised only when lightgbm is absent
    LGBMRegressor = None


FREQ = "15min"


@dataclass(frozen=True)
class ScheduleResult:
    schedule: pd.DataFrame
    metadata: dict[str, Any]


def _standardize_power_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean 15-minute frame with datetime and power_actual columns."""
    data = df.copy()
    if "datetime" not in data.columns:
        if "timestamp" in data.columns:
            data = data.rename(columns={"timestamp": "datetime"})
        else:
            raise ValueError("Input DataFrame must contain datetime or timestamp")
    if "power_actual" not in data.columns:
        if "pv_power_mw" in data.columns:
            data = data.rename(columns={"pv_power_mw": "power_actual"})
        else:
            raise ValueError("Input DataFrame must contain power_actual or pv_power_mw")

    data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors="coerce")
    data["power_actual"] = pd.to_numeric(data["power_actual"], errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values("datetime")
    data = data.drop_duplicates("datetime", keep="last")
    data = data.set_index("datetime")[["power_actual"]]
    data = data.resample(FREQ).mean()
    full_index = pd.date_range(data.index.min().floor(FREQ), data.index.max().ceil(FREQ), freq=FREQ, tz="UTC")
    data = data.reindex(full_index)
    data.index.name = "datetime"
    data["power_actual"] = data["power_actual"].interpolate(method="time", limit_direction="both")
    data["power_actual"] = data["power_actual"].ffill().bfill().clip(lower=0)
    return data[["power_actual"]].reset_index()


def _target_index(target_date: Any, periods: int = 96) -> pd.DatetimeIndex:
    target = pd.Timestamp(target_date)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    start = target.floor(FREQ)
    return pd.date_range(start, periods=periods, freq=FREQ, tz="UTC")


def _safe_replace_year(ts: pd.Timestamp, year: int) -> pd.Timestamp | None:
    try:
        return ts.replace(year=year)
    except ValueError:
        # Feb 29 fallback for non-leap years.
        if ts.month == 2 and ts.day == 29:
            return ts.replace(year=year, day=28)
        return None


def _lookup_power(series: pd.Series, ts: pd.Timestamp) -> float | None:
    if ts in series.index:
        value = series.loc[ts]
        return None if pd.isna(value) else float(value)
    return None


def _base_curve_for_index(clean: pd.DataFrame, index: pd.DatetimeIndex, years_back: int = 3) -> tuple[pd.Series, str]:
    series = clean.set_index("datetime")["power_actual"].sort_index()
    available_years = sorted(set(series.index.year))
    values: list[float] = []
    fallback_count = 0
    for ts in index:
        years = [year for year in available_years if year < ts.year]
        years = years[-years_back:]
        candidates = []
        for year in years:
            hist_ts = _safe_replace_year(ts, year)
            if hist_ts is not None:
                value = _lookup_power(series, hist_ts)
                if value is not None:
                    candidates.append(value)
        if candidates:
            values.append(float(np.mean(candidates)))
            continue

        fallback_count += 1
        same_month_time = clean[
            (clean["datetime"].dt.month == ts.month)
            & (clean["datetime"].dt.hour == ts.hour)
            & (clean["datetime"].dt.minute == ts.minute)
        ]["power_actual"]
        if not same_month_time.empty:
            values.append(float(same_month_time.mean()))
            continue

        same_time = clean[
            (clean["datetime"].dt.hour == ts.hour)
            & (clean["datetime"].dt.minute == ts.minute)
        ]["power_actual"]
        values.append(float(same_time.mean()) if not same_time.empty else 0.0)

    source = "same_month_day_time_previous_years"
    if fallback_count:
        source = f"mixed_with_fallback_for_{fallback_count}_intervals"
    return pd.Series(values, index=index, dtype=float), source


def _daily_base_total(clean: pd.DataFrame, day: pd.Timestamp) -> float:
    index = _target_index(day.normalize(), periods=96)
    base, _ = _base_curve_for_index(clean, index)
    # 15-minute power MW to energy MWh.
    return float(base.sum() * 0.25)


def _actual_daily_total(clean: pd.DataFrame, day: pd.Timestamp) -> float | None:
    start = day.normalize()
    end = start + pd.Timedelta(days=1)
    mask = (clean["datetime"] >= start) & (clean["datetime"] < end)
    subset = clean.loc[mask, "power_actual"]
    if subset.empty or subset.isna().all():
        return None
    return float(subset.sum() * 0.25)


def _recent_alpha(clean: pd.DataFrame, target_start: pd.Timestamp) -> tuple[float, str]:
    actual_total = 0.0
    base_total = 0.0
    complete_days = 0
    for i in range(1, 8):
        day = target_start.normalize() - pd.Timedelta(days=i)
        actual = _actual_daily_total(clean, day)
        base = _daily_base_total(clean, day)
        if actual is None:
            continue
        actual_total += actual
        base_total += base
        complete_days += 1

    if complete_days and base_total > 0:
        return float(actual_total / base_total), f"target_previous_{complete_days}_days"

    max_day = clean["datetime"].max().normalize()
    actual_total = 0.0
    base_total = 0.0
    complete_days = 0
    for i in range(0, 7):
        day = max_day - pd.Timedelta(days=i)
        actual = _actual_daily_total(clean, day)
        base = _daily_base_total(clean, day)
        if actual is None:
            continue
        actual_total += actual
        base_total += base
        complete_days += 1
    if complete_days and base_total > 0:
        return float(actual_total / base_total), f"latest_available_{complete_days}_days"
    return 1.0, "default_alpha_1_no_recent_actuals"


def predict_baseline_statistical(df: pd.DataFrame, target_date: Any) -> pd.DataFrame:
    """Model A: multiplicative seasonal statistical baseline.

    The input can be the user-requested schema (`datetime`, `power_actual`) or
    the project schema (`timestamp`, `pv_power_mw`). Output is a 15-minute daily
    schedule for the target start timestamp.
    """
    result = predict_baseline_statistical_with_metadata(df, target_date)
    return result.schedule


def predict_baseline_statistical_with_metadata(df: pd.DataFrame, target_date: Any) -> ScheduleResult:
    clean = _standardize_power_frame(df)
    index = _target_index(target_date, periods=96)
    base, base_source = _base_curve_for_index(clean, index)
    alpha, alpha_source = _recent_alpha(clean, index[0])
    schedule = pd.DataFrame(
        {
            "timestamp": index,
            "p_base_mw": base.to_numpy(),
            "alpha": alpha,
            "schedule_model_a_mw": (base * alpha).clip(lower=0).to_numpy(),
        }
    )
    return ScheduleResult(
        schedule=schedule,
        metadata={
            "model": "Model A statistical multiplicative seasonal baseline",
            "frequency": FREQ,
            "base_curve_source": base_source,
            "alpha": alpha,
            "alpha_source": alpha_source,
        },
    )


def _add_cyclical_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.DatetimeIndex(out["datetime"])
    minute_of_day = ts.hour * 60 + ts.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    out["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * ts.month / 12)
    out["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    return out


def _feature_frame(clean: pd.DataFrame) -> pd.DataFrame:
    series = clean.set_index("datetime")["power_actual"].sort_index()
    data = clean.copy()
    idx = pd.DatetimeIndex(data["datetime"])
    data["last_year_same_time"] = [series.get(_safe_replace_year(ts, ts.year - 1), np.nan) for ts in idx]
    data["two_years_same_time"] = [series.get(_safe_replace_year(ts, ts.year - 2), np.nan) for ts in idx]
    data["yesterday_same_time"] = [series.get(ts - pd.Timedelta(days=1), np.nan) for ts in idx]
    three_day_values = []
    for ts in idx:
        vals = [series.get(ts - pd.Timedelta(days=i), np.nan) for i in [1, 2, 3]]
        three_day_values.append(float(np.nanmean(vals)) if not np.isnan(vals).all() else np.nan)
    data["three_day_same_time_mean"] = three_day_values
    data = _add_cyclical_time_features(data)
    return data


def _feature_columns() -> list[str]:
    return [
        "last_year_same_time",
        "two_years_same_time",
        "yesterday_same_time",
        "three_day_same_time_mean",
        "tod_sin",
        "tod_cos",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]


def train_baseline_lgbm(df: pd.DataFrame, target_date: Any | None = None, random_state: int = 42) -> tuple[Any, dict[str, Any]]:
    """Train Model B using only data before target_date when supplied."""
    clean = _standardize_power_frame(df)
    if target_date is not None:
        cutoff = _target_index(target_date, periods=1)[0]
        clean = clean[clean["datetime"] < cutoff].copy()
    features = _feature_frame(clean)
    feature_cols = _feature_columns()
    train = features.dropna(subset=["power_actual"]).copy()
    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        train[col] = train[col].interpolate(limit_direction="both").ffill().bfill().fillna(0)

    if LGBMRegressor is not None:
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbosity=-1,
        )
        backend = "lightgbm"
    else:
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=random_state)
        backend = "sklearn_hist_gradient_boosting"

    model.fit(train[feature_cols], train["power_actual"])
    return model, {
        "model": "Model B machine learning baseline",
        "backend": backend,
        "features": feature_cols,
        "train_rows": int(len(train)),
        "train_start": str(train["datetime"].min()),
        "train_end": str(train["datetime"].max()),
    }


def predict_baseline_lgbm(df: pd.DataFrame, target_date: Any, model: Any | None = None) -> pd.DataFrame:
    """Model B prediction for a 15-minute daily schedule."""
    result = predict_baseline_lgbm_with_metadata(df, target_date, model=model)
    return result.schedule


def predict_baseline_lgbm_with_metadata(df: pd.DataFrame, target_date: Any, model: Any | None = None) -> ScheduleResult:
    clean = _standardize_power_frame(df)
    if model is None:
        model, metadata = train_baseline_lgbm(clean, target_date=target_date)
    else:
        metadata = {"model": "Model B machine learning baseline", "backend": type(model).__name__}

    index = _target_index(target_date, periods=96)
    target_frame = pd.DataFrame({"datetime": index, "power_actual": np.nan})
    combined = pd.concat([clean, target_frame], ignore_index=True).drop_duplicates("datetime", keep="last")
    features = _feature_frame(combined)
    features = features[features["datetime"].isin(index)].copy()
    feature_cols = _feature_columns()
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features[col].interpolate(limit_direction="both").ffill().bfill().fillna(0)
    prediction = np.clip(model.predict(features[feature_cols]), 0, None)
    schedule = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(features["datetime"]),
            "schedule_model_b_mw": prediction,
        }
    )
    return ScheduleResult(schedule=schedule, metadata=metadata)


def build_next_24h_schedule(
    df: pd.DataFrame,
    start_time: Any,
    schedule_model: str = "Model A",
) -> ScheduleResult:
    """Build a 0-24h 15-minute schedule using Model A, Model B, or both."""
    start = _target_index(start_time, periods=1)[0]
    periods = 97
    result_a = predict_baseline_statistical_with_metadata(df, start)
    schedule = result_a.schedule[["timestamp", "schedule_model_a_mw"]].copy()
    metadata: dict[str, Any] = {"model_a": result_a.metadata}

    if schedule_model in {"Model B", "Blend"}:
        result_b = predict_baseline_lgbm_with_metadata(df, start)
        schedule = schedule.merge(result_b.schedule, on="timestamp", how="left")
        metadata["model_b"] = result_b.metadata
    else:
        schedule["schedule_model_b_mw"] = np.nan

    if schedule_model == "Model B":
        schedule["scheduled_power_mw"] = schedule["schedule_model_b_mw"].fillna(schedule["schedule_model_a_mw"])
        selected = "Model B"
    elif schedule_model == "Blend":
        schedule["scheduled_power_mw"] = schedule[["schedule_model_a_mw", "schedule_model_b_mw"]].mean(axis=1)
        selected = "Blend"
    else:
        schedule["scheduled_power_mw"] = schedule["schedule_model_a_mw"]
        selected = "Model A"

    full_index = pd.date_range(start, periods=periods, freq=FREQ, tz="UTC")
    schedule = schedule.set_index("timestamp").reindex(full_index).interpolate(method="time").ffill().bfill()
    schedule.index.name = "timestamp"
    schedule = schedule.reset_index()
    schedule["horizon_hours"] = (schedule["timestamp"] - start).dt.total_seconds() / 3600
    metadata["selected_schedule_model"] = selected
    metadata["start_time"] = str(start)
    return ScheduleResult(schedule=schedule, metadata=metadata)
