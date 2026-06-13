from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pvlib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_adapters import attach_demo_adapters
from src.evaluate import classify_weather
from src.train import split_time_series
from src.config import get_paths
from src.utils import safe_divide, set_random_seed, write_json

LOGGER = logging.getLogger(__name__)

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - environment dependent
    LGBMRegressor = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


IRRADIANCE_FEATURES = [
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
    "clear_sky_index",
    "diffuse_fraction",
    "beam_fraction",
    "irradiance_lag_1h",
    "irradiance_lag_2h",
    "irradiance_lag_3h",
    "irradiance_rolling_mean_3h",
    "irradiance_rolling_std_3h",
    "irradiance_change_1h",
    "temperature_lag_1h",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "missing_satellite",
    "missing_ground",
    "missing_nwp",
    "horizon_hours",
]


if nn is not None:

    class QuantileMoE(nn.Module):
        """Small horizon-conditioned Mixture-of-Experts for irradiance quantiles."""

        def __init__(self, input_dim: int, hidden_dim: int, n_quantiles: int, dropout: float) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                    for _ in range(4)
                ]
            )
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 4))
            self.kstar_head = nn.Linear(hidden_dim, n_quantiles)
            self.fd_head = nn.Linear(hidden_dim, n_quantiles)

        def forward(self, x):
            encoded = self.encoder(x)
            gate_weights = torch.softmax(self.gate(encoded), dim=1)
            expert_stack = torch.stack([expert(encoded) for expert in self.experts], dim=1)
            fused = (expert_stack * gate_weights.unsqueeze(-1)).sum(dim=1)
            return _monotonic_quantiles(self.kstar_head(fused)), _monotonic_quantiles(self.fd_head(fused)), gate_weights


def _monotonic_quantiles(raw):
    first = raw[:, :1]
    increments = torch.nn.functional.softplus(raw[:, 1:])
    return torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)


def _make_quantile_model(alpha: float, seed: int):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=90,
            learning_rate=0.06,
            num_leaves=24,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=80, learning_rate=0.06, random_state=seed)


def _stack_horizons(frame: pd.DataFrame, horizons: list[int], target_prefix: str) -> pd.DataFrame:
    chunks = []
    for h in horizons:
        cols = IRRADIANCE_FEATURES[:-1] + [f"target_{target_prefix}_h{h}"]
        chunk = frame[cols].copy()
        chunk["horizon_hours"] = float(h)
        chunk["target"] = chunk[f"target_{target_prefix}_h{h}"]
        chunk = chunk.drop(columns=[f"target_{target_prefix}_h{h}"])
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True).dropna(subset=["target"])


def _stack_horizons_multi_target(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    chunks = []
    for h in horizons:
        cols = IRRADIANCE_FEATURES[:-1] + [f"target_kstar_h{h}", f"target_diffuse_fraction_h{h}"]
        chunk = frame[cols].copy()
        chunk["horizon_hours"] = float(h)
        chunk["target_kstar"] = chunk[f"target_kstar_h{h}"]
        chunk["target_fd"] = chunk[f"target_diffuse_fraction_h{h}"]
        chunk = chunk.drop(columns=[f"target_kstar_h{h}", f"target_diffuse_fraction_h{h}"])
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True).dropna(subset=["target_kstar", "target_fd"])


def _pinball_loss(pred, target, quantiles):
    errors = target.unsqueeze(1) - pred
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, -1)
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


def _train_pytorch_quantile_moe(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizons: list[int],
    quantiles: list[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is not installed")

    cfg = config["irradiance_model"]
    seed = int(config["project"]["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = _stack_horizons_multi_target(train_df, horizons)
    val = _stack_horizons_multi_target(val_df, horizons)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[IRRADIANCE_FEATURES].astype(float))
    x_val = scaler.transform(val[IRRADIANCE_FEATURES].astype(float))
    yk_train = np.clip(train["target_kstar"].to_numpy(dtype=np.float32), 0, float(cfg["clear_sky_index_max"]))
    yf_train = np.clip(train["target_fd"].to_numpy(dtype=np.float32), 0, 1)
    yk_val = np.clip(val["target_kstar"].to_numpy(dtype=np.float32), 0, float(cfg["clear_sky_index_max"]))
    yf_val = np.clip(val["target_fd"].to_numpy(dtype=np.float32), 0, 1)

    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(yk_train, dtype=torch.float32),
        torch.tensor(yf_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["torch_batch_size"]), shuffle=True)
    model = QuantileMoE(
        input_dim=len(IRRADIANCE_FEATURES),
        hidden_dim=int(cfg["torch_hidden_dim"]),
        n_quantiles=len(quantiles),
        dropout=float(cfg["torch_dropout"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["torch_learning_rate"]), weight_decay=1e-4)
    best_state = None
    best_val = float("inf")

    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    yk_val_t = torch.tensor(yk_val, dtype=torch.float32)
    yf_val_t = torch.tensor(yf_val, dtype=torch.float32)
    for _ in range(int(cfg["torch_epochs"])):
        model.train()
        for xb, ykb, yfb in loader:
            optimizer.zero_grad()
            pred_k, pred_f, _ = model(xb)
            loss = _pinball_loss(pred_k, ykb, quantiles) + _pinball_loss(pred_f, yfb, quantiles)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_k, val_f, _ = model(x_val_t)
            val_loss = float((_pinball_loss(val_k, yk_val_t, quantiles) + _pinball_loss(val_f, yf_val_t, quantiles)).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "scaler": scaler, "validation_pinball_loss": best_val}


def _predict_pytorch_quantiles(payload: dict[str, Any], frame: pd.DataFrame, quantiles: list[float], h: int, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model: QuantileMoE = payload["model"]
    scaler: StandardScaler = payload["scaler"]
    current = frame.copy()
    current["horizon_hours"] = float(h)
    x = scaler.transform(current[IRRADIANCE_FEATURES].astype(float))
    model.eval()
    with torch.no_grad():
        pred_k, pred_f, gates = model(torch.tensor(x, dtype=torch.float32))
    pred_k_np = np.clip(pred_k.numpy(), 0, float(config["irradiance_model"]["clear_sky_index_max"]))
    pred_f_np = np.clip(pred_f.numpy(), 0, 1)
    pred_k_np = np.sort(pred_k_np, axis=1)
    pred_f_np = np.sort(pred_f_np, axis=1)

    k_cols = {i: f"irradiance_kstar_q{int(q * 100):02d}_h{h}" for i, q in enumerate(quantiles)}
    f_cols = {i: f"irradiance_fd_q{int(q * 100):02d}_h{h}" for i, q in enumerate(quantiles)}
    kpred = pd.DataFrame({col: pred_k_np[:, i] for i, col in k_cols.items()}, index=frame.index)
    fpred = pd.DataFrame({col: pred_f_np[:, i] for i, col in f_cols.items()}, index=frame.index)
    gate_names = ["satellite", "ground", "nwp", "static"]
    gate_json = [json.dumps({name: float(value) for name, value in zip(gate_names, row)}) for row in gates.numpy()]
    kpred[f"learned_expert_weights_h{h}"] = gate_json
    return kpred, fpred


def _fit_quantile_family(train: pd.DataFrame, target_prefix: str, quantiles: list[float], seed: int) -> dict[str, Any]:
    models = {}
    for q in quantiles:
        model = _make_quantile_model(q, seed + int(q * 1000))
        model.fit(train[IRRADIANCE_FEATURES], train["target"])
        models[f"q{int(q * 100):02d}"] = model
    return models


def _predict_quantiles(models: dict[str, Any], frame: pd.DataFrame, prefix: str, max_value: float | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for key, model in models.items():
        values = model.predict(frame[IRRADIANCE_FEATURES])
        if max_value is not None:
            values = np.clip(values, 0.0, max_value)
        else:
            values = np.clip(values, 0.0, None)
        out[f"{prefix}_{key}"] = values

    q_cols = [c for c in out.columns if c.startswith(prefix)]
    ordered = np.sort(out[q_cols].to_numpy(), axis=1)
    out[q_cols] = ordered
    return out


def _pvlib_surface_azimuth(config: dict[str, Any]) -> float:
    # Project config follows PVGIS convention where 0 means south-facing.
    return (180.0 + float(config["site"].get("azimuth_deg", 0.0))) % 360.0


def _reconstruct_physical_irradiance(pred: pd.DataFrame, config: dict[str, Any], horizon: int) -> pd.DataFrame:
    cfg = config["irradiance_model"]
    site = config["site"]
    cos_zenith = np.cos(np.deg2rad(pred[f"solar_zenith_h{horizon}"].astype(float).to_numpy()))
    low_sun = cos_zenith < float(cfg["minimum_cos_zenith"])

    for q in cfg["quantiles"]:
        key = f"q{int(float(q) * 100):02d}"
        ghi_col = f"irradiance_ghi_{key}_h{horizon}"
        fd_col = f"irradiance_fd_{key}_h{horizon}"
        pred[ghi_col] = np.where(
            low_sun,
            0.0,
            pred[f"irradiance_kstar_{key}_h{horizon}"] * pred[f"clear_sky_ghi_h{horizon}"],
        )
        pred[fd_col] = pred[fd_col].clip(float(cfg["diffuse_fraction_min"]), float(cfg["diffuse_fraction_max"]))

    p50 = "q50"
    pred[f"irradiance_dhi_p50_h{horizon}"] = np.where(
        low_sun,
        0.0,
        pred[f"irradiance_fd_{p50}_h{horizon}"] * pred[f"irradiance_ghi_{p50}_h{horizon}"],
    )
    pred[f"irradiance_dni_p50_h{horizon}"] = np.where(
        low_sun,
        0.0,
        np.maximum(
            (pred[f"irradiance_ghi_{p50}_h{horizon}"] - pred[f"irradiance_dhi_p50_h{horizon}"])
            / np.maximum(cos_zenith, float(cfg["minimum_cos_zenith"])),
            0.0,
        ),
    )
    valid_times = pd.DatetimeIndex(pd.to_datetime(pred["timestamp"], utc=True) + pd.to_timedelta(horizon, unit="h"))
    dni_extra = pvlib.irradiance.get_extra_radiation(valid_times)
    airmass = pvlib.atmosphere.get_relative_airmass(pred[f"solar_zenith_h{horizon}"].astype(float))
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(site["tilt_deg"]),
        surface_azimuth=_pvlib_surface_azimuth(config),
        solar_zenith=pred[f"solar_zenith_h{horizon}"].astype(float).to_numpy(),
        solar_azimuth=pred[f"solar_azimuth_h{horizon}"].astype(float).to_numpy(),
        dni=pred[f"irradiance_dni_p50_h{horizon}"].astype(float).to_numpy(),
        ghi=pred[f"irradiance_ghi_{p50}_h{horizon}"].astype(float).to_numpy(),
        dhi=pred[f"irradiance_dhi_p50_h{horizon}"].astype(float).to_numpy(),
        dni_extra=np.asarray(dni_extra, dtype=float),
        airmass=np.asarray(airmass, dtype=float),
        albedo=float(cfg["albedo"]),
        model=str(cfg["poa_model"]),
    )
    pred[f"irradiance_poa_p50_h{horizon}"] = np.where(low_sun, 0.0, np.maximum(np.asarray(poa["poa_global"], dtype=float), 0.0))
    pred[f"night_flag_h{horizon}"] = low_sun
    return pred


def _expert_weights(row: pd.Series, horizon: int) -> dict[str, float]:
    satellite_base = max(0.15, 0.55 - 0.08 * max(horizon - 1, 0))
    ground_base = max(0.15, 0.30 - 0.03 * max(horizon - 1, 0))
    nwp_base = min(0.45, 0.08 + 0.10 * horizon)
    static_base = 0.12
    if bool(row.get("missing_nwp", True)):
        static_base += nwp_base * 0.55
        ground_base += nwp_base * 0.25
        satellite_base += nwp_base * 0.20
        nwp_base = 0.0
    if bool(row.get("missing_satellite", False)):
        static_base += satellite_base * 0.45
        ground_base += satellite_base * 0.35
        nwp_base += satellite_base * 0.20
        satellite_base = 0.0
    total = satellite_base + ground_base + nwp_base + static_base
    return {
        "satellite": satellite_base / total,
        "ground": ground_base / total,
        "nwp": nwp_base / total,
        "static": static_base / total,
    }


def _quality_flags(row: pd.Series, horizon: int) -> list[str]:
    flags = ["satellite_irradiance", "weather_context"]
    if bool(row.get("satellite_archive_available", False)):
        flags.append("openmeteo_satellite_archive")
    else:
        flags.append("pvgis_sarah3_satellite")
    if bool(row.get(f"night_flag_h{horizon}", False)):
        flags.append("night_or_low_sun")
    if float(row.get("clear_sky_index", 0.0)) > 1.0:
        flags.append("cloud_enhancement_possible")
    return flags


def train_irradiance_model(features_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Train a horizon-conditioned probabilistic irradiance layer and save demo forecasts."""
    if not config.get("irradiance_model", {}).get("enabled", True):
        LOGGER.info("Irradiance model disabled in config")
        return pd.DataFrame()

    paths = get_paths()
    set_random_seed(int(config["project"]["random_seed"]))
    data, statuses = attach_demo_adapters(features_df)
    data["weather_condition"] = classify_weather(data)
    horizons = [int(h) for h in config["irradiance_model"].get("active_horizons_hours", config["forecast"]["horizons_hours"])]
    quantiles = [float(q) for q in config["irradiance_model"]["quantiles"]]
    train_df, val_df, test_df, split_meta = split_time_series(data, config)
    max_kstar = float(config["irradiance_model"]["clear_sky_index_max"])

    use_torch = bool(config["irradiance_model"].get("use_pytorch_if_available", True)) and torch is not None
    implementation = "pytorch_quantile_moe" if use_torch else (
        "LightGBM quantile" if LGBMRegressor is not None else "sklearn GradientBoostingRegressor quantile"
    )
    if use_torch:
        model_payload = _train_pytorch_quantile_moe(train_df, val_df, horizons, quantiles, config)
        torch.save(
            {
                "state_dict": model_payload["model"].state_dict(),
                "features": IRRADIANCE_FEATURES,
                "quantiles": quantiles,
                "horizons": horizons,
                "validation_pinball_loss": model_payload["validation_pinball_loss"],
                "scaler": model_payload["scaler"],
                "config": {
                    "hidden_dim": int(config["irradiance_model"]["torch_hidden_dim"]),
                    "dropout": float(config["irradiance_model"]["torch_dropout"]),
                },
            },
            paths.models_dir / "irradiance_quantile_moe.pt",
        )
    else:
        train_k = _stack_horizons(train_df, horizons, "kstar")
        train_fd = _stack_horizons(train_df, horizons, "diffuse_fraction")
        kstar_models = _fit_quantile_family(train_k, "kstar", quantiles, int(config["project"]["random_seed"]) + 700)
        fd_models = _fit_quantile_family(train_fd, "diffuse_fraction", quantiles, int(config["project"]["random_seed"]) + 1700)
        model_payload = {
            "kstar_models": kstar_models,
            "fd_models": fd_models,
            "features": IRRADIANCE_FEATURES,
            "quantiles": quantiles,
            "horizons": horizons,
            "targets": ["clear_sky_index", "diffuse_fraction"],
        }
        joblib.dump(model_payload, paths.models_dir / "irradiance_quantile_moe.pkl")

    frames = []
    metrics = []
    for h in horizons:
        current = test_df.copy()
        current["horizon_hours"] = float(h)
        if use_torch:
            kpred, fpred = _predict_pytorch_quantiles(model_payload, current, quantiles, h, config)
            learned_gate_col = f"learned_expert_weights_h{h}"
        else:
            kpred = _predict_quantiles(model_payload["kstar_models"], current, f"irradiance_kstar_h{h}", max_value=max_kstar)
            fpred = _predict_quantiles(model_payload["fd_models"], current, f"irradiance_fd_h{h}", max_value=1.0)
            # Rename from prefix_qXX to irradiance_*_qXX_hN for easier dashboard access.
            kpred = kpred.rename(columns={c: c.replace(f"irradiance_kstar_h{h}_", f"irradiance_kstar_") + f"_h{h}" for c in kpred.columns})
            fpred = fpred.rename(columns={c: c.replace(f"irradiance_fd_h{h}_", f"irradiance_fd_") + f"_h{h}" for c in fpred.columns})
            learned_gate_col = None
        current = pd.concat([current.reset_index(drop=True), kpred.reset_index(drop=True), fpred.reset_index(drop=True)], axis=1)
        current = _reconstruct_physical_irradiance(current, config, h)
        if learned_gate_col and learned_gate_col in current.columns:
            current[f"expert_weights_h{h}"] = current[learned_gate_col]
        else:
            current[f"expert_weights_h{h}"] = current.apply(lambda row: json.dumps(_expert_weights(row, h)), axis=1)
        current[f"quality_flags_h{h}"] = current.apply(lambda row: json.dumps(_quality_flags(row, h)), axis=1)
        current[f"valid_time_h{h}"] = pd.to_datetime(current["timestamp"], utc=True) + pd.to_timedelta(h, unit="h")
        frames.append(
            current[
                [
                    "timestamp",
                    f"valid_time_h{h}",
                    f"target_ghi_h{h}",
                    f"target_kstar_h{h}",
                    f"target_diffuse_fraction_h{h}",
                    f"irradiance_kstar_q50_h{h}",
                    f"irradiance_fd_q50_h{h}",
                    f"irradiance_ghi_q05_h{h}",
                    f"irradiance_ghi_q10_h{h}",
                    f"irradiance_ghi_q25_h{h}",
                    f"irradiance_ghi_q50_h{h}",
                    f"irradiance_ghi_q75_h{h}",
                    f"irradiance_ghi_q90_h{h}",
                    f"irradiance_ghi_q95_h{h}",
                    f"irradiance_dni_p50_h{h}",
                    f"irradiance_dhi_p50_h{h}",
                    f"irradiance_poa_p50_h{h}",
                    f"night_flag_h{h}",
                    f"expert_weights_h{h}",
                    f"quality_flags_h{h}",
                ]
            ]
        )
        daylight = ~current[f"night_flag_h{h}"].astype(bool)
        yt = current.loc[daylight, f"target_ghi_h{h}"]
        yp = current.loc[daylight, f"irradiance_ghi_q50_h{h}"]
        sp = current.loc[daylight, f"clear_sky_index"] * current.loc[daylight, f"clear_sky_ghi_h{h}"]
        mae = mean_absolute_error(yt, yp)
        sp_mae = mean_absolute_error(yt, sp)
        coverage80 = ((yt >= current.loc[daylight, f"irradiance_ghi_q10_h{h}"]) & (yt <= current.loc[daylight, f"irradiance_ghi_q90_h{h}"])).mean()
        coverage90 = ((yt >= current.loc[daylight, f"irradiance_ghi_q05_h{h}"]) & (yt <= current.loc[daylight, f"irradiance_ghi_q95_h{h}"])).mean()
        metrics.append(
            {
                "horizon_h": h,
                "target": "GHI",
                "segment": "daylight",
                "mae_wm2": float(mae),
                "rmse_wm2": float(np.sqrt(mean_squared_error(yt, yp))),
                "smart_persistence_skill": float(1 - mae / sp_mae) if sp_mae > 0 else np.nan,
                "coverage_80": float(coverage80),
                "width_80_wm2": float((current.loc[daylight, f"irradiance_ghi_q90_h{h}"] - current.loc[daylight, f"irradiance_ghi_q10_h{h}"]).mean()),
                "coverage_90": float(coverage90),
                "width_90_wm2": float((current.loc[daylight, f"irradiance_ghi_q95_h{h}"] - current.loc[daylight, f"irradiance_ghi_q05_h{h}"]).mean()),
            }
        )

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="timestamp", how="inner")

    result.to_csv(paths.predictions_dir / "irradiance_test_predictions.csv", index=False)
    try:
        result.to_parquet(paths.predictions_dir / "irradiance_test_predictions.parquet", index=False)
    except Exception as exc:  # pragma: no cover - optional parquet engine
        LOGGER.warning("Could not write irradiance parquet output: %s", exc)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(paths.metrics_dir / "irradiance_metrics.csv", index=False)
    metadata = {
        "model_family": config["irradiance_model"]["model_family"],
        "implementation": implementation,
        "note": "The model uses satellite irradiance from Open-Meteo satellite archive and PVGIS/SARAH-3, then reconstructs physical GHI/DHI/DNI/POA forecasts for operations.",
        "split": split_meta,
        "adapter_status": [asdict(status) for status in statuses],
        "features": IRRADIANCE_FEATURES,
        "quantiles": quantiles,
        "horizons": horizons,
    }
    write_json(paths.models_dir / "irradiance_model_metadata.json", metadata)
    write_json(paths.predictions_dir / "irradiance_forecast_sample.json", _forecast_json_sample(result, config, horizons))
    LOGGER.info("Saved irradiance probabilistic model outputs")
    return result


def _forecast_json_sample(result: pd.DataFrame, config: dict[str, Any], horizons: list[int]) -> dict[str, Any]:
    row = result.iloc[0]
    forecasts = []
    for h in horizons:
        forecasts.append(
            {
                "horizon_minutes": h * 60,
                "valid_time": str(row[f"valid_time_h{h}"]),
                "ghi_p05": float(row[f"irradiance_ghi_q05_h{h}"]),
                "ghi_p10": float(row[f"irradiance_ghi_q10_h{h}"]),
                "ghi_p50": float(row[f"irradiance_ghi_q50_h{h}"]),
                "ghi_p90": float(row[f"irradiance_ghi_q90_h{h}"]),
                "ghi_p95": float(row[f"irradiance_ghi_q95_h{h}"]),
                "dni_p50": float(row[f"irradiance_dni_p50_h{h}"]),
                "dhi_p50": float(row[f"irradiance_dhi_p50_h{h}"]),
                "poa_p50": float(row[f"irradiance_poa_p50_h{h}"]),
                "expert_weights": json.loads(row[f"expert_weights_h{h}"]),
                "quality_flags": json.loads(row[f"quality_flags_h{h}"]),
            }
        )
    return {
        "issue_time": str(row["timestamp"]),
        "site_id": config["site"]["name"],
        "latitude": config["site"]["latitude"],
        "longitude": config["site"]["longitude"],
        "forecasts": forecasts,
    }
