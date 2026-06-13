# SolarOps Munich

SolarOps converts satellite-derived irradiance and cloud information into uncertainty-aware PV generation forecasts and operational recommendations for intraday energy decisions.

This project is an end-to-end solar irradiance and photovoltaic power forecasting system for Munich, Germany. The architecture is satellite-first: satellite-derived irradiance is the preferred input, weather APIs provide fallback and forecast context, and every forecast should be benchmarked against persistence before being used for operational decisions.

## Product Goal

1. Ingest satellite-derived irradiance and cloud information.
2. Compute solar geometry and clear-sky irradiance.
3. Build a physical irradiance baseline from cloud and atmospheric transmittance.
4. Benchmark forecasts against persistence and clear-sky persistence.
5. Convert irradiance forecasts into operational PV output forecasts.
6. Estimate uncertainty for intraday energy decisions.
7. Compare good and bad Munich-area site behavior.
8. Produce operational recommendations for intraday trading and grid decisions.

## Target Location

- Name: Munich, Germany
- Latitude: `48.137`
- Longitude: `11.575`
- Timezone: `Europe/Berlin`
- Altitude: `520 m`

## Configured Sites

- Munich Centre
- Munich North
- Munich East
- Munich South
- Munich West

These sites support good-vs-bad site comparison across the Munich region.

## Forecast Horizons

Operational horizons:

- 60 minutes
- 3 hours
- 6 hours
- 12 hours
- 24 hours

High-frequency diagnostic horizons:

- 15 minutes
- 30 minutes

## Data Source Positioning

- `eumetsat_ssi`: primary operational satellite-derived irradiance source
- `nasa_power`: satellite/model-derived historical solar baseline
- `openmeteo`: weather forecast and fallback irradiance source

NASA POWER is not treated as a raw satellite feed. Open-Meteo remains useful for weather context, forecast weather variables, and fallback irradiance.

## Canonical Data Schema

All data adapters should map source-specific fields into the canonical schema in `src/core/schema.py`.

Canonical source metadata is tracked in:

- `data/processed/data_source_registry.json`

Missing source fields must remain missing as `NaN`; adapters should not invent unavailable observations.

## Project Structure

```text
solar_pv_forecast_munich/
    README.md
    requirements.txt
    config.yaml

    data/
        raw/
        processed/
        manual/
            era5/
            cams/
            eumetsat/
            dwd/
            entsoe/

    src/
        core/
        data_download/
        preprocessing/
        physics/
        models/
        evaluation/
        visualization/

    outputs/
        models/
        forecasts/
        figures/
```

## Configuration

All product positioning, data-source priority, site definitions, location settings, PV system assumptions, train/validation dates, and forecast horizons live in `config.yaml`.

## Execution Order

1. Register available data sources and inspect satellite-first source priority.
2. Ingest EUMETSAT SSI when manual operational satellite data is available.
3. Use NASA POWER as a satellite/model-derived historical solar baseline when added.
4. Use Open-Meteo as weather forecast and fallback irradiance source.
5. Align timestamps and normalize units into the canonical schema.
6. Compute solar geometry and clear-sky irradiance with `pvlib`.
7. Build atmospheric transmittance and physical irradiance baselines.
8. Benchmark against persistence and clear-sky persistence.
9. Train LightGBM residual models for the configured forecast horizons.
10. Train quantile models for uncertainty estimation.
11. Generate multi-horizon irradiance and PV power forecasts.
12. Compare good-vs-bad site outcomes.
13. Visualize forecasts, uncertainty intervals, and intraday operational recommendations.

## Open-Meteo Download

From the project directory:

```bash
python src/data_download/openmeteo_download.py --config config.yaml
```

This writes:

- `data/raw/openmeteo_historical_munich.csv`
- `data/raw/openmeteo_forecast_munich.csv`

## NASA POWER Download

NASA POWER is used as a satellite/model-derived historical solar baseline. It is not the primary short-term satellite nowcasting source.

From the project directory:

```bash
python src/data_download/nasa_power_download.py --config config.yaml
```

Use `--force` to redownload files even when valid cached site CSV files already exist:

```bash
python src/data_download/nasa_power_download.py --config config.yaml --force
```

This writes one standardized file per configured site:

- `data/raw/nasa_power_<site_id>.csv`

It also writes the combined standardized dataset:

- `data/processed/nasa_power_all_sites.parquet`

The data source registry is updated at:

- `data/processed/data_source_registry.json`

## EUMETSAT SSI Optional Adapter

EUMETSAT surface solar irradiance is the preferred operational satellite-derived irradiance source for SolarOps. The system still works when EUMETSAT credentials or files are unavailable.

From the project directory:

```bash
python src/data_download/eumetsat_download.py --config config.yaml
```

Automatic mode only attempts product discovery when credentials are available through environment variables or an existing local EUMETSAT configuration. Credentials are never hard-coded.

Manual-file mode reads NetCDF or GRIB files from:

- `data/manual/eumetsat/`

If EUMETSAT is unavailable, the script prints manual enablement instructions and updates the registry as unavailable with manual action required. It does not create synthetic satellite data.

When manual files are available, this writes:

- `data/processed/eumetsat_ssi_all_sites.parquet`

## Solar Features

From the project directory:

```bash
python src/physics/clear_sky.py --config config.yaml --input data/raw/openmeteo_historical_munich.csv --output data/processed/openmeteo_with_solar_features.parquet
```

This computes solar geometry and clear-sky irradiance.

## Physical Baseline

From the project directory:

```bash
python src/physics/atmospheric_transmittance.py
```

This writes:

- `data/processed/openmeteo_with_physical_baseline.parquet`

## Supervised Dataset And Persistence Benchmarks

From the project directory:

```bash
python src/preprocessing/feature_engineering.py --config config.yaml --input data/processed/fused_solar_dataset.parquet --summary data/processed/fusion_summary.json
```

This writes:

- `data/processed/supervised_multi_horizon.parquet`
- `data/processed/train_dataset.parquet`
- `data/processed/validation_dataset.parquet`
- `data/processed/test_dataset.parquet`
- `outputs/forecasts/persistence_baseline_predictions.csv`

Target-time Open-Meteo weather fields are named with the `_forecast_proxy` suffix. They are hindcast-development approximations from historical data, not live operational forecasts. Metrics built with these proxy columns must not be presented as live operational performance.

## Deterministic Hybrid Residual Model

From the project directory:

```bash
python src/models/train_hybrid_lightgbm.py --output-root .
```

The model does not predict sunlight from scratch. It learns the residual:

```text
GHI_pred = GHI_phys_target + residual_model(features)
```

It writes:

- `outputs/models/hybrid_ghi_horizon_<h>.pkl`
- `outputs/forecasts/test_hybrid_predictions.csv`
- `outputs/metrics/benchmark_metrics.csv`
- `outputs/metrics/forecast_skill_by_horizon.csv`
- `outputs/metrics/satellite_ablation.csv`
- `outputs/metrics/feature_importance_horizon_<h>.csv`

This is an offline hindcast evaluation. It uses historical target-time weather proxy fields where live forecast products would be used operationally. Do not describe the hindcast metrics as live operational forecast performance.

The training script prefers LightGBM. If the local LightGBM runtime is unavailable, it falls back to sklearn histogram gradient boosting and records the backend in the saved model payloads and feature-importance files.

## Probabilistic GHI Forecasts

From the project directory:

```bash
python src/models/train_quantile_forecast.py --output-root .
```

This trains P10, P50, and P90 GHI quantile models by horizon, calibrates the P10-P90 interval on the validation split only, and evaluates uncertainty on the test split.

It writes:

- `outputs/models/quantile_ghi_horizon_<h>_p10.pkl`
- `outputs/models/quantile_ghi_horizon_<h>_p50.pkl`
- `outputs/models/quantile_ghi_horizon_<h>_p90.pkl`
- `outputs/forecasts/test_probabilistic_predictions.csv`
- `outputs/metrics/uncertainty_metrics.csv`
- `outputs/metrics/coverage_by_cloud_regime.csv`

The calibrated P10-P90 interval targets roughly 80% coverage. If observed coverage is above 80%, the interval is conservative; if it falls below 80%, the uncertainty layer is under-covering and should not be used for risk decisions without further calibration.

## Operational Product Layer

From the project directory:

```bash
python src/operations/decision_engine.py --config config.yaml --input outputs/forecasts/test_probabilistic_predictions.csv
```

This converts probabilistic GHI forecasts to estimated DNI/DHI, POA irradiance, PV power, limiting-factor labels, operator actions, and site ranking.

It writes:

- `outputs/forecasts/operational_forecast.csv`
- `outputs/operations/operator_actions.json`
- `outputs/operations/site_ranking.csv`
- `outputs/operations/site_ranking_summary.json`

Operator actions are operational suggestions only. They are not guaranteed financial outcomes.

## Live Forecast Orchestrator

From the project directory:

```bash
python src/pipeline/run_live_forecast.py --config config.yaml
```

The live orchestrator:

1. inspects the data-source registry;
2. fetches the current Open-Meteo forecast;
3. uses EUMETSAT SSI only when it is genuinely available;
4. keeps NASA POWER as historical and fallback context only;
5. computes target-time solar geometry and clear-sky irradiance;
6. constructs inference rows using the same saved model encoders used in training;
7. loads horizon-specific quantile models;
8. predicts GHI P10/P50/P90;
9. converts GHI to DNI/DHI, POA, and PV power;
10. writes UI-ready forecast, actions, source status, and site ranking files.

It writes:

- `outputs/live/forecast.csv`
- `outputs/live/operator_actions.json`
- `outputs/live/site_ranking.csv`
- `outputs/live/site_ranking_summary.json`
- `outputs/live/source_status.json`
- `outputs/forecasts/live_operational_forecast.csv`

Every live forecast row includes source transparency fields:

- `primary_satellite_source`
- `weather_forecast_source`
- `satellite_data_available`
- `fallback_active`
- `data_freshness_minutes`
- `data_quality_level`

When EUMETSAT is unavailable, the system must not describe fallback data as live EUMETSAT data.

15-minute and 30-minute operational forecasts are shown only when both high-frequency satellite/observation input and matching trained models are available. Otherwise they are marked:

- `Not operationally available`
- `High-frequency satellite input required`

## Demo Snapshot

For reproducible videos and demos that do not depend on live internet:

```bash
python src/pipeline/prepare_demo_snapshot.py --config config.yaml
```

This snapshot is produced from cached real downloaded or previously processed data. UI surfaces must display:

```text
Demo Snapshot
```

Do not describe cached demo data as live.

It writes:

- `outputs/demo/demo_snapshot.json`
- `outputs/demo/demo_forecast.csv`
- `outputs/demo/demo_site_ranking.csv`
- `outputs/demo/demo_operator_actions.json`
- `outputs/demo/demo_source_status.json`

## Backend API

Start the local product API with:

```bash
uvicorn src.api.app:app --reload
```

Endpoints:

- `GET /api/health`
- `GET /api/overview`
- `GET /api/forecast`
- `GET /api/forecast/{site_id}`
- `GET /api/benchmark`
- `GET /api/sites`
- `GET /api/sites/{site_id}`
- `GET /api/actions`
- `GET /api/drivers`
- `GET /api/data-sources`

The API loads live outputs from `outputs/live/` when available. If live outputs are unavailable, it falls back to the demo snapshot and returns `demo_mode = true`.

Benchmark responses are labelled `offline hindcast` unless a genuine operational evaluation dataset is connected.

## One-Command Product Runs

Full product mode:

```bash
scripts/run_product.sh
```

Demo video mode:

```bash
scripts/run_demo.sh
```

`run_demo.sh` forces the backend into demo mode with `SOLAROPS_DATA_MODE=demo`, verifies the demo snapshot files, starts FastAPI, starts the frontend, and prints the frontend URL.

Final challenge evidence and validation docs:

- `docs/CHALLENGE_REQUIREMENTS.md`
- `docs/DEMO_VIDEO_SCRIPT.md`
- `docs/FINAL_VALIDATION.md`

## Current Implementation

- Project structure and config
- Satellite-first product positioning
- Canonical data schema and data-source registry
- Open-Meteo, NASA POWER, and optional EUMETSAT adapters
- Source fusion
- Solar geometry, clear-sky irradiance, and physical irradiance baseline
- Leakage-safe supervised dataset and persistence baselines
- Deterministic hybrid residual model
- Probabilistic calibrated GHI forecasts
- PV conversion, site ranking, and decision support
- Live/demo forecast orchestrators
- Lightweight FastAPI backend
