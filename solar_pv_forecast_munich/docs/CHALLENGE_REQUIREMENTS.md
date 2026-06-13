# Challenge Requirement Validation

SolarOps product statement:

> SolarOps converts satellite-derived irradiance and cloud information into uncertainty-aware PV generation forecasts and operational recommendations for intraday energy decisions.

| Requirement | Status | Evidence file | Limitation |
|---|---|---|---|
| Satellite-derived irradiance | Partially implemented | `src/data_download/eumetsat_download.py`, `src/preprocessing/eumetsat_ingest.py`, `src/data_download/nasa_power_download.py`, `data/processed/data_source_registry.json`, `outputs/live/source_status.json` | EUMETSAT SSI is implemented as optional automatic/manual ingestion, but no live EUMETSAT credentials or manual SSI file are currently available. NASA POWER is used only as satellite/model-derived historical baseline. |
| Forecast vs persistence | Implemented | `src/models/persistence_baseline.py`, `src/evaluation/evaluate_benchmarks.py`, `outputs/metrics/benchmark_metrics.csv`, `outputs/metrics/forecast_skill_by_horizon.csv` | Metrics are offline hindcast metrics on the held-out test split, not live operational performance. |
| Nowcasting / intraday relevance | Partially implemented | `src/pipeline/run_live_forecast.py`, `outputs/live/forecast.csv`, `frontend/src/App.tsx` | 60-minute and longer operational horizons are active. 15-minute and 30-minute horizons are explicitly disabled until high-frequency satellite or observation inputs and trained models exist. |
| Operational PV output | Implemented | `src/physics/irradiance_decomposition.py`, `src/physics/poa_model.py`, `src/physics/pv_power_model.py`, `outputs/live/forecast.csv`, `outputs/forecasts/operational_forecast.csv` | DNI/DHI are estimated from GHI quantiles rather than directly predicted. |
| Grid / battery / trading decision support | Implemented | `src/operations/decision_engine.py`, `outputs/live/operator_actions.json`, `outputs/operations/operator_actions.json`, `frontend/src/App.tsx` | Recommendations are structured operational suggestions, not guaranteed financial outcomes or market bids. |
| Good-vs-bad site comparison | Implemented | `config.yaml`, `src/operations/site_ranking.py`, `outputs/live/site_ranking.csv`, `outputs/operations/site_ranking.csv`, frontend Site Intelligence view | Site score is transparent and heuristic; it is intended for demonstration and operations triage. |
| Uncertainty | Implemented | `src/models/train_quantile_forecast.py`, `src/models/conformal_calibration.py`, `src/evaluation/evaluate_uncertainty.py`, `outputs/forecasts/test_probabilistic_predictions.csv`, `outputs/metrics/uncertainty_metrics.csv`, `outputs/live/forecast.csv` | P10-P90 calibration uses validation data. Live uncertainty quality depends on live source quality and current model calibration. |
| Data-source transparency | Implemented | `src/core/data_registry.py`, `src/pipeline/source_health.py`, `outputs/live/source_status.json`, `outputs/demo/demo_source_status.json`, `src/api/data_service.py`, frontend source cards | Fallback sources are disclosed. EUMETSAT is shown as unavailable/manual-required unless real credentials or files are present. |

## Evidence Notes

- `outputs/live/forecast.csv` contains `primary_satellite_source`, `weather_forecast_source`, `satellite_data_available`, `fallback_active`, `data_freshness_minutes`, and `data_quality_level`.
- `outputs/live/source_status.json` records high-frequency horizon support and fallback state.
- `outputs/demo/demo_snapshot.json` and frontend `Demo Snapshot` badge prevent cached demo data from being presented as live.
