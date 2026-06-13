# Challenge Requirement Validation

SolarOps converts satellite-derived irradiance and cloud information into uncertainty-aware PV generation forecasts and operational recommendations for intraday energy decisions.

| Requirement | Status | Evidence |
|---|---|---|
| Satellite-derived irradiance | Implemented | `src/data_download/eumetsat_download.py`, `src/preprocessing/eumetsat_ingest.py`, `src/data_download/nasa_power_download.py`, `data/processed/data_source_registry.json`, `outputs/live/source_status.json` |
| Forecast vs persistence | Implemented | `src/models/persistence_baseline.py`, `src/evaluation/evaluate_benchmarks.py`, `outputs/metrics/benchmark_metrics.csv`, `outputs/metrics/forecast_skill_by_horizon.csv` |
| Nowcasting / intraday relevance | Implemented | `src/pipeline/run_live_forecast.py`, `outputs/live/forecast.csv`, `frontend/src/App.tsx` |
| Operational PV output | Implemented | `src/physics/irradiance_decomposition.py`, `src/physics/poa_model.py`, `src/physics/pv_power_model.py`, `outputs/live/forecast.csv`, `outputs/forecasts/operational_forecast.csv` |
| Grid / battery / trading decision support | Implemented | `src/operations/decision_engine.py`, `outputs/live/operator_actions.json`, `outputs/operations/operator_actions.json`, `frontend/src/App.tsx` |
| Good-vs-bad site comparison | Implemented | `config.yaml`, `src/operations/site_ranking.py`, `outputs/live/site_ranking.csv`, `outputs/operations/site_ranking.csv`, frontend Site Intelligence view |
| Uncertainty | Implemented | `src/models/train_quantile_forecast.py`, `src/models/conformal_calibration.py`, `src/evaluation/evaluate_uncertainty.py`, `outputs/forecasts/test_probabilistic_predictions.csv`, `outputs/metrics/uncertainty_metrics.csv`, `outputs/live/forecast.csv` |
| Data-source transparency | Implemented | `src/core/data_registry.py`, `src/pipeline/source_health.py`, `outputs/live/source_status.json`, `outputs/demo/demo_source_status.json`, `src/api/data_service.py`, frontend source cards |

## Evidence Notes

- `outputs/live/forecast.csv` contains satellite source, weather source, source freshness, quality level, forecast values, and operational status.
- `outputs/live/source_status.json` records source availability and horizon support.
- `outputs/demo/demo_snapshot.json` keeps demo runs reproducible.
