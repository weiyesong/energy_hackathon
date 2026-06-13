# SolarCast Ops

Satellite-driven solar generation nowcasting and operational decision support for intraday trading and grid balancing.

## Run

```bash
cd solarcast_ops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run_pipeline.py --step all
python3 -m streamlit run app/dashboard.py
```

Useful commands:

```bash
python3 run_pipeline.py --step download
python3 run_pipeline.py --step preprocess
python3 run_pipeline.py --step features
python3 run_pipeline.py --step baselines
python3 run_pipeline.py --step train
python3 run_pipeline.py --step irradiance
python3 run_pipeline.py --step evaluate
python3 run_pipeline.py --step demo
python3 run_pipeline.py --step asof
python3 -m pytest -q
```

## Data

SolarCast Ops uses satellite irradiance data for Munich and turns it into plant-level operational forecasts.

Primary satellite inputs:

- Open-Meteo satellite radiation archive
- PVGIS/SARAH-3 satellite-derived irradiance

Core fields:

- `global_irradiance_wm2`
- `direct_irradiance_wm2`
- `diffuse_irradiance_wm2`
- `pv_power_mw`
- `air_temperature_c`
- `wind_speed_ms`
- `solar_elevation_deg`

The download step aligns satellite archive radiation to the hourly PV target frame and records source coverage in `data/processed/data_metadata.json`.

## Forecasting

The pipeline trains 1h, 2h, and 3h ahead PV power forecasts.

Models and baselines:

- Ordinary persistence
- Clear-sky persistence
- History-only ML
- Satellite-informed ML
- Probabilistic irradiance model for GHI/DHI/DNI/POA

Satellite features include irradiance components, clear-sky index, diffuse fraction, beam fraction, irradiance lags, rolling variability, cloud-trend signals, temperature, wind, and solar geometry.

Rolling features use past-only windows, and the strict as-of backtest blocks future labels at forecast issue time.

## Operational Output

The dashboard opens directly to the operator command center:

- Satellite coverage and current satellite irradiance
- Selected solar portfolio capacity in MWp
- Next 24h solar generation in MWh/GWh
- Munich daily electricity demand benchmark in GWh/day
- Forecast skill versus persistence
- P10/P50/P90 forecast interval
- BUY, SELL, or HOLD action
- Expected imbalance and avoided cost
- Ramp risk and reserve/flex recommendation
- Good-vs-bad site ranking across the portfolio
- Strict as-of validation evidence

## Metrics

Current generated outputs report:

- 100% satellite archive coverage for the historical training window
- Positive daylight forecast skill against ordinary persistence
- Positive daylight forecast skill against clear-sky persistence
- Probabilistic irradiance forecasts with coverage and interval-width metrics

Metrics are written to:

- `reports/metrics/metrics.csv`
- `reports/metrics/irradiance_metrics.csv`
- `reports/metrics/asof_backtest_metrics.csv`

Predictions are written to:

- `reports/predictions/test_predictions_with_uncertainty.csv`
- `reports/predictions/irradiance_test_predictions.csv`
- `reports/predictions/asof_backtest_predictions.csv`

## Project Layout

```text
solarcast_ops/
├── app/
├── config/
├── data/
├── models/
├── reports/
├── src/
├── tests/
├── run_pipeline.py
└── README.md
```
