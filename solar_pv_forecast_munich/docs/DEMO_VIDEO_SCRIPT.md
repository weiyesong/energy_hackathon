# Demo Video Script

Target length: 3 minutes 30 seconds.

## 0:00-0:25 Problem

Solar production is increasingly operational, not just analytical. Grid operators, battery controllers, and energy traders need to know how much PV power is coming, when it will change, how uncertain the forecast is, and whether the forecast is good enough to act on.

The hard part is clouds. A consumer weather forecast is not enough. For intraday solar decisions, the product needs satellite-derived irradiance, physics, machine learning, uncertainty, and clear benchmark evidence.

## 0:25-1:10 Satellite Forecast Cockpit

This is SolarOps for Munich. The first screen is an energy-operations cockpit, not a weather app.

At the top we show current PV output, the next expected peak, expected energy today, forecast risk, and skill versus persistence.

The main chart shows the P50 PV forecast, the P10-P90 uncertainty band, and a persistence baseline. Cloud-front and operator-action markers make the forecast usable in a control-room workflow.

The Munich map shows the configured PV sites, irradiance intensity, cloud-risk overlay, and active source. If EUMETSAT SSI is unavailable, the UI does not pretend it is active. It clearly shows the fallback source.

## 1:10-1:50 Forecast vs Persistence

A solar forecast is only valuable if it beats a simple benchmark.

SolarOps compares the hybrid satellite-aware model against clear-sky-index persistence on the held-out test set. The benchmark panel reports offline hindcast RMSE and forecast skill.

The model is not predicting sunlight from scratch. It starts from solar geometry, clear-sky physics, and atmospheric correction, then learns the residual using satellite, cloud, weather, and source-disagreement features.

Positive skill means the model beats persistence for that horizon. If the result is not positive, the product does not claim it beats persistence.

## 1:50-2:25 Operator Recommendation

The forecast is converted into an operational suggestion.

Here SolarOps identifies the expected event, its operational impact, a recommended action, confidence, and the valid time window.

Examples include monitoring a high-uncertainty period, preserving battery, charging before an expected PV drop, shifting flexible load into a high-solar window, or reducing expected feed-in commitment.

These are operational recommendations, not guaranteed financial outcomes.

## 2:25-2:55 Site Intelligence

SolarOps also compares multiple Munich sites.

Each site is scored using expected daily energy, peak output, uncertainty width, cloud risk, forecast volatility, and data quality.

The site table and map make good-vs-bad site behavior visible. The expandable explanation shows why a site received its grade, so the ranking is transparent rather than a black box.

## 2:55-3:20 Hybrid Architecture and Uncertainty

The Model and Data view shows the full pipeline:

Satellite SSI and cloud information → solar geometry and clear-sky physics → atmospheric correction → ML residual forecast → uncertainty calibration → PV power → operator action.

It also shows live source status for EUMETSAT, NASA POWER, Open-Meteo, and the physical solar model.

Uncertainty is generated with LightGBM quantile models and calibrated on the validation set only. The P10-P90 interval is carried through the PV model into operational decisions.

## 3:20-3:30 Closing Statement

SolarOps turns satellite-derived irradiance into operational solar decisions: how much power, when it changes, how certain the forecast is, and what the operator should do next.
