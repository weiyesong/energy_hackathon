# Demo Video Script

Target length: 3 minutes 30 seconds.

## 0:00-0:25 Problem

Solar production is increasingly operational, not just analytical. Grid operators, battery controllers, and energy traders need to know how much PV power is coming, when it will change, how uncertain the forecast is, and whether the forecast is good enough to act on.

The hard part is clouds. A consumer weather forecast is not enough. For intraday solar decisions, the product needs satellite-derived irradiance, physics, machine learning, uncertainty, and clear benchmark evidence.

## 0:25-1:10 Satellite Forecast Cockpit

This is SolarOps for Munich. The first screen is an energy-operations cockpit, not a weather app.

In the sidebar we make two things explicit before we say anything else: the cockpit site and the active data source. In this demo the cockpit site is Munich Centre. The current PV output card is a near-real-time estimate from current Open-Meteo weather and irradiance. It is not plant telemetry.

At the top we show five KPI cards: current PV output for Munich Centre, the next expected peak, expected energy today, forecast risk, and skill versus persistence.

The main chart is for the cockpit site only. It shows the P50 PV forecast line, the calibrated P10-P90 uncertainty band, and the persistence baseline as a dashed line. We also mark cloud-front timing, the operator action point, and the high-risk window.

On the right, the Munich map shows all configured PV sites, irradiance intensity, cloud-risk overlay, and cloud movement direction. The source pill names the active source. If EUMETSAT SSI is unavailable, the UI does not pretend it is active. It clearly shows the fallback source.

Below the map, the operator recommendation card translates the forecast into an operational decision window for the cockpit site.

## 1:10-1:50 Forecast vs Persistence

A solar forecast is only valuable if it beats a simple benchmark.

The KPI strip already shows skill versus persistence, and later in the Model and Data view we open the benchmark evidence in more detail.

SolarOps compares the hybrid satellite-aware model against clear-sky-index persistence on the held-out test set. The benchmark panel reports offline hindcast RMSE and forecast skill.

The model is not predicting sunlight from scratch. It starts from solar geometry, clear-sky physics, and atmospheric correction, then learns the residual using satellite, cloud, weather, and source-disagreement features.

Positive skill means the model beats persistence for that horizon. If the result is not positive, the product does not claim it beats persistence.

## 1:50-2:25 Operator Recommendation

The forecast is converted into an operational suggestion.

Here SolarOps identifies the expected event, its operational impact, a recommended action, confidence, and the valid time window. This card is designed to be narratable in one glance during the demo.

Examples include monitoring a high-uncertainty period, preserving battery, charging before an expected PV drop, shifting flexible load into a high-solar window, or reducing expected feed-in commitment.

These are operational recommendations, not guaranteed financial outcomes.

## 2:25-2:55 Site Intelligence

SolarOps also compares multiple Munich sites.

In the Site Intelligence view we move from one cockpit site to the full Munich portfolio. Each site is scored using expected daily energy, peak output, uncertainty width, cloud risk, forecast volatility, and data quality.

The site table and map make good-vs-bad site behavior visible. The comparison chart shows relative energy opportunity. The expandable explanation shows why a site received its grade, so the ranking is transparent rather than a black box.

## 2:55-3:20 Hybrid Architecture and Uncertainty

The Model and Data view shows the full pipeline:

Satellite SSI and cloud information → solar geometry and clear-sky physics → atmospheric correction → ML residual forecast → uncertainty calibration → PV power → operator action.

It also shows live source status for EUMETSAT, NASA POWER, Open-Meteo, and the physical solar model.

This view is also where we explain status honesty: Live, Cached, Fallback, Unavailable, and Manual download required. We can point directly to the source cards and show that fallback never masquerades as live satellite input.

Uncertainty is generated with LightGBM quantile models and calibrated on the validation set only. The P10-P90 interval is carried through the PV model into operational decisions.

## 3:20-3:30 Closing Statement

SolarOps turns satellite-derived irradiance into operational solar decisions: how much power, when it changes, how certain the forecast is, and what the operator should do next.
