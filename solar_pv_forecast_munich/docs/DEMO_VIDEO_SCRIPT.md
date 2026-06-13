# Demo Video Script

Target length: 3 minutes.

## 0:00-0:25 What I Built

This is SolarCast Ops, a satellite-driven operating system for short-term solar decisions.

for the Invertix OpenTrack challenge.

I will walk through those outputs directly.

## 0:25-0:55 Satellite Data to Forecast

The first thing the operator sees is not a weather map. It is the decision state.

At the top, the dashboard shows the recommended action, trade size, forecast skill versus persistence, skill versus clear-sky persistence, satellite data coverage, and the number of sites needing action.

The satellite feed provides GHI, direct irradiance, diffuse irradiance, cloud-state signals, wind, temperature, and solar geometry. 我通过复杂的物理建模 combines those with recent PV behavior to forecast power over the next operational horizons.值得注意的是，真实的PV数据目前是不公开的，所以PV数据（这里写数据怎么来的），但是只要接入真实的grid数据，这套系统就能立刻大展身手。

So the raw question, "can satellite data become a usable output?", is answered immediately: yes, it becomes a PV forecast with uncertainty and an operator action.

## 0:55-1:30 Beating Persistence

The first worth-exploring requirement is a generation forecast that beats persistence.

Here, persistence means the simple baseline: assume the future looks like now. I tested against that baseline and also against clear-sky persistence.

In the command center, the skill numbers are shown directly. Positive skill means the satellite-informed model reduces error compared with persistence.

The Model Evaluation view gives the evidence behind that claim: MAE, RMSE, normalized error, and forecast skill by horizon and sky condition. This is important because the model is not just drawing a nice curve; it is benchmarked against the simplest thing an operator could do.

## 1:30-2:05 Nowcast to Trading and Grid Balancing

The second requirement is a nowcast that helps intraday trading or grid-balancing decisions.

SolarCast converts the forecast into the language operators use: expected imbalance, no-action exposure, avoided cost, reserve or flexibility need, and the action to prepare.

If the forecast is below schedule, the system prepares a BUY action. If generation is above schedule, it prepares a SELL action. If the deviation is too small, it holds.

For grid balancing, the same forecast becomes ramp direction, ramp magnitude, ramp risk, and reserve or downward-flexibility recommendations. The operator does not need to interpret satellite irradiance manually; the system turns it into the next operational move.

## 2:05-2:35 Good vs Bad Sites

The third requirement is to reveal good versus bad sites.

This table ranks the portfolio using the satellite-informed signal. Each site is scored by schedule deviation, ramp risk, forecasted power, trade action, and avoided cost.

Good sites are stable or close to schedule. Watch sites have meaningful uncertainty or emerging ramp risk. Bad or action-needed sites are the ones where the satellite forecast says the operator should intervene.

This turns satellite coverage into a prioritized work queue: which site is safe, which site needs monitoring, and which site needs action first.

## 2:35-2:55 Operational Output

The last requirement is any tool that takes satellite feeds and turns them into operational output.

SolarCast does that end to end. Satellite irradiance becomes clear-sky index and cloud signals. Those become probabilistic GHI, DNI, DHI, and POA forecasts. Those become PV power forecasts. Then the decision engine converts the power forecast into trading, balancing, reserve, and site-ranking outputs.

The strict as-of backtest shows the forecast uses information available at issue time, and the dashboard keeps the benchmark evidence visible.

## 2:55-3:00 Close

SolarCast Ops turns satellite data into decisions operators can act on: forecast power, quantify uncertainty, beat persistence, rank sites, and prepare the next market or grid action.
