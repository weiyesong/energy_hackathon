# Demo Video Script: SolarCast Ops

This is **SolarCast Ops**, a satellite-driven operating system for short-term solar decisions developed for the Invertix OpenTrack challenge. 

Instead of walking through point by point chronologically, this demo first highlights the core capabilities of the product, and then directly addresses how it solves the specific requirements of the challenge.

---

## Part 1: Key Product Capabilities (Important Features)

### 1. The SolarOps Cockpit (Decision-First UI)
The first thing the operator sees is not just a raw weather map—it is the **decision state**. 
The system instantly converts complex forecasts into the language operators actually use: expected imbalance, no-action exposure, avoided cost, reserve/flexibility needs, and the immediate action to take. *Note: As real high-frequency PV data is currently not public, the power output is synthetically generated using physical models to emulate operational dynamics, but the system is fully plug-and-play to ingest real grid data.*

### 2. Multi-Horizon Intelligence & End-to-End Automation
SolarCast operates as a fully automated后台调度器 (Orchestrator). The "Model & Data" view reveals the 7-step engine under the hood: taking raw Satellite SSI -> clear-sky physics -> atmospheric correction -> ML residual forecast -> uncertainty calibration -> PV power -> and finally, operator action. Operators can monitor data source health (fallback, active, manual) transparently. 

### 3. Site Intelligence & Prioritization (Work Queue)
Instead of overwhelming the operator with data from hundreds of solar farms, the product ranks the portfolio from most critical to least critical. By combining expected energy, confidence, data quality, cloud risk, and volatility, it assigns a concrete grade (e.g., A, B, C, D) to every site, turning satellite surveillance into a prioritized daily work queue.

---

## Part 2: How We Addressed the Core Challenge Requirements

### 1. Challenge: Weather vs. Physical vs. Hybrid Models (Forecast the curve & Beat Persistence)
**How we solved it:** We built a hybrid modeling pipeline that uses satellite data combined with physical clear-sky models and machine learning. But we didn't just draw a curve; we built an evaluation dashboard that proves it beats the baseline. The system tracks MAE, RMSE, and forecast skill against both "ordinary persistence" (tomorrow looks like today) and "clear-sky persistence". Positive skill displayed directly in the command center demonstrates that the satellite-informed model actively reduces operational error.

### 2. Challenge: A nowcast for intraday trading or grid balancing decisions
**How we solved it:** The decision engine explicitly translates MW deviations into exact trading and balancing instructions in the Forecast Cockpit:
*   **Intraday Trading:** If the forecast indicates a shortfall below schedule, it generates a `BUY` action. If above schedule, a `SELL` action. If the deviation is minimal, it outputs `HOLD`.
*   **Grid Balancing:** The tool automatically calculates ramp direction, ramp magnitude, ramp risk, and reserve or downward-flexibility metrics. The operator never has to manually interpret irradiance; the system dictates the next move.

### 3. Challenge: Reveal good vs bad sites
**How we solved it:** This requirement is answered by our **Site Intelligence** dashboard. Sites are explicitly assessed:
*   **Good sites** are stable, close to schedule, and have low cloud risk.
*   **Watch sites** have emerging uncertainty or elevated ramp risk.
*   **Bad sites** (Grade D) are those where irradiance deviation is severe enough that the operator must intervene.
This allows dispatchers to know exactly which sites are safe and which ones require immediate attention.

### 4. Challenge: Any tool that takes raw satellite feeds and turns them into an operational output
**How we solved it:** We didn't just write a single notebook; we engineered an **end-to-end software product** (FastAPI backend + React frontend). The pipeline seamlessly auto-ingests EUMETSAT/CAMS satellite signals, converts irradiance to cloud-state metrics, feeds them through the probabilistic PV forecast model, and uses a built-in Decision Engine to print out actionable limit orders, trade sizes, and risk warnings directly to the user's screen.
