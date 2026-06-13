# Final Validation

Run all commands from the project root:

```bash
cd solar_pv_forecast_munich
```

## Environment

### LightGBM Runtime

Command:

```bash
python3 - <<'PY'
import lightgbm as lgb
print("lightgbm import ok", lgb.__version__)
PY
```

Expected output:

```text
lightgbm import ok 4.6.0
```

Note: `libomp.dylib` is linked at `/opt/homebrew/opt/libomp/lib/libomp.dylib`.

## One-Command Scripts

### Product Mode

Command:

```bash
scripts/run_product.sh
```

Expected output:

```text
Python dependencies OK, including LightGBM.
Saved live UI forecast: .../outputs/live/forecast.csv
SolarOps product is running.
Frontend URL: http://127.0.0.1:5173
```

### Demo Mode

Command:

```bash
scripts/run_demo.sh
```

Expected output:

```text
Saved demo forecast: .../outputs/demo/demo_forecast.csv
SolarOps demo is running.
Frontend URL: http://127.0.0.1:5173
The UI should display: Demo Snapshot
```

## Automated Tests

Command:

```bash
python3 -m pytest -q
```

Expected output:

```text
35 passed
```

## Frontend Build

Command:

```bash
npm --prefix frontend run build
```

Expected output:

```text
tsc -b && vite build
✓ built
```

## API Schema Consistency

Command:

```bash
python3 - <<'PY'
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)
for path in [
    "/api/health",
    "/api/overview",
    "/api/forecast",
    "/api/forecast/munich_centre",
    "/api/benchmark",
    "/api/sites",
    "/api/sites/munich_centre",
    "/api/actions",
    "/api/drivers",
    "/api/data-sources",
]:
    response = client.get(path)
    print(path, response.status_code)
    assert response.status_code == 200
PY
```

Expected output:

```text
/api/health 200
/api/overview 200
/api/forecast 200
/api/forecast/munich_centre 200
/api/benchmark 200
/api/sites 200
/api/sites/munich_centre 200
/api/actions 200
/api/drivers 200
/api/data-sources 200
```

## Product Integrity Checks

### No Target Leakage

Command:

```bash
python3 -m pytest tests/test_feature_engineering.py tests/test_hybrid_model.py -q
```

Expected output:

```text
passed
```

Evidence:

- `src/preprocessing/feature_engineering.py`
- `src/models/train_hybrid_lightgbm.py`
- `tests/test_feature_engineering.py`
- `tests/test_hybrid_model.py`

### No Negative Irradiance, Zero Nighttime Irradiance, Ordered Quantiles

Command:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("outputs/live/forecast.csv")
operational = df[df["operational_status"].eq("Operational")].copy()
for column in ["GHI_P50", "POA_P50", "PV_P50", "DNI_P50_estimated", "DHI_P50_estimated"]:
    assert (pd.to_numeric(operational[column], errors="coerce").fillna(0) >= 0).all(), column
night = operational[pd.to_numeric(operational["target_solar_elevation"], errors="coerce") <= 0]
if not night.empty:
    assert (pd.to_numeric(night["GHI_P50"], errors="coerce").fillna(0) == 0).all()
    assert (pd.to_numeric(night["PV_P50"], errors="coerce").fillna(0) == 0).all()
assert (operational["GHI_P10_calibrated"] <= operational["GHI_P50"]).all()
assert (operational["GHI_P50"] <= operational["GHI_P90_calibrated"]).all()
assert (operational["PV_P10"] <= operational["PV_P50"]).all()
assert (operational["PV_P50"] <= operational["PV_P90"]).all()
print("physical forecast checks passed")
PY
```

Expected output:

```text
physical forecast checks passed
```

### Fallback Source Is Always Disclosed

Command:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("outputs/live/forecast.csv")
required = [
    "primary_satellite_source",
    "weather_forecast_source",
    "satellite_data_available",
    "fallback_active",
    "data_freshness_minutes",
    "data_quality_level",
]
assert set(required).issubset(df.columns)
assert not df["fallback_active"].isna().any()
assert not df["weather_forecast_source"].isna().any()
print("source transparency checks passed")
PY
```

Expected output:

```text
source transparency checks passed
```

### Demo Snapshot Is Clearly Labelled

Command:

```bash
python3 - <<'PY'
import json
import pandas as pd

forecast = pd.read_csv("outputs/demo/demo_forecast.csv")
assert forecast["demo_mode"].eq(True).all()
assert forecast["display_mode"].eq("Demo Snapshot").all()
with open("outputs/demo/demo_snapshot.json", "r", encoding="utf-8") as f:
    snapshot = json.load(f)
assert snapshot["demo_mode"] is True
assert snapshot["display_mode"] == "Demo Snapshot"
print("demo labelling checks passed")
PY
```

Expected output:

```text
demo labelling checks passed
```

### Unsupported 15/30-Minute Horizons Are Not Operational

Command:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("outputs/live/forecast.csv")
hf = df[df["horizon_minutes"].isin([15, 30])]
assert not hf.empty
assert hf["operational_status"].eq("Not operationally available").all()
assert hf["operational_unavailable_reason"].eq("High-frequency satellite input required").all()
assert hf["GHI_P50"].isna().all()
print("high-frequency availability checks passed")
PY
```

Expected output:

```text
high-frequency availability checks passed
```

### Benchmark Metrics Use Test Data

Command:

```bash
python3 - <<'PY'
import pandas as pd

metrics = pd.read_csv("outputs/metrics/benchmark_metrics.csv")
skill = pd.read_csv("outputs/metrics/forecast_skill_by_horizon.csv")
assert not metrics.empty
assert not skill.empty
test = pd.read_parquet("data/processed/test_dataset.parquet")
local_time = pd.to_datetime(test["timestamp"], utc=True).dt.tz_convert("Europe/Berlin")
start = local_time.min()
end = local_time.max()
assert str(start.date()) >= "2025-07-01"
assert str(end.date()) <= "2025-12-31"
print("benchmark test-period checks passed")
PY
```

Expected output:

```text
benchmark test-period checks passed
```

### Uncertainty Calibration Uses Validation Data Only

Command:

```bash
python3 - <<'PY'
from pathlib import Path

quantile_code = Path("src/models/train_quantile_forecast.py").read_text()
conformal_code = Path("src/models/conformal_calibration.py").read_text()
assert "validation_pred" in quantile_code
assert "fit_split_conformal_interval(validation_pred)" in quantile_code
assert "test" not in conformal_code.lower()
print("uncertainty calibration split checks passed")
PY
```

Expected output:

```text
uncertainty calibration split checks passed
```
