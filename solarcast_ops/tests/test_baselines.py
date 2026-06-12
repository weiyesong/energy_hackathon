from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines import add_baseline_predictions


def test_persistence_and_clear_sky_are_bounded(tmp_path, monkeypatch):
    import src.baselines as bl

    class Paths:
        predictions_dir = tmp_path

    monkeypatch.setattr(bl, "get_paths", lambda: Paths())
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"),
            "pv_power_mw": [0.0, 0.5, 1.2],
            "clear_sky_power_mw": [0.0, 0.8, 0.8],
            "clear_sky_power_h1": [0.0, 0.7, 0.9],
        }
    )
    out = add_baseline_predictions(df, {"site": {"peak_power_mw": 1.0}, "forecast": {"horizons_hours": [1]}})
    assert np.allclose(out["pred_persistence_h1"], [0.0, 0.5, 1.05])
    assert np.isfinite(out["pred_clear_sky_persistence_h1"]).all()
    assert out["pred_clear_sky_persistence_h1"].iloc[0] == 0
    assert out["pred_clear_sky_persistence_h1"].max() <= 1.05
