"""Pydantic response models for the SolarOps API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """API health response."""

    status: str
    demo_mode: bool
    data_mode: str


class OverviewResponse(BaseModel):
    """Dashboard overview response."""

    region: str
    current_pv_output: float
    current_output_time: str | None
    current_output_basis: str
    next_peak_time: str | None
    next_peak_power: float
    expected_daily_energy: float
    forecast_risk: str
    forecast_skill_vs_persistence: float | None
    primary_satellite_source: str
    satellite_data_available: bool
    next_expected_pv_drop: dict[str, Any] | None
    recommended_action: dict[str, Any] | None
    selected_site_id: str
    selected_site_name: str
    demo_mode: bool


class ForecastPoint(BaseModel):
    """One forecast time-series point."""

    site_id: str | None = None
    target_time: str | None
    horizon_minutes: int | None
    GHI_P10: float | None
    GHI_P50: float | None
    GHI_P90: float | None
    persistence_GHI: float | None
    POA_P50: float | None
    PV_P10: float | None
    PV_P50: float | None
    PV_P90: float | None
    cloud_cover: float | None
    solar_elevation: float | None
    main_limiting_factor: str | None
    uncertainty_level: str | None
    cloud_event: bool


class BenchmarkResponse(BaseModel):
    """Offline benchmark response."""

    model_RMSE: float | None
    persistence_RMSE: float | None
    skill_score: float | None
    satellite_value_add: float | None
    evaluation_mode: str
    evaluation_period: str | None


class SiteResponse(BaseModel):
    """Site ranking/detail response."""

    site_id: str
    rank_grade: str | None = None
    site_score: float | None = None
    expected_daily_energy: float | None = None
    peak_PV_P50: float | None = None
    mean_uncertainty_width: float | None = None
    cloud_risk: float | None = None
    forecast_volatility: float | None = None
    data_quality: float | None = None


class DataSourceResponse(BaseModel):
    """Data-source transparency response."""

    source_name: str
    source_role: str
    status: str
    last_update: str | None
    temporal_resolution: str | None
    coverage: dict[str, Any] | list[Any] | str | None
    manual_action_required: bool
    fallback_active: bool
