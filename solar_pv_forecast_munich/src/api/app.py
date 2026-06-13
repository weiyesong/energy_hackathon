"""FastAPI application for the SolarOps product UI."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.data_service import SolarOpsDataService
from src.api.models import (
    BenchmarkResponse,
    DataSourceResponse,
    ForecastPoint,
    HealthResponse,
    OverviewResponse,
    SiteResponse,
)

app = FastAPI(title="SolarOps API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = SolarOpsDataService()


@app.get("/api/health", response_model=HealthResponse)
def health() -> dict:
    """Return API and data-mode health."""
    return {"status": "ok", "demo_mode": service.demo_mode(), "data_mode": service.data_mode()}


@app.get("/api/overview", response_model=OverviewResponse)
def overview() -> dict:
    """Return dashboard overview information."""
    return service.overview()


@app.get("/api/forecast", response_model=list[ForecastPoint])
def forecast() -> list[dict]:
    """Return all active forecast points."""
    return service.forecast_points()


@app.get("/api/forecast/{site_id}", response_model=list[ForecastPoint])
def forecast_for_site(site_id: str) -> list[dict]:
    """Return forecast points for a single site."""
    points = service.forecast_points(site_id)
    if not points:
        raise HTTPException(status_code=404, detail=f"No forecast found for site_id={site_id}")
    return points


@app.get("/api/benchmark", response_model=BenchmarkResponse)
def benchmark() -> dict:
    """Return offline hindcast benchmark metrics."""
    return service.benchmark()


@app.get("/api/sites", response_model=list[SiteResponse])
def sites() -> list[dict]:
    """Return site ranking records."""
    frame = service.load_sites()
    return frame.where(frame.notna(), None).to_dict(orient="records") if not frame.empty else []


@app.get("/api/sites/{site_id}", response_model=SiteResponse)
def site(site_id: str) -> dict:
    """Return one site ranking record."""
    frame = service.load_sites()
    if frame.empty or site_id not in set(frame["site_id"].astype(str)):
        raise HTTPException(status_code=404, detail=f"No site found for site_id={site_id}")
    row = frame[frame["site_id"].astype(str) == site_id].iloc[0]
    return row.where(row.notna(), None).to_dict()


@app.get("/api/actions")
def actions() -> list[dict]:
    """Return operational action suggestions."""
    return service.load_actions()


@app.get("/api/drivers")
def drivers() -> list[dict]:
    """Return limiting-factor distribution."""
    return service.drivers()


@app.get("/api/data-sources", response_model=list[DataSourceResponse])
def data_sources() -> list[dict]:
    """Return data-source transparency records."""
    return service.data_sources()
