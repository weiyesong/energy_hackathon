import type { ForecastPoint } from './types';

export function formatPower(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Unavailable';
  const abs = Math.abs(value);
  if (abs >= 1000) return `${(value / 1000).toFixed(abs >= 10000 ? 1 : 2)} GW`;
  return `${value.toFixed(abs >= 10 ? 1 : 2)} MW`;
}

export function formatEnergy(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Unavailable';
  const abs = Math.abs(value);
  if (abs >= 1000) return `${(value / 1000).toFixed(abs >= 10000 ? 1 : 2)} GWh`;
  return `${value.toFixed(abs >= 10 ? 1 : 2)} MWh`;
}

export function formatPercent(value: number | null | undefined, fraction = true, decimals = 0): string {
  if (value == null || !Number.isFinite(value)) return 'Unavailable';
  const pct = fraction ? value * 100 : value;
  return `${pct.toFixed(decimals)}%`;
}

export function shortTime(value: string | null | undefined): string {
  if (!value) return 'Unavailable';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value.slice(11, 16);
  return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function shortDateTime(value: string | null | undefined): string {
  if (!value) return 'Unavailable';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function sourceLabel(source: string): string {
  if (source === 'eumetsat_ssi') return 'EUMETSAT SSI';
  if (source === 'nasa_power') return 'NASA POWER';
  if (source === 'openmeteo' || source === 'unavailable') return 'Open-Meteo satellite';
  return source.replaceAll('_', ' ');
}

export function horizonLabel(minutes: number | null): string {
  if (minutes == null) return 'Unavailable';
  if (minutes < 60) return `${minutes} min`;
  if (minutes === 60) return '60 min';
  return `${minutes / 60} h`;
}

export function horizonBand(minutes: number | null): string {
  if (minutes == null) return 'Unavailable';
  if (minutes <= 60) return 'satellite / persistence dominated';
  if (minutes <= 360) return 'satellite and weather blend';
  return 'weather forecast dominated';
}

export function operationalPoints(forecast: ForecastPoint[]): ForecastPoint[] {
  return forecast.filter((point) => point.GHI_P50 != null && point.PV_P50 != null);
}

export function byHorizon(forecast: ForecastPoint[]): ForecastPoint[] {
  const map = new Map<number, ForecastPoint>();
  for (const point of forecast) {
    if (point.horizon_minutes == null) continue;
    if (!map.has(point.horizon_minutes)) map.set(point.horizon_minutes, point);
  }
  return Array.from(map.entries())
    .sort(([a], [b]) => a - b)
    .map(([, point]) => point);
}

export function riskClass(value: string | null | undefined): string {
  const normalized = String(value ?? '').toLowerCase();
  if (normalized.includes('high')) return 'riskHigh';
  if (normalized.includes('medium')) return 'riskMedium';
  if (normalized.includes('low')) return 'riskLow';
  return 'riskNeutral';
}
