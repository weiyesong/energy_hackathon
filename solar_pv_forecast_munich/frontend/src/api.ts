import type { Benchmark, DashboardData, DataSource, Driver, ForecastPoint, OperatorAction, Overview, Site } from './types';

const endpoints = {
  overview: '/api/overview',
  forecast: '/api/forecast',
  sites: '/api/sites',
  actions: '/api/actions',
  drivers: '/api/drivers',
  benchmark: '/api/benchmark',
  sources: '/api/data-sources',
};

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function loadDashboardData(): Promise<DashboardData> {
  const [overview, forecast, sites, actions, drivers, benchmark, sources] = await Promise.all([
    getJson<Overview>(endpoints.overview),
    getJson<ForecastPoint[]>(endpoints.forecast),
    getJson<Site[]>(endpoints.sites),
    getJson<OperatorAction[]>(endpoints.actions),
    getJson<Driver[]>(endpoints.drivers),
    getJson<Benchmark>(endpoints.benchmark),
    getJson<DataSource[]>(endpoints.sources),
  ]);

  return { overview, forecast, sites, actions, drivers, benchmark, sources };
}
