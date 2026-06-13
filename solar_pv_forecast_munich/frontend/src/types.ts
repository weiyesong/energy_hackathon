export type ViewKey = 'cockpit' | 'horizons' | 'sites' | 'model';

export type Overview = {
  region: string;
  current_pv_output: number;
  current_output_time: string | null;
  current_output_basis: string;
  next_peak_time: string | null;
  next_peak_power: number;
  expected_daily_energy: number;
  forecast_risk: 'low' | 'medium' | 'high' | string;
  forecast_skill_vs_persistence: number | null;
  primary_satellite_source: string;
  satellite_data_available: boolean;
  next_expected_pv_drop: { time: string; drop_fraction: number } | null;
  recommended_action: OperatorAction | null;
  selected_site_id: string;
  selected_site_name: string;
  demo_mode: boolean;
};

export type ForecastPoint = {
  site_id: string | null;
  target_time: string | null;
  horizon_minutes: number | null;
  GHI_P10: number | null;
  GHI_P50: number | null;
  GHI_P90: number | null;
  persistence_GHI: number | null;
  POA_P50: number | null;
  PV_P10: number | null;
  PV_P50: number | null;
  PV_P90: number | null;
  cloud_cover: number | null;
  solar_elevation: number | null;
  main_limiting_factor: string | null;
  uncertainty_level: string | null;
  cloud_event: boolean;
};

export type Site = {
  site_id: string;
  rank_grade: string | null;
  site_score: number | null;
  expected_daily_energy: number | null;
  peak_PV_P50: number | null;
  mean_uncertainty_width: number | null;
  cloud_risk: number | null;
  forecast_volatility: number | null;
  data_quality: number | null;
};

export type OperatorAction = {
  site_id?: string;
  action_type: string;
  priority: string;
  valid_from: string | null;
  valid_until: string | null;
  reason: string;
  confidence: string;
  basis?: string;
};

export type Driver = {
  driver: string;
  count: number;
  share: number;
};

export type Benchmark = {
  model_RMSE: number | null;
  persistence_RMSE: number | null;
  skill_score: number | null;
  satellite_value_add: number | null;
  evaluation_mode: string;
  evaluation_period: string | null;
};

export type DataSource = {
  source_name: string;
  source_role: string;
  status: string;
  last_update: string | null;
  temporal_resolution: string | null;
  coverage: Record<string, unknown> | unknown[] | string | null;
  manual_action_required: boolean;
  fallback_active: boolean;
};

export type DashboardData = {
  overview: Overview;
  forecast: ForecastPoint[];
  sites: Site[];
  actions: OperatorAction[];
  drivers: Driver[];
  benchmark: Benchmark;
  sources: DataSource[];
};
