import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Activity, AlertTriangle, BarChart3, BatteryCharging, Database, Gauge, MapPinned, RadioTower, Satellite, Zap } from 'lucide-react';
import { loadDashboardData } from './api';
import { ForecastChart } from './components/ForecastChart';
import { MunichMap } from './components/MunichMap';
import type { DashboardData, ForecastPoint, OperatorAction, Site, ViewKey } from './types';
import { byHorizon, formatKwh, formatKw, formatPercent, horizonBand, horizonLabel, operationalPoints, riskClass, shortDateTime, sourceLabel } from './utils';
import './styles.css';

const navItems: Array<{ key: ViewKey; label: string; icon: typeof Gauge }> = [
  { key: 'cockpit', label: 'Forecast Cockpit', icon: Gauge },
  { key: 'horizons', label: 'Multi-Horizon Intelligence', icon: Activity },
  { key: 'sites', label: 'Site Intelligence', icon: MapPinned },
  { key: 'model', label: 'Model & Data', icon: Database },
];

function App() {
  const [activeView, setActiveView] = useState<ViewKey>('cockpit');
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData()
      .then((payload) => {
        setData(payload);
        setError(null);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <StateFrame title="Loading SolarOps cockpit" detail="Connecting to the FastAPI product backend." />;
  if (error) return <StateFrame title="API connection unavailable" detail={error} />;
  if (!data) return <StateFrame title="No data available" detail="Run the live orchestrator or prepare a demo snapshot." />;

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark"><Zap size={22} /></div>
          <div>
            <strong>SolarOps</strong>
            <span>Munich PV Control</span>
          </div>
        </div>
        <nav>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.key} className={activeView === item.key ? 'active' : ''} onClick={() => setActiveView(item.key)}>
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sourceStack">
          <span>Active source</span>
          <strong>{sourceLabel(data.overview.primary_satellite_source)}</strong>
          <small>{data.overview.satellite_data_available ? 'satellite available' : 'fallback active'}</small>
        </div>
      </aside>

      <main className="mainStage">
        <header className="topBar">
          <div>
            <p className="eyebrow">Energy operations cockpit</p>
            <h1>{navItems.find((item) => item.key === activeView)?.label}</h1>
          </div>
          <div className="topBadges">
            {data.overview.demo_mode && <span className="demoBadge">Demo Snapshot</span>}
            <span className={`riskBadge ${riskClass(data.overview.forecast_risk)}`}>{data.overview.forecast_risk} risk</span>
          </div>
        </header>

        {activeView === 'cockpit' && <ForecastCockpit data={data} />}
        {activeView === 'horizons' && <MultiHorizon data={data} />}
        {activeView === 'sites' && <SiteIntelligence data={data} />}
        {activeView === 'model' && <ModelAndData data={data} />}
      </main>
    </div>
  );
}

function ForecastCockpit({ data }: { data: DashboardData }) {
  const operational = operationalPoints(data.forecast);
  const action = data.overview.recommended_action ?? data.actions[0] ?? null;
  return (
    <div className="viewGrid cockpitGrid">
      <section className="kpiStrip">
        <Kpi title="Current PV Output" value={formatKw(data.overview.current_pv_output)} icon={<Zap size={18} />} />
        <Kpi title="Next Peak" value={formatKw(data.overview.next_peak_power)} detail={shortDateTime(data.overview.next_peak_time)} icon={<BarChart3 size={18} />} />
        <Kpi title="Expected Energy Today" value={formatKwh(data.overview.expected_daily_energy)} icon={<BatteryCharging size={18} />} />
        <Kpi title="Forecast Risk" value={data.overview.forecast_risk} tone={riskClass(data.overview.forecast_risk)} icon={<AlertTriangle size={18} />} />
        <Kpi title="Skill vs Persistence" value={formatPercent(data.overview.forecast_skill_vs_persistence)} icon={<Gauge size={18} />} />
      </section>
      <ForecastChart forecast={operational} overview={data.overview} action={action} />
      <MunichMap sites={data.sites} forecast={data.forecast} overview={data.overview} />
      <RecommendationCard action={action} overview={data.overview} />
    </div>
  );
}

function MultiHorizon({ data }: { data: DashboardData }) {
  const horizons = byHorizon(data.forecast);
  return (
    <div className="viewGrid horizonGrid">
      {horizons.map((point) => {
        const unsupported = point.GHI_P50 == null || point.uncertainty_level === 'Not operationally available';
        return (
          <section key={point.horizon_minutes} className={`panel horizonCard ${unsupported ? 'disabledCard' : ''}`}>
            <div className="horizonTop">
              <div>
                <p className="eyebrow">{horizonBand(point.horizon_minutes)}</p>
                <h2>{horizonLabel(point.horizon_minutes)}</h2>
              </div>
              <span className={unsupported ? 'availability off' : 'availability on'}>{unsupported ? 'Not operationally available' : 'Operational'}</span>
            </div>
            {unsupported ? (
              <div className="unsupported">
                <Satellite size={30} />
                <strong>High-frequency satellite input required</strong>
                <span>No invented predictions are displayed for this horizon.</span>
              </div>
            ) : (
              <>
                <div className="metricRows">
                  <span>Forecast PV output <strong>{formatKw(point.PV_P50)}</strong></span>
                  <span>Uncertainty <strong>{formatKw((point.PV_P90 ?? 0) - (point.PV_P10 ?? 0))}</strong></span>
                  <span>Main driver <strong>{point.main_limiting_factor ?? 'Unavailable'}</strong></span>
                  <span>Data source <strong>{sourceLabel(data.overview.primary_satellite_source)}</strong></span>
                </div>
                <div className="horizonBar"><i style={{ width: `${Math.min((point.PV_P50 ?? 0) * 100, 100)}%` }} /></div>
              </>
            )}
          </section>
        );
      })}
    </div>
  );
}

function SiteIntelligence({ data }: { data: DashboardData }) {
  return (
    <div className="viewGrid siteGrid">
      <MunichMap sites={data.sites} forecast={data.forecast} overview={data.overview} />
      <section className="panel siteTablePanel">
        <div className="panelHeader">
          <div>
            <p className="eyebrow">Good-vs-bad comparison</p>
            <h2>Configured Munich sites</h2>
          </div>
        </div>
        <div className="siteTable">
          <div className="siteRow header">
            <span>Rank</span><span>Site</span><span>Energy</span><span>Peak</span><span>Cloud risk</span><span>Confidence</span><span>Quality</span><span>Grade</span>
          </div>
          {data.sites.map((site, index) => (
            <details className="siteRowWrap" key={site.site_id}>
              <summary className="siteRow">
                <span>#{index + 1}</span>
                <strong>{siteName(site.site_id)}</strong>
                <span>{formatKwh(site.expected_daily_energy)}</span>
                <span>{formatKw(site.peak_PV_P50)}</span>
                <span>{formatPercent(site.cloud_risk)}</span>
                <span>{formatPercent(1 - (site.mean_uncertainty_width ?? 0))}</span>
                <span>{formatPercent(site.data_quality)}</span>
                <b className={`grade grade${site.rank_grade ?? 'D'}`}>{site.rank_grade ?? 'D'}</b>
              </summary>
              <p>Why this site? Score combines expected energy, confidence, data quality, cloud risk, and volatility. Current score: {(site.site_score ?? 0).toFixed(3)}.</p>
            </details>
          ))}
        </div>
      </section>
      <section className="panel comparisonPanel">
        <p className="eyebrow">Energy comparison</p>
        <h2>Expected daily energy by site</h2>
        <div className="barList">
          {data.sites.map((site) => {
            const max = Math.max(...data.sites.map((s) => s.expected_daily_energy ?? 0), 0.01);
            return (
              <div className="barItem" key={site.site_id}>
                <span>{siteName(site.site_id)}</span>
                <i><b style={{ width: `${((site.expected_daily_energy ?? 0) / max) * 100}%` }} /></i>
                <strong>{formatKwh(site.expected_daily_energy)}</strong>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

function ModelAndData({ data }: { data: DashboardData }) {
  const stages = ['Satellite SSI and cloud information', 'solar geometry and clear-sky physics', 'atmospheric correction', 'ML residual forecast', 'uncertainty calibration', 'PV power', 'operator action'];
  const physicalModel = { source_name: 'physical solar model', source_role: 'solar geometry and clear-sky physics', status: 'Live', last_update: null, temporal_resolution: 'deterministic', coverage: null, manual_action_required: false, fallback_active: false };
  return (
    <div className="viewGrid modelGrid">
      <section className="panel pipelinePanel">
        <p className="eyebrow">Forecast chain</p>
        <h2>Satellite-first hybrid model</h2>
        <div className="pipeline">
          {stages.map((stage, index) => (
            <div className="pipelineNode" key={stage}>
              <span>{index + 1}</span>
              <strong>{stage}</strong>
            </div>
          ))}
        </div>
      </section>
      <section className="panel sourcesPanel">
        <p className="eyebrow">Live source status</p>
        <h2>Data transparency</h2>
        <div className="sourceCards">
          {[...data.sources, physicalModel].map((source) => (
            <div className="sourceCard" key={source.source_name}>
              <div>
                <strong>{sourceName(source.source_name)}</strong>
                <span>{source.source_role}</span>
              </div>
              <b className={`status ${statusClass(source.status, source.manual_action_required, source.fallback_active)}`}>
                {source.manual_action_required ? 'Manual download required' : source.fallback_active ? 'Fallback' : source.status}
              </b>
            </div>
          ))}
        </div>
      </section>
      <section className="panel benchmarkPanel">
        <p className="eyebrow">Offline hindcast benchmark</p>
        <h2>Model evidence</h2>
        <div className="benchmarkGrid">
          <Kpi title="Hybrid RMSE" value={`${(data.benchmark.model_RMSE ?? 0).toFixed(1)} W/m²`} />
          <Kpi title="CSI Persistence RMSE" value={`${(data.benchmark.persistence_RMSE ?? 0).toFixed(1)} W/m²`} />
          <Kpi title="Forecast Skill" value={formatPercent(data.benchmark.skill_score)} />
          <Kpi title="Satellite Value Add" value={`${(data.benchmark.satellite_value_add ?? 0).toFixed(1)} W/m²`} />
        </div>
        <p className="smallNote">Evaluation mode: {data.benchmark.evaluation_mode}. Period: {data.benchmark.evaluation_period ?? 'Unavailable'}.</p>
      </section>
      <section className="panel driversPanel">
        <p className="eyebrow">Forecast drivers</p>
        <h2>Current limiting factors</h2>
        <div className="driverList">
          {data.drivers.map((driver) => (
            <div key={driver.driver} className="driverItem">
              <span>{driver.driver}</span>
              <i><b style={{ width: `${driver.share * 100}%` }} /></i>
              <strong>{formatPercent(driver.share)}</strong>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function RecommendationCard({ action, overview }: { action: OperatorAction | null; overview: DashboardData['overview'] }) {
  if (!action) return <section className="panel recommendation"><h2>No intervention required</h2><p>No operator action is currently available from the API.</p></section>;
  return (
    <section className="panel recommendation">
      <p className="eyebrow">Operator recommendation</p>
      <h2>{action.action_type.replaceAll('_', ' ')}</h2>
      <dl>
        <div><dt>Expected event</dt><dd>{overview.next_expected_pv_drop ? `PV drop around ${shortDateTime(overview.next_expected_pv_drop.time)}` : action.reason}</dd></div>
        <div><dt>Operational impact</dt><dd>{action.reason}</dd></div>
        <div><dt>Recommended action</dt><dd>{action.action_type.replaceAll('_', ' ')}</dd></div>
        <div><dt>Confidence</dt><dd>{action.confidence}</dd></div>
        <div><dt>Valid window</dt><dd>{shortDateTime(action.valid_from)} → {shortDateTime(action.valid_until)}</dd></div>
      </dl>
    </section>
  );
}

function Kpi({ title, value, detail, tone, icon }: { title: string; value: string; detail?: string; tone?: string; icon?: ReactNode }) {
  return (
    <div className={`kpi ${tone ?? ''}`}>
      <span className="kpiIcon">{icon}</span>
      <small>{title}</small>
      <strong>{value}</strong>
      {detail && <em>{detail}</em>}
    </div>
  );
}

function StateFrame({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="stateFrame">
      <RadioTower size={34} />
      <h1>{title}</h1>
      <p>{detail}</p>
    </div>
  );
}

function siteName(id: string) {
  return id.replace('munich_', 'Munich ').replaceAll('_', ' ');
}

function sourceName(id: string) {
  if (id === 'eumetsat_ssi') return 'EUMETSAT';
  if (id === 'nasa_power') return 'NASA POWER';
  if (id === 'openmeteo') return 'Open-Meteo';
  return id.replaceAll('_', ' ');
}

function statusClass(status: string, manual: boolean, fallback: boolean) {
  if (manual || status === 'unavailable') return 'unavailable';
  if (fallback) return 'fallback';
  if (status.toLowerCase() === 'available' || status.toLowerCase() === 'live') return 'live';
  return 'cached';
}

export default App;
