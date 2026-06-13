import { Navigation } from 'lucide-react';
import type { ForecastPoint, Overview, Site } from '../types';
import { sourceLabel } from '../utils';

type Props = {
  sites: Site[];
  forecast: ForecastPoint[];
  overview: Overview;
};

const sitePositions: Record<string, { x: number; y: number; label: string }> = {
  munich_centre: { x: 51, y: 50, label: 'Centre' },
  munich_north: { x: 50, y: 25, label: 'North' },
  munich_east: { x: 75, y: 51, label: 'East' },
  munich_south: { x: 52, y: 77, label: 'South' },
  munich_west: { x: 25, y: 51, label: 'West' },
};

export function MunichMap({ sites, forecast, overview }: Props) {
  const source = sourceLabel(overview.primary_satellite_source);
  const cloudRisk = average(forecast.map((point) => point.cloud_cover));
  const irradiance = average(forecast.map((point) => point.GHI_P50));

  return (
    <section className="panel mapPanel">
      <div className="panelHeader compact">
        <div>
          <p className="eyebrow">Munich Field Layer</p>
          <h2>Site status and active source</h2>
        </div>
        <span className="sourcePill">{source}</span>
      </div>
      <div className="mapCanvas">
        <svg viewBox="0 0 560 360" role="img" aria-label="Munich PV site map">
          <defs>
            <radialGradient id="irradianceGradient" cx="45%" cy="50%" r="70%">
              <stop offset="0%" stopColor={`rgba(255, 168, 60, ${0.24 + Math.min(irradiance / 900, 0.35)})`} />
              <stop offset="55%" stopColor="rgba(255, 168, 60, 0.10)" />
              <stop offset="100%" stopColor="rgba(49, 168, 255, 0.04)" />
            </radialGradient>
            <linearGradient id="cloudOverlay" x1="0" x2="1">
              <stop offset="0" stopColor={`rgba(48, 168, 255, ${0.05 + cloudRisk / 320})`} />
              <stop offset="1" stopColor={`rgba(255, 92, 92, ${0.03 + cloudRisk / 420})`} />
            </linearGradient>
          </defs>
          <rect x="18" y="18" width="524" height="324" rx="18" className="mapBase" />
          <path d="M86 248 C134 136, 226 72, 326 88 C425 104, 494 178, 468 264 C438 324, 256 326, 144 298 C104 288, 74 276, 86 248 Z" fill="url(#irradianceGradient)" stroke="rgba(255,255,255,.12)" />
          <path d="M30 104 C130 76, 242 76, 330 116 C410 152, 456 194, 532 204 L532 342 L30 342 Z" fill="url(#cloudOverlay)" />
          <path d="M104 66 C210 104, 326 156, 462 112" className="movementPath" />
          <g transform="translate(445 82)" className="cloudDirection">
            <Navigation size={20} />
            <text x="26" y="16">cloud drift</text>
          </g>
          {sites.map((site) => {
            const position = sitePositions[site.site_id] ?? { x: 50, y: 50, label: site.site_id };
            const status = site.rank_grade ?? 'D';
            return (
              <g key={site.site_id} transform={`translate(${(position.x / 100) * 560} ${(position.y / 100) * 360})`} className="siteNode">
                <circle r="13" className={`siteCircle grade${status}`} />
                <circle r="4" className="siteCore" />
                <text x="18" y="-4">{position.label}</text>
                <text x="18" y="12" className="siteTiny">Grade {status}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mapStats">
        <span>SSI intensity proxy: {Math.round(irradiance)} W/m²</span>
        <span>Cloud risk: {Math.round(cloudRisk)}%</span>
      </div>
    </section>
  );
}

function average(values: Array<number | null>) {
  const clean = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (!clean.length) return 0;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}
