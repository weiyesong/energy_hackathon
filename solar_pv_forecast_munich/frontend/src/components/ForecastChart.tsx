import type { ForecastPoint, OperatorAction, Overview } from '../types';
import { formatPower, shortTime } from '../utils';

type Props = {
  forecast: ForecastPoint[];
  overview: Overview;
  action: OperatorAction | null;
};

export function ForecastChart({ forecast, overview, action }: Props) {
  const points = forecast.filter((point) => point.PV_P50 != null);
  if (!points.length) return <div className="emptyPanel">No operational PV forecast available.</div>;

  const width = 920;
  const height = 340;
  const pad = { top: 24, right: 28, bottom: 44, left: 52 };
  const maxY = Math.max(...points.flatMap((p) => [p.PV_P90 ?? 0, p.PV_P50 ?? 0, p.persistence_GHI ? p.persistence_GHI / 1000 : 0]), 0.1);
  const x = (index: number) => pad.left + (index / Math.max(points.length - 1, 1)) * (width - pad.left - pad.right);
  const y = (value: number | null) => height - pad.bottom - ((value ?? 0) / maxY) * (height - pad.top - pad.bottom);
  const line = (key: keyof ForecastPoint) => points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y((p[key] as number | null) ?? 0).toFixed(1)}`).join(' ');
  const persistenceLine = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y((p.persistence_GHI ?? 0) / 1000).toFixed(1)}`)
    .join(' ');
  const upper = points.map((p, i) => `${x(i).toFixed(1)},${y(p.PV_P90).toFixed(1)}`).join(' ');
  const lower = points
    .map((p, i) => `${x(points.length - 1 - i).toFixed(1)},${y(points[points.length - 1 - i].PV_P10).toFixed(1)}`)
    .join(' ');
  const cloudIndex = points.findIndex((point) => point.cloud_event);
  const actionIndex = action?.valid_from ? nearestIndex(points, action.valid_from) : -1;
  const highRiskStart = points.findIndex((point) => point.uncertainty_level === 'High' || point.cloud_event);
  const labelStride = Math.max(Math.ceil(points.length / 8), 1);

  return (
    <section className="panel chartPanel">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">PV Forecast</p>
          <h2>P50 output with calibrated uncertainty</h2>
        </div>
        <div className="legend">
          <span><i className="legendLine p50" /> P50 PV</span>
          <span><i className="legendBand" /> P10-P90</span>
          <span><i className="legendLine dashed" /> persistence</span>
        </div>
      </div>
      <svg className="forecastSvg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="PV forecast chart">
        <defs>
          <linearGradient id="riskWindow" x1="0" x2="1">
            <stop offset="0" stopColor="rgba(255, 178, 55, 0.18)" />
            <stop offset="1" stopColor="rgba(255, 92, 92, 0.10)" />
          </linearGradient>
        </defs>
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis" />
        {[0, 0.5, 1].map((tick) => {
          const yy = y(maxY * tick);
          return (
            <g key={tick}>
              <line x1={pad.left} x2={width - pad.right} y1={yy} y2={yy} className="grid" />
              <text x={pad.left - 10} y={yy + 4} className="axisText" textAnchor="end">{formatPower(maxY * tick)}</text>
            </g>
          );
        })}
        {highRiskStart >= 0 && (
          <rect
            x={x(highRiskStart)}
            y={pad.top}
            width={Math.max(width - pad.right - x(highRiskStart), 12)}
            height={height - pad.top - pad.bottom}
            fill="url(#riskWindow)"
            rx="6"
          />
        )}
        <polygon points={`${upper} ${lower}`} className="uncertaintyBand" />
        <path d={persistenceLine} className="persistenceLine" />
        <path d={line('PV_P50')} className="p50Line" />
        {cloudIndex >= 0 && (
          <g transform={`translate(${x(cloudIndex)} ${pad.top + 10})`}>
            <line y1="0" y2={height - pad.top - pad.bottom - 8} className="eventLine" />
            <text x="8" y="14" className="eventLabel">cloud front</text>
          </g>
        )}
        {actionIndex >= 0 && (
          <g transform={`translate(${x(actionIndex)} ${y(points[actionIndex].PV_P50)})`}>
            <circle r="7" className="actionMarker" />
            <text x="12" y="-10" className="eventLabel">action</text>
          </g>
        )}
        {points.map((point, index) => (index % labelStride === 0 || index === points.length - 1) && (
          <text key={`${point.target_time}-${index}`} x={x(index)} y={height - 14} className="axisText" textAnchor="middle">
            {shortTime(point.target_time)}
          </text>
        ))}
      </svg>
      <div className="chartFooter">
        <span>Next peak: {shortTime(overview.next_peak_time)} at {formatPower(overview.next_peak_power)}</span>
        <span>High-risk windows are derived from cloud events and interval width.</span>
      </div>
    </section>
  );
}

function nearestIndex(points: ForecastPoint[], time: string) {
  const target = new Date(time).getTime();
  if (Number.isNaN(target)) return -1;
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  points.forEach((point, index) => {
    const value = point.target_time ? new Date(point.target_time).getTime() : Number.NaN;
    if (Number.isNaN(value)) return;
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}
