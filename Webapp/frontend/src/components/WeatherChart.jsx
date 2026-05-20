import {
  ResponsiveContainer, ComposedChart, Bar, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts'
import './Chart.css'

const formatDate = (s) => {
  const d = new Date(s)
  return `${d.getDate()} ${d.toLocaleString('en', { month: 'short' })}`
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  return (
    <div className="chart-tooltip">
      <div className="tooltip-date">{formatDate(label)}</div>
      <div className="tooltip-row">
        <span style={{ color: 'var(--blue)' }}>Rainfall</span>
        <span>{row?.rain_mm?.toFixed(1)} mm</span>
      </div>
      <div className="tooltip-row">
        <span style={{ color: 'var(--amber)' }}>Cloud cover</span>
        <span>{row?.cloud_pct?.toFixed(0)}%</span>
      </div>
      {row?.description && (
        <div style={{ marginTop: 4, fontSize: '0.75rem', color: 'var(--text-3)', textTransform: 'capitalize' }}>
          {row.description}
        </div>
      )}
    </div>
  )
}

export default function WeatherChart({ data }) {
  if (!data?.length) {
    return (
      <div className="chart-card card">
        <div className="chart-header">
          <div>
            <span className="chart-title">7-Day Weather Forecast</span>
            <span className="chart-sub">No forecast data available</span>
          </div>
        </div>
        <div className="chart-loading" style={{ height: 160 }}>
          <span style={{ color: 'var(--text-3)', fontSize: '0.82rem' }}>
            Check OWM_API_KEY in environment
          </span>
        </div>
      </div>
    )
  }

  const rows = data.map(d => ({
    date: d.date,
    rain_mm: d.rain_mm,
    cloud_pct: d.cloud_cover_pct,
    description: d.description,
  }))

  return (
    <div className="chart-card card">
      <div className="chart-header">
        <div>
          <span className="chart-title">7-Day Weather Forecast</span>
          <span className="chart-sub">Rainfall & cloud cover · Rawalpindi</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={rows} margin={{ top: 8, right: 40, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            tick={{ fill: 'var(--text-2)', fontSize: 11 }}
            axisLine={false} tickLine={false}
          />
          <YAxis
            yAxisId="rain"
            tick={{ fill: 'var(--text-2)', fontSize: 11 }}
            axisLine={false} tickLine={false}
            width={30}
            label={{ value: 'mm', position: 'insideTopLeft', fill: 'var(--text-3)', fontSize: 10, dy: -2 }}
          />
          <YAxis
            yAxisId="cloud" orientation="right"
            domain={[0, 100]}
            tick={{ fill: 'var(--text-2)', fontSize: 11 }}
            axisLine={false} tickLine={false}
            width={36}
            tickFormatter={v => `${v}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: 'var(--text-2)', paddingTop: 8 }}
            iconSize={8}
          />
          <ReferenceLine yAxisId="rain" y={5} stroke="var(--blue)" strokeDasharray="3 3" strokeOpacity={0.5} />
          <ReferenceLine yAxisId="cloud" y={60} stroke="var(--amber)" strokeDasharray="3 3" strokeOpacity={0.5} />
          <Bar
            yAxisId="rain"
            dataKey="rain_mm" name="Rainfall (mm)"
            fill="var(--blue)" opacity={0.65} radius={[2, 2, 0, 0]}
          />
          <Line
            yAxisId="cloud"
            dataKey="cloud_pct" name="Cloud cover (%)"
            stroke="var(--amber)" strokeWidth={2}
            dot={{ fill: 'var(--amber)', r: 3 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
