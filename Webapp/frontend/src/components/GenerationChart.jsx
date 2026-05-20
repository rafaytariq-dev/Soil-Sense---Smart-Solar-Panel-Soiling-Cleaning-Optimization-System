import {
  ResponsiveContainer, ComposedChart, Line, Scatter,
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
      {row?.generation != null && (
        <div className="tooltip-row">
          <span style={{ color: row.is_anomaly ? 'var(--red)' : 'var(--green)' }}>
            {row.is_anomaly ? '⚠ Anomaly' : '● Normal'}
          </span>
          <span>{row.generation?.toFixed(2)} kWh</span>
        </div>
      )}
      {row?.baseline != null && (
        <div className="tooltip-row">
          <span style={{ color: 'var(--blue)' }}>Baseline</span>
          <span>{row.baseline?.toFixed(2)} kWh</span>
        </div>
      )}
    </div>
  )
}

export default function GenerationChart({ data, loading }) {
  const rows = buildRows(data)

  return (
    <div className="chart-card card">
      <div className="chart-header">
        <div>
          <span className="chart-title">6-Month Generation Trend</span>
          <span className="chart-sub">Daily kWh · anomalies below 85% efficiency</span>
        </div>
        {data?.total_loss_pkr > 0 && (
          <div className="chart-stat">
            <span className="label">Estimated soiling loss</span>
            <span className="chart-stat-val" style={{ color: 'var(--red)' }}>
              PKR {data.total_loss_pkr?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
        )}
      </div>

      {loading ? (
        <div className="chart-loading"><div className="spinner" /></div>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fill: 'var(--text-2)', fontSize: 11 }}
              axisLine={false} tickLine={false}
              interval={Math.floor(rows.length / 6)}
            />
            <YAxis
              tick={{ fill: 'var(--text-2)', fontSize: 11 }}
              axisLine={false} tickLine={false}
              width={36}
              tickFormatter={v => `${v}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, color: 'var(--text-2)', paddingTop: 8 }}
              iconSize={8}
            />
            <Line
              dataKey="generation" name="Generation (kWh)"
              stroke="var(--green)" strokeWidth={1.5}
              dot={false} activeDot={{ r: 3 }}
            />
            <Line
              dataKey="baseline" name="Baseline (kWh)"
              stroke="var(--blue)" strokeWidth={1.5}
              strokeDasharray="5 3" dot={false}
            />
            <Scatter
              dataKey="anomaly_gen" name="Anomaly day"
              fill="var(--red)" shape="cross" size={40}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

function buildRows(data) {
  if (!data?.dates?.length) return []
  return data.dates.map((date, i) => ({
    date,
    generation: data.is_anomaly[i] ? null : data.generations[i],
    anomaly_gen: data.is_anomaly[i] ? data.generations[i] : null,
    baseline: data.baselines[i],
    is_anomaly: data.is_anomaly[i],
  }))
}
