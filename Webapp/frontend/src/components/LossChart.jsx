import {
  ResponsiveContainer, ComposedChart, Bar, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'
import './Chart.css'

const formatDate = (s) => {
  const d = new Date(s)
  return `${d.getDate()} ${d.toLocaleString('en', { month: 'short' })}`
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="tooltip-date">{formatDate(label)}</div>
      {payload.map((p, i) => (
        <div key={i} className="tooltip-row">
          <span style={{ color: p.color }}>{p.name}</span>
          <span>PKR {p.value?.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

export default function LossChart({ data, loading }) {
  const rows = buildRows(data)

  return (
    <div className="chart-card card">
      <div className="chart-header">
        <div>
          <span className="chart-title">PKR Loss from Soiling</span>
          <span className="chart-sub">Daily & cumulative estimated loss</span>
        </div>
      </div>

      {loading ? (
        <div className="chart-loading"><div className="spinner" /></div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={rows} margin={{ top: 8, right: 40, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fill: 'var(--text-2)', fontSize: 11 }}
              axisLine={false} tickLine={false}
              interval={Math.floor(rows.length / 5)}
            />
            <YAxis
              yAxisId="left"
              tick={{ fill: 'var(--text-2)', fontSize: 11 }}
              axisLine={false} tickLine={false}
              width={36}
            />
            <YAxis
              yAxisId="right" orientation="right"
              tick={{ fill: 'var(--text-2)', fontSize: 11 }}
              axisLine={false} tickLine={false}
              width={44}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, color: 'var(--text-2)', paddingTop: 8 }}
              iconSize={8}
            />
            <Bar
              yAxisId="left"
              dataKey="daily_loss" name="Daily Loss (PKR)"
              fill="var(--red)" opacity={0.7} radius={[2, 2, 0, 0]}
            />
            <Line
              yAxisId="right"
              dataKey="cumulative" name="Cumulative (PKR)"
              stroke="var(--amber)" strokeWidth={2}
              dot={false}
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
    daily_loss: data.daily_losses[i],
    cumulative: data.cumulative_losses[i],
    is_anomaly: data.is_anomaly[i],
  }))
}
