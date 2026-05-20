import './RecommendationCard.css'

const ACTION_CONFIG = {
  'CLEAN NOW': { color: 'var(--red)',   bg: 'rgba(239,68,68,0.08)',   label: 'CLEAN NOW' },
  'WAIT':      { color: 'var(--blue)',  bg: 'rgba(96,165,250,0.08)',  label: 'WAIT' },
  'MONITOR':   { color: 'var(--green)', bg: 'rgba(34,197,94,0.08)',   label: 'MONITOR' },
}

const CROSS_CHECK_CONFIG = {
  confirmed:      { color: 'var(--red)',    bg: 'rgba(239,68,68,0.10)',   label: 'CONFIRMED SOILING',       icon: '✓' },
  localised:      { color: '#f59e0b',       bg: 'rgba(245,158,11,0.10)',  label: 'LOCALISED DUST',          icon: '⚠' },
  non_dust_loss:  { color: 'var(--red)',    bg: 'rgba(239,68,68,0.10)',   label: 'NON-DUST LOSS',           icon: '⚠' },
  healthy:        { color: 'var(--green)',  bg: 'rgba(34,197,94,0.10)',   label: 'ALL CLEAR',               icon: '✓' },
  unknown:        { color: 'var(--muted)',  bg: 'rgba(148,163,184,0.08)', label: 'CROSS-CHECK UNAVAILABLE', icon: '—' },
  skipped:        { color: 'var(--muted)',  bg: 'rgba(148,163,184,0.08)', label: 'CROSS-CHECK SKIPPED',     icon: '○' },
}

export default function RecommendationCard({ analysis, weather }) {
  if (!analysis) {
    return (
      <div className="rec-placeholder card">
        <span className="label">Cleaning Recommendation</span>
        <div className="rec-empty">
          Upload a panel photo to get a cleaning recommendation.
        </div>
      </div>
    )
  }

  const rec = analysis.recommendation
  const cfg = ACTION_CONFIG[rec?.action] || ACTION_CONFIG['MONITOR']
  const cc = analysis.cross_check
  const ccCfg = CROSS_CHECK_CONFIG[cc?.status] || CROSS_CHECK_CONFIG.unknown
  const loss = analysis

  return (
    <div className="rec-card card">
      <span className="label">Cleaning Recommendation</span>

      {cc && (
        <div className="rec-crosscheck" style={{ background: ccCfg.bg, borderColor: ccCfg.color }}>
          <div className="rec-crosscheck-header">
            <span className="rec-crosscheck-icon" style={{ color: ccCfg.color }}>{ccCfg.icon}</span>
            <span className="rec-crosscheck-label" style={{ color: ccCfg.color }}>{ccCfg.label}</span>
          </div>
          <div className="rec-crosscheck-grid">
            <div className="rec-crosscheck-cell">
              <span className="label">This panel</span>
              <span className="rec-crosscheck-val">{cc.image_dusty ? 'Dust detected' : 'Looks clean'}</span>
            </div>
            <div className="rec-crosscheck-cell">
              <span className="label">Inverter ({cc.window_days || 0}-day avg)</span>
              <span className="rec-crosscheck-val">
                {cc.inverter_efficiency_pct != null
                  ? `${cc.inverter_efficiency_pct}% efficient`
                  : 'No data'}
              </span>
            </div>
          </div>
          <div className="rec-crosscheck-msg">{cc.message}</div>
        </div>
      )}

      <div className="rec-action-block" style={{ background: cfg.bg, borderColor: cfg.color }}>
        <div className="rec-action" style={{ color: cfg.color }}>{cfg.label}</div>
        <div className="rec-reason">{rec?.reason}</div>
      </div>

      {rec?.suppress_loss ? (
        <div className="rec-suppressed">
          Overcast ({rec.today_cloud_pct?.toFixed(0)}% cloud cover) — loss figures hidden, readings unreliable.
        </div>
      ) : (
        <div className="rec-metrics">
          <RecMetric label="Daily Loss" value={`PKR ${loss.daily_loss_pkr?.toLocaleString()}`} />
          <RecMetric label="Monthly Loss" value={`PKR ${loss.monthly_loss_pkr?.toLocaleString()}`} />
          <RecMetric
            label="Breakeven"
            value={
              loss.days_to_breakeven === 0 ? 'DIY — instant' :
              loss.days_to_breakeven === -1 ? 'No loss' :
              `${loss.days_to_breakeven} day${loss.days_to_breakeven !== 1 ? 's' : ''}`
            }
          />
        </div>
      )}

      <div className="rec-weather-row">
        {rec?.rain_5day_mm > 0 && (
          <span className="rec-tag">⛆ {rec.rain_5day_mm?.toFixed(1)} mm rain (5 days)</span>
        )}
        {rec?.today_cloud_pct > 0 && (
          <span className="rec-tag">☁ {rec.today_cloud_pct?.toFixed(0)}% cloud today</span>
        )}
      </div>
    </div>
  )
}

function RecMetric({ label, value }) {
  return (
    <div className="rec-metric">
      <span className="label">{label}</span>
      <span className="rec-metric-val">{value}</span>
    </div>
  )
}
