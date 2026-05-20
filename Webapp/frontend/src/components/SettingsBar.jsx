import './SettingsBar.css'

export default function SettingsBar({ params, onChange }) {
  const set = (key, val) => onChange(p => ({ ...p, [key]: val }))

  return (
    <div className="settings-bar card">
      <span className="label">System Parameters</span>
      <div className="settings-fields">
        <div className="field-group">
          <label className="label">Capacity</label>
          <div className="range-row">
            <input
              type="range" min={1} max={50} step={0.5}
              value={params.system_kw}
              onChange={e => set('system_kw', Number(e.target.value))}
            />
            <span className="range-val">{Number(params.system_kw).toFixed(1)} kW</span>
          </div>
        </div>

        <div className="field-group">
          <label className="label">Tariff (PKR/kWh)</label>
          <input
            type="number" min={1} step={5}
            value={params.tariff}
            onChange={e => set('tariff', Number(e.target.value))}
          />
        </div>

        <div className="field-group">
          <label className="label">Cleaning Cost (PKR)</label>
          <input
            type="number" min={0} step={50}
            value={params.cleaning_cost}
            onChange={e => set('cleaning_cost', Number(e.target.value))}
          />
        </div>

        <div className="settings-info">
          <span className="label">Location</span>
          <span className="settings-loc">Rawalpindi · 33.60°N 73.07°E</span>
        </div>
      </div>
    </div>
  )
}
