import { useState, useRef } from 'react'
import axios from 'axios'
import './UploadSection.css'

const CLASS_COLOR = { Clean: '#22C55E', 'Low Dust': '#F59E0B', 'Heavy Dust': '#EF4444' }
const CLASS_LABEL_COLOR = { Clean: 'green', 'Low Dust': 'amber', 'Heavy Dust': 'red' }

export default function UploadSection({ params, onResult }) {
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState('idle') // idle | loading | done | error
  const [result, setResult] = useState(null)
  const [errMsg, setErrMsg] = useState('')
  const [step, setStep] = useState('')
  const [isMyPanel, setIsMyPanel] = useState(true)
  const inputRef = useRef()

  const runAnalysis = async (file) => {
    setStatus('loading')
    setResult(null)
    setErrMsg('')

    const form = new FormData()
    form.append('file', file)
    form.append('system_kw', params.system_kw)
    form.append('tariff', params.tariff)
    form.append('cleaning_cost', params.cleaning_cost)
    form.append('is_my_panel', isMyPanel)

    const steps = [
      'Validating image…',
      'Running DIP pipeline…',
      'Running model inference…',
      'Calculating PKR loss…',
    ]
    let i = 0
    const tick = setInterval(() => { if (i < steps.length) setStep(steps[i++]) }, 800)

    try {
      const { data } = await axios.post('/api/analyze', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      clearInterval(tick)

      if (!data.valid) {
        setErrMsg(data.reason || 'Image rejected — not a valid solar panel photo.')
        setStatus('error')
        onResult(null)
        return
      }

      setResult(data)
      setStatus('done')
      onResult(data)
    } catch (err) {
      clearInterval(tick)
      setErrMsg('Server error — make sure the FastAPI backend is running.')
      setStatus('error')
      onResult(null)
    }
  }

  const handleFile = (file) => {
    if (!file) return
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setErrMsg('Only JPG or PNG files are accepted.')
      setStatus('error')
      return
    }
    runAnalysis(file)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const reset = () => {
    setStatus('idle')
    setResult(null)
    setErrMsg('')
    onResult(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="upload-section">
      {status === 'idle' && (
        <label className="my-panel-toggle">
          <input
            type="checkbox"
            checked={isMyPanel}
            onChange={e => setIsMyPanel(e.target.checked)}
          />
          <span className="my-panel-text">
            <span className="my-panel-title">This is my own solar system</span>
            <span className="my-panel-sub">
              {isMyPanel
                ? 'Result will be cross-checked against my FoxESS inverter data.'
                : 'Classifier only — inverter cross-check disabled for non-personal images.'}
            </span>
          </span>
        </label>
      )}
      {status !== 'done' && (
        <div
          className={`drop-zone card ${dragging ? 'dragging' : ''} ${status === 'loading' ? 'loading' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => status !== 'loading' && inputRef.current?.click()}
        >
          <input
            ref={inputRef} type="file" accept=".jpg,.jpeg,.png"
            style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])}
          />
          {status === 'idle' && (
            <>
              <div className="drop-icon">↑</div>
              <div className="drop-title">Drop a panel photo here</div>
              <div className="drop-sub">or click to browse · JPG / PNG</div>
            </>
          )}
          {status === 'loading' && (
            <>
              <div className="spinner" />
              <div className="drop-title">{step || 'Processing…'}</div>
              <div className="drop-sub">This takes a few seconds</div>
            </>
          )}
          {status === 'error' && (
            <>
              <div className="drop-icon error-icon">✕</div>
              <div className="drop-title" style={{ color: 'var(--red)' }}>Analysis failed</div>
              <div className="drop-sub">{errMsg}</div>
              <button className="btn btn-outline" style={{ marginTop: 12 }} onClick={e => { e.stopPropagation(); reset() }}>
                Try again
              </button>
            </>
          )}
        </div>
      )}

      {status === 'done' && result && (
        <AnalysisResult result={result} onReset={reset} />
      )}
    </div>
  )
}

function AnalysisResult({ result, onReset }) {
  const color = CLASS_COLOR[result.class_label] || '#888'
  const conf = result.confidence

  return (
    <div className="result-wrap">
      <div className="result-header card">
        <div className="result-badge" style={{ borderColor: color, color }}>
          {result.class_label}
        </div>
        <div className="result-conf">
          <span className="label">Model confidence</span>
          <div className="conf-row">
            <div className="progress-track" style={{ flex: 1 }}>
              <div className="progress-fill" style={{ width: `${conf * 100}%`, background: color }} />
            </div>
            <span style={{ color, fontWeight: 700, fontSize: '0.88rem', minWidth: 44, textAlign: 'right' }}>
              {(conf * 100).toFixed(1)}%
            </span>
          </div>
          {result.mock && (
            <span className="mock-badge">Mock mode — add model file for real predictions</span>
          )}
        </div>
        <button className="btn btn-outline reset-btn" onClick={onReset}>New image</button>
      </div>

      <div className="images-row">
        <div className="image-card card">
          <span className="label">Preprocessed · 224×224 px</span>
          <img src={`data:image/jpeg;base64,${result.preprocessed_image}`} alt="Preprocessed" />
        </div>
        <div className="image-card card">
          <span className="label">Grad-CAM saliency</span>
          {result.gradcam_image
            ? <img src={`data:image/jpeg;base64,${result.gradcam_image}`} alt="Grad-CAM" />
            : <div className="no-cam">Not available in mock mode</div>
          }
        </div>
      </div>

      <div className="metrics-row">
        <MetricTile label="Expected Output" value={`${result.expected_watts?.toLocaleString()} W`} />
        <MetricTile
          label="Predicted Output"
          value={`${result.predicted_watts?.toLocaleString()} W`}
          delta={`−${result.watt_loss?.toLocaleString()} W`}
          deltaColor="var(--red)"
        />
        <MetricTile label="Daily PKR Loss" value={`PKR ${result.daily_loss_pkr?.toLocaleString()}`} accent />
        <MetricTile label="Monthly PKR Loss" value={`PKR ${result.monthly_loss_pkr?.toLocaleString()}`} />
      </div>

      <div className="loss-bar-wrap card">
        <div className="loss-bar-header">
          <span className="label">Output loss</span>
          <span style={{ fontWeight: 700, color: 'var(--red)', fontSize: '0.95rem' }}>
            {result.loss_pct?.toFixed(1)}%
          </span>
        </div>
        <div className="progress-track" style={{ height: 8, marginTop: 8 }}>
          <div
            className="progress-fill"
            style={{
              width: `${Math.min(result.loss_pct, 100)}%`,
              background: result.loss_pct > 30 ? 'var(--red)' : result.loss_pct > 15 ? 'var(--amber)' : 'var(--green)',
            }}
          />
        </div>
      </div>
    </div>
  )
}

function MetricTile({ label, value, delta, deltaColor, accent }) {
  return (
    <div className={`metric-tile card ${accent ? 'metric-accent' : ''}`}>
      <span className="label">{label}</span>
      <span className="metric-value" style={{ fontSize: '1.1rem' }}>{value}</span>
      {delta && <span style={{ fontSize: '0.78rem', color: deltaColor || 'var(--text-2)', marginTop: 2 }}>{delta}</span>}
    </div>
  )
}
