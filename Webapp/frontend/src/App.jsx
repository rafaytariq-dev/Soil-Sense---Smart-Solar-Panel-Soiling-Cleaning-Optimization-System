import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import Header from './components/Header'
import SettingsBar from './components/SettingsBar'
import UploadSection from './components/UploadSection'
import RecommendationCard from './components/RecommendationCard'
import GenerationChart from './components/GenerationChart'
import LossChart from './components/LossChart'
import WeatherChart from './components/WeatherChart'
import './App.css'

const DEFAULT_PARAMS = { system_kw: 10, tariff: 65, cleaning_cost: 800 }

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [analysis, setAnalysis] = useState(null)
  const [dashboard, setDashboard] = useState(null)
  const [weather, setWeather] = useState([])
  const [dashLoading, setDashLoading] = useState(true)

  // Load initial config from backend
  useEffect(() => {
    axios.get('/api/config').then(r => {
      setParams({
        system_kw: r.data.system_kw,
        tariff: r.data.tariff,
        cleaning_cost: r.data.cleaning_cost,
      })
    }).catch(() => {})
  }, [])

  // Fetch dashboard whenever system params change
  const fetchDashboard = useCallback(() => {
    setDashLoading(true)
    axios.get('/api/dashboard', { params: { system_kw: params.system_kw, tariff: params.tariff } })
      .then(r => setDashboard(r.data))
      .catch(() => setDashboard(null))
      .finally(() => setDashLoading(false))
  }, [params.system_kw, params.tariff])

  useEffect(() => { fetchDashboard() }, [fetchDashboard])

  useEffect(() => {
    axios.get('/api/weather').then(r => setWeather(r.data.data || [])).catch(() => {})
  }, [])

  return (
    <div className="app">
      <Header />
      <div className="page-content">
        <SettingsBar params={params} onChange={setParams} />

        <div className="main-grid">
          <div className="col-left">
            <UploadSection params={params} onResult={setAnalysis} />
          </div>
          <div className="col-right">
            <RecommendationCard analysis={analysis} weather={weather} />
          </div>
        </div>

        <div className="charts-section">
          <div className="charts-grid">
            <div className="chart-full">
              <GenerationChart data={dashboard} loading={dashLoading} />
            </div>
            <div className="chart-half">
              <LossChart data={dashboard} loading={dashLoading} />
            </div>
            <div className="chart-half">
              <WeatherChart data={weather} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
