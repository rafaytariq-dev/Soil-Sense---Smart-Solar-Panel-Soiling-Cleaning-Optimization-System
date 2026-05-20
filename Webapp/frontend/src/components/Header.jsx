import './Header.css'

export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <span className="header-icon">☀</span>
          <div>
            <div className="header-name">SoilSense</div>
            <div className="header-sub">Solar Panel Soiling Detection · Rawalpindi, Pakistan</div>
          </div>
        </div>
        <div className="header-badge">
          <span className="badge-dot" />
          System Online
        </div>
      </div>
    </header>
  )
}
