import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { api, remember, store } from './api.js'
import Collect from './Collect.jsx'
import Insights from './Insights.jsx'
import Onboarding from './Onboarding.jsx'
import './styles.css'

const route = () => (window.location.hash === '#/insights' ? 'insights' : 'collect')

// Theme: the user's choice if they made one, otherwise the system setting.
const systemTheme = () =>
  (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
const initialTheme = () => remember('theme', '') || systemTheme()
document.documentElement.dataset.theme = initialTheme()   // before first paint: no flash

function ThemeToggle() {
  const [theme, setTheme] = useState(initialTheme())
  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])
  const flip = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    store('theme', next)
    setTheme(next)
  }
  return (
    <button className="link" onClick={flip} aria-label="Toggle dark mode">
      {theme === 'dark' ? 'Light mode' : 'Dark mode'}
    </button>
  )
}

function Quit({ onQuit }) {
  const quit = async () => {
    if (!window.confirm('Quit Focus Tracker? Recording stops until you open it again from the desktop shortcut.')) return
    try { await api.quit() } catch { /* server already gone */ }
    onQuit()
  }
  return <button className="link link-danger" onClick={quit}>Quit</button>
}

function App() {
  const [page, setPage] = useState(route())
  const [settings, setSettings] = useState(null)
  const [quit, setQuit] = useState(false)
  const [error, setError] = useState(null)

  const loadSettings = () => api.settings().then(setSettings).catch((e) => setError(e.message))
  useEffect(() => { loadSettings() }, [])
  useEffect(() => {
    const on = () => setPage(route())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  if (quit) {
    return (
      <div className="onboard">
        <h1>Focus Tracker has stopped</h1>
        <p className="lead">Nothing is being recorded now. To start again, double-click
          <strong> Focus Tracker</strong> on your desktop. You can close this tab.</p>
      </div>
    )
  }
  if (error) return <div className="onboard"><h1>Can&apos;t reach Focus Tracker</h1><p className="lead">{error}. Open it again from the desktop shortcut.</p></div>
  if (!settings) return null
  if (!settings.onboarded) return <Onboarding settings={settings} onDone={loadSettings} />

  return (
    <>
      <nav className="top">
        <span className="brand">Focus Tracker</span>
        <a href="#/" className={`tab${page === 'collect' ? ' active' : ''}`}>Collect &amp; label</a>
        <a href="#/insights" className={`tab${page === 'insights' ? ' active' : ''}`}>Insights</a>
        <span className="right">
          <span className="who">{settings.participant ? `${settings.participant} · ` : ''}stays on this computer</span>
          <ThemeToggle />
          <Quit onQuit={() => setQuit(true)} />
        </span>
      </nav>
      <main>{page === 'insights' ? <Insights />
        : <Collect settings={settings} onSettings={setSettings} />}</main>
    </>
  )
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
