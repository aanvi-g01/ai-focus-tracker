import { useCallback, useEffect, useState } from 'react'
import { api, fmtTime, LABELS } from './api.js'
import { LabelChip, ModelLine, PredChip, Section } from './components.jsx'

const today = () => {
  const d = new Date()
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10)
}
const hm = (minutes) => {
  const m = Math.round(minutes)
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`
}
const hhmm = (iso) => fmtTime(iso).slice(0, 5)

function DaySummary({ summary, personal }) {
  const total = LABELS.reduce((s, c) => s + summary.minutes[c], 0)
  const { n_labelled: n, matches } = summary.agreement
  return (
    <>
      <div className="stack-bar" role="img"
           aria-label={LABELS.map((c) => `${c} ${hm(summary.minutes[c])}`).join(', ')}>
        {LABELS.map((c) => summary.minutes[c] > 0 && (
          <div key={c} className={`fill fill-${c.toLowerCase()}`}
               style={{ width: `${(100 * summary.minutes[c]) / total}%` }} />
        ))}
      </div>
      <div className="tally">
        {LABELS.map((c) => (
          <div key={c} className="tally-item">
            <div className="tally-num">{hm(summary.minutes[c])}</div>
            <div className="tally-label" style={{ color: `var(--${c.toLowerCase()})` }}>{c}</div>
          </div>
        ))}
        {n > 0 && !personal && (
          <div className="tally-item">
            <div className="tally-num">{matches}<small> / {n}</small></div>
            <div className="tally-label">match your labels</div>
          </div>
        )}
      </div>
    </>
  )
}

function PlainWords({ date }) {
  const [keySet, setKeySet] = useState(null)
  const [fromEnv, setFromEnv] = useState(false)
  const [key, setKey] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [showSent, setShowSent] = useState(false)

  useEffect(() => {
    api.settings().then((s) => { setKeySet(s.llm_key_set); setFromEnv(s.llm_key_from_env) })
      .catch(() => setKeySet(false))
  }, [])
  useEffect(() => { setResult(null); setError(null); setShowSent(false) }, [date])

  const saveKey = async (value) => {
    setError(null)
    try {
      const s = await api.updateSettings({ llm_api_key: value })
      setKeySet(s.llm_key_set); setFromEnv(s.llm_key_from_env); setKey('')
    } catch (e) { setError(e.message) }
  }
  const explain = async () => {
    setBusy(true); setError(null)
    try { setResult(await api.explainDay(date)) } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  if (keySet === null) return null
  return (
    <Section title="In plain words">
      {keySet ? (
        <>
          <div className="row">
            <button className="btn btn-primary" disabled={busy} onClick={explain}>
              {busy ? 'Asking…' : 'Explain this day'}</button>
            {!fromEnv && <button className="link" onClick={() => saveKey('')}>Remove API key</button>}
          </div>
          <p className="fine">Sends a summary of this day&apos;s estimates to Groq: minutes per state, states per hour,
            the model&apos;s reasons and app/website <em>categories</em>. Never app names, websites, video titles
            or what you typed.</p>
        </>
      ) : (
        <form className="row" onSubmit={(e) => { e.preventDefault(); if (key.trim()) saveKey(key.trim()) }}>
          <input type="password" className="grow" placeholder="Groq API key (gsk_…)" value={key}
                 autoComplete="off" onChange={(e) => setKey(e.target.value)} />
          <button className="btn btn-primary" disabled={!key.trim()}>Save key</button>
          <p className="fine" style={{ flexBasis: '100%' }}>An AI model can describe your day in plain words. Add a
            Groq API key to turn it on; it is stored only on this computer.</p>
        </form>
      )}
      {error && <p className="note note-err">{error}</p>}
      {result && (
        <>
          <div className="llm-text">{result.text}</div>
          <p className="fine">
            Written by {result.model} from the model&apos;s estimates{result.cached ? ' (saved answer)' : ''}.
            AI text can be wrong. <button className="link" onClick={() => setShowSent(!showSent)}>
              {showSent ? 'Hide' : 'Show'} what was sent</button>
          </p>
          {showSent && <pre className="sent">{JSON.stringify(result.sent, null, 2)}</pre>}
        </>
      )}
    </Section>
  )
}

export default function Insights() {
  const [date, setDate] = useState(today())
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [jumped, setJumped] = useState(false)

  const load = useCallback(
    () => api.predictions({ date }).then((r) => {
      setData(r); setError(null)
      // first visit with nothing today: show the most recent day that has windows
      if (!jumped && r.windows.length === 0 && r.latest_date && r.latest_date !== date) setDate(r.latest_date)
      setJumped(true)
    }).catch((e) => { setData(null); setError(e.message) }),
    [date, jumped])
  useEffect(() => {
    load()
    const id = setInterval(load, 60000)
    return () => clearInterval(id)
  }, [load])

  const m = data?.model
  return (
    <>
      <header className="page-head">
        <h1>Insights</h1>
        <p className="lede">The model&apos;s <strong>estimate</strong> for each finished window, worked out on this
          computer. Estimates are never saved or exported; your own labels always win.</p>
      </header>

      <Section title="The day at a glance"
               actions={<input type="date" value={date} onChange={(e) => setDate(e.target.value)} />}>
        {error && <p className="note note-err">Predictions unavailable: {error}</p>}
        {data && data.windows.length === 0 && <p className="empty">No finished windows on this day.</p>}
        {data && data.windows.length > 0 &&
          <DaySummary summary={data.summary} personal={m?.source === 'personal'} />}
        {m && <p className="fine"><ModelLine model={m} /></p>}
      </Section>

      {data && data.windows.length > 0 && <PlainWords date={date} />}

      {data && data.windows.length > 0 && (
        <Section title="Window by window">
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Window</th><th>Model</th><th className="num">Confidence</th>
                  <th>Why</th><th>Your label</th></tr>
              </thead>
              <tbody>
                {data.windows.map((w) => (
                  <tr key={w.window_start} className="static">
                    <td className="time">{hhmm(w.window_start)}–{hhmm(w.window_end)}</td>
                    <td><PredChip label={w.predicted} confidence={w.probabilities[w.predicted]} /></td>
                    <td className="num">{Math.round(100 * w.probabilities[w.predicted])}%</td>
                    <td className="why">{w.reasons.length ? w.reasons.join('; ') : '—'}</td>
                    <td><LabelChip label={w.user_label} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </>
  )
}
