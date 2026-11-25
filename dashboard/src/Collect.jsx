import { useCallback, useEffect, useState } from 'react'
import { api, fmtTime, LABELS } from './api.js'
import { LabelBar, LabelChip, ModelLine, PredChip, Section, SelectableTable, summarise, useToast } from './components.jsx'

const TARGET_PER_CLASS = 40
const today = () => {
  const d = new Date()
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10)
}
const mins = (s) => `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`
const hhmm = (iso) => fmtTime(iso).slice(0, 5)

function Logger({ status, refresh, toast, settings, onSettings }) {
  const lg = status?.logger
  const [websites, setWebsites] = useState(true)
  const [titles, setTitles] = useState(true)
  const [busy, setBusy] = useState(false)
  const act = async (fn) => {
    setBusy(true)
    try { await fn(); await refresh() } catch (e) { toast(e.message, 'err') } finally { setBusy(false) }
  }
  return (
    <Section title="Recording">
      <p className="status-line">
        <span className={`dot ${lg?.running ? 'dot-on' : ''}`} />
        {lg?.running ? `Recording since ${hhmm(lg.started_at)}` : 'Not recording'}
      </p>
      {!lg?.running && (
        <div className="options">
          <label><input type="checkbox" checked={websites} onChange={(e) => setWebsites(e.target.checked)} />
            Website names (e.g. github.com)</label>
          <label><input type="checkbox" checked={titles} disabled={!websites}
                        onChange={(e) => setTitles(e.target.checked)} />
            Video titles on video sites</label>
        </div>
      )}
      <div className="row">
        {lg?.running
          ? <button className="btn btn-stop" disabled={busy} onClick={() => act(api.stopLogger)}>Stop recording</button>
          : <button className="btn btn-primary" disabled={busy}
                    onClick={() => act(() => api.startLogger({ websites, read_titles: titles }))}>Start recording</button>}
      </div>
      {settings?.desktop && (
        <label className="check-line small">
          <input type="checkbox" checked={!!settings.autostart} onChange={async (e) => {
            try { onSettings(await api.updateSettings({ autostart: e.target.checked })) }
            catch (err) { toast(err.message, 'err') }
          }} />
          Start with Windows</label>
      )}
      {lg?.recovered_session && <p className="note">{lg.recovered_session}</p>}
      {lg?.last_error && <p className="note note-err">Logger error: {lg.last_error}</p>}
      <p className="fine">App names, switches, idle time and key/mouse <em>counts</em>. Never keystrokes,
        screenshots, full addresses or page text. Closing this tab doesn't stop recording; <strong>Quit</strong> does.</p>
    </Section>
  )
}

function Session({ status, refresh, toast }) {
  const s = status?.session
  const [task, setTask] = useState('')
  const [category, setCategory] = useState('')
  const act = async (fn) => {
    try { await fn(); await refresh() } catch (e) { toast(e.message, 'err') }
  }
  return (
    <Section title="Focus session">
      {s ? (
        <>
          <p className="status-line">{s.task}
            <span className="meta">{s.task_category} · {s.status} · {mins(s.active_seconds)} active</span></p>
          <div className="row">
            {s.status === 'active'
              ? <button className="btn" onClick={() => act(() => api.sessionAction('pause'))}>Pause</button>
              : <button className="btn" onClick={() => act(() => api.sessionAction('resume'))}>Resume</button>}
            <button className="btn btn-stop" onClick={() => act(() => api.sessionAction('end'))}>End session</button>
          </div>
        </>
      ) : (
        <form className="row" onSubmit={(e) => {
          e.preventDefault()
          act(async () => { await api.startSession(task, category || null); setTask('') })
        }}>
          <input className="grow" maxLength={80} placeholder="What are you working on?"
                 value={task} onChange={(e) => setTask(e.target.value)} />
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">auto category</option>
            {(status?.task_categories || []).map((c) => <option key={c}>{c}</option>)}
          </select>
          <button className="btn btn-primary" disabled={!task.trim()}>Start</button>
        </form>
      )}
      <p className="fine">A session only records what you meant to do. It never labels your windows.</p>
    </Section>
  )
}

function Tally({ counts }) {
  return (
    <div className="tally">
      {LABELS.map((l) => {
        const n = counts?.[l] || 0
        return (
          <div key={l} className="tally-item">
            <div className="tally-num">{n}<small> / {TARGET_PER_CLASS}</small></div>
            <div className="tally-label" style={{ color: `var(--${l.toLowerCase()})` }}>{l}</div>
            <div className="bar"><div className={`fill fill-${l.toLowerCase()}`}
                                      style={{ width: `${Math.min(100, (100 * n) / TARGET_PER_CLASS)}%` }} /></div>
          </div>
        )
      })}
    </div>
  )
}

const columns = (preds) => [
  { key: 'time', title: 'Window', className: 'time',
    render: (w) => `${hhmm(w.window_start)}–${hhmm(w.window_end)}` },
  { key: 'top_app', title: 'Main app', render: (w) => <span className="clip">{w.top_app || '–'}</span> },
  { key: 'active_seconds', title: 'Active s', num: true },
  { key: 'idle_seconds', title: 'Idle s', num: true },
  { key: 'switch_count', title: 'App sw.', num: true },
  { key: 'site_switch_count', title: 'Site sw.', num: true },
  { key: 'keyboard_count', title: 'Keys', num: true },
  { key: 'mouse_count', title: 'Mouse', num: true },
  ...(preds ? [{
    key: 'model', title: 'Model',
    render: (w) => {
      const p = preds[w.window_start]
      return p ? <PredChip label={p.predicted} confidence={p.probabilities[p.predicted]} />
        : <span className="tag tag-none">{w.complete ? '—' : ''}</span>
    },
  }] : []),
  { key: 'label', title: 'Your label', render: (w) => (
    w.complete ? <LabelChip label={w.label} accepted={w.label_source === 'accepted'} />
      : <span className="tag tag-none">in progress</span>) },
]

export default function Collect({ settings, onSettings }) {
  const [status, setStatus] = useState(null)
  const [rows, setRows] = useState([])
  const [preds, setPreds] = useState(null)
  const [model, setModel] = useState(null)
  const [predError, setPredError] = useState(null)
  const [date, setDate] = useState(today())
  const [unlabelled, setUnlabelled] = useState(false)
  const [selected, setSelected] = useState(new Set())
  const [busy, setBusy] = useState(false)
  const [participant, setParticipant] = useState(settings?.participant || '')
  const [toastNode, toast] = useToast()

  const refreshStatus = useCallback(() => api.status().then(setStatus).catch((e) => toast(e.message, 'err')), [])
  const refreshWindows = useCallback(() => Promise.all([
    api.windows({ date, unlabelled }).then((r) => {
      setRows(r)
      const keys = new Set(r.map((w) => w.window_start))
      setSelected((s) => new Set([...s].filter((k) => keys.has(k))))
    }).catch((e) => toast(e.message, 'err')),
    api.predictions({ date }).then((r) => {
      setPreds(Object.fromEntries(r.windows.map((w) => [w.window_start, w])))
      setModel(r.model)
      setPredError(null)
    }).catch((e) => { setPreds(null); setPredError(e.message) }),
  ]), [date, unlabelled])

  useEffect(() => { refreshStatus(); const id = setInterval(refreshStatus, 5000); return () => clearInterval(id) }, [refreshStatus])
  useEffect(() => { refreshWindows(); const id = setInterval(refreshWindows, 60000); return () => clearInterval(id) }, [refreshWindows])

  const run = async (fn) => {
    setBusy(true)
    try { await fn(); setSelected(new Set()); await Promise.all([refreshWindows(), refreshStatus()]) }
    catch (e) { toast(e.message, 'err') } finally { setBusy(false) }
  }
  const onLabel = (label) => run(async () => {
    const res = await api.setLabels([...selected], label)
    toast(...summarise(res.results))
  })
  const onClear = () => run(async () => {
    const res = await api.clearLabels([...selected])
    toast(`${res.cleared} label(s) removed`)
  })
  // Accepting is the user's decision, saved as a label marked "accepted" so it is
  // never used to train a model (a model must not learn from its own guesses).
  const onAccept = () => run(async () => {
    const groups = {}
    for (const ws of selected) {
      const p = preds?.[ws]
      if (p) (groups[p.predicted] ||= []).push(ws)
    }
    const results = []
    for (const [label, starts] of Object.entries(groups)) {
      results.push(...(await api.setLabels(starts, label, 'accepted')).results)
    }
    const skipped = selected.size - results.length
    const [text, kind] = summarise(results, 'accepted')
    toast(skipped ? `${text}; ${skipped} had no model estimate` : text, kind)
  })
  const selectUnlabelled = () =>
    setSelected(new Set(rows.filter((w) => w.complete && !w.label).map((w) => w.window_start)))

  const validPseudonym = /^[A-Za-z0-9_-]{1,20}$/.test(participant)
  const lg = status?.logger

  return (
    <>
      {toastNode}
      <header className="page-head">
        <h1>Collect &amp; label</h1>
        <p className="lede">
          <strong>{lg?.running ? 'Recording.' : 'Not recording.'}</strong> Every five minutes becomes a window.
          Check the model&apos;s guess, then confirm it or set your own label.
        </p>
      </header>

      <div className="cols">
        <Logger status={status} refresh={refreshStatus} toast={toast}
                settings={settings} onSettings={onSettings} />
        <Session status={status} refresh={refreshStatus} toast={toast} />
      </div>

      <Section title="Windows" actions={<Tally counts={status?.label_counts} />}>
        <div className="toolbar">
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
          <label><input type="checkbox" checked={unlabelled} onChange={(e) => setUnlabelled(e.target.checked)} />
            Unlabelled only</label>
          <button className="link" onClick={selectUnlabelled}>Select all unlabelled</button>
          <button className="link" onClick={refreshWindows}>Refresh</button>
        </div>
        <LabelBar count={selected.size} onLabel={onLabel} onClear={onClear}
                  onAccept={preds ? onAccept : undefined} busy={busy} />
        <SelectableTable rows={rows} rowKey={(w) => w.window_start} columns={columns(preds)}
                         selected={selected} setSelected={setSelected} disabled={(w) => !w.complete} />
        <p className="fine">
          Click a row to select it, shift-click for a range. Press <strong>F</strong>, <strong>D</strong> or <strong>I</strong> to
          label, <strong>A</strong> to accept the model&apos;s guess (marked ✓), <strong>C</strong> to clear. The dashed model
          column is only an estimate; only your label is saved. Labels you pick yourself teach your personal model;
          accepted guesses don&apos;t.</p>
        <p className="fine">{predError ? <>Model estimates are unavailable: {predError}.</> : <ModelLine model={model} />}
        </p>
      </Section>

      <div className="cols">
        <Section title="How to label">
          <ul className="guide">
            <li><LabelChip label="Focused" />on your intended task for most of the window, including reading or watching a lecture without typing.</li>
            <li><LabelChip label="Distracted" />on the computer but off-task: feeds, unrelated videos, aimless tab-hopping.</li>
            <li><LabelChip label="Inactive" />away from the computer or on your phone for most of the window.</li>
            <li><strong>Mixed or unsure?</strong> Leave it unlabelled, even if the model has a guess.</li>
          </ul>
        </Section>
        <Section title="Export">
          <p className="small">Saves a CSV of your <strong>labelled</strong> windows to Downloads: the numbers, websites,
            video titles, apps and your labels. Model guesses you didn&apos;t accept are not included.</p>
          <div className="row">
            <input placeholder="Code, e.g. P3" value={participant} maxLength={20}
                   onChange={(e) => setParticipant(e.target.value.trim())}
                   onBlur={async () => {
                     if (validPseudonym && participant !== settings?.participant) {
                       try { onSettings(await api.updateSettings({ participant })) } catch (err) { toast(err.message, 'err') }
                     }
                   }} />
            <a className={`btn btn-primary${validPseudonym ? '' : ' disabled'}`}
               href={validPseudonym ? api.exportUrl(participant) : undefined}
               aria-disabled={!validPseudonym}>Export</a>
          </div>
          {!validPseudonym && participant && <p className="note note-err">Letters, digits, - or _ only (no real names).</p>}
        </Section>
      </div>
    </>
  )
}
