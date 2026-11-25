import { useCallback, useEffect, useState } from 'react'
import { api, fmtTime, LABELS, remember, store } from './api.js'
import { Card, LabelBar, LabelChip, SelectableTable, summarise, useToast } from './components.jsx'

const keyOf = (w) => `${w.participant}|${w.window_start}|${w.window_seconds}`
const itemOf = (k) => {
  const [participant, window_start, window_seconds] = k.split('|')
  return { participant, window_start, window_seconds: Number(window_seconds) }
}

function Clip({ text }) {
  if (!text) return <span className="muted">–</span>
  return <span className="clip" title={text}>{text}</span>
}

function Agreement({ data }) {
  if (!data) return null
  const rows = Object.entries(data).filter(([, s]) => s.n > 0)
  if (!rows.length) return <p className="muted small">No windows labelled by both a participant and a developer yet.</p>
  return (
    <table className="compact">
      <thead><tr><th>Participant</th><th className="num">Both labelled</th>
        <th className="num">Agreement</th><th className="num">Cohen's κ</th></tr></thead>
      <tbody>
        {rows.map(([p, s]) => (
          <tr key={p} className={p === 'ALL' ? 'total' : ''}>
            <td>{p}</td><td className="num">{s.n}</td>
            <td className="num">{(100 * s.agreement).toFixed(0)}%</td>
            <td className="num">{s.kappa === null ? '–' : s.kappa.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function Developer() {
  const [annotator, setAnnotator] = useState(remember('ft.annotator'))
  const [participants, setParticipants] = useState([])
  const [participant, setParticipant] = useState('')
  const [blind, setBlind] = useState(true)
  const [unlabelled, setUnlabelled] = useState(false)
  const [rows, setRows] = useState([])
  const [selected, setSelected] = useState(new Set())
  const [agreement, setAgreement] = useState(null)
  const [busy, setBusy] = useState(false)
  const [toastNode, toast] = useToast()

  const load = useCallback(async () => {
    try {
      const [ps, ws, ag] = await Promise.all([
        api.devParticipants(),
        api.devWindows({ participant, unlabelled, blind }),
        api.devAgreement(),
      ])
      setParticipants(ps); setRows(ws); setAgreement(ag)
      const keys = new Set(ws.map(keyOf))
      setSelected((s) => new Set([...s].filter((k) => keys.has(k))))
    } catch (e) { toast(e.message, 'err') }
  }, [participant, unlabelled, blind])
  useEffect(() => { load() }, [load])

  const run = async (fn) => {
    setBusy(true)
    try { await fn(); setSelected(new Set()); await load() } catch (e) { toast(e.message, 'err') } finally { setBusy(false) }
  }
  const onLabel = (label) => {
    if (!annotator.trim()) { toast('Enter your annotator name first', 'err'); return }
    run(async () => {
      const res = await api.devSetLabels([...selected].map(itemOf), label, annotator)
      toast(...summarise(res.results))
    })
  }
  const onClear = () => run(async () => {
    const res = await api.devClearLabels([...selected].map(itemOf))
    toast(`${res.cleared} developer label(s) removed`)
  })
  const onUpload = (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (file) run(async () => {
      const r = await api.devImport(file)
      toast(`Imported ${r.rows} windows: ${Object.entries(r.participants).map(([p, n]) => `${p} (${n})`).join(', ')}`)
    })
  }

  const columns = [
    { key: 'participant', title: 'Participant' },
    { key: 'time', title: 'Window', render: (w) => `${w.window_start.slice(0, 10)} ${fmtTime(w.window_start)}` },
    { key: 'active_seconds', title: 'Active s', num: true },
    { key: 'idle_seconds', title: 'Idle s', num: true },
    { key: 'switch_count', title: 'App sw.', num: true },
    { key: 'site_switch_count', title: 'Site sw.', num: true },
    { key: 'keyboard_count', title: 'Keys', num: true },
    { key: 'mouse_count', title: 'Mouse', num: true },
    { key: 'session_task_category', title: 'Task cat.', render: (w) => w.session_task_category || '–' },
    { key: 'app_seconds', title: 'Apps (s)', render: (w) => <Clip text={w.app_seconds} /> },
    { key: 'hostnames', title: 'Sites (s)', render: (w) => <Clip text={w.hostnames} /> },
    { key: 'video_titles', title: 'Video titles', render: (w) => <Clip text={w.video_titles} /> },
    ...(blind ? [] : [{ key: 'user_label', title: 'User label', render: (w) => <LabelChip label={w.user_label} muted /> }]),
    { key: 'developer_label', title: 'Developer label', render: (w) => (
      <>{<LabelChip label={w.developer_label} />} {w.annotator && <span className="muted small">{w.annotator}</span>}</>) },
  ]

  return (
    <>
      {toastNode}
      <div className="banner">
        Developer labels are a <strong>separate annotation</strong>. They never change participants' own labels,
        which remain the ground truth. You only see numeric features, so use these labels for agreement
        analysis, not to evaluate a model trained on the same features.
      </div>
      <div className="grid2">
        <Card title="Import participant exports">
          <p className="small">Upload CSV files exported from the app's Collect page. Other files are rejected.</p>
          <input type="file" accept=".csv,text/csv" onChange={onUpload} disabled={busy} />
          <div className="row">
            <input placeholder="Your annotator name" value={annotator} maxLength={40}
                   onChange={(e) => { setAnnotator(e.target.value); store('ft.annotator', e.target.value) }} />
            <a className="btn" href={api.devExportUrl()}>Export merged CSV</a>
          </div>
        </Card>
        <Card title="Agreement with participants' labels">
          <Agreement data={agreement} />
          <p className="muted small">Cohen's κ corrects agreement for chance: above 0.6 is usually considered substantial.</p>
        </Card>
      </div>
      <Card title="Label participants' windows">
        <div className="toolbar">
          <select value={participant} onChange={(e) => setParticipant(e.target.value)}>
            <option value="">All participants</option>
            {participants.map((p) => <option key={p}>{p}</option>)}
          </select>
          <label><input type="checkbox" checked={blind} onChange={(e) => setBlind(e.target.checked)} />
            Blind (hide participants' labels)</label>
          <label><input type="checkbox" checked={unlabelled} onChange={(e) => setUnlabelled(e.target.checked)} />
            Not yet labelled by a developer</label>
        </div>
        <LabelBar count={selected.size} onLabel={onLabel} onClear={onClear} busy={busy} />
        <SelectableTable rows={rows} rowKey={keyOf} columns={columns}
                         selected={selected} setSelected={setSelected} />
      </Card>
    </>
  )
}
