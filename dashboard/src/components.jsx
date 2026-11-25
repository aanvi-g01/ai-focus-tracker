import { useEffect, useRef, useState } from 'react'
import { LABELS } from './api.js'

export function LabelChip({ label, muted, accepted }) {
  if (!label) return <span className="tag tag-none">—</span>
  return (
    <span className={`tag tag-${label.toLowerCase()}${muted ? ' tag-muted' : ''}`}
          title={accepted ? "You accepted the model's guess (not used to train your personal model)" : undefined}>
      {label}{accepted && <span className="accepted-mark"> ✓</span>}
    </span>
  )
}

/** One line saying which model made the guesses. */
export function ModelLine({ model }) {
  if (!model) return null
  const p = model.personal || {}
  if (model.source === 'personal') {
    const est = p.estimate
    return (
      <>Guesses come from <strong>your personal model</strong>, trained on {p.windows} windows you labelled
        yourself{est ? <>; on your most recent {est.n_test} of them it was right {Math.round(100 * est.accuracy)}% of the time</> : ''}.
        {p.state === 'training' && ' Updating with your newest labels…'}</>
    )
  }
  const s = model.shared || {}
  const left = Math.max(0, (p.needed || 20) - (p.have || 0))
  return (
    <>Guesses come from the <strong>shared model</strong> ({s.windows} windows from {s.participants} study
      participants), which can be far off for a new person.
      {p.state === 'training' ? ' Your personal model is being trained…'
        : left > 0 ? ` Label ${left} more window${left === 1 ? '' : 's'} yourself (at least two different labels) to get your own model.`
        : p.state === 'collecting' ? ' Label at least two different kinds of window to get your own model.' : ''}</>
  )
}

/** The model's estimate: dashed and italic so it never reads as the user's own label. */
export function PredChip({ label, confidence }) {
  if (!label) return <span className="tag tag-none">—</span>
  const title = `Model estimate${confidence != null ? `, ${Math.round(100 * confidence)}% confident` : ''}. Not a label.`
  return <span className={`tag tag-pred tag-${label.toLowerCase()}`} title={title}>{label}</span>
}

/** A page section: serif heading over a hairline rule, no box. */
export function Section({ title, children, actions, className = '' }) {
  return (
    <section className={`sec ${className}`}>
      {(title || actions) && (
        <header className="sec-head">
          {title && <h2>{title}</h2>}
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}
export const Card = Section

/** Label buttons for the current selection; F / D / I / C (and A = accept model) shortcuts. */
export function LabelBar({ count, onLabel, onClear, onAccept, busy }) {
  useEffect(() => {
    const onKey = (e) => {
      if (count === 0 || busy) return
      const tag = (e.target.tagName || '').toLowerCase()
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.ctrlKey || e.metaKey) return
      const k = e.key.toLowerCase()
      const map = { f: 'Focused', d: 'Distracted', i: 'Inactive' }
      if (map[k]) { e.preventDefault(); onLabel(map[k]) }
      if (k === 'c') { e.preventDefault(); onClear() }
      if (k === 'a' && onAccept) { e.preventDefault(); onAccept() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [count, busy, onLabel, onClear, onAccept])

  return (
    <div className="label-bar">
      <span className="count">{count} selected</span>
      {LABELS.map((l) => (
        <button key={l} className={`btn btn-${l.toLowerCase()}`} disabled={!count || busy}
                onClick={() => onLabel(l)}>
          {l} <kbd>{l[0]}</kbd>
        </button>
      ))}
      {onAccept && (
        <button className="btn btn-primary" disabled={!count || busy} onClick={onAccept}
                title="Use the model's estimate as your label for the selected windows">
          Accept model <kbd>A</kbd>
        </button>
      )}
      <button className="link" disabled={!count || busy} onClick={onClear}>Clear label <kbd>C</kbd></button>
    </div>
  )
}

/**
 * Table with checkbox selection. Shift-click selects a range. Rows for which
 * `disabled(row)` is true cannot be selected.
 */
export function SelectableTable({ rows, rowKey, columns, selected, setSelected, disabled = () => false }) {
  const lastIdx = useRef(null)
  const allowed = rows.filter((r) => !disabled(r))
  const allOn = allowed.length > 0 && allowed.every((r) => selected.has(rowKey(r)))

  const toggle = (idx, e) => {
    const next = new Set(selected)
    const key = rowKey(rows[idx])
    const on = !selected.has(key)
    if (e.shiftKey && lastIdx.current !== null) {
      const [a, b] = [Math.min(lastIdx.current, idx), Math.max(lastIdx.current, idx)]
      for (let i = a; i <= b; i++) {
        if (disabled(rows[i])) continue
        on ? next.add(rowKey(rows[i])) : next.delete(rowKey(rows[i]))
      }
    } else {
      on ? next.add(key) : next.delete(key)
    }
    lastIdx.current = idx
    setSelected(next)
  }

  const toggleAll = () => setSelected(allOn ? new Set() : new Set(allowed.map(rowKey)))

  if (!rows.length) return <p className="empty">No windows recorded on this day.</p>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th className="check"><input type="checkbox" checked={allOn} onChange={toggleAll}
                                         aria-label="select all" /></th>
            {columns.map((c) => <th key={c.key} className={c.num ? 'num' : ''}>{c.title}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => {
            const key = rowKey(r)
            const off = disabled(r)
            return (
              <tr key={key} className={`${selected.has(key) ? 'sel' : ''}${off ? ' off' : ''}`}
                  onClick={(e) => { if (!off && e.target.tagName !== 'INPUT') toggle(idx, e) }}>
                <td className="check">
                  <input type="checkbox" disabled={off} checked={selected.has(key)}
                         onChange={(e) => toggle(idx, e.nativeEvent)} aria-label="select window" />
                </td>
                {columns.map((c) => (
                  <td key={c.key} className={c.num ? 'num' : c.className || ''}>{c.render ? c.render(r) : r[c.key]}</td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function useToast() {
  const [msg, setMsg] = useState(null)
  const timer = useRef(null)
  const show = (text, kind = 'ok') => {
    setMsg({ text, kind })
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setMsg(null), 5000)
  }
  const node = msg ? <div className={`toast toast-${msg.kind}`} role="status">{msg.text}</div> : null
  return [node, show]
}

/** Summarise a bulk-label response: "3 labelled, 1 corrected, 1 failed: ..." */
export function summarise(results, verb = 'labelled') {
  const ok = results.filter((r) => r.ok)
  const corrected = ok.filter((r) => r.previous).length
  const failed = results.filter((r) => !r.ok)
  let text = `${ok.length} ${verb}`
  if (corrected) text += `, ${corrected} corrected`
  if (failed.length) text += `, ${failed.length} failed: ${failed[0].error}`
  return [text, failed.length ? 'warn' : 'ok']
}
