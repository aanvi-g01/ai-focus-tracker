import { useState } from 'react'
import { api } from './api.js'
import { LabelChip } from './components.jsx'

const VALID = /^[A-Za-z0-9_-]{1,20}$/

/** First-run screen: pseudonym, what is recorded, how to label, start logging. */
export default function Onboarding({ settings, onDone }) {
  const [participant, setParticipant] = useState(settings.participant || '')
  const [autostart, setAutostart] = useState(false)
  const [understood, setUnderstood] = useState(false)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const valid = VALID.test(participant)

  const finish = async () => {
    setBusy(true); setError(null)
    try {
      await api.updateSettings({ participant, onboarded: true,
                                 ...(settings.desktop ? { autostart } : {}) })
      try { await api.startLogger({ websites: true, read_titles: true }) } catch { /* already running is fine */ }
      onDone()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="onboard">
      <h1>Welcome to Focus Tracker</h1>
      <p className="lead">For the next few days this app records <em>how</em> you use your computer, and you
        tell it every so often whether you were focused. That's all you need to do.</p>

      <ol className="steps">
        <li>
          <h3>Your participant code</h3>
          <p className="small muted">Your team gives you a code like <strong>P3</strong>. Please don't use your name.</p>
          <input value={participant} maxLength={20} placeholder="e.g. P3" autoFocus
                 onChange={(e) => setParticipant(e.target.value.trim())} />
          {participant && !valid && <p className="note note-err">Letters, digits, - or _ only.</p>}
        </li>
        <li>
          <h3>What is recorded</h3>
          <ul className="small">
            <li>Which app is in front, and when you switch apps or websites</li>
            <li>How long you are idle, and <strong>how many</strong> keys and mouse events (never which keys)</li>
            <li>Website names only, like github.com (never full addresses, searches or page text)</li>
            <li>Video titles on video sites like YouTube, to tell lectures from entertainment (you can turn this off)</li>
          </ul>
          <p className="small muted">Everything stays on this computer until you export it. The export includes the
            websites and video titles, so your team will see them.</p>
        </li>
        <li>
          <h3>Label your 5-minute windows, once or twice a day</h3>
          <ul className="small guide">
            <li><LabelChip label="Focused" /> doing what you meant to do (reading counts, even without typing)</li>
            <li><LabelChip label="Distracted" /> on the computer but off-task: feeds, unrelated videos</li>
            <li><LabelChip label="Inactive" /> away from the computer or on your phone</li>
            <li>Not sure? Leave it unlabelled.</li>
          </ul>
        </li>
      </ol>

      {settings.desktop && (
        <label className="check-line"><input type="checkbox" checked={autostart}
                                             onChange={(e) => setAutostart(e.target.checked)} />
          Start Focus Tracker automatically when Windows starts (recommended)</label>
      )}
      <label className="check-line"><input type="checkbox" checked={understood}
                                           onChange={(e) => setUnderstood(e.target.checked)} />
        I understand what is recorded and agree to take part</label>
      {error && <p className="note note-err">{error}</p>}
      <button className="btn btn-primary btn-big" disabled={!valid || !understood || busy} onClick={finish}>
        Save and start recording
      </button>
    </div>
  )
}
