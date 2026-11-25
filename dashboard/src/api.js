// Thin fetch wrapper for the local FastAPI app (same origin, or Vite proxy).
async function call(method, path, body) {
  const opts = { method, headers: {} }
  if (body instanceof FormData) opts.body = body
  else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  const res = await fetch(path, opts)
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    const detail = data && data.detail
    throw new Error(typeof detail === 'string' ? detail : `HTTP ${res.status}`)
  }
  return data
}

const qs = (params) => {
  const p = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '' && v !== false) p.set(k, v)
  })
  const s = p.toString()
  return s ? `?${s}` : ''
}

export const api = {
  status: () => call('GET', '/api/status'),
  settings: () => call('GET', '/api/settings'),
  updateSettings: (changes) => call('POST', '/api/settings', changes),
  quit: () => call('POST', '/api/app/quit'),
  startLogger: (opts) => call('POST', '/api/logger/start', opts),
  stopLogger: () => call('POST', '/api/logger/stop'),
  startSession: (task, category) => call('POST', '/api/session/start', { task, category }),
  sessionAction: (action) => call('POST', `/api/session/${action}`),
  windows: (params) => call('GET', `/api/windows${qs(params)}`),
  setLabels: (window_starts, label, source = 'user') =>
    call('POST', '/api/labels', { window_starts, label, source }),
  clearLabels: (window_starts) => call('DELETE', '/api/labels', { window_starts }),
  predictions: (params) => call('GET', `/api/predictions${qs(params)}`),
  explainDay: (date) => call('POST', '/api/insights/explain', { date }),
  exportUrl: (participant) => `/api/export${qs({ participant })}`,

  devImport: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return call('POST', '/api/dev/import', fd)
  },
  devParticipants: () => call('GET', '/api/dev/participants'),
  devWindows: (params) => call('GET', `/api/dev/windows${qs(params)}`),
  devSetLabels: (items, label, annotator) =>
    call('POST', '/api/dev/labels', { items, label, annotator }),
  devClearLabels: (items) => call('DELETE', '/api/dev/labels', { items }),
  devAgreement: () => call('GET', '/api/dev/agreement'),
  devExportUrl: () => '/api/dev/export',
}

export const LABELS = ['Focused', 'Distracted', 'Inactive']

export function fmtTime(iso) {
  return iso ? iso.slice(11, 19) : '-'
}

export function remember(key, fallback = '') {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}

export function store(key, value) {
  try { localStorage.setItem(key, value) } catch { /* private mode: fine */ }
}
