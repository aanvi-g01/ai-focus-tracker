"""Fixed-length window aggregation (default 5 minutes) of raw logger events.

Turns the event stream into one structured record per window. Windows are
anchored at the moment the logger started (its `agent_start` row, to the
second): a run started at 10:03:17 gives 10:03:17-10:08:17, 10:08:17-10:13:17,
... Each new run of the logger starts a new grid. The start time is the
window's stable identity, so labels attached to it survive re-aggregation.

Timeline reconstruction
-----------------------
* Observed time: from `agent_start` to `agent_stop`. If the agent crashed (no
  stop), the run ends at its last event. Time the agent was not running is
  MISSING DATA: it is neither active nor idle (see `observed_seconds`).
* Foreground app: every agent row carries the current app; the app holds until
  the next row with a different app, or the end of the run.
* Idle: an `idle_duration` row gives the exact episode [end - D, end]. An `idle`
  start without its duration (older data, crash) is closed at the next sign of
  input (active / switch / input count) or at the end of the run.
* Session: from lifecycle rows (start/pause/resume/end), intersected with
  observed time. A window belongs to the session covering the majority of its
  observed time; `session_seconds` says how much.
* Completeness: a window is complete once it has ended, or once its run has
  ended (agent_stop, or a later agent_start after a crash) - it cannot grow.

Approximations: input counts are flushed every 5 s and attributed to the
window of the flush timestamp; switch counts use the event timestamp.

Privacy: works only on what the logger stores (app names, event types,
numbers). Nothing here is a focus label - labels come from the user.
"""
import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

DEFAULT_WINDOW_SECONDS = 300

_SESSION_OPEN = {"session_start", "session_resume"}
_SESSION_CLOSE = {"session_pause", "session_end"}
_INPUT_SIGNS = {"active", "switched", "keyboard_count", "mouse_count"}


@dataclass
class Window:
    window_start: datetime
    window_end: datetime
    window_seconds: int
    session_id: Optional[int]
    session_seconds: int          # observed seconds inside session_id's active time
    observed_seconds: int         # seconds the agent was running in this window
    coverage: float               # observed_seconds / window_seconds
    complete: bool                # False while the window is still in progress
    active_seconds: int           # observed - idle
    idle_seconds: int
    idle_episodes: int            # idle episodes that started in this window
    switch_count: int             # foreground application changes
    site_switch_count: int        # website changes inside the same browser
    switches_per_min: Optional[float]  # per OBSERVED minute; None if unobserved
    distinct_apps: int
    app_seconds: Dict[str, int] = field(default_factory=dict)  # foreground time
    keyboard_count: int = 0
    mouse_count: int = 0


@dataclass
class _Timeline:
    runs: list = field(default_factory=list)      # (start, end) observed
    apps: list = field(default_factory=list)      # (start, end, app)
    idle: list = field(default_factory=list)      # (start, end)
    sessions: list = field(default_factory=list)  # (start, end, session_id)
    points: list = field(default_factory=list)    # (ts, event, value) countable
    starts: list = field(default_factory=list)    # every run's start, in order
    end: Optional[datetime] = None                # end of the last observed run
    open_run: Optional[datetime] = None           # start of a run not yet ended

    def anchor(self, ts):
        """Grid anchor (start of the run containing `ts`, to the second)."""
        i = bisect_right(self.starts, ts) - 1
        return self.starts[max(i, 0)].replace(microsecond=0)


def validate_window_seconds(window_seconds):
    if (isinstance(window_seconds, bool) or not isinstance(window_seconds, int)
            or window_seconds < 60):
        raise ValueError("window_seconds must be an integer >= 60 (e.g. 60, 300, 600)")
    return window_seconds


def window_start(ts: datetime, anchor: datetime,
                 window_seconds=DEFAULT_WINDOW_SECONDS) -> datetime:
    """Start of the window containing `ts` on the grid that begins at `anchor`."""
    step = timedelta(seconds=window_seconds)
    return anchor + ((ts - anchor) // step) * step


def load_rows(store):
    """(ts, app, event, value, session_id) for every event, in time order."""
    return [(datetime.fromisoformat(ts), app, event, value, sid)
            for ts, app, event, value, sid in store.conn.execute(
                "SELECT ts, app, event, value, session_id FROM events ORDER BY ts, id")]


def build_timeline(rows, live_until: Optional[datetime] = None) -> _Timeline:
    """Reconstruct observed runs, app stays, idle episodes and session spans.

    `live_until`: the caller IS the running logger, so a run not yet ended is
    known to be observed up to that moment (not just up to its last row).
    """
    tl = _Timeline()
    run_start = last_ts = cur_app = app_since = pending_idle = None
    open_sessions = {}
    latest = None

    def close_run(end, still_open=False):
        nonlocal run_start, cur_app, app_since, pending_idle
        if run_start is None:
            return
        tl.starts.append(run_start)
        if still_open:
            tl.open_run = run_start
            if live_until is not None:
                end = max(end or run_start, live_until)
        end = max(end or run_start, run_start)
        if cur_app is not None and end > app_since:
            tl.apps.append((app_since, end, cur_app))
        if pending_idle is not None and end > pending_idle:
            tl.idle.append((pending_idle, end))
        if end > run_start:
            tl.runs.append((run_start, end))
            tl.end = end
        run_start = cur_app = app_since = pending_idle = None

    for ts, app, event, value, sid in rows:
        latest = ts if latest is None else max(latest, ts)
        if event in _SESSION_OPEN and sid is not None:
            open_sessions[sid] = ts
            continue
        if event in _SESSION_CLOSE and sid is not None:
            if sid in open_sessions:
                tl.sessions.append((open_sessions.pop(sid), ts, sid))
            continue
        if event == "agent_start":
            close_run(last_ts)                 # previous run crashed: ends at its last row
            run_start = last_ts = ts
            continue
        if run_start is None:                  # data without agent_start (old logs)
            run_start = ts
        if event == "agent_stop":
            close_run(ts)
            last_ts = None
            continue
        last_ts = ts
        if app and app != cur_app:
            if cur_app is not None and ts > app_since:
                tl.apps.append((app_since, ts, cur_app))
            cur_app, app_since = app, ts
        if event == "idle":
            if pending_idle is None:
                pending_idle = ts
        elif event == "idle_duration":
            start = max(ts - timedelta(seconds=int(value)), run_start)
            if ts > start:
                tl.idle.append((start, ts))
            pending_idle = None
        elif pending_idle is not None and event in _INPUT_SIGNS:
            if ts > pending_idle:
                tl.idle.append((pending_idle, ts))
            pending_idle = None
        if event in ("switched", "site_switched", "keyboard_count", "mouse_count"):
            tl.points.append((ts, event, int(value)))
    close_run(last_ts, still_open=True)
    if tl.open_run is not None and live_until is not None:
        latest = max(latest, live_until)
    for sid, start in open_sessions.items():   # still-open session: up to the data end
        if latest and latest > start:
            tl.sessions.append((start, latest, sid))
    tl.sessions.sort()
    return tl


def _intersect(a, b):
    """Intersection of two lists of non-overlapping (start, end, *tag) intervals
    (linear merge after sorting); keeps b's tag."""
    a, b, out, i, j = sorted(a), sorted(b), [], 0, 0
    while i < len(a) and j < len(b):
        (s1, e1), (s2, e2, *tag) = a[i][:2], b[j]
        s, e = max(s1, s2), min(e1, e2)
        if s < e:
            out.append((s, e, *tag))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return out


def _pieces(start, end, anchor, w):
    """Split [start, end) at window boundaries: yields (window_start, seconds)."""
    t = start
    while t < end:
        ws = window_start(t, anchor, w)
        piece_end = min(end, ws + timedelta(seconds=w))
        yield ws, int((piece_end - t).total_seconds())
        t = piece_end


def aggregate_windows(rows, window_seconds=DEFAULT_WINDOW_SECONDS,
                      live_until: Optional[datetime] = None) -> List[Window]:
    """One Window per window (anchored at each logger start) with data, in order.

    Windows the agent never observed are omitted (missing data, not zeros).
    Partly observed windows are kept with coverage < 1; the window still in
    progress in a run that has not ended has complete=False.
    """
    w = validate_window_seconds(window_seconds)
    tl = build_timeline(rows, live_until)
    anchors = {}                                   # window_start -> its run's anchor

    def pieces(s, e):
        a = tl.anchor(s)
        for ws, n in _pieces(s, e, a, w):
            anchors[ws] = a
            yield ws, n

    def at(ts):
        a = tl.anchor(ts)
        ws = window_start(ts, a, w)
        anchors[ws] = a
        return ws

    acc = defaultdict(lambda: {"observed": 0, "idle": 0, "idle_episodes": 0,
                               "apps": defaultdict(int), "sessions": defaultdict(int),
                               "switched": 0, "site_switched": 0,
                               "keyboard_count": 0, "mouse_count": 0})

    for s, e in tl.runs:
        for ws, n in pieces(s, e):
            acc[ws]["observed"] += n
    for s, e in _intersect(tl.runs, tl.idle):
        acc[at(s)]["idle_episodes"] += 1
        for ws, n in pieces(s, e):
            acc[ws]["idle"] += n
    for s, e, app in tl.apps:
        for ws, n in pieces(s, e):
            acc[ws]["apps"][app] += n
    for s, e, sid in _intersect(tl.runs, tl.sessions):
        for ws, n in pieces(s, e):
            acc[ws]["sessions"][sid] += n
    for ts, event, value in tl.points:
        a = acc[at(ts)]
        a[event] += 1 if event in ("switched", "site_switched") else value

    windows = []
    for ws in sorted(acc):
        a = acc[ws]
        observed = min(a["observed"], w)
        if observed == 0 and not any(a[k] for k in ("keyboard_count", "mouse_count",
                                                    "switched", "site_switched")):
            continue
        idle = min(a["idle"], observed)
        sid, sid_secs = None, 0
        if a["sessions"]:
            best = max(a["sessions"].items(), key=lambda kv: (kv[1], -kv[0]))
            sid_secs = best[1]
            if observed and best[1] * 2 >= observed:   # majority of observed time
                sid = best[0]
        apps = {k: v for k, v in sorted(a["apps"].items(), key=lambda kv: (-kv[1], kv[0]))
                if v > 0}
        we = ws + timedelta(seconds=w)
        windows.append(Window(
            window_start=ws, window_end=we, window_seconds=w,
            session_id=sid, session_seconds=sid_secs,
            observed_seconds=observed, coverage=round(observed / w, 3),
            complete=(tl.open_run is None
                      or anchors[ws] != tl.open_run.replace(microsecond=0)
                      or we <= tl.end),
            active_seconds=observed - idle, idle_seconds=idle,
            idle_episodes=a["idle_episodes"],
            switch_count=a["switched"], site_switch_count=a["site_switched"],
            switches_per_min=round(a["switched"] / (observed / 60), 3) if observed else None,
            distinct_apps=len(apps), app_seconds=apps,
            keyboard_count=a["keyboard_count"], mouse_count=a["mouse_count"],
        ))
    return windows


# --------------------------------------------------------------------------
# Persistence: the `windows` table is a rebuildable cache of the above
# --------------------------------------------------------------------------
_STORED = ("window_start", "window_seconds", "session_id", "session_seconds",
           "observed_seconds", "coverage", "complete", "active_seconds",
           "idle_seconds", "idle_episodes", "switch_count", "site_switch_count",
           "switches_per_min", "distinct_apps", "app_seconds", "keyboard_count",
           "mouse_count")


def _to_row(w: Window):
    d = asdict(w)
    d["window_start"] = w.window_start.isoformat(timespec="seconds")
    d["complete"] = int(w.complete)
    d["app_seconds"] = json.dumps(w.app_seconds)
    return tuple(d[k] for k in _STORED)


def _from_row(row) -> Window:
    d = dict(zip(_STORED, row))
    d["window_start"] = datetime.fromisoformat(d["window_start"])
    d["window_end"] = d["window_start"] + timedelta(seconds=d["window_seconds"])
    d["complete"] = bool(d["complete"])
    d["app_seconds"] = json.loads(d["app_seconds"])
    return Window(**d)


def refresh_windows(store, window_seconds=DEFAULT_WINDOW_SECONDS, now=None,
                    live=False) -> List[Window]:
    """Re-aggregate all events and REPLACE the stored windows of this length.

    `live=True` only from the running logger itself: its current run counts
    as observed up to `now`. One transaction, so readers never see a
    half-written table. Labels live in `window_labels` and are never touched.
    """
    now = now or datetime.now()
    windows = aggregate_windows(load_rows(store), window_seconds,
                                live_until=now if live else None)
    computed_at = now.isoformat(timespec="seconds")
    placeholders = ", ".join("?" * (len(_STORED) + 1))
    with store.conn:
        store.conn.execute("DELETE FROM windows WHERE window_seconds = ?", (window_seconds,))
        store.conn.executemany(
            f"INSERT INTO windows ({', '.join(_STORED)}, computed_at) VALUES ({placeholders})",
            [_to_row(w) + (computed_at,) for w in windows])
    return windows


def stored_windows(store, window_seconds=DEFAULT_WINDOW_SECONDS,
                   session_id=None) -> List[Window]:
    """Windows from the table (as last refreshed), oldest first."""
    sql = f"SELECT {', '.join(_STORED)} FROM windows WHERE window_seconds = ?"
    args = [window_seconds]
    if session_id is not None:
        sql += " AND session_id = ?"
        args.append(session_id)
    return [_from_row(r) for r in store.conn.execute(sql + " ORDER BY window_start", args)]


def get_window(store, start: datetime, window_seconds=DEFAULT_WINDOW_SECONDS):
    """The stored window starting exactly at `start`, or None."""
    row = store.conn.execute(
        f"SELECT {', '.join(_STORED)} FROM windows WHERE window_start = ? AND window_seconds = ?",
        (start.isoformat(timespec="seconds"), window_seconds)).fetchone()
    return _from_row(row) if row else None


def find_window(store, when: datetime, window_seconds=DEFAULT_WINDOW_SECONDS):
    """The stored window containing `when` (start <= when < end), or None."""
    row = store.conn.execute(
        f"SELECT {', '.join(_STORED)} FROM windows WHERE window_seconds = ?"
        " AND window_start <= ? ORDER BY window_start DESC LIMIT 1",
        (window_seconds, when.isoformat(timespec="seconds"))).fetchone()
    w = _from_row(row) if row else None
    return w if w is not None and when < w.window_end else None
