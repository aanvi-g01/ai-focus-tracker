"""SQLite event store.

Privacy by construction: the schema has no column that can hold a key,
a window title or screen content, and event names are whitelisted.
The only free-text columns beyond the app name are:
- `events.website`: a bare hostname (validated on every write), browsers only;
- `sessions.task`: the short intention the user typed themselves when starting
  a focus session (e.g. "studying DSA").
- `events.content_title`: the cleaned video title, ONLY when `website` is a
  video-streaming platform (content.VIDEO_HOSTS) and the user opted in.
`events.content_category` holds one word from content.CONTENT_CATEGORIES;
`events.session_id` links a row to the session that was active.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

from .browser import is_browser, is_hostname
from .content import CONTENT_CATEGORIES, is_clean_title, is_video_host

# Everything the logger is allowed to write. Nothing else is accepted.
ALLOWED_EVENTS = {
    "agent_start", "agent_stop",
    "active",          # first window seen / user returned from idle
    "switched",        # foreground application changed
    "site_switched",   # website changed while the same browser stayed in front
    "content_changed", # content category changed on the same website
    "idle",            # no input for >= idle threshold (episode started)
    "idle_duration",   # episode ended: value = total idle SECONDS (a number only)
    "keyboard_count",  # number of key events in the interval (no identity)
    "mouse_count",     # number of mouse events in the interval
    "session_start", "session_pause", "session_resume",
    "session_end",     # value = active (unpaused) SECONDS of the session
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    TEXT    NOT NULL,   -- ISO-8601 local time
    app   TEXT,               -- process name only, e.g. 'Code.exe'
    event TEXT    NOT NULL,
    value INTEGER NOT NULL DEFAULT 1,
    website TEXT,             -- hostname only, e.g. 'github.com'; NULL unless app is a browser
    session_id INTEGER,       -- sessions.id while a session is ACTIVE (NULL when none / paused)
    content_category TEXT,    -- one word, e.g. 'educational'; NULL unless app is a browser
    content_title TEXT        -- video title; NULL unless website is a video platform
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task           TEXT    NOT NULL,  -- user-declared intention, <= 80 chars
    task_category  TEXT    NOT NULL,  -- e.g. 'studying', 'coding'
    started_at     TEXT    NOT NULL,
    ended_at       TEXT,
    status         TEXT    NOT NULL CHECK (status IN ('active', 'paused', 'ended')),
    paused_at      TEXT,              -- set while paused
    paused_seconds INTEGER NOT NULL DEFAULT 0,
    end_reason     TEXT               -- 'user' | 'recovered' (closed at agent start)
);
-- At most one open (active or paused) session, even across processes.
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_open_session
    ON sessions(status = 'ended') WHERE status != 'ended';

-- Derived: one row per clock-aligned window (features/windows.py). Fully
-- rebuildable from `events` at any time; never edited by hand.
CREATE TABLE IF NOT EXISTS windows (
    window_start      TEXT    NOT NULL,   -- ISO-8601 local time, aligned
    window_seconds    INTEGER NOT NULL,
    session_id        INTEGER,            -- majority session, NULL if none
    session_seconds   INTEGER NOT NULL,
    observed_seconds  INTEGER NOT NULL,
    coverage          REAL    NOT NULL,
    complete          INTEGER NOT NULL,   -- 0 while the window may still grow
    active_seconds    INTEGER NOT NULL,
    idle_seconds      INTEGER NOT NULL,
    idle_episodes     INTEGER NOT NULL,
    switch_count      INTEGER NOT NULL,
    site_switch_count INTEGER NOT NULL,
    switches_per_min  REAL,
    distinct_apps     INTEGER NOT NULL,
    app_seconds       TEXT    NOT NULL,   -- JSON {"Code.exe": 240, ...}
    keyboard_count    INTEGER NOT NULL,
    mouse_count       INTEGER NOT NULL,
    computed_at       TEXT    NOT NULL,
    PRIMARY KEY (window_start, window_seconds)
);

-- User-given ground truth (logger/labels.py). Kept apart from `windows` so a
-- rebuild never touches it, and from `sessions` so a session name never
-- implies a label. Only the latest label per window is kept.
CREATE TABLE IF NOT EXISTS window_labels (
    window_start   TEXT    NOT NULL,
    window_seconds INTEGER NOT NULL,
    session_id     INTEGER,               -- the window's session when labelled
    label          TEXT    NOT NULL CHECK (label IN ('Focused', 'Distracted', 'Inactive')),
    labelled_at    TEXT    NOT NULL,      -- first labelled
    updated_at     TEXT    NOT NULL,      -- last added or corrected
    source         TEXT    NOT NULL DEFAULT 'user'
                   CHECK (source IN ('user', 'accepted')),  -- typed, or model guess accepted
    PRIMARY KEY (window_start, window_seconds)
);
"""

# Columns added after the first release: name -> SQL type.
_ADDED_COLUMNS = {"website": "TEXT", "session_id": "INTEGER",
                  "content_category": "TEXT", "content_title": "TEXT"}


class EventStore:
    def __init__(self, path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        # The agent and the `focus-session` CLI share this file; wait on locks.
        self.conn = sqlite3.connect(str(path), timeout=10)
        self._migrate()
        self.conn.executescript(SCHEMA)
        self._migrate_sessions()

    def _migrate_sessions(self):
        """Session tables created before end_reason existed get the column; label
        tables created before `source` existed get it (old labels were all typed)."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(sessions)")]
        if "end_reason" not in cols:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN end_reason TEXT")
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(window_labels)")]
        if "source" not in cols:
            self.conn.execute("ALTER TABLE window_labels ADD COLUMN source TEXT NOT NULL"
                              " DEFAULT 'user' CHECK (source IN ('user', 'accepted'))")
        self.conn.commit()

    def _migrate(self):
        """Older databases get the new columns (existing rows keep NULL)."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(events)")]
        if not cols:
            return  # fresh database: SCHEMA creates everything
        for name, typ in _ADDED_COLUMNS.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE events ADD COLUMN {name} {typ}")
        self.conn.commit()

    def log(self, ts: datetime, app, event: str, value: int = 1, website=None,
            session_id=None, content_category=None, content_title=None):
        if event not in ALLOWED_EVENTS:
            raise ValueError(f"event '{event}' is not allowed by the privacy policy")
        if website is not None:
            if not is_browser(app):
                raise ValueError("website must be NULL for non-browser applications")
            if not is_hostname(website):
                raise ValueError("website must be a bare hostname (no path or query)")
        if content_category is not None:
            if not is_browser(app):
                raise ValueError("content_category must be NULL for non-browser applications")
            if content_category not in CONTENT_CATEGORIES:
                raise ValueError("content_category must be one of CONTENT_CATEGORIES")
        if content_title is not None:
            if website is None or not is_video_host(website):
                raise ValueError("content_title is only allowed on video-streaming websites")
            if not is_clean_title(content_title):
                raise ValueError("content_title must be one printable line of "
                                 "at most MAX_TITLE_CHARS characters")
        if session_id is not None and (isinstance(session_id, bool)
                                       or not isinstance(session_id, int)):
            raise ValueError("session_id must be an integer")
        self.conn.execute(
            "INSERT INTO events (ts, app, event, value, website, session_id,"
            " content_category, content_title) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts.isoformat(timespec="seconds"), app, event, int(value), website,
             session_id, content_category, content_title),
        )
        self.conn.commit()

    def active_session_id(self):
        """Id of the session currently ACTIVE (not paused), or None.

        Read from the database each time, so a session started/paused by the
        CLI in another process - or before a restart - is picked up.
        """
        row = self.conn.execute(
            "SELECT id FROM sessions WHERE status = 'active'").fetchone()
        return row[0] if row else None

    def last_event_ts(self):
        """Timestamp (datetime) of the latest event of any kind, or None."""
        row = self.conn.execute("SELECT MAX(ts) FROM events").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def rows(self, include_website=False, include_context=False):
        cols = "ts, app, event, value"
        if include_website:
            cols += ", website"
        if include_context:
            cols += ", session_id, content_category, content_title"
        return self.conn.execute(
            f"SELECT {cols} FROM events ORDER BY id"
        ).fetchall()

    def close(self):
        self.conn.close()
