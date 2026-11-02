"""SQLite event store.

Privacy by construction: the schema has no column that can hold a key,
a window title or screen content, and event names are whitelisted.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

# Everything the logger is allowed to write. Nothing else is accepted.
ALLOWED_EVENTS = {
    "agent_start", "agent_stop",
    "active",          # first window seen / user returned from idle
    "switched",        # foreground application changed
    "idle",            # no input for >= idle threshold
    "keyboard_count",  # number of key events in the interval (no identity)
    "mouse_count",     # number of mouse events in the interval
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    TEXT    NOT NULL,   -- ISO-8601 local time
    app   TEXT,               -- process name only, e.g. 'Code.exe'
    event TEXT    NOT NULL,
    value INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class EventStore:
    def __init__(self, path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)

    def log(self, ts: datetime, app, event: str, value: int = 1):
        if event not in ALLOWED_EVENTS:
            raise ValueError(f"event '{event}' is not allowed by the privacy policy")
        self.conn.execute(
            "INSERT INTO events (ts, app, event, value) VALUES (?, ?, ?, ?)",
            (ts.isoformat(timespec="seconds"), app, event, int(value)),
        )
        self.conn.commit()

    def rows(self):
        return self.conn.execute(
            "SELECT ts, app, event, value FROM events ORDER BY id"
        ).fetchall()

    def close(self):
        self.conn.close()
