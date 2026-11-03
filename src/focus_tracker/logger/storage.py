"""SQLite event store.

Privacy by construction: the schema has no column that can hold a key,
a window title or screen content, and event names are whitelisted.
The only text column beyond the app name is `website`, and it can only hold a
bare hostname (validated on every write) and only for browser apps.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

from .browser import is_browser, is_hostname

# Everything the logger is allowed to write. Nothing else is accepted.
ALLOWED_EVENTS = {
    "agent_start", "agent_stop",
    "active",          # first window seen / user returned from idle
    "switched",        # foreground application changed
    "site_switched",   # website changed while the same browser stayed in front
    "idle",            # no input for >= idle threshold (episode started)
    "idle_duration",   # episode ended: value = total idle SECONDS (a number only)
    "keyboard_count",  # number of key events in the interval (no identity)
    "mouse_count",     # number of mouse events in the interval
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    TEXT    NOT NULL,   -- ISO-8601 local time
    app   TEXT,               -- process name only, e.g. 'Code.exe'
    event TEXT    NOT NULL,
    value INTEGER NOT NULL DEFAULT 1,
    website TEXT              -- hostname only, e.g. 'github.com'; NULL unless app is a browser
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
        self._migrate()

    def _migrate(self):
        """Databases created before website tracking get the new column."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(events)")]
        if "website" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN website TEXT")
            self.conn.commit()

    def log(self, ts: datetime, app, event: str, value: int = 1, website=None):
        if event not in ALLOWED_EVENTS:
            raise ValueError(f"event '{event}' is not allowed by the privacy policy")
        if website is not None:
            if not is_browser(app):
                raise ValueError("website must be NULL for non-browser applications")
            if not is_hostname(website):
                raise ValueError("website must be a bare hostname (no path or query)")
        self.conn.execute(
            "INSERT INTO events (ts, app, event, value, website) VALUES (?, ?, ?, ?, ?)",
            (ts.isoformat(timespec="seconds"), app, event, int(value), website),
        )
        self.conn.commit()

    def rows(self, include_website=False):
        cols = "ts, app, event, value" + (", website" if include_website else "")
        return self.conn.execute(
            f"SELECT {cols} FROM events ORDER BY id"
        ).fetchall()

    def close(self):
        self.conn.close()
