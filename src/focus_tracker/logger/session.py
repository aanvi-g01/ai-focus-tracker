"""Focus sessions: the user's DECLARED intention for a stretch of time.

    focus-session start "studying DSA"    # optional: --category studying
    focus-session pause | resume | end | status

All state lives in the `sessions` table of the same database the agent writes,
so a session survives restarts of the agent or the machine, and the CLI can be
used from another terminal while the agent runs. The agent stamps every event
row with the id of the session that is ACTIVE at that moment (none while paused).

A session records what the user *intended* to do (its name is stored in the
`task` column). It is context for the later classifier, never a verdict:
nothing here says whether the user was focused. Focus labels are given by the
user per window, separately (see the labels step).

Unfinished sessions: a session left open while the agent was not running for
longer than STALE_AFTER_SECONDS (forgotten, crash, shutdown) is closed by the
agent on its next start, at the time of the session's last activity, with
end_reason = 'recovered'. Shorter breaks (a reboot) simply continue it.
"""
import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .storage import EventStore

MAX_TASK_CHARS = 80
STALE_AFTER_SECONDS = 2 * 60 * 60   # agent-off gap after which an open session is closed

TASK_CATEGORIES = ("studying", "coding", "research", "writing", "other")

_TASK_KEYWORDS = {
    "studying": ("study", "studying", "revise", "revision", "exam", "exams",
                 "homework", "assignment", "lecture", "lectures", "course",
                 "learn", "learning", "dsa", "notes", "practice", "chapter"),
    "coding": ("code", "coding", "program", "programming", "debug", "debugging",
               "implement", "implementing", "develop", "developing", "refactor",
               "bug", "bugs", "feature", "build", "building"),
    "research": ("research", "researching", "paper", "papers", "literature",
                 "survey", "investigate", "experiment", "experiments"),
    "writing": ("write", "writing", "essay", "report", "thesis", "draft",
                "blog", "article", "documentation"),
}


class SessionError(ValueError):
    """Invalid lifecycle step, e.g. pausing when no session is active."""


@dataclass(frozen=True)
class Session:
    id: int
    task: str
    task_category: str
    started_at: datetime
    ended_at: Optional[datetime]
    status: str
    paused_at: Optional[datetime]
    paused_seconds: int
    end_reason: Optional[str] = None   # 'user' | 'recovered' once ended

    def active_seconds(self, now: datetime) -> int:
        """Wall time since start minus all paused time (incl. a pause in progress)."""
        end = self.ended_at or now
        paused = self.paused_seconds
        if self.status == "paused" and self.paused_at:
            paused += _secs(self.paused_at, now)
        return max(0, _secs(self.started_at, end) - paused)


def _secs(a: datetime, b: datetime) -> int:
    return max(0, int((b - a).total_seconds()))


def _ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _dt(text):
    return datetime.fromisoformat(text) if text else None


def clean_task(task) -> str:
    """Collapse whitespace, drop control characters, cap the length."""
    if not isinstance(task, str):
        raise SessionError("task must be text")
    task = "".join(ch for ch in task if ch.isprintable() or ch.isspace())
    task = " ".join(task.split())[:MAX_TASK_CHARS].strip()
    if not task:
        raise SessionError("task must not be empty")
    return task


def classify_task(task: str) -> str:
    """Broad category of a declared task; the earliest keyword in the text wins
    ("write code for X" -> writing, "coding DSA problems" -> coding)."""
    text = task.lower()
    best, best_pos = "other", len(text) + 1
    for cat, words in _TASK_KEYWORDS.items():
        m = re.search(r"(?<![a-z0-9])(?:" + "|".join(words) + r")(?![a-z0-9])", text)
        if m and m.start() < best_pos:
            best, best_pos = cat, m.start()
    return best


class SessionManager:
    def __init__(self, store: EventStore):
        self.store = store
        self.conn = store.conn

    def _row_to_session(self, row):
        if row is None:
            return None
        sid, task, cat, started, ended, status, paused_at, paused_secs, reason = row
        return Session(sid, task, cat, _dt(started), _dt(ended), status,
                       _dt(paused_at), paused_secs, reason)

    _COLS = ("id, task, task_category, started_at, ended_at, status, "
             "paused_at, paused_seconds, end_reason")

    def get(self, session_id) -> Optional[Session]:
        return self._row_to_session(self.conn.execute(
            f"SELECT {self._COLS} FROM sessions WHERE id = ?", (session_id,)).fetchone())

    def current(self) -> Optional[Session]:
        """The open (active or paused) session, if any."""
        return self._row_to_session(self.conn.execute(
            f"SELECT {self._COLS} FROM sessions WHERE status != 'ended'").fetchone())

    def _require(self, *statuses) -> Session:
        s = self.current()
        if s is None or s.status not in statuses:
            state = "no open session" if s is None else f"session is {s.status}"
            raise SessionError(f"cannot do that: {state}")
        return s

    def start(self, task, now: datetime, category=None) -> Session:
        task = clean_task(task)
        if category is None:
            category = classify_task(task)
        elif category not in TASK_CATEGORIES:
            raise SessionError(f"category must be one of {', '.join(TASK_CATEGORIES)}")
        if self.current() is not None:
            raise SessionError("a session is already open; end it first")
        try:
            cur = self.conn.execute(
                "INSERT INTO sessions (task, task_category, started_at, status)"
                " VALUES (?, ?, ?, 'active')", (task, category, _ts(now)))
        except sqlite3.IntegrityError as e:  # unique index: another process won the race
            self.conn.rollback()
            raise SessionError("a session is already open; end it first") from e
        self.conn.commit()
        self.store.log(now, None, "session_start", session_id=cur.lastrowid)
        return self.get(cur.lastrowid)

    def pause(self, now: datetime) -> Session:
        s = self._require("active")
        self.conn.execute("UPDATE sessions SET status = 'paused', paused_at = ? WHERE id = ?",
                          (_ts(now), s.id))
        self.conn.commit()
        self.store.log(now, None, "session_pause", session_id=s.id)
        return self.get(s.id)

    def resume(self, now: datetime) -> Session:
        s = self._require("paused")
        self.conn.execute(
            "UPDATE sessions SET status = 'active', paused_at = NULL,"
            " paused_seconds = paused_seconds + ? WHERE id = ?",
            (_secs(s.paused_at, now), s.id))
        self.conn.commit()
        self.store.log(now, None, "session_resume", session_id=s.id)
        return self.get(s.id)

    def end(self, now: datetime) -> Session:
        return self._close(self._require("active", "paused"), now, "user")

    def _close(self, s: Session, at: datetime, reason: str) -> Session:
        extra = _secs(s.paused_at, at) if s.status == "paused" else 0
        self.conn.execute(
            "UPDATE sessions SET status = 'ended', ended_at = ?, paused_at = NULL,"
            " paused_seconds = paused_seconds + ?, end_reason = ? WHERE id = ?",
            (_ts(at), extra, reason, s.id))
        self.conn.commit()
        ended = self.get(s.id)
        self.store.log(at, None, "session_end", ended.active_seconds(at), session_id=s.id)
        return ended

    def last_activity(self, s: Session) -> datetime:
        """Latest event recorded for the session (falls back to its start)."""
        row = self.conn.execute(
            "SELECT MAX(ts) FROM events WHERE session_id = ?", (s.id,)).fetchone()
        last = _dt(row[0]) if row else None
        return max(last, s.started_at) if last else s.started_at

    def recover_stale(self, now: datetime,
                      stale_after=STALE_AFTER_SECONDS) -> Optional[Session]:
        """Close an open session left behind while nothing was being recorded.

        Call ONLY when the agent starts, i.e. when we know it was not running:
        while it runs, a long idle stretch also produces no rows. If the latest
        event of any kind is older than `stale_after` seconds, the open session
        is ended at its own last activity (not `now`), so the off-time is never
        counted, and marked end_reason = 'recovered'. Returns it, or None.
        """
        s = self.current()
        last_any = self.store.last_event_ts()
        if s is None or last_any is None or _secs(last_any, now) < stale_after:
            return None
        return self._close(s, self.last_activity(s), "recovered")


def _describe(s: Session, now: datetime) -> str:
    mins, secs = divmod(s.active_seconds(now), 60)
    note = " (auto-closed: unfinished)" if s.end_reason == "recovered" else ""
    return (f"#{s.id} [{s.status}] {s.task!r} ({s.task_category}) "
            f"- {mins}m {secs:02d}s active{note}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="focus-session",
                                description="Start, pause, resume and end focus sessions")
    p.add_argument("--db", default="data/activity.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start", help="start a session with your intended task")
    st.add_argument("task", help='what you intend to do, e.g. "studying DSA"')
    st.add_argument("--category", choices=TASK_CATEGORIES,
                    help="override the automatically detected task category")
    for name in ("pause", "resume", "end", "status"):
        sub.add_parser(name)
    args = p.parse_args(argv)

    store = EventStore(args.db)
    mgr, now = SessionManager(store), datetime.now()
    try:
        if args.cmd == "start":
            s = mgr.start(args.task, now, category=args.category)
            print("Started", _describe(s, now))
        elif args.cmd == "status":
            s = mgr.current()
            print(_describe(s, now) if s else "No open session.")
        else:
            s = getattr(mgr, args.cmd)(now)
            print(f"{args.cmd.capitalize()}d" if args.cmd != "end" else "Ended",
                  _describe(s, now))
    except SessionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
