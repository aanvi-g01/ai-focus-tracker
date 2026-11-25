"""Self-labelling of windows: the user says Focused, Distracted or Inactive.

    focus-label list                     # today's windows and their labels
    focus-label list --session 3 --unlabelled
    focus-label set 10:05 focused        # add, or correct, a label
    focus-label set "2026-11-16 10:05" Distracted
    focus-label clear 10:05

Labels are the ONLY source of focus states in this project, and only the user
creates them. Nothing here reads session names or categories, and no label is
ever inferred: a session called "Studying DSA" does not make a window Focused.

Labels live in `window_labels`, keyed by (window_start, window_seconds), apart
from the rebuildable `windows` table, so recomputing features never loses them.
A correction overwrites the label (updated_at changes, labelled_at is kept).
"""
import argparse
import sys
from datetime import date, datetime, time
from typing import List, Optional, Tuple

from ..features.windows import (DEFAULT_WINDOW_SECONDS, Window, find_window,
                                refresh_windows, stored_windows, validate_window_seconds)
from .storage import EventStore

LABELS = ("Focused", "Distracted", "Inactive")


class LabelError(ValueError):
    """Bad label, bad time, or a window that cannot be labelled."""


def _ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def normalize_label(text) -> str:
    """Case-insensitive: 'focused' -> 'Focused'."""
    for label in LABELS:
        if isinstance(text, str) and text.strip().lower() == label.lower():
            return label
    raise LabelError(f"label must be one of {', '.join(LABELS)}")


def parse_when(text: str, today: date) -> datetime:
    """'10:05' (today) or '2026-11-16 10:05' / '2026-11-16T10:05'."""
    text = (text or "").strip()
    try:
        if len(text) <= 8 and ":" in text and "-" not in text:
            return datetime.combine(today, time.fromisoformat(text))
        return datetime.fromisoformat(text)
    except ValueError:
        raise LabelError(f"cannot read time {text!r}; use HH:MM or YYYY-MM-DD HH:MM") from None


def get_label(store, start: datetime, window_seconds=DEFAULT_WINDOW_SECONDS) -> Optional[str]:
    row = store.conn.execute(
        "SELECT label FROM window_labels WHERE window_start = ? AND window_seconds = ?",
        (_ts(start), window_seconds)).fetchone()
    return row[0] if row else None


LABEL_SOURCES = ("user", "accepted")


def set_label(store, when: datetime, label, now: datetime,
              window_seconds=DEFAULT_WINDOW_SECONDS,
              source="user") -> Tuple[Window, Optional[str]]:
    """Label the window containing `when`. Returns (window, previous label).

    `source` is "user" when the person chose the label and "accepted" when they
    confirmed the model's guess. Both are the person's decision; accepted ones
    are kept apart so models are never trained on their own guesses.

    The window must exist in the `windows` table (refresh first) and be
    complete: a window still in progress may change after being labelled.
    """
    label = normalize_label(label)
    if source not in LABEL_SOURCES:
        raise LabelError(f"source must be one of {', '.join(LABEL_SOURCES)}")
    w = find_window(store, when, window_seconds)
    if w is None:
        raise LabelError(f"no recorded activity in a window at {when:%Y-%m-%d %H:%M:%S}")
    start = w.window_start
    if not w.complete:
        raise LabelError(f"the window starting {start:%H:%M:%S} is still in progress")
    previous = get_label(store, start, window_seconds)
    with store.conn:
        store.conn.execute(
            "INSERT INTO window_labels (window_start, window_seconds, session_id, label,"
            " labelled_at, updated_at, source) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (window_start, window_seconds) DO UPDATE SET"
            " label = excluded.label, session_id = excluded.session_id,"
            " updated_at = excluded.updated_at, source = excluded.source",
            (_ts(start), window_seconds, w.session_id, label, _ts(now), _ts(now), source))
    return w, previous


def clear_label(store, when: datetime, window_seconds=DEFAULT_WINDOW_SECONDS) -> bool:
    """Remove the label of the window containing `when`. True if one existed."""
    w = find_window(store, when, window_seconds)
    if w is None:
        return False
    with store.conn:
        cur = store.conn.execute(
            "DELETE FROM window_labels WHERE window_start = ? AND window_seconds = ?",
            (_ts(w.window_start), window_seconds))
    return cur.rowcount > 0


def labelled_windows(store, window_seconds=DEFAULT_WINDOW_SECONDS, session_id=None,
                     on_date: Optional[date] = None,
                     unlabelled_only=False) -> List[Tuple[Window, Optional[str]]]:
    """Stored windows paired with their label (None if unlabelled)."""
    labels = {datetime.fromisoformat(ws): label for ws, label in store.conn.execute(
        "SELECT window_start, label FROM window_labels WHERE window_seconds = ?",
        (window_seconds,))}
    out = []
    for w in stored_windows(store, window_seconds, session_id=session_id):
        if on_date is not None and w.window_start.date() != on_date:
            continue
        label = labels.get(w.window_start)
        if unlabelled_only and label is not None:
            continue
        out.append((w, label))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _describe(w: Window, label: Optional[str]) -> str:
    top = next(iter(w.app_seconds), "-")
    notes = []
    if not w.complete:
        notes.append("in progress")
    elif w.coverage < 1:
        notes.append(f"partial {w.coverage:.0%}")
    session = f"#{w.session_id}" if w.session_id is not None else "-"
    return (f"{w.window_start:%Y-%m-%d %H:%M:%S}-{w.window_end:%H:%M:%S}  {session:>4}  "
            f"active {w.active_seconds:>3}s idle {w.idle_seconds:>3}s  "
            f"switches {w.switch_count:>2}  keys {w.keyboard_count:>4} mouse {w.mouse_count:>5}  "
            f"{top:<16} {label or '(unlabelled)'}"
            + (f"  [{', '.join(notes)}]" if notes else ""))


def main(argv=None, now: Optional[datetime] = None):
    p = argparse.ArgumentParser(prog="focus-label",
                                description="Label your activity windows as "
                                            + " / ".join(LABELS))
    p.add_argument("--db", default="data/activity.db")
    p.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="show windows (default: today)")
    ls.add_argument("--session", type=int, help="only windows of this session id")
    ls.add_argument("--date", type=date.fromisoformat, help="YYYY-MM-DD")
    ls.add_argument("--all", action="store_true", help="every date")
    ls.add_argument("--unlabelled", action="store_true")
    st = sub.add_parser("set", help="add or correct the label of a window")
    st.add_argument("when", help='any time inside the window, e.g. 10:05 or "2026-11-16 10:05"')
    st.add_argument("label", help=" | ".join(LABELS) + " (any case)")
    cl = sub.add_parser("clear", help="remove the label of a window")
    cl.add_argument("when", help="any time inside the window")
    args = p.parse_args(argv)

    now = now or datetime.now()
    store = EventStore(args.db)
    try:
        validate_window_seconds(args.window_seconds)
        refresh_windows(store, args.window_seconds, now)    # always label current data
        if args.cmd == "list":
            on_date = None if (args.all or args.session is not None) else (args.date or now.date())
            rows = labelled_windows(store, args.window_seconds, args.session, on_date,
                                    args.unlabelled)
            for w, label in rows:
                print(_describe(w, label))
            if not rows:
                print("No windows to show.")
        elif args.cmd == "set":
            w, previous = set_label(store, parse_when(args.when, now.date()), args.label, now,
                                    args.window_seconds)
            new = get_label(store, w.window_start, args.window_seconds)
            verb = f"Corrected {previous} ->" if previous and previous != new else "Labelled"
            print(f"{verb} {new}: {_describe(w, new)}")
        else:
            when = parse_when(args.when, now.date())
            print("Label removed." if clear_label(store, when, args.window_seconds)
                  else "That window had no label.")
    except (LabelError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
