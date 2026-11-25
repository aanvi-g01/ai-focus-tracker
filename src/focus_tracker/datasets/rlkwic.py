"""RLKWiC adapter: external knowledge-work logs -> our 5-minute window features.

    python -m focus_tracker.datasets.rlkwic --root data/external/RLKWiC --out <dir>

What the label is (and is not)
------------------------------
RLKWiC sessions carry `in_context`: whether the session's activity was judged
to belong to the work context the participant had selected. It is a
session-level RELEVANCE judgement, not a focus label: off-context work can be
focused work on another project. This adapter exports it as
`label_in_context` and never maps it to Focused / Distracted / Inactive.
`KW_action(s)` is never read: it is "Unknown" for every off-context session,
so it would leak the label.

Fixed design decisions (agreed before implementation)
------------------------------------------------------
* Gap threshold: RLKWiC has no logger start/stop or heartbeat. A gap of more
  than GAP_MINUTES (15) between consecutive events ends a logging run, so the
  quiet time is treated as unobserved instead of being invented as activity.
  `observed_seconds` / `coverage` are therefore DERIVED from this threshold.
* Sidebar events (reporter "cSpaces sidebar app": context switches, tags,
  searches, observation on/off) are excluded entirely: they cluster at session
  boundaries and would leak the session label.

Missing features are missing, never zero
----------------------------------------
RLKWiC records no idle time and no keyboard / mouse counts. Our aggregator
would report idle_seconds = 0, active_seconds = observed, counts = 0 for such
input, so those columns are overwritten with None (empty in the CSV).

Privacy
-------
Only `active_app`, timestamps, reporter and the URL of browser-companion
events are read. URLs are reduced to a hostname immediately
(logger.browser.normalize_site). Window titles, clipboard text, file and web
contents are never read or written.

The existing pipeline is reused unchanged: rows are built in the same
(ts, app, event, value, session_id) shape the logger database yields, then
passed to features.windows.aggregate_windows.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from ..features.windows import (DEFAULT_WINDOW_SECONDS, aggregate_windows,
                                build_timeline, window_start)
from ..logger.browser import normalize_site
from ..logger.content import CONTENT_CATEGORIES, classify_content

GAP_MINUTES = 15
SIDEBAR_REPORTER = "cSpaces sidebar app"
BROWSER_REPORTER = "cSpaces browser companion"
APP_REPORTERS = {"window tracker", "explorer tracker", "clipboard tracker"}
UNKNOWN_APPS = {"", "None", "default browser"}
# Not recorded by RLKWiC: exported as None, never 0.
MISSING_FEATURES = ("active_seconds", "idle_seconds", "idle_episodes",
                    "keyboard_count", "mouse_count")
CATEGORIES = tuple(sorted(CONTENT_CATEGORIES))
_EPOCH = datetime(1970, 1, 1)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class Event:
    ts: datetime          # participant-local wall clock
    reporter: str
    app: Optional[str]    # None when unknown
    host: Optional[str]   # browser-companion events only


@dataclass
class Session:
    key: str              # e.g. "p1-c3-s1"
    sid: int              # integer id for our pipeline (position in sessions.csv)
    context_id: str
    in_context: Optional[bool]
    start: datetime
    end: datetime


@dataclass
class Participant:
    name: str
    utc_offset_minutes: int
    events: List[Event] = field(default_factory=list)
    sessions: List[Session] = field(default_factory=list)
    clipped_sessions: int = 0
    excluded_sidebar_events: int = 0


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _read_csv(path: Path):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        yield from csv.DictReader(f)


def _utc(ms: str) -> datetime:
    return _EPOCH + timedelta(milliseconds=int(ms))


def _id_wall_clock(event_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(event_id[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def utc_offset_minutes(rows) -> int:
    """Participant's UTC offset: event ids encode local wall-clock time,
    timestamps are UTC epoch ms. Median difference, rounded to 15 minutes."""
    diffs = []
    for r in rows:
        local = _id_wall_clock(r["id"])
        if local is not None and r["timestamp_received"].isdigit():
            diffs.append((local - _utc(r["timestamp_received"])).total_seconds() / 60)
    if not diffs:
        raise ValueError("no event ids with a readable wall-clock time")
    return int(round(statistics.median(diffs) / 15) * 15)


def _app(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    return None if raw in UNKNOWN_APPS else raw


def _host(address: str) -> Optional[str]:
    if not address or not address.startswith(("http://", "https://")):
        return None
    host = normalize_site(address)
    if host and host.startswith("www."):
        host = host[4:]
    return host


def _parse_bool(text: str) -> Optional[bool]:
    t = (text or "").strip().lower()
    return True if t == "true" else False if t == "false" else None


def read_participant(folder: Path) -> Participant:
    folder = Path(folder)
    rows = [r for r in _read_csv(folder / "events.csv")
            if r.get("timestamp_received", "").isdigit()]
    offset = utc_offset_minutes(rows)
    p = Participant(folder.name, offset)
    shift = timedelta(minutes=offset)

    ts_by_id: Dict[str, datetime] = {}
    for r in rows:
        if r["reporter"] == SIDEBAR_REPORTER:
            p.excluded_sidebar_events += 1
            continue
        ts = _utc(r["timestamp_received"]) + shift
        ts_by_id[r["id"]] = ts
        if r["reporter"] == BROWSER_REPORTER:
            p.events.append(Event(ts, r["reporter"], None, _host(r["address"])))
        elif r["reporter"] in APP_REPORTERS:
            p.events.append(Event(ts, r["reporter"], _app(r["active_app"]), None))
    p.events.sort(key=lambda e: e.ts)

    for sid, r in enumerate(_read_csv(folder / "sessions.csv"), start=1):
        ids = [x.strip() for x in r["events_list"].strip("[] \n").split(",")]
        times = sorted(ts_by_id[i] for i in ids if i in ts_by_id)
        if not times:      # only sidebar events: nothing observable left
            continue
        p.sessions.append(Session(r["id"], sid, r["context_id"],
                                  _parse_bool(r["in_context"]), times[0], times[-1]))
    p.sessions.sort(key=lambda s: s.start)
    # Spans come from first/last event; interleaved sessions would overlap.
    # Our interval code needs disjoint spans: cut the earlier one short.
    for a, b in zip(p.sessions, p.sessions[1:]):
        if b.start < a.end:
            a.end = b.start
            p.clipped_sessions += 1
    return p


# --------------------------------------------------------------------------
# Rows in our pipeline's shape
# --------------------------------------------------------------------------
def split_runs(events: List[Event], gap_minutes=GAP_MINUTES) -> List[List[Event]]:
    """Consecutive events with gaps <= gap_minutes form one logging run."""
    gap = timedelta(minutes=gap_minutes)
    runs: List[List[Event]] = []
    for e in events:
        if runs and e.ts - runs[-1][-1].ts <= gap:
            runs[-1].append(e)
        else:
            runs.append([e])
    return runs


def build_rows(p: Participant, gap_minutes=GAP_MINUTES):
    """(ts, app, event, value, session_id) rows for features.windows."""
    rows = []
    for run in split_runs(p.events, gap_minutes):
        rows.append((run[0].ts, 0, None, "agent_start", 1, None))
        app = host = None
        for e in run:
            if e.reporter == BROWSER_REPORTER:
                # First readable host of a run is not a switch, as in our logger.
                if e.host is not None:
                    if host is not None and e.host != host:
                        rows.append((e.ts, 1, None, "site_switched", 1, None))
                    host = e.host
            elif e.app is not None and e.app != app:
                rows.append((e.ts, 1, e.app, "switched" if app else "active", 1, None))
                app = e.app
        rows.append((run[-1].ts, 2, None, "agent_stop", 1, None))
    for s in p.sessions:
        rows.append((s.start, 1, None, "session_start", 1, s.sid))
        rows.append((s.end, 1, None, "session_end", 1, s.sid))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [(ts, app, ev, val, sid) for ts, _, app, ev, val, sid in rows]


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def _ratio(n, d):
    return round(n / d, 4) if d else None


def participant_windows(p: Participant, gap_minutes=GAP_MINUTES,
                        window_seconds=DEFAULT_WINDOW_SECONDS) -> List[dict]:
    rows = build_rows(p, gap_minutes)
    windows = aggregate_windows(rows, window_seconds)
    tl = build_timeline(rows)

    # Dataset-only browser features, on the same window grid.
    browser = defaultdict(lambda: {"events": 0, "hosts": set(), "cats": Counter()})
    for e in p.events:
        if e.reporter != BROWSER_REPORTER:
            continue
        ws = window_start(e.ts, tl.anchor(e.ts), window_seconds)
        b = browser[ws]
        b["events"] += 1
        if e.host:
            b["hosts"].add(e.host)
        b["cats"][classify_content(e.host) if e.host else None] += 1

    sessions = {s.sid: s for s in p.sessions}
    out = []
    for w in windows:
        d = asdict(w)
        for k in MISSING_FEATURES:
            d[k] = None
        d["app_seconds"] = json.dumps(w.app_seconds)
        s = sessions.get(w.session_id)
        b = browser.get(w.window_start)
        known = sum(n for c, n in b["cats"].items() if c) if b else 0
        d.update({
            "participant": p.name,
            "session_key": s.key if s else None,
            "context_id": s.context_id if s else None,
            "label_in_context": s.in_context if s else None,
            "label_purity": _ratio(w.session_seconds, w.observed_seconds) if s else None,
            "site_switches_per_min": (_ratio(w.site_switch_count * 60, w.observed_seconds)
                                      if w.observed_seconds else None),
            "browser_events": b["events"] if b else 0,
            "distinct_hosts": len(b["hosts"]) if b else 0,
            "unknown_category_share": _ratio(b["cats"][None], b["events"]) if b else None,
        })
        for c in CATEGORIES:
            d[f"share_{c}"] = _ratio(b["cats"][c], known) if b and known else None
        out.append(d)
    return out


COLUMNS = (["participant", "window_start", "window_end", "window_seconds",
            "session_key", "context_id", "label_in_context", "label_purity",
            "complete", "observed_seconds", "coverage",
            "switch_count", "switches_per_min", "site_switch_count",
            "site_switches_per_min", "distinct_apps", "app_seconds",
            "browser_events", "distinct_hosts", "unknown_category_share"]
           + [f"share_{c}" for c in CATEGORIES] + list(MISSING_FEATURES))


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.isoformat(timespec="seconds")
    return v


def export(root: Path, out_dir: Path, gap_minutes=GAP_MINUTES,
           window_seconds=DEFAULT_WINDOW_SECONDS) -> dict:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folders = sorted((f for f in root.iterdir() if (f / "events.csv").exists()),
                     key=lambda f: (len(f.name), f.name))
    card = {"source": "RLKWiC", "label": "label_in_context = session-level relevance "
            "to the selected work context; NOT a focus label",
            "gap_minutes": gap_minutes, "window_seconds": window_seconds,
            "sidebar_events": "excluded",
            "missing_features_exported_as_empty": list(MISSING_FEATURES),
            "observed_seconds": f"derived: runs split at gaps > {gap_minutes} min",
            "participants": {}}
    with open(out_dir / "rlkwic_windows.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for folder in folders:
            p = read_participant(folder)
            ws = participant_windows(p, gap_minutes, window_seconds)
            for w in ws:
                writer.writerow({k: _cell(w.get(k)) for k in COLUMNS})
            labels = Counter(w["label_in_context"] for w in ws if w["complete"])
            card["participants"][p.name] = {
                "utc_offset_minutes": p.utc_offset_minutes,
                "events_used": len(p.events),
                "sidebar_events_excluded": p.excluded_sidebar_events,
                "sessions": len(p.sessions),
                "sessions_clipped_for_overlap": p.clipped_sessions,
                "runs": len(split_runs(p.events, gap_minutes)),
                "windows": len(ws),
                "windows_in_context": labels[True],
                "windows_off_context": labels[False],
                "windows_unlabelled": labels[None],
            }
    with open(out_dir / "rlkwic_datacard.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export RLKWiC as 5-minute windows")
    ap.add_argument("--root", required=True, help="folder containing p1 ... p8")
    ap.add_argument("--out", required=True, help="output folder (keep it out of git)")
    ap.add_argument("--gap-minutes", type=float, default=GAP_MINUTES)
    ap.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    args = ap.parse_args(argv)
    card = export(args.root, args.out, args.gap_minutes, args.window_seconds)
    print(json.dumps(card["participants"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
