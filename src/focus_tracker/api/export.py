"""Export a participant's labelled windows for pooled training.

Per complete, user-labelled window:
* numeric window features (from the `windows` table);
* browser context computed from the raw events: seconds of browser time,
  share of that time per content category, distinct sites, site switches
  per minute, the HOSTNAMES visited (with seconds) and any VIDEO TITLES
  recorded (titles exist only when the logger ran with video titles opted in);
* the participant's own label.
* the foreground apps: `top_app` and `app_seconds` (e.g. "Code.exe:240;
  chrome.exe:60"), for cross-checking labels.
Still excluded: raw events and the free-text session task (only its broad
category).

Browser time is reconstructed from the event rows: a row stamped with a
website covers the time until the next row, capped at BROWSER_GAP_CAP seconds
(the logger writes input counts every 5 s, so longer gaps mean idle time or a
stopped logger and are not counted as browsing).
"""
import csv
import io
import re
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta

from ..features.windows import DEFAULT_WINDOW_SECONDS, stored_windows
from ..logger.content import CONTENT_CATEGORIES

CATEGORIES = tuple(sorted(CONTENT_CATEGORIES))
BROWSER_GAP_CAP = 60

BASE_COLUMNS = (
    "participant", "window_start", "window_end", "window_seconds",
    "session_id", "session_task_category",
    "observed_seconds", "coverage", "active_seconds", "idle_seconds", "idle_episodes",
    "switch_count", "site_switch_count", "switches_per_min", "distinct_apps",
    "keyboard_count", "mouse_count",
)
APP_COLUMNS = ("top_app", "app_seconds")
WEB_COLUMNS = (("browser_seconds", "distinct_hosts", "site_switches_per_min")
               + tuple(f"share_{c}" for c in CATEGORIES)
               + ("share_unknown", "hostnames", "video_titles"))
LABEL_COLUMNS = ("label", "label_source", "labelled_at", "updated_at")
EXPORT_COLUMNS = BASE_COLUMNS + APP_COLUMNS + WEB_COLUMNS + LABEL_COLUMNS
_PSEUDONYM = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


def validate_pseudonym(text) -> str:
    text = (text or "").strip()
    if not _PSEUDONYM.fullmatch(text):
        raise ValueError("participant must be a short pseudonym such as P3 "
                         "(letters, digits, - or _; no real names)")
    return text


def browser_context(store, windows):
    """{window_start: dict of WEB_COLUMNS} for the given (sorted) windows."""
    starts = [w.window_start for w in windows]
    ends = {w.window_start: w.window_end for w in windows}
    secs = defaultdict(lambda: defaultdict(float))     # ws -> host -> seconds
    cats = defaultdict(lambda: defaultdict(float))     # ws -> category -> seconds
    titles = defaultdict(dict)                         # ws -> title -> None (ordered)

    def window_of(ts):
        i = bisect_right(starts, ts) - 1
        return starts[i] if i >= 0 and ts < ends[starts[i]] else None

    rows = [(datetime.fromisoformat(ts), event, site, cat, title)
            for ts, event, site, cat, title in store.conn.execute(
                "SELECT ts, event, website, content_category, content_title"
                " FROM events ORDER BY ts, id")]
    for i, (ts, event, site, cat, title) in enumerate(rows):
        if title:
            ws = window_of(ts)
            if ws is not None:
                titles[ws][title] = None
        if not site or event == "agent_stop":
            continue
        nxt = rows[i + 1][0] if i + 1 < len(rows) else ts
        t, end = ts, min(nxt, ts + timedelta(seconds=BROWSER_GAP_CAP))
        while t < end:                                  # split at window boundaries
            ws = window_of(t)
            if ws is None:
                j = bisect_right(starts, t)
                if j >= len(starts):
                    break
                t = starts[j]
                continue
            piece = min(end, ends[ws])
            d = (piece - t).total_seconds()
            secs[ws][site] += d
            cats[ws][cat or "unknown"] += d
            t = piece

    out = {}
    for w in windows:
        ws = w.window_start
        total = sum(secs[ws].values()) if ws in secs else 0.0
        hosts = sorted(secs[ws].items(), key=lambda kv: -kv[1]) if ws in secs else []
        row = {
            "browser_seconds": int(round(total)),
            "distinct_hosts": len(hosts),
            "site_switches_per_min": (round(w.site_switch_count * 60 / w.observed_seconds, 3)
                                      if w.observed_seconds else None),
            "hostnames": "; ".join(f"{h}:{int(round(s))}" for h, s in hosts) or None,
            "video_titles": " | ".join(titles[ws]) if ws in titles else None,
        }
        for c in CATEGORIES + ("unknown",):
            row[f"share_{c}"] = round(cats[ws].get(c, 0.0) / total, 4) if total else None
        out[ws] = row
    return out


def feature_rows(store, participant: str, window_seconds=DEFAULT_WINDOW_SECONDS,
                 complete_only=True, labelled_only=False):
    """Windows (oldest first) as dicts with the feature columns of EXPORT_COLUMNS
    (everything except the label columns). `participant` is written as given."""
    labelled = None
    if labelled_only:
        labelled = {ws for (ws,) in store.conn.execute(
            "SELECT window_start FROM window_labels WHERE window_seconds = ?",
            (window_seconds,))}
    categories = dict(store.conn.execute("SELECT id, task_category FROM sessions"))
    windows = [w for w in stored_windows(store, window_seconds)
               if (w.complete or not complete_only)
               and (labelled is None or w.window_start.isoformat(timespec="seconds") in labelled)]
    web = browser_context(store, windows)
    rows = []
    for w in windows:
        rows.append({
            "participant": participant,
            "window_start": w.window_start.isoformat(timespec="seconds"),
            "window_end": w.window_end.isoformat(timespec="seconds"),
            "window_seconds": w.window_seconds, "session_id": w.session_id,
            "session_task_category": categories.get(w.session_id),
            "observed_seconds": w.observed_seconds, "coverage": w.coverage,
            "active_seconds": w.active_seconds, "idle_seconds": w.idle_seconds,
            "idle_episodes": w.idle_episodes, "switch_count": w.switch_count,
            "site_switch_count": w.site_switch_count,
            "switches_per_min": w.switches_per_min, "distinct_apps": w.distinct_apps,
            "keyboard_count": w.keyboard_count, "mouse_count": w.mouse_count,
            "top_app": next(iter(w.app_seconds), None),     # sorted longest first
            "app_seconds": "; ".join(f"{a}:{s}" for a, s in w.app_seconds.items()) or None,
            **web[w.window_start],
        })
    return rows


def export_rows(store, participant: str, window_seconds=DEFAULT_WINDOW_SECONDS):
    """Complete, user-labelled windows as dicts with EXPORT_COLUMNS."""
    participant = validate_pseudonym(participant)
    labels = {ws: (label, la, ua, src) for ws, label, la, ua, src in store.conn.execute(
        "SELECT window_start, label, labelled_at, updated_at, source FROM window_labels"
        " WHERE window_seconds = ?", (window_seconds,))}
    rows = feature_rows(store, participant, window_seconds, labelled_only=True)
    for r in rows:
        label, labelled_at, updated_at, source = labels[r["window_start"]]
        r.update({"label": label, "label_source": source,
                  "labelled_at": labelled_at, "updated_at": updated_at})
    return rows


def to_csv(rows, columns=EXPORT_COLUMNS) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: "" if r.get(k) is None else r.get(k) for k in columns})
    return buf.getvalue()


def export_filename(participant: str, now: datetime) -> str:
    return f"focus_windows_{participant}_{now:%Y%m%d_%H%M}.csv"
