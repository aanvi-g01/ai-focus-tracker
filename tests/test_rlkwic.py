"""RLKWiC adapter: synthetic fixtures only (no real dataset rows in tests)."""
import csv
import json
from datetime import datetime, timedelta

import pytest

from focus_tracker.datasets import rlkwic as R

# Wall clock 2023-05-06 10:00:00 at UTC+2 -> 08:00:00 UTC
LOCAL0 = datetime(2023, 5, 6, 10, 0, 0)
UTC_MS0 = int((LOCAL0 - timedelta(hours=2) - datetime(1970, 1, 1)).total_seconds() * 1000)
EVENT_COLS = ["id", "timestamp_received", "timestamp_processed", "reporter", "cause",
              "title", "address", "selected_context_id", "spo_id", "active_app"]
SESSION_COLS = ["id", "context_id", "session_number", "in_context", "KW_action(s)",
                "events_list", "sidebar_actions", "window_titles", "browsed_webpages",
                "used_apps", "google_searches", "clipboard", "start",
                "approximate_duration"]
SECRET_TITLE = "Secret Title Must Never Appear"
SECRET_PATH = "/private/path?q=secret"


def evt(sec, reporter="window tracker", app="Code", address="None", offset_h=2):
    local = LOCAL0 + timedelta(seconds=sec)
    ms = UTC_MS0 + sec * 1000 + int((2 - offset_h) * 3600 * 1000)
    return {"id": local.strftime("%Y%m%d%H%M%S") + f"{sec % 1000:03d}",
            "timestamp_received": str(ms), "timestamp_processed": str(ms + 5),
            "reporter": reporter, "cause": "active window changed",
            "title": SECRET_TITLE, "address": address, "selected_context_id": "1",
            "spo_id": "None", "active_app": app}


def session(key, events, in_context, kw="['Learning']"):
    return {"id": key, "context_id": "1", "session_number": "1",
            "in_context": in_context, "KW_action(s)": kw,
            "events_list": "[" + ", ".join(e["id"] for e in events) + "]",
            "sidebar_actions": "[]", "window_titles": f"['{SECRET_TITLE}']",
            "browsed_webpages": "[]", "used_apps": "[]", "google_searches": "[]",
            "clipboard": "['secret clipboard']", "start": "x", "approximate_duration": "9 hours"}


def write_participant(root, name, events, sessions):
    d = root / name
    d.mkdir(parents=True)
    for fname, cols, rows in (("events.csv", EVENT_COLS, events),
                              ("sessions.csv", SESSION_COLS, sessions)):
        with open(d / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return d


def windows_of(tmp_path, events, sessions):
    return R.participant_windows(R.read_participant(
        write_participant(tmp_path, "p1", events, sessions)))


def by_start(ws):
    return {w["window_start"]: w for w in ws}


def test_local_time_from_id_and_offset(tmp_path):
    e = [evt(0, offset_h=3.5), evt(60, offset_h=3.5)]
    p = R.read_participant(write_participant(tmp_path, "p1", e, []))
    assert p.utc_offset_minutes == 210
    assert p.events[0].ts == LOCAL0


def test_gap_over_threshold_is_unobserved_not_active(tmp_path):
    # 10:00-10:04 activity, then nothing until 10:40 (gap 36 min > 15)
    e = [evt(0), evt(240, app="chrome"), evt(2400), evt(2640, app="chrome")]
    ws = by_start(windows_of(tmp_path, e, []))
    assert sum(w["observed_seconds"] for w in ws.values()) == 480
    assert all(w["observed_seconds"] <= 300 for w in ws.values())
    assert LOCAL0 + timedelta(minutes=20) not in ws          # gap: no window at all


def test_gap_within_threshold_stays_one_run(tmp_path):
    e = [evt(0), evt(600, app="chrome")]                    # 10 min gap <= 15
    assert R.split_runs(R.read_participant(
        write_participant(tmp_path, "p1", e, [])).events) and \
        sum(w["observed_seconds"] for w in windows_of(tmp_path / "b", e, [])) == 600


def test_missing_features_are_none_never_zero(tmp_path):
    ws = windows_of(tmp_path, [evt(0), evt(120, app="chrome"), evt(200)], [])
    for w in ws:
        for k in R.MISSING_FEATURES:
            assert w[k] is None


def test_sidebar_events_are_excluded(tmp_path):
    e = [evt(0), evt(100, reporter=R.SIDEBAR_REPORTER, app="cSpaces"), evt(200)]
    p = R.read_participant(write_participant(tmp_path, "p1", e, []))
    assert p.excluded_sidebar_events == 1
    assert all(ev.reporter != R.SIDEBAR_REPORTER for ev in p.events)
    assert sum(w["switch_count"] for w in R.participant_windows(p)) == 0


def test_sidebar_events_do_not_bridge_a_gap(tmp_path):
    # Only a sidebar click in the middle of a 30-minute silence.
    e = [evt(0), evt(60), evt(900, reporter=R.SIDEBAR_REPORTER, app="cSpaces"),
         evt(1800), evt(1860)]
    p = R.read_participant(write_participant(tmp_path, "p1", e, []))
    assert len(R.split_runs(p.events)) == 2


def test_only_real_app_changes_are_switches(tmp_path):
    # Same app repeated (title changes), unknown apps and the browser companion's
    # "default browser" must not create switches.
    e = [evt(0, app="Code"), evt(10, app="Code"), evt(20, app="None"), evt(30, app=""),
         evt(40, reporter=R.BROWSER_REPORTER, app="default browser",
             address="https://github.com/x"),
         evt(50, app="Code"), evt(60, app="msedge"), evt(70, app="Code")]
    ws = windows_of(tmp_path, e, [])
    assert sum(w["switch_count"] for w in ws) == 2


def test_site_switches_and_hostname_only(tmp_path, tmp_path_factory):
    e = [evt(0, app="msedge"),
         evt(10, reporter=R.BROWSER_REPORTER, app="default browser",
             address="https://www.github.com" + SECRET_PATH),
         evt(20, reporter=R.BROWSER_REPORTER, app="default browser",
             address="https://github.com/other"),
         evt(30, reporter=R.BROWSER_REPORTER, app="default browser",
             address="https://docs.python.org/3/"),
         evt(40, reporter=R.BROWSER_REPORTER, app="default browser", address="None"),
         evt(250, app="Code")]
    w = windows_of(tmp_path, e, [])[0]
    assert w["site_switch_count"] == 1                        # github -> docs only
    assert w["distinct_hosts"] == 2
    assert w["share_development"] == 0.6667 and w["share_documentation"] == 0.3333  # per event
    assert w["unknown_category_share"] == 0.25                # the unreadable one


def test_label_comes_from_majority_session_and_kw_action_is_ignored(tmp_path):
    a = [evt(0), evt(120, app="chrome"), evt(240)]
    b = [evt(400), evt(500, app="chrome"), evt(590)]
    ss = [session("p1-c1-s1", a, "TRUE"), session("p1-c1-s2", b, "False", kw="Unknown")]
    ws = by_start(windows_of(tmp_path, a + b, ss))
    first, second = ws[LOCAL0], ws[LOCAL0 + timedelta(minutes=5)]
    assert first["label_in_context"] is True and first["session_key"] == "p1-c1-s1"
    assert second["label_in_context"] is False
    assert 0 < second["label_purity"] <= 1
    assert "KW_action(s)" not in first and "kw" not in json.dumps(first, default=str).lower()


def test_window_without_majority_session_is_unlabelled(tmp_path):
    a = [evt(0), evt(30, app="chrome")]                     # 30 s of a 240 s window
    e = a + [evt(120), evt(240, app="chrome")]
    w = by_start(windows_of(tmp_path, e, [session("s1", a, "True")]))[LOCAL0]
    assert w["label_in_context"] is None


def test_interleaved_sessions_are_clipped_to_be_disjoint(tmp_path):
    a1, b1, a2 = evt(0), evt(60, app="chrome"), evt(120)
    ss = [session("s1", [a1, a2], "True"), session("s2", [b1], "False")]
    p = R.read_participant(write_participant(tmp_path, "p1", [a1, b1, a2], ss))
    assert p.clipped_sessions == 1
    assert p.sessions[0].end <= p.sessions[1].start


def test_export_never_writes_titles_urls_or_clipboard(tmp_path):
    e = [evt(0), evt(10, reporter=R.BROWSER_REPORTER, app="default browser",
                     address="https://example.com" + SECRET_PATH),
         evt(20, reporter="clipboard tracker", app="Code"), evt(200, app="chrome")]
    root = tmp_path / "root"
    write_participant(root, "p1", e, [session("s1", e, "True")])
    card = R.export(root, tmp_path / "out")
    text = (tmp_path / "out" / "rlkwic_windows.csv").read_text(encoding="utf-8")
    for secret in (SECRET_TITLE, "secret", "/private", "example.com"):
        assert secret not in text
    assert card["sidebar_events"] == "excluded" and card["gap_minutes"] == 15
    row = next(csv.DictReader(open(tmp_path / "out" / "rlkwic_windows.csv",
                                   encoding="utf-8")))
    assert all(row[k] == "" for k in R.MISSING_FEATURES)
    assert "NOT a focus label" in card["label"]


def test_no_focus_labels_are_produced(tmp_path):
    e = [evt(0), evt(120, app="chrome")]
    ws = windows_of(tmp_path, e, [session("s1", e, "True")])
    text = json.dumps(ws, default=str)
    for label in ("Focused", "Distracted", "Inactive"):
        assert label not in text
