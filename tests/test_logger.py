import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta
from unittest import mock

import pytest

from focus_tracker.logger import browser, probes, storage
from focus_tracker.logger.agent import Agent
from focus_tracker.logger.idle import IdleTracker
from focus_tracker.logger.input_counter import InputCounter
from focus_tracker.logger.storage import EventStore

T0 = datetime(2026, 11, 3, 10, 0, 0)


# --------------------------------------------------------------------------
# Privacy / storage / counter / basic agent behaviour
# --------------------------------------------------------------------------
def test_rejects_non_whitelisted_event():
    s = EventStore(":memory:")
    with pytest.raises(ValueError):
        s.log(datetime.now(), "x.exe", "keystroke")


def test_schema_has_no_content_columns():
    s = EventStore(":memory:")
    cols = [r[1] for r in s.conn.execute("PRAGMA table_info(events)")]
    # `website` is the only free-text column besides app: hostname-only, browsers-only;
    # content_category is a validated one-word enum, session_id an integer
    assert cols == ["id", "ts", "app", "event", "value", "website",
                    "session_id", "content_category", "content_title"]


def test_counter_never_reads_key_identity():
    c = InputCounter()
    c._on_key("secret-key-object")
    c._on_key()
    assert c.drain() == (2, 0)
    assert c.drain() == (0, 0)


def test_agent_sequence():
    store = EventStore(":memory:")
    counter = InputCounter()
    apps = iter(["Code.exe", "Code.exe", "chrome.exe", "chrome.exe", "chrome.exe"])
    idles = iter([0, 0, 0, 0, 120])
    agent = Agent(store, counter, get_app=lambda: next(apps),
                  get_idle=lambda: next(idles), idle_threshold=60, flush_seconds=2)
    t0 = datetime(2026, 10, 29, 10, 0, 0)
    counter._on_key()
    for i in range(5):
        agent.step(t0 + timedelta(seconds=i))
    events = [(r[1], r[2]) for r in store.rows()]
    assert ("Code.exe", "active") in events
    assert ("chrome.exe", "switched") in events
    assert ("chrome.exe", "idle") in events
    assert ("Code.exe", "keyboard_count") in events or ("chrome.exe", "keyboard_count") in events


# --------------------------------------------------------------------------
# Idle detection: tick maths (pure, runs on any OS)
# --------------------------------------------------------------------------
def test_ticks_simple_difference():
    assert probes.idle_seconds_from_ticks(10_500, 10_000) == 0.5


def test_ticks_survive_32bit_wraparound():
    # last input just before the counter wrapped, now just after: 21 ms
    assert probes.idle_seconds_from_ticks(5, 0xFFFFFFF0) == pytest.approx(0.021)


def test_ticks_zero_when_input_is_now():
    assert probes.idle_seconds_from_ticks(1234, 1234) == 0.0


# --------------------------------------------------------------------------
# Idle detection: probe selection and fallback
# --------------------------------------------------------------------------
def test_windows_path_used_when_available():
    with mock.patch.object(probes.sys, "platform", "win32"), \
         mock.patch.object(probes, "_windows_idle_seconds", return_value=12.5):
        assert probes.get_idle_seconds() == 12.5


def test_falls_back_when_windows_api_fails():
    with mock.patch.object(probes.sys, "platform", "win32"), \
         mock.patch.object(probes, "_windows_idle_seconds", return_value=None):
        got = probes.get_idle_seconds(time.monotonic() - 5)
    assert got == pytest.approx(5.0, abs=0.5)


def test_falls_back_when_windows_api_raises():
    with mock.patch.object(probes.sys, "platform", "win32"), \
         mock.patch.object(probes, "_windows_idle_seconds", side_effect=OSError):
        got = probes.get_idle_seconds(time.monotonic() - 3)
    assert got == pytest.approx(3.0, abs=0.5)


def test_idle_seconds_never_negative():
    with mock.patch.object(probes.sys, "platform", "win32"), \
         mock.patch.object(probes, "_windows_idle_seconds", return_value=-1.0):
        assert probes.get_idle_seconds() == 0.0


@pytest.mark.skipif(sys.platform != "win32", reason="needs real Windows")
def test_real_windows_call_returns_sane_number():
    secs = probes._windows_idle_seconds()
    assert secs is not None
    assert 0.0 <= secs < 60 * 60 * 24 * 50


# --------------------------------------------------------------------------
# Idle detection: episode tracking (start + duration)
# --------------------------------------------------------------------------
def test_tracker_no_transition_below_threshold():
    t = IdleTracker(60)
    assert [t.update(s) for s in (0, 10, 59.9)] == [None, None, None]
    assert not t.is_idle


def test_tracker_episode_start_and_end_with_duration():
    t = IdleTracker(60)
    assert t.update(60).kind == "started"
    assert t.update(75) is None
    assert t.update(90) is None
    end = t.update(2)  # input resumed
    assert (end.kind, end.seconds) == ("ended", 90)
    assert not t.is_idle


def test_tracker_second_episode_does_not_inherit_first_peak():
    t = IdleTracker(10)
    t.update(10); t.update(500); t.update(1)
    t.update(10)
    assert t.update(0).seconds == 10


def test_tracker_close_reports_in_progress_episode():
    t = IdleTracker(60)
    t.update(61); t.update(130)
    end = t.close()
    assert (end.kind, end.seconds) == ("ended", 130)
    assert t.close() is None  # idempotent


def test_tracker_close_when_not_idle_is_none():
    assert IdleTracker(60).close() is None


# --------------------------------------------------------------------------
# Idle detection: agent integration
# --------------------------------------------------------------------------
def _idle_agent(readings, threshold=60):
    store = EventStore(":memory:")
    it = iter(readings)
    agent = Agent(store, InputCounter(), get_app=lambda: "Code.exe",
                  get_idle=lambda: next(it), idle_threshold=threshold)
    return agent, store


def _run(agent, n):
    _run_from(agent, 0, n)


def _run_from(agent, start, n):
    for i in range(start, start + n):
        agent.step(T0 + timedelta(seconds=i))


def test_idle_episode_logged_with_duration():
    agent, store = _idle_agent([0, 30, 60, 61, 90, 2, 3])
    _run(agent, 7)
    assert [(r[2], r[3]) for r in store.rows()] == [
        ("active", 1),          # first window seen
        ("idle", 1),            # crossed threshold
        ("idle_duration", 90),  # episode length in seconds
        ("active", 1),          # input resumed
    ]


def test_shutdown_while_idle_still_records_duration():
    agent, store = _idle_agent([0, 70, 100])
    _run(agent, 3)
    agent.finish(T0 + timedelta(seconds=3))
    last = store.rows()[-1]
    assert (last[2], last[3]) == ("idle_duration", 100)


def test_is_idle_property_tracks_tracker():
    agent, _ = _idle_agent([0, 70, 1])
    _run(agent, 2)
    assert agent.is_idle
    agent.step(T0 + timedelta(seconds=2))
    assert not agent.is_idle


# --------------------------------------------------------------------------
# Idle time is a raw signal: never a label
# --------------------------------------------------------------------------
def test_long_idle_alone_never_produces_a_label():
    """A 10-minute idle stretch must yield raw signals only."""
    agent, store = _idle_agent([0] + [60 + i for i in range(600)] + [1])
    _run(agent, 602)
    assert {r[2] for r in store.rows()} == {"active", "idle", "idle_duration"}


@pytest.mark.parametrize("label", ["distracted", "focused", "inactive", "Distracted"])
def test_label_events_are_not_allowed_in_store(label):
    assert label not in storage.ALLOWED_EVENTS
    with pytest.raises(ValueError):
        EventStore(":memory:").log(T0, None, label)


def test_idle_duration_is_whitelisted():
    assert "idle_duration" in storage.ALLOWED_EVENTS


def test_idle_duration_stored_as_plain_integer():
    store = EventStore(":memory:")
    store.log(T0, "Code.exe", "idle_duration", 90.7)
    (val, typ), = store.conn.execute("SELECT value, typeof(value) FROM events").fetchall()
    assert (val, typ) == (90, "integer")


# --------------------------------------------------------------------------
# Website tracking: hostname normalisation (the privacy-critical part)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("https://www.GitHub.com/user/repo?tab=readme#top", "github.com"),
    ("github.com/some/private-repo", "github.com"),          # scheme hidden by browser
    ("https://mail.google.com/mail/u/0/#inbox", "mail.google.com"),
    ("https://www.google.com/search?q=my+medical+question", "google.com"),
    ("http://localhost:3000/dashboard", "localhost"),
    ("https://user:pass@example.com:8443/p", "example.com"),
    ("https://example.com./", "example.com"),
    ("https://m\u00fcnchen.de/", "xn--mnchen-3ya.de"),
    ("  https://news.ycombinator.com/item?id=1  ", "news.ycombinator.com"),
])
def test_normalize_site_keeps_hostname_only(raw, expected):
    assert browser.normalize_site(raw) == expected


@pytest.mark.parametrize("raw", [
    None, "", "   ",
    "how to cheat on exam",        # user typing a search in the address bar
    "github",                      # partial typing, no dot
    "chrome://settings", "edge://history", "about:blank",
    "file:///C:/Users/me/secret.pdf", "javascript:alert(1)",
    "https://a..b.com/",
])
def test_normalize_site_rejects_non_hostnames(raw):
    assert browser.normalize_site(raw) is None


def test_normalize_site_never_leaks_path_or_query():
    out = browser.normalize_site("https://bank.example.com/accounts/12345?token=abc#x")
    assert out == "bank.example.com"
    assert not any(ch in out for ch in "/?#=@: ")


@pytest.mark.parametrize("app, expected", [
    ("chrome.exe", True), ("Chrome.EXE", True), ("msedge.exe", True),
    ("firefox.exe", True), ("Code.exe", False), ("", False), (None, False),
])
def test_is_browser(app, expected):
    assert browser.is_browser(app) is expected


def test_get_active_site_is_none_off_windows():
    with mock.patch.object(browser.sys, "platform", "linux"):
        assert browser.get_active_site() is None


def test_get_active_site_never_raises():
    with mock.patch.object(browser.sys, "platform", "win32"), \
         mock.patch.object(browser, "_foreground_hwnd", side_effect=OSError):
        assert browser.get_active_site() is None


def test_get_active_site_reduces_raw_address_to_hostname():
    with mock.patch.object(browser.sys, "platform", "win32"), \
         mock.patch.object(browser, "_foreground_hwnd", return_value=1), \
         mock.patch.object(browser, "_read_address_bar",
                           return_value="https://www.youtube.com/watch?v=abc"):
        assert browser.get_active_site() == "youtube.com"


# --------------------------------------------------------------------------
# Website tracking: storage (column, validation, migration)
# --------------------------------------------------------------------------
def test_website_defaults_to_null():
    s = EventStore(":memory:")
    s.log(T0, "chrome.exe", "active")
    s.log(T0, "Code.exe", "active")
    assert [r[4] for r in s.rows(include_website=True)] == [None, None]


def test_website_stored_for_browser():
    s = EventStore(":memory:")
    s.log(T0, "chrome.exe", "site_switched", website="github.com")
    assert s.rows(include_website=True)[0][4] == "github.com"


def test_website_rejected_for_non_browser_app():
    with pytest.raises(ValueError):
        EventStore(":memory:").log(T0, "Code.exe", "active", website="github.com")


@pytest.mark.parametrize("bad", [
    "https://github.com", "github.com/user/repo", "github.com?q=1",
    "GitHub.com", "my search terms", "localhost:3000", "",
])
def test_storage_rejects_anything_but_a_bare_hostname(bad):
    with pytest.raises(ValueError):
        EventStore(":memory:").log(T0, "chrome.exe", "site_switched", website=bad)


def test_rows_default_shape_is_unchanged():
    s = EventStore(":memory:")
    s.log(T0, "chrome.exe", "active", website="github.com")
    assert s.rows() == [(T0.isoformat(timespec="seconds"), "chrome.exe", "active", 1)]


def test_old_database_is_migrated_without_losing_rows():
    with tempfile.TemporaryDirectory() as d:
        path = d + "/old.db"
        old = sqlite3.connect(path)
        old.executescript(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
            " app TEXT, event TEXT NOT NULL, value INTEGER NOT NULL DEFAULT 1);"
            "INSERT INTO events (ts, app, event, value) VALUES ('2026-11-03T01:02:22','chrome.exe','idle',1);")
        old.commit(); old.close()

        s = EventStore(path)
        cols = [r[1] for r in s.conn.execute("PRAGMA table_info(events)")]
        assert cols == ["id", "ts", "app", "event", "value", "website",
                        "session_id", "content_category", "content_title"]
        assert s.rows(include_website=True)[0][4] is None      # old row kept, NULL site
        s.log(T0, "chrome.exe", "site_switched", website="github.com")
        assert s.rows(include_website=True)[-1][4] == "github.com"
        s.close()
        EventStore(path).close()                                # re-open is a no-op


# --------------------------------------------------------------------------
# Website tracking: agent behaviour
# --------------------------------------------------------------------------
def _site_agent(apps, sites, idles=None, **kw):
    """Agent fed by scripted app / site readings (site_poll_seconds=0 => read every tick)."""
    store = EventStore(":memory:")
    counter = InputCounter()
    app_it, site_it = iter(apps), iter(sites)
    idle_it = iter(idles) if idles is not None else None
    kw.setdefault("site_poll_seconds", 0)
    agent = Agent(store, counter, get_app=lambda: next(app_it),
                  get_site=lambda: next(site_it),
                  get_idle=(lambda: next(idle_it)) if idle_it else (lambda: 0), **kw)
    return agent, store, counter


def _events(store):
    return [(r[1], r[2], r[4]) for r in store.rows(include_website=True)]


def test_site_switch_inside_browser_is_logged():
    agent, store, _ = _site_agent(["chrome.exe"] * 4,
                                  ["github.com", "github.com", "youtube.com", "youtube.com"])
    _run(agent, 4)
    assert _events(store) == [
        ("chrome.exe", "active", "github.com"),
        ("chrome.exe", "site_switched", "youtube.com"),
    ]


def test_non_browser_never_reads_site_and_website_is_null():
    store = EventStore(":memory:")
    def boom():
        raise AssertionError("address bar must not be read for non-browser apps")
    agent = Agent(store, InputCounter(), get_app=lambda: "Code.exe",
                  get_idle=lambda: 0, get_site=boom, site_poll_seconds=0)
    _run(agent, 3)
    assert _events(store) == [("Code.exe", "active", None)]


def test_app_switches_carry_site_only_for_the_browser():
    agent, store, _ = _site_agent(["Code.exe", "chrome.exe", "Code.exe"], ["github.com"])
    _run(agent, 3)
    assert _events(store) == [
        ("Code.exe", "active", None),
        ("chrome.exe", "switched", "github.com"),
        ("Code.exe", "switched", None),
    ]


def test_input_counts_and_idle_rows_are_stamped_with_current_site():
    agent, store, counter = _site_agent(
        ["chrome.exe"] * 4, ["github.com"] * 4, idles=[0, 61, 62, 2], flush_seconds=3)
    counter._on_key()
    _run(agent, 4)
    by_event = {e: site for _, e, site in _events(store)}
    assert by_event["idle"] == "github.com"
    assert by_event["idle_duration"] == "github.com"
    assert by_event["keyboard_count"] == "github.com"


def test_unreadable_address_bar_is_not_a_switch():
    # typing in the address bar / new tab -> None; returning to same site -> no event
    agent, store, _ = _site_agent(["chrome.exe"] * 4, ["github.com", None, None, "github.com"])
    _run(agent, 4)
    assert _events(store) == [("chrome.exe", "active", "github.com")]


def test_site_rows_while_unreadable_are_null_not_stale():
    agent, store, counter = _site_agent(["chrome.exe"] * 3, ["github.com", None, None],
                                        flush_seconds=2)
    _run(agent, 1)
    counter._on_key()
    _run_from(agent, 1, 2)
    kb = [r for r in store.rows(include_website=True) if r[2] == "keyboard_count"]
    assert kb and kb[0][4] is None


def test_switching_between_browsers_is_an_app_switch_not_a_site_switch():
    agent, store, _ = _site_agent(["chrome.exe", "msedge.exe"], ["github.com", "github.com"])
    _run(agent, 2)
    assert _events(store) == [
        ("chrome.exe", "active", "github.com"),
        ("msedge.exe", "switched", "github.com"),
    ]


def test_address_bar_is_polled_at_most_every_site_poll_seconds():
    calls = []
    store = EventStore(":memory:")
    agent = Agent(store, InputCounter(), get_app=lambda: "chrome.exe",
                  get_idle=lambda: 0, site_poll_seconds=5,
                  get_site=lambda: calls.append(1) or "github.com")
    _run(agent, 6)               # ticks at t = 0..5 s
    assert len(calls) == 2       # t=0 (browser came to front) and t=5


def test_get_site_failure_does_not_crash_the_agent():
    store = EventStore(":memory:")
    def broken():
        raise RuntimeError("UI Automation failed")
    agent = Agent(store, InputCounter(), get_app=lambda: "chrome.exe",
                  get_idle=lambda: 0, get_site=broken, site_poll_seconds=0)
    _run(agent, 3)
    assert _events(store) == [("chrome.exe", "active", None)]


def test_raw_url_from_a_custom_probe_is_sanitised_before_storage():
    agent, store, _ = _site_agent(
        ["chrome.exe"], ["https://github.com/me/secret-repo?tab=x#frag"])
    _run(agent, 1)
    assert _events(store) == [("chrome.exe", "active", "github.com")]
    dump = repr(store.conn.execute("SELECT * FROM events").fetchall())
    assert "secret-repo" not in dump and "tab=x" not in dump


def test_websites_are_off_by_default():
    store = EventStore(":memory:")
    agent = Agent(store, InputCounter(), get_app=lambda: "chrome.exe", get_idle=lambda: 0)
    _run(agent, 2)
    assert _events(store) == [("chrome.exe", "active", None)]
