import dataclasses
from datetime import datetime, timedelta

import pytest

from focus_tracker.features import windows as W
from focus_tracker.features.windows import aggregate_windows, load_rows, window_start
from focus_tracker.logger.agent import Agent
from focus_tracker.logger.input_counter import InputCounter
from focus_tracker.logger.session import SessionManager
from focus_tracker.logger.storage import EventStore

T0 = datetime(2026, 11, 16, 10, 0, 0)


def t(m, s=0):
    return T0 + timedelta(minutes=m, seconds=s)


def ev(ts, event, app=None, value=1, sid=None):
    return (ts, app, event, value, sid)


def by_start(ws):
    return {w.window_start: w for w in ws}


# --------------------------------------------------------------------------
# Window boundaries and configuration
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ts, secs, expected", [
    (t(3, 17), 300, t(0)), (t(5), 300, t(5)), (t(4, 59), 300, t(0)),
    (t(14, 59), 600, t(10)), (datetime(2026, 11, 16, 23, 59, 59), 300,
                              datetime(2026, 11, 16, 23, 55)),
])
def test_window_start_is_clock_aligned(ts, secs, expected):
    assert window_start(ts, secs) == expected


@pytest.mark.parametrize("bad", [0, 59, 420, 300.0, True, "300"])
def test_invalid_window_lengths_rejected(bad):
    with pytest.raises(ValueError):
        aggregate_windows([], bad)


def test_no_data_gives_no_windows():
    assert aggregate_windows([]) == []


# --------------------------------------------------------------------------
# Core features
# --------------------------------------------------------------------------
BASIC = [
    ev(t(0), "agent_start"), ev(t(0), "active", "Code.exe"),
    ev(t(1), "keyboard_count", "Code.exe", 10), ev(t(1), "mouse_count", "Code.exe", 5),
    ev(t(2), "switched", "chrome.exe"), ev(t(2, 30), "site_switched", "chrome.exe"),
    ev(t(3), "switched", "Code.exe"), ev(t(6), "agent_stop"),
]


def test_basic_window_features():
    w0, w1 = aggregate_windows(BASIC)
    assert dataclasses.asdict(w0) == {
        "window_start": t(0), "window_end": t(5), "window_seconds": 300,
        "session_id": None, "session_seconds": 0,
        "observed_seconds": 300, "coverage": 1.0, "complete": True,
        "active_seconds": 300, "idle_seconds": 0, "idle_episodes": 0,
        "switch_count": 2, "site_switch_count": 1, "switches_per_min": 0.4,
        "distinct_apps": 2, "app_seconds": {"Code.exe": 240, "chrome.exe": 60},
        "keyboard_count": 10, "mouse_count": 5,
    }
    # last window: agent stopped at 10:06 -> partly observed, may still grow
    assert (w1.window_start, w1.observed_seconds, w1.coverage, w1.complete) == \
        (t(5), 60, 0.2, False)
    assert w1.app_seconds == {"Code.exe": 60} and w1.switches_per_min == 0.0


def test_window_length_is_configurable():
    (w,) = aggregate_windows(BASIC, 600)
    assert (w.window_seconds, w.observed_seconds, w.app_seconds["Code.exe"]) == \
        (600, 360, 300)


def test_app_seconds_never_exceed_observed_time():
    for w in aggregate_windows(BASIC, 60):
        assert sum(w.app_seconds.values()) <= w.observed_seconds <= 60


# --------------------------------------------------------------------------
# Idle
# --------------------------------------------------------------------------
def test_idle_episode_spanning_two_windows_is_split():
    rows = [ev(t(0), "agent_start"), ev(t(0), "active", "Code.exe"),
            ev(t(4), "idle", "Code.exe"),                     # detected after threshold
            ev(t(7), "idle_duration", "Code.exe", 240),       # really idle 10:03-10:07
            ev(t(7), "active", "Code.exe"), ev(t(10), "agent_stop")]
    w0, w1 = aggregate_windows(rows)
    assert (w0.idle_seconds, w0.active_seconds, w0.idle_episodes) == (120, 180, 1)
    assert (w1.idle_seconds, w1.active_seconds, w1.idle_episodes) == (120, 180, 0)


def test_idle_without_duration_ends_at_next_input():
    rows = [ev(t(0), "agent_start"), ev(t(0), "active", "Code.exe"),
            ev(t(2), "idle", "Code.exe"), ev(t(3), "keyboard_count", "Code.exe", 3),
            ev(t(5), "agent_stop")]
    (w,) = aggregate_windows(rows)
    assert (w.idle_seconds, w.active_seconds) == (60, 240)


def test_idle_without_duration_ends_at_agent_stop():
    rows = [ev(t(0), "agent_start"), ev(t(0), "active", "Code.exe"),
            ev(t(1), "idle", "Code.exe"), ev(t(5), "agent_stop")]
    (w,) = aggregate_windows(rows)
    assert (w.idle_seconds, w.active_seconds) == (240, 60)


def test_idle_reported_before_agent_start_is_clipped():
    rows = [ev(t(0), "agent_start"), ev(t(0), "active", "Code.exe"),
            ev(t(1), "idle_duration", "Code.exe", 600), ev(t(5), "agent_stop")]
    (w,) = aggregate_windows(rows)
    assert (w.idle_seconds, w.observed_seconds) == (60, 300)


# --------------------------------------------------------------------------
# Missing data: agent off, crashes, legacy rows
# --------------------------------------------------------------------------
def test_unobserved_windows_are_omitted_not_zero_filled():
    rows = [ev(t(0), "agent_start"), ev(t(0), "active", "Code.exe"), ev(t(2), "agent_stop"),
            ev(t(30), "agent_start"), ev(t(30), "active", "Code.exe"), ev(t(35), "agent_stop")]
    assert [w.window_start for w in aggregate_windows(rows)] == [t(0), t(30)]


def test_crashed_run_ends_at_its_last_event():
    rows = [ev(t(0), "agent_start"), ev(t(0), "active", "Code.exe"),
            ev(t(2), "mouse_count", "Code.exe", 7),
            ev(t(60), "agent_start"), ev(t(60), "active", "Code.exe"), ev(t(61), "agent_stop")]
    ws = by_start(aggregate_windows(rows))
    assert ws[t(0)].observed_seconds == 120 and ws[t(0)].app_seconds == {"Code.exe": 120}
    assert set(ws) == {t(0), t(60)}


def test_legacy_rows_without_agent_start():
    rows = [ev(t(0), "active", "Code.exe"), ev(t(2), "agent_stop")]
    (w,) = aggregate_windows(rows)
    assert w.observed_seconds == 120


def test_session_time_while_agent_off_is_not_counted():
    rows = [ev(t(0), "session_start", sid=1), ev(t(0), "agent_start"),
            ev(t(0), "active", "Code.exe"), ev(t(2), "agent_stop"),
            ev(t(10), "session_end", value=600, sid=1)]
    (w,) = aggregate_windows(rows)
    assert (w.session_id, w.session_seconds, w.observed_seconds) == (1, 120, 120)


# --------------------------------------------------------------------------
# Session association
# --------------------------------------------------------------------------
def _observed(start_m, end_m, extra):
    return sorted([ev(t(start_m), "agent_start"), ev(t(start_m), "active", "Code.exe"),
                   ev(t(end_m), "agent_stop")] + extra, key=lambda r: r[0])


def test_window_takes_the_majority_session():
    rows = _observed(0, 10, [ev(t(0, 30), "session_start", sid=1),
                             ev(t(7), "session_end", sid=1)])
    w0, w1 = aggregate_windows(rows)
    assert (w0.session_id, w0.session_seconds) == (1, 270)
    assert (w1.session_id, w1.session_seconds) == (None, 120)   # minority: not assigned


def test_paused_time_is_not_session_time():
    rows = _observed(0, 5, [ev(t(0), "session_start", sid=1), ev(t(1), "session_pause", sid=1),
                            ev(t(4), "session_resume", sid=1), ev(t(5), "session_end", sid=1)])
    (w,) = aggregate_windows(rows)
    assert (w.session_id, w.session_seconds) == (None, 120)


def test_back_to_back_sessions_in_one_window():
    rows = _observed(0, 5, [ev(t(0), "session_start", sid=1), ev(t(1), "session_end", sid=1),
                            ev(t(1), "session_start", sid=2), ev(t(5), "session_end", sid=2)])
    (w,) = aggregate_windows(rows)
    assert (w.session_id, w.session_seconds) == (2, 240)


def test_open_session_extends_to_end_of_data():
    rows = _observed(0, 10, [ev(t(1), "session_start", sid=3)])
    assert [w.session_id for w in aggregate_windows(rows)] == [3, 3]


# --------------------------------------------------------------------------
# Separation of concerns: no labels, no session names
# --------------------------------------------------------------------------
def test_windows_carry_no_label_or_session_name():
    names = {f.name for f in dataclasses.fields(W.Window)}
    assert not names & {"label", "focus", "state", "task", "session_name", "task_category"}


# --------------------------------------------------------------------------
# End to end: real Agent + EventStore + SessionManager
# --------------------------------------------------------------------------
def test_end_to_end_from_agent_logs():
    store = EventStore(":memory:")
    counter = InputCounter()
    apps = iter(["Code.exe"] * 200 + ["chrome.exe"] * 100 + ["Code.exe"] * 300)
    idles = iter([0] * 400 + [61 + i for i in range(120)] + [0] * 80)
    agent = Agent(store, counter, get_app=lambda: next(apps),
                  get_idle=lambda: next(idles), idle_threshold=60, flush_seconds=5)
    sessions = SessionManager(store)
    store.log(t(0), None, "agent_start")
    for i in range(600):                                   # 10 minutes, 1 tick/s
        if i == 0:
            sessions.start("studying DSA", t(0))
        if i % 10 == 0:
            counter._on_key(); counter._on_mouse()
        agent.step(t(0, i))
    agent.finish(t(10)); store.log(t(10), None, "agent_stop")
    sessions.end(t(10))

    w0, w1 = aggregate_windows(load_rows(store))
    assert (w0.observed_seconds, w1.observed_seconds) == (300, 300)
    assert w0.session_id == w1.session_id == 1
    assert w0.app_seconds == {"Code.exe": 200, "chrome.exe": 100}
    assert w0.switch_count == 1 and w1.switch_count == 1
    assert w0.keyboard_count + w1.keyboard_count == 60     # one key every 10 ticks, none lost
    assert w1.idle_seconds > 100 and w0.idle_seconds == 0
    assert w0.complete and w1.complete
