import sqlite3
import sys
import time
from datetime import datetime, timedelta
from unittest import mock

import pytest

from focus_tracker.logger import probes, storage
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
    assert cols == ["id", "ts", "app", "event", "value"]


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
    for i in range(n):
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
