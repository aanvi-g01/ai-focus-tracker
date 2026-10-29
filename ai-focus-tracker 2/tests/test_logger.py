import sqlite3
from datetime import datetime, timedelta

import pytest

from focus_tracker.logger.agent import Agent
from focus_tracker.logger.input_counter import InputCounter
from focus_tracker.logger.storage import EventStore


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
