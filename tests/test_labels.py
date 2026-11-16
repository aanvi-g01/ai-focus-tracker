import sqlite3
import tempfile
from datetime import date, datetime, timedelta

import pytest

from focus_tracker.features.windows import (aggregate_windows, load_rows, refresh_windows,
                                            stored_windows)
from focus_tracker.logger import labels as L
from focus_tracker.logger.labels import LabelError, clear_label, get_label, set_label
from focus_tracker.logger.session import SessionManager
from focus_tracker.logger.storage import EventStore

T0 = datetime(2026, 11, 16, 10, 0, 0)
NOW = datetime(2026, 11, 16, 18, 0, 0)


def t(m, s=0):
    return T0 + timedelta(minutes=m, seconds=s)


def _record(store, start_m=0, end_m=12, task="Studying DSA"):
    """Agent observed start..end (minutes) inside a session with the given name."""
    sid = SessionManager(store).start(task, t(start_m)).id
    store.log(t(start_m), None, "agent_start", session_id=sid)
    store.log(t(start_m), "Code.exe", "active", session_id=sid)
    for m in range(start_m, end_m):
        store.log(t(m, 30), "Code.exe", "keyboard_count", 20, session_id=sid)
    store.log(t(end_m), None, "agent_stop", session_id=sid)
    SessionManager(store).end(t(end_m))
    return sid


@pytest.fixture
def store():
    s = EventStore(":memory:")
    _record(s)
    refresh_windows(s, now=NOW)      # windows 10:00, 10:05 complete; 10:10 in progress
    yield s
    s.close()


def _label_rows(store):
    return store.conn.execute(
        "SELECT window_start, window_seconds, session_id, label, labelled_at, updated_at"
        " FROM window_labels ORDER BY window_start").fetchall()


# --------------------------------------------------------------------------
# Window persistence
# --------------------------------------------------------------------------
def test_stored_windows_round_trip(store):
    assert stored_windows(store) == aggregate_windows(load_rows(store))
    assert [w.window_start for w in stored_windows(store)] == [t(0), t(5), t(10)]


def test_refresh_replaces_instead_of_duplicating(store):
    store.log(t(20), None, "agent_start")
    store.log(t(20), "chrome.exe", "active")
    store.log(t(26), None, "agent_stop")
    refresh_windows(store, now=NOW)
    starts = [w.window_start for w in stored_windows(store)]
    assert starts == [t(0), t(5), t(10), t(20), t(25)]
    assert stored_windows(store)[2].complete            # 10:10 no longer the last window


def test_window_lengths_are_stored_independently(store):
    refresh_windows(store, 600, now=NOW)
    assert len(stored_windows(store, 600)) == 2 and len(stored_windows(store, 300)) == 3


def test_stored_windows_filter_by_session(store):
    sid2 = _record(store, 30, 36, task="coding")
    refresh_windows(store, now=NOW)
    assert {w.session_id for w in stored_windows(store, session_id=sid2)} == {sid2}


# --------------------------------------------------------------------------
# Labelling: add, correct, clear
# --------------------------------------------------------------------------
def test_set_label_stores_it_against_window_and_session(store):
    w, previous = set_label(store, t(5), "focused", NOW)
    assert previous is None and w.window_start == t(5)
    assert _label_rows(store) == [
        (t(5).isoformat(), 300, 1, "Focused", NOW.isoformat(), NOW.isoformat())]


def test_any_time_inside_the_window_selects_it(store):
    set_label(store, t(7, 42), "Inactive", NOW)
    assert get_label(store, t(5)) == "Inactive"


def test_correction_overwrites_and_keeps_first_labelled_time(store):
    set_label(store, t(0), "Focused", NOW)
    later = NOW + timedelta(days=1)
    _, previous = set_label(store, t(0), "Distracted", later)
    assert previous == "Focused"
    assert _label_rows(store) == [
        (t(0).isoformat(), 300, 1, "Distracted", NOW.isoformat(), later.isoformat())]


def test_clear_label(store):
    set_label(store, t(0), "Focused", NOW)
    assert clear_label(store, t(0)) is True
    assert clear_label(store, t(0)) is False
    assert _label_rows(store) == []


@pytest.mark.parametrize("bad", ["focus", "Happy", "", None, "unfocused"])
def test_invalid_labels_rejected(store, bad):
    with pytest.raises(LabelError):
        set_label(store, t(0), bad, NOW)


def test_database_rejects_labels_outside_the_three(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("INSERT INTO window_labels VALUES ('x', 300, NULL, 'focused', 'a', 'a')")


def test_cannot_label_a_window_without_data(store):
    with pytest.raises(LabelError, match="no recorded activity"):
        set_label(store, t(45), "Focused", NOW)


def test_cannot_label_a_window_still_in_progress(store):
    with pytest.raises(LabelError, match="in progress"):
        set_label(store, t(10), "Focused", NOW)


# --------------------------------------------------------------------------
# Separation: session names never become labels; rebuilds never lose labels
# --------------------------------------------------------------------------
def test_no_label_is_ever_created_automatically(store):
    _record(store, 30, 41, task="Focused deep work on DSA")   # suggestive name
    refresh_windows(store, now=NOW)
    assert _label_rows(store) == []
    assert all(label is None for _, label in L.labelled_windows(store, on_date=T0.date()))


def test_user_label_may_contradict_the_session_name(store):
    set_label(store, t(0), "Distracted", NOW)                # session: "Studying DSA"
    assert get_label(store, t(0)) == "Distracted"


def test_labels_survive_window_rebuilds(store):
    set_label(store, t(0), "Focused", NOW)
    store.conn.execute("DELETE FROM windows")
    refresh_windows(store, now=NOW)
    refresh_windows(store, now=NOW)
    assert get_label(store, t(0)) == "Focused"


def test_label_tables_are_separate_from_sessions_and_windows(store):
    def cols(table):
        return {r[1] for r in store.conn.execute(f"PRAGMA table_info({table})")}
    assert "label" not in cols("windows") | cols("sessions")
    assert not {"task", "task_category"} & cols("window_labels")


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------
def test_labelled_windows_filters(store):
    set_label(store, t(0), "Focused", NOW)
    all_rows = L.labelled_windows(store, on_date=T0.date())
    assert [(w.window_start, lab) for w, lab in all_rows] == \
        [(t(0), "Focused"), (t(5), None), (t(10), None)]
    assert [w.window_start for w, _ in L.labelled_windows(store, unlabelled_only=True)] == \
        [t(5), t(10)]
    assert L.labelled_windows(store, on_date=date(2026, 11, 17)) == []
    assert L.labelled_windows(store, session_id=99) == []


@pytest.mark.parametrize("text, expected", [
    ("10:05", datetime(2026, 11, 16, 10, 5)),
    ("09:30", datetime(2026, 11, 16, 9, 30)),
    ("2026-11-15 23:55", datetime(2026, 11, 15, 23, 55)),
    ("2026-11-15T23:55", datetime(2026, 11, 15, 23, 55)),
])
def test_parse_when(text, expected):
    assert L.parse_when(text, date(2026, 11, 16)) == expected


@pytest.mark.parametrize("bad", ["", "tomorrow", "25:00", "2026-13-01 10:00"])
def test_parse_when_rejects_garbage(bad):
    with pytest.raises(LabelError):
        L.parse_when(bad, date(2026, 11, 16))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_round_trip(capsys):
    with tempfile.TemporaryDirectory() as d:
        db = d + "/cli.db"
        s = EventStore(db); _record(s); s.close()           # no windows stored yet

        def run(*args):
            return L.main(["--db", db, *args], now=NOW)

        assert run("list") == 0                              # list refreshes windows first
        out = capsys.readouterr().out
        assert out.count("(unlabelled)") == 3 and "in progress" in out
        assert run("set", "10:05", "focused") == 0
        assert "Labelled Focused" in capsys.readouterr().out
        assert run("set", "10:05", "Distracted") == 0
        assert "Corrected Focused -> Distracted" in capsys.readouterr().out
        assert run("list", "--unlabelled") == 0
        assert "10:05-10:10" not in capsys.readouterr().out
        assert run("set", "10:10", "Focused") == 1           # in progress
        assert "in progress" in capsys.readouterr().err
        assert run("set", "10:00", "great") == 1
        assert run("clear", "10:05") == 0
        assert "Label removed" in capsys.readouterr().out
        assert run("--window-seconds", "420", "list") == 1   # does not divide a day
