import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest import mock

import pytest

from focus_tracker.features.session_context import RELATIONS, relation, session_context
from focus_tracker.logger import browser, content, session, storage
from focus_tracker.logger.agent import Agent
from focus_tracker.logger.input_counter import InputCounter
from focus_tracker.logger.session import SessionError, SessionManager
from focus_tracker.logger.storage import EventStore

T0 = datetime(2026, 11, 15, 9, 0, 0)


def at(seconds):
    return T0 + timedelta(seconds=seconds)


@pytest.fixture
def store():
    s = EventStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def mgr(store):
    return SessionManager(store)


def _events(store):
    return [(r[2], r[1], r[4], r[5]) for r in  # event, app, session_id, category
            store.rows(include_context=True)]


# --------------------------------------------------------------------------
# Session lifecycle
# --------------------------------------------------------------------------
def test_start_records_task_category_and_timestamp(mgr):
    s = mgr.start("Studying DSA", at(0))
    assert (s.task, s.task_category, s.status, s.started_at) == \
        ("Studying DSA", "studying", "active", T0)
    assert mgr.current() == s


def test_full_lifecycle_tracks_active_time_excluding_pauses(mgr):
    mgr.start("coding the logger", at(0))
    mgr.pause(at(600))                      # 10 min active
    assert mgr.current().active_seconds(at(900)) == 600   # pause in progress
    mgr.resume(at(900))                     # 5 min paused
    s = mgr.end(at(1500))                   # 10 more min active
    assert (s.status, s.paused_seconds, s.active_seconds(at(9999))) == ("ended", 300, 1200)
    assert s.ended_at == at(1500)
    assert mgr.current() is None


def test_ending_while_paused_counts_the_pause(mgr):
    mgr.start("research", at(0))
    mgr.pause(at(100))
    s = mgr.end(at(400))
    assert (s.paused_seconds, s.active_seconds(at(400))) == (300, 100)


def test_lifecycle_events_are_logged_with_session_id(mgr, store):
    s = mgr.start("research", at(0))
    mgr.pause(at(10)); mgr.resume(at(20)); mgr.end(at(30))
    rows = [(r[2], r[3], r[4]) for r in store.rows(include_context=True)]
    assert rows == [("session_start", 1, s.id), ("session_pause", 1, s.id),
                    ("session_resume", 1, s.id), ("session_end", 20, s.id)]


@pytest.mark.parametrize("step", ["pause", "resume", "end"])
def test_steps_without_open_session_are_rejected(mgr, step):
    with pytest.raises(SessionError):
        getattr(mgr, step)(at(0))


def test_invalid_transitions_are_rejected(mgr):
    mgr.start("coding", at(0))
    with pytest.raises(SessionError):
        mgr.resume(at(1))                   # not paused
    with pytest.raises(SessionError):
        mgr.start("another", at(1))         # one open session at a time
    mgr.pause(at(2))
    with pytest.raises(SessionError):
        mgr.pause(at(3))


def test_database_allows_only_one_open_session(store):
    """Guard against two processes racing past the application check."""
    store.conn.execute("INSERT INTO sessions (task, task_category, started_at, status)"
                       " VALUES ('a', 'other', 'x', 'paused')")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("INSERT INTO sessions (task, task_category, started_at, status)"
                           " VALUES ('b', 'other', 'x', 'active')")


def test_new_session_after_previous_ended(mgr):
    a = mgr.start("one", at(0)); mgr.end(at(1))
    b = mgr.start("two", at(2))
    assert b.id != a.id and mgr.current().id == b.id


@pytest.mark.parametrize("task, expected", [
    ("studying DSA", "studying"),
    ("revise for OS exam", "studying"),
    ("coding DSA problems", "coding"),
    ("debug the logger", "coding"),
    ("research on attention models", "research"),
    ("write thesis chapter", "writing"),
    ("write code for parser", "writing"),     # earliest keyword wins
    ("planning my week", "other"),
    ("decoder rings", "other"),               # 'code' inside a word is not a match
])
def test_classify_task(task, expected):
    assert session.classify_task(task) == expected


def test_category_override_and_validation(mgr):
    assert mgr.start("misc stuff", at(0), category="research").task_category == "research"
    mgr.end(at(1))
    with pytest.raises(SessionError):
        mgr.start("x", at(2), category="gaming")


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 42])
def test_empty_or_non_text_task_rejected(mgr, bad):
    with pytest.raises(SessionError):
        mgr.start(bad, at(0))


def test_task_is_cleaned_and_capped(mgr):
    s = mgr.start("  studying\x00  graphs \x1b[31m" + "x" * 200, at(0))
    assert "\x00" not in s.task and "\x1b" not in s.task and "  " not in s.task
    assert len(s.task) <= session.MAX_TASK_CHARS
    assert s.task.startswith("studying graphs")


# --------------------------------------------------------------------------
# Persistence across restarts (file-backed database, fresh objects)
# --------------------------------------------------------------------------
def test_session_survives_restart_and_pause_time_is_kept():
    with tempfile.TemporaryDirectory() as d:
        path = d + "/a.db"
        st = EventStore(path); SessionManager(st).start("studying DSA", at(0))
        SessionManager(st).pause(at(60)); st.close()

        st = EventStore(path)                       # "restart"
        m = SessionManager(st)
        assert (m.current().task, m.current().status) == ("studying DSA", "paused")
        m.resume(at(160))
        s = m.end(at(260))
        assert (s.paused_seconds, s.active_seconds(at(260))) == (100, 160)
        st.close()


def test_agent_started_after_restart_tags_rows_with_open_session():
    with tempfile.TemporaryDirectory() as d:
        path = d + "/a.db"
        st = EventStore(path); sid = SessionManager(st).start("coding", at(0)).id; st.close()

        st = EventStore(path)
        Agent(st, InputCounter(), get_app=lambda: "Code.exe", get_idle=lambda: 0).step(at(5))
        assert _events(st)[-1] == ("active", "Code.exe", sid, None)
        st.close()


def test_session_changes_from_another_connection_are_seen_by_agent():
    """The CLI runs in a separate process: simulate with a second connection."""
    with tempfile.TemporaryDirectory() as d:
        path = d + "/a.db"
        agent_store, cli_store = EventStore(path), EventStore(path)
        apps = iter(["Code.exe", "chrome.exe", "Code.exe"])
        agent = Agent(agent_store, InputCounter(), get_app=lambda: next(apps),
                      get_idle=lambda: 0)
        agent.step(at(0))
        sid = SessionManager(cli_store).start("coding", at(1)).id
        agent.step(at(2))
        SessionManager(cli_store).pause(at(3))
        agent.step(at(4))
        tagged = [(e, a, s) for e, a, s, _ in _events(agent_store) if a]
        assert tagged == [("active", "Code.exe", None), ("switched", "chrome.exe", sid),
                          ("switched", "Code.exe", None)]       # paused => untagged
        agent_store.close(); cli_store.close()


# --------------------------------------------------------------------------
# Unfinished sessions: recovery on agent start
# --------------------------------------------------------------------------
HOUR = 3600


def _activity(store, sid, seconds):
    store.log(at(seconds), "Code.exe", "keyboard_count", 5, session_id=sid)


def test_user_end_is_marked_user(mgr):
    mgr.start("coding", at(0))
    assert mgr.end(at(10)).end_reason == "user"


def test_stale_open_session_is_closed_at_its_last_activity(mgr, store):
    sid = mgr.start("studying DSA", at(0)).id
    _activity(store, sid, 600)                          # last sign of life at +10 min
    s = mgr.recover_stale(at(600 + 3 * HOUR))           # agent restarted 3 h later
    assert (s.status, s.end_reason, s.ended_at) == ("ended", "recovered", at(600))
    assert s.active_seconds(at(99 * HOUR)) == 600       # off-time is not counted
    assert mgr.current() is None
    last = store.rows(include_context=True)[-1]
    assert (last[0], last[2], last[3], last[4]) == (at(600).isoformat(), "session_end", 600, sid)


def test_recent_session_survives_a_quick_restart(mgr, store):
    sid = mgr.start("coding", at(0)).id
    _activity(store, sid, 600)
    assert mgr.recover_stale(at(600 + 30 * 60)) is None  # 30 min: reboot, keep going
    assert mgr.current().status == "active"


def test_stale_paused_session_ends_at_pause_without_counting_pause(mgr, store):
    sid = mgr.start("research", at(0)).id
    _activity(store, sid, 300)
    mgr.pause(at(400))
    s = mgr.recover_stale(at(400 + 5 * HOUR))
    assert (s.ended_at, s.end_reason, s.active_seconds(at(400))) == (at(400), "recovered", 400)


def test_untagged_activity_since_session_counts_as_recent(mgr, store):
    """Activity of any kind (e.g. while paused) means the agent was alive."""
    mgr.start("coding", at(0)); mgr.pause(at(10))
    store.log(at(3 * HOUR), "Code.exe", "keyboard_count", 3)   # untagged (paused)
    assert mgr.recover_stale(at(3 * HOUR + 60)) is None


def test_recover_without_open_session_is_noop(mgr):
    assert mgr.recover_stale(at(10 * HOUR)) is None
    mgr.start("x", at(0)); mgr.end(at(5))
    assert mgr.recover_stale(at(10 * HOUR)) is None


def test_custom_stale_threshold(mgr, store):
    mgr.start("coding", at(0))
    assert mgr.recover_stale(at(20 * 60), stale_after=10 * 60).end_reason == "recovered"


def test_new_session_can_start_after_recovery(mgr):
    mgr.start("one", at(0))
    mgr.recover_stale(at(5 * HOUR))
    assert mgr.start("two", at(5 * HOUR)).task == "two"


def test_session_ids_stay_unique_across_restarts():
    with tempfile.TemporaryDirectory() as d:
        path, ids = d + "/a.db", []
        for i in range(3):
            st = EventStore(path); m = SessionManager(st)
            ids.append(m.start(f"s{i}", at(i * 10)).id); m.end(at(i * 10 + 5)); st.close()
        assert len(set(ids)) == 3


def test_sessions_table_without_end_reason_is_migrated():
    with tempfile.TemporaryDirectory() as d:
        path = d + "/old.db"
        old = sqlite3.connect(path)
        old.executescript(
            "CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL,"
            " task_category TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT,"
            " status TEXT NOT NULL, paused_at TEXT, paused_seconds INTEGER NOT NULL DEFAULT 0);"
            "INSERT INTO sessions (task, task_category, started_at, status)"
            " VALUES ('old', 'other', '2026-11-15T09:00:00', 'active');")
        old.commit(); old.close()
        st = EventStore(path); m = SessionManager(st)
        assert (m.current().task, m.current().end_reason) == ("old", None)
        assert m.end(at(60)).end_reason == "user"
        st.close()


# --------------------------------------------------------------------------
# Storage: new columns, validation, migration
# --------------------------------------------------------------------------
def test_new_lifecycle_events_are_whitelisted_but_labels_still_are_not():
    for e in ("session_start", "session_pause", "session_resume", "session_end",
              "content_changed"):
        assert e in storage.ALLOWED_EVENTS
    assert not {"focused", "distracted", "related", "unrelated"} & storage.ALLOWED_EVENTS


@pytest.mark.parametrize("bad", ["Educational", "fun", "Two Sum - LeetCode", ""])
def test_storage_rejects_unknown_content_category(store, bad):
    with pytest.raises(ValueError):
        store.log(T0, "chrome.exe", "active", content_category=bad)


def test_storage_rejects_category_for_non_browser(store):
    with pytest.raises(ValueError):
        store.log(T0, "Code.exe", "active", content_category="development")


@pytest.mark.parametrize("bad", ["1", 1.5, True])
def test_storage_rejects_non_integer_session_id(store, bad):
    with pytest.raises(ValueError):
        store.log(T0, "Code.exe", "active", session_id=bad)


def test_sessions_table_has_no_url_or_title_columns(store):
    cols = [r[1] for r in store.conn.execute("PRAGMA table_info(sessions)")]
    assert cols == ["id", "task", "task_category", "started_at", "ended_at", "status",
                    "paused_at", "paused_seconds", "end_reason"]


def test_website_era_database_gets_session_columns():
    with tempfile.TemporaryDirectory() as d:
        path = d + "/old.db"
        old = sqlite3.connect(path)
        old.executescript(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
            " app TEXT, event TEXT NOT NULL, value INTEGER NOT NULL DEFAULT 1, website TEXT);"
            "INSERT INTO events (ts, app, event, value, website)"
            " VALUES ('2026-11-03T01:02:22','chrome.exe','active',1,'github.com');")
        old.commit(); old.close()
        s = EventStore(path)
        assert s.rows(include_website=True, include_context=True) == [
            ("2026-11-03T01:02:22", "chrome.exe", "active", 1, "github.com",
             None, None, None)]
        SessionManager(s).start("coding", T0)
        s.close()


# --------------------------------------------------------------------------
# Content categorisation (pure, local)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("host, expected", [
    ("github.com", "development"), ("gist.github.com", "development"),
    ("docs.python.org", "documentation"), ("project.readthedocs.io", "documentation"),
    ("en.wikipedia.org", "educational"), ("cs.stanford.edu", "educational"),
    ("netflix.com", "entertainment"), ("www.reddit.com", "social"),
    ("mail.google.com", "communication"),     # more specific than google.com
    ("google.com", "search"), ("amazon.in", "shopping"),
    ("news.ycombinator.com", "news"),
    ("dropbox.com", "other"),                 # 'x.com' must not match as a suffix
    ("example.org", "other"),
    (None, None),
])
def test_classify_content_by_hostname(host, expected):
    assert content.classify_content(host) == expected


@pytest.mark.parametrize("title, expected", [
    ("Dynamic Programming Lecture 12 | MIT 6.006 - YouTube", "educational"),
    ("Two Sum - LeetCode Explained in Python - YouTube", "educational"),
    ("Artist - Song Name (Official Music Video) - YouTube", "entertainment"),
    ("Funny cat prank compilation - YouTube", "entertainment"),
    ("YouTube", "other"),
    ("Python song", "other"),                 # tie => no guess
])
def test_category_from_title(title, expected):
    assert content.category_from_title(title) == expected


def test_mixed_platform_without_title_is_unknown_not_guessed():
    assert content.classify_content("youtube.com") is None
    assert content.classify_content("m.youtube.com", "DSA lecture") == "educational"


def test_only_mixed_content_and_video_platforms_need_a_title():
    for host in ("youtube.com", "m.youtube.com", "netflix.com", "twitch.tv", "medium.com"):
        assert content.needs_title(host)
    for host in ("mail.google.com", "google.com", "bank.example.com", "github.com", None):
        assert not content.needs_title(host)


def test_get_active_title_is_none_off_windows():
    with mock.patch.object(browser.sys, "platform", "linux"):
        assert browser.get_active_title() is None


def test_get_active_title_never_raises():
    with mock.patch.object(browser.sys, "platform", "win32"), \
         mock.patch.object(browser, "_foreground_hwnd", side_effect=OSError):
        assert browser.get_active_title() is None


# --------------------------------------------------------------------------
# Agent: category stamping, title privacy, content_changed
# --------------------------------------------------------------------------
def _agent(store, apps, sites, titles=None):
    app_it, site_it = iter(apps), iter(sites)
    title_calls = []
    get_title = None
    if titles is not None:
        title_it = iter(titles)
        def get_title():
            title_calls.append(1)
            return next(title_it)
    agent = Agent(store, InputCounter(), get_app=lambda: next(app_it),
                  get_site=lambda: next(site_it), get_idle=lambda: 0,
                  site_poll_seconds=0, get_title=get_title)
    return agent, title_calls


def _run(agent, n):
    for i in range(n):
        agent.step(at(i))


def test_rows_carry_content_category_only_for_browser(store):
    agent, _ = _agent(store, ["chrome.exe", "chrome.exe", "Code.exe"],
                      ["github.com", "netflix.com"])
    _run(agent, 3)
    assert [(e, a, c) for e, a, _, c in _events(store)] == [
        ("active", "chrome.exe", "development"),
        ("site_switched", "chrome.exe", "entertainment"),
        ("switched", "Code.exe", None),
    ]


def test_title_is_never_read_unless_opted_in(store):
    agent, _ = _agent(store, ["chrome.exe"], ["youtube.com"])  # get_title=None
    _run(agent, 1)
    assert _events(store) == [("active", "chrome.exe", None, None)]


def test_title_is_never_read_for_other_sites(store):
    agent, calls = _agent(store, ["chrome.exe"] * 3,
                          ["mail.google.com", "google.com", "github.com"], titles=[])
    _run(agent, 3)
    assert calls == []


def test_video_change_on_same_site_logs_content_changed(store):
    agent, _ = _agent(store, ["chrome.exe"] * 4, ["youtube.com"] * 4, titles=[
        "Graph algorithms lecture - YouTube", "Graph algorithms lecture - YouTube",
        None,                                    # unreadable: not a change
        "Movie trailer reaction - YouTube"])
    _run(agent, 4)
    assert [(e, c) for e, _, _, c in _events(store)] == [
        ("active", "educational"), ("content_changed", "entertainment")]


def _titles(store):
    return [(r[2], r[5], r[6]) for r in store.rows(include_context=True)]


def test_video_title_is_stored_cleaned_on_video_platforms(store):
    agent, _ = _agent(store, ["chrome.exe", "chrome.exe"], ["youtube.com", "netflix.com"],
                      titles=["(3) Graph algorithms lecture - YouTube - Google Chrome",
                              "Dark | Netflix - Google Chrome"])
    _run(agent, 2)
    assert _titles(store) == [
        ("active", "educational", "Graph algorithms lecture"),
        ("site_switched", "entertainment", "Dark"),   # netflix: category from host
    ]


def test_title_used_but_not_stored_on_non_video_title_hosts(store):
    secret = "My Private Health Notes - Medium - Google Chrome"
    agent, calls = _agent(store, ["chrome.exe"], ["medium.com"], titles=[secret])
    _run(agent, 1)
    assert calls == [1]                     # read for the category ...
    dump = repr(store.conn.execute("SELECT * FROM events").fetchall())
    assert "Private" not in dump and "Health" not in dump    # ... but not stored
    assert _titles(store) == [("active", "other", None)]


def test_no_title_stored_without_opt_in(store):
    agent, _ = _agent(store, ["chrome.exe"], ["netflix.com"])   # get_title=None
    _run(agent, 1)
    assert _titles(store) == [("active", "entertainment", None)]


def test_next_video_with_same_category_logs_content_changed(store):
    agent, _ = _agent(store, ["chrome.exe"] * 4, ["youtube.com"] * 4, titles=[
        "DSA lecture 1 - YouTube", "DSA lecture 1 - YouTube", None,
        "DSA lecture 2 - YouTube"])
    _run(agent, 4)
    assert _titles(store) == [("active", "educational", "DSA lecture 1"),
                              ("content_changed", "educational", "DSA lecture 2")]


def test_title_not_stored_after_leaving_video_site(store):
    agent, _ = _agent(store, ["chrome.exe"] * 2, ["youtube.com", "github.com"],
                      titles=["Some video - YouTube"])
    _run(agent, 2)
    assert _titles(store)[-1] == ("site_switched", "development", None)


@pytest.mark.parametrize("raw, expected", [
    ("(12) Two Sum Explained - YouTube - Google Chrome", "Two Sum Explained"),
    ("Two Sum - YouTube and 3 more pages - Microsoft​ Edge", "Two Sum"),
    ("Lecture 5 — Mozilla Firefox", "Lecture 5"),
    ("Stranger Things | Netflix", "Stranger Things"),
    ("YouTube - Google Chrome", None),       # home page: no video
    ("Netflix", None), ("", None), (None, None), ("   ", None),
    ("tab\tnew\x00line\x1b", "tab newline"),
])
def test_clean_title(raw, expected):
    assert content.clean_title(raw) == expected


def test_clean_title_caps_length():
    assert len(content.clean_title("x" * 1000)) == content.MAX_TITLE_CHARS


@pytest.mark.parametrize("website, title", [
    ("github.com", "Some title"),            # not a video platform
    (None, "Some title"),
    ("youtube.com", "two\nlines"),
    ("youtube.com", "x" * 201),
    ("youtube.com", ""),
])
def test_storage_rejects_title_outside_video_sites_or_unclean(store, website, title):
    with pytest.raises(ValueError):
        store.log(T0, "chrome.exe", "active", website=website, content_title=title)


def test_storage_accepts_title_on_video_site(store):
    store.log(T0, "chrome.exe", "active", website="www.youtube.com",
              content_category="educational", content_title="DSA lecture 1")
    assert store.rows(include_context=True)[0][-1] == "DSA lecture 1"


def test_title_reader_failure_does_not_crash(store):
    def boom():
        raise RuntimeError("no window")
    agent = Agent(store, InputCounter(), get_app=lambda: "chrome.exe",
                  get_site=lambda: "youtube.com", get_idle=lambda: 0,
                  site_poll_seconds=0, get_title=boom)
    _run(agent, 2)
    assert _events(store) == [("active", "chrome.exe", None, None)]


# --------------------------------------------------------------------------
# Context features: task x content, never a focus label
# --------------------------------------------------------------------------
@pytest.mark.parametrize("task, cat, expected", [
    ("studying", "educational", "related"),
    ("studying", "entertainment", "unrelated"),
    ("coding", "development", "related"),
    ("coding", "social", "unrelated"),
    ("writing", "documentation", "related"),
    ("other", "educational", "unknown"),
    ("studying", None, "unknown"),
    ("studying", "other", "unknown"),
    ("coding", "communication", "unknown"),
])
def test_relation(task, cat, expected):
    assert relation(task, cat) == expected


def test_relation_never_returns_a_focus_label():
    tasks = session.TASK_CATEGORIES + (None,)
    cats = tuple(content.CONTENT_CATEGORIES) + (None,)
    outputs = {relation(t, c) for t in tasks for c in cats}
    assert outputs <= set(RELATIONS)
    assert not outputs & {"focused", "distracted", "inactive"}


def test_session_context_summarises_only_that_sessions_rows(store, mgr):
    store.log(at(0), "chrome.exe", "active", content_category="social")  # before session
    s = mgr.start("studying DSA", at(1))
    agent, _ = _agent(store, ["chrome.exe"] * 3, ["en.wikipedia.org", "leetcode.com",
                                                  "instagram.com"])
    _run(agent, 3)
    mgr.pause(at(10))
    ctx = session_context(store, s.id, now=at(70))
    assert ctx == {
        "session_id": s.id, "task_category": "studying", "status": "paused",
        "active_seconds": 9,
        "content_category_counts": {"educational": 1, "development": 1, "social": 1},
        "relation_counts": {"related": 1, "unrelated": 2, "unknown": 0},
    }
    assert "task" not in ctx            # the free-text task is not part of features


def test_session_context_unknown_id(store):
    with pytest.raises(KeyError):
        session_context(store, 999)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_round_trip(capsys):
    with tempfile.TemporaryDirectory() as d:
        db = d + "/cli.db"
        assert session.main(["--db", db, "start", "studying DSA"]) == 0
        assert session.main(["--db", db, "pause"]) == 0
        assert session.main(["--db", db, "status"]) == 0
        assert "[paused]" in capsys.readouterr().out
        assert session.main(["--db", db, "pause"]) == 1          # invalid step
        assert "Error" in capsys.readouterr().err
        assert session.main(["--db", db, "resume"]) == 0
        assert session.main(["--db", db, "end"]) == 0
        assert session.main(["--db", db, "status"]) == 0
        assert "No open session" in capsys.readouterr().out
