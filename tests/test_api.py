"""Local app API: temp databases, fixed clock, fake desktop probes."""
import csv
import io
import os
import time
from datetime import datetime, timedelta

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from focus_tracker.api.app import create_app  # noqa: E402
from focus_tracker.api.developer import TeamStore, agreement_stats  # noqa: E402
from focus_tracker.api.export import EXPORT_COLUMNS  # noqa: E402
from focus_tracker.api.logger_control import LoggerController, LoggerError  # noqa: E402
from focus_tracker.logger.agent import Agent  # noqa: E402
from focus_tracker.logger.storage import EventStore  # noqa: E402

T0 = datetime(2026, 11, 23, 10, 0, 0)
NOW = T0 + timedelta(minutes=20)


def t(m, s=0):
    return T0 + timedelta(minutes=m, seconds=s)


class FakeCounter:
    last_input = 0.0

    def drain(self):
        return 3, 10


def fake_factory(store, websites, read_titles):
    agent = Agent(store, FakeCounter(), get_app=lambda: "Code.exe",
                  get_idle=lambda: 0.0, window_seconds=300)
    return agent, lambda: None


@pytest.fixture
def paths(tmp_path):
    db = tmp_path / "activity.db"
    store = EventStore(db)
    # One logger run 10:00-10:12 -> windows 10:00, 10:05 (complete), 10:10 (run ended)
    store.log(t(0), None, "agent_start")
    store.log(t(0, 1), "Code.exe", "active")
    store.log(t(3), "chrome.exe", "switched", website="github.com")
    store.log(t(6), "Code.exe", "switched")
    store.log(t(7), "Code.exe", "keyboard_count", 40)
    store.log(t(12), None, "agent_stop")
    store.close()
    return {"db": db, "team": tmp_path / "team.db", "dir": tmp_path}


@pytest.fixture
def client(paths):
    ctl = LoggerController(paths["db"], agent_factory=fake_factory,
                           clock=lambda: NOW, poll_seconds=0.01)
    app = create_app(paths["db"], paths["team"], controller=ctl,
                     clock=lambda: NOW, static_dir=None)
    with TestClient(app) as c:
        yield c
    ctl.shutdown()


def ws(m):
    return t(m).isoformat(timespec="seconds")


# -- windows & labels --------------------------------------------------------
def test_windows_listed_newest_first_with_features(client):
    rows = client.get("/api/windows", params={"date": "2026-11-23"}).json()
    assert [r["window_start"] for r in rows] == [ws(10), ws(5), ws(0)]
    first = rows[-1]
    assert first["switch_count"] == 1 and first["complete"] and first["label"] is None
    assert "website" not in first and "app_seconds" not in first


def test_bulk_labelling_correction_and_clear(client):
    r = client.post("/api/labels", json={"window_starts": [ws(0), ws(5)],
                                          "label": "focused"}).json()
    assert all(x["ok"] and x["previous"] is None for x in r["results"])
    r = client.post("/api/labels", json={"window_starts": [ws(5)],
                                          "label": "Distracted"}).json()
    assert r["results"][0]["previous"] == "Focused"
    labels = {x["window_start"]: x["label"] for x in client.get("/api/windows").json()}
    assert labels[ws(0)] == "Focused" and labels[ws(5)] == "Distracted"
    assert client.request("DELETE", "/api/labels",
                          json={"window_starts": [ws(0)]}).json() == {"cleared": 1}
    counts = client.get("/api/status").json()["label_counts"]
    assert counts == {"Focused": 0, "Distracted": 1, "Inactive": 0}


def test_bulk_labelling_reports_per_window_errors(client):
    r = client.post("/api/labels", json={"window_starts": [ws(0), ws(30)],
                                          "label": "Inactive"}).json()["results"]
    assert r[0]["ok"] and not r[1]["ok"] and "no recorded activity" in r[1]["error"]
    assert client.post("/api/labels", json={"window_starts": [ws(0)],
                                             "label": "bored"}).status_code == 422


def test_in_progress_window_cannot_be_labelled(paths):
    store = EventStore(paths["db"])
    store.log(t(30), None, "agent_start")
    store.log(t(31), "Code.exe", "active")        # run still open at NOW + no stop
    store.close()
    app = create_app(paths["db"], paths["team"], clock=lambda: t(32), static_dir=None,
                     controller=LoggerController(paths["db"], agent_factory=fake_factory))
    with TestClient(app) as c:
        r = c.post("/api/labels", json={"window_starts": [ws(30)],
                                         "label": "Focused"}).json()["results"][0]
    assert not r["ok"] and "in progress" in r["error"]


# -- sessions -----------------------------------------------------------------------
def test_session_lifecycle_and_errors(client):
    s = client.post("/api/session/start", json={"task": "studying DSA"}).json()
    assert s["status"] == "active" and s["task_category"] == "studying"
    assert client.post("/api/session/start", json={"task": "x"}).status_code == 409
    assert client.post("/api/session/pause").json()["status"] == "paused"
    assert client.post("/api/session/resume").json()["status"] == "active"
    assert client.post("/api/session/end").json()["end_reason"] == "user"
    assert client.post("/api/session/end").status_code == 409
    assert client.post("/api/session/explode").status_code == 404


# -- export -------------------------------------------------------------------------
def test_export_contains_only_features_and_user_labels(client):
    client.post("/api/session/start", json={"task": "secret project name"})
    client.post("/api/labels", json={"window_starts": [ws(0)], "label": "Focused"})
    r = client.get("/api/export", params={"participant": "P3"})
    assert r.status_code == 200 and "focus_windows_P3" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert tuple(rows[0].keys()) == EXPORT_COLUMNS
    assert len(rows) == 1 and rows[0]["label"] == "Focused"
    assert rows[0]["label_source"] == "user" and rows[0]["participant"] == "P3"
    assert "secret project" not in r.text                      # task text stays out
    assert rows[0]["top_app"] == "Code.exe"            # 10:00:01-10:03 = 179 s
    assert rows[0]["app_seconds"] == "Code.exe:179; chrome.exe:120"
    assert rows[0]["hostnames"] == "github.com:60"     # 3-min gap capped at 60 s
    assert rows[0]["browser_seconds"] == "60" and rows[0]["share_unknown"] == "1.0"


@pytest.mark.parametrize("bad", ["", "Aanvi Gunjiyal", "x" * 30, "a/b"])
def test_export_rejects_non_pseudonyms(client, bad):
    assert client.get("/api/export", params={"participant": bad}).status_code == 422


# -- logger control -----------------------------------------------------------------
def _wait(pred, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_logger_start_stop_writes_run_boundaries(client, paths):
    assert client.post("/api/logger/start", json={"websites": False}).status_code == 200
    assert client.get("/api/status").json()["logger"]["running"]
    assert client.post("/api/logger/start", json={}).status_code == 409
    assert (paths["dir"] / "logger.lock").exists()
    assert client.post("/api/logger/stop").json()["running"] is False
    assert not (paths["dir"] / "logger.lock").exists()
    events = [r[2] for r in EventStore(paths["db"]).rows()]
    assert events[-1] == "agent_stop" and events.count("agent_start") == 2
    assert client.post("/api/logger/stop").status_code == 409


def test_lock_held_by_another_live_process_blocks_start(paths):
    import subprocess
    import sys
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (paths["dir"] / "logger.lock").write_text(str(other.pid))
        ctl = LoggerController(paths["db"], agent_factory=fake_factory)
        with pytest.raises(LoggerError, match="another logger"):
            ctl.start()
    finally:
        other.kill()


def test_stale_lock_from_dead_process_is_taken_over(paths):
    (paths["dir"] / "logger.lock").write_text("999999")
    ctl = LoggerController(paths["db"], agent_factory=fake_factory, poll_seconds=0.01)
    ctl.start()
    try:
        assert ctl.running
    finally:
        ctl.stop()


def test_factory_failure_is_reported_not_fatal(paths):
    def broken(store, websites, read_titles):
        raise OSError("no keyboard hook")
    ctl = LoggerController(paths["db"], agent_factory=broken)
    with pytest.raises(LoggerError, match="no keyboard hook"):
        ctl.start()
    assert not ctl.running and not (paths["dir"] / "logger.lock").exists()


# -- developer annotations ------------------------------------------------------------
def _export(client, participant="P3"):
    client.post("/api/labels", json={"window_starts": [ws(0)], "label": "Focused"})
    client.post("/api/labels", json={"window_starts": [ws(5)], "label": "Distracted"})
    return client.get("/api/export", params={"participant": participant}).text


def test_developer_import_label_agreement_and_export(client, paths):
    text = _export(client)
    r = client.post("/api/dev/import", files={"file": ("e.csv", text, "text/csv")})
    assert r.json() == {"rows": 2, "participants": {"P3": 2}}
    assert client.get("/api/dev/participants").json() == ["P3"]
    blind = client.get("/api/dev/windows", params={"blind": True}).json()
    assert all(w["user_label"] is None for w in blind)
    items = [{"participant": "P3", "window_start": ws(0)},
             {"participant": "P3", "window_start": ws(5)}]
    r = client.post("/api/dev/labels", json={"items": items, "label": "Focused",
                                              "annotator": "dev1"}).json()
    assert all(x["ok"] for x in r["results"])
    ag = client.get("/api/dev/agreement").json()
    assert ag["P3"]["n"] == 2 and ag["P3"]["agreement"] == 0.5
    assert ag["P3"]["confusion"]["Distracted"]["Focused"] == 1
    out = list(csv.DictReader(io.StringIO(client.get("/api/dev/export").text)))
    assert {(o["user_label"], o["developer_label"], o["annotator"]) for o in out} == \
        {("Focused", "Focused", "dev1"), ("Distracted", "Focused", "dev1")}
    # user ground truth untouched
    labels = dict(EventStore(paths["db"]).conn.execute(
        "SELECT window_start, label FROM window_labels"))
    assert labels == {ws(0): "Focused", ws(5): "Distracted"}


def test_developer_label_requires_annotator_and_known_window(client):
    client.post("/api/dev/import", files={"file": ("e.csv", _export(client), "text/csv")})
    item = {"participant": "P3", "window_start": ws(0)}
    assert client.post("/api/dev/labels", json={"items": [item], "label": "Focused",
                                                 "annotator": " "}).status_code == 422
    r = client.post("/api/dev/labels", json={
        "items": [{"participant": "P9", "window_start": ws(0)}], "label": "Focused",
        "annotator": "dev1"}).json()
    assert r["results"][0]["error"] == "unknown window"


@pytest.mark.parametrize("mutate, msg", [
    (lambda h: h + ",website", "unexpected columns"),
    (lambda h: h.replace(",label_source", ""), "missing columns"),
])
def test_developer_import_rejects_other_files(client, mutate, msg):
    text = _export(client)
    header, body = text.split("\n", 1)
    r = client.post("/api/dev/import",
                    files={"file": ("e.csv", mutate(header) + "\n", "text/csv")})
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_developer_import_rejects_real_names_and_bad_labels(client):
    text = _export(client)
    for old, new in (("P3,", "Aanvi G,"), ("Focused", "Sleepy")):
        r = client.post("/api/dev/import",
                        files={"file": ("e.csv", text.replace(old, new), "text/csv")})
        assert r.status_code == 422


def test_agreement_stats_kappa():
    pairs = [("Focused", "Focused")] * 6 + [("Distracted", "Distracted")] * 2 + \
            [("Focused", "Distracted")] * 2
    s = agreement_stats(pairs)
    # po = 0.8; pe = (8*6 + 2*4)/100 = 0.56; kappa = 0.24/0.44
    assert s["agreement"] == 0.8 and s["kappa"] == round(0.24 / 0.44, 4)
    assert agreement_stats([])["kappa"] is None
    assert agreement_stats([("Focused", "Focused")])["kappa"] is None   # pe == 1


def test_team_store_never_opens_activity_db(paths):
    team = TeamStore(paths["team"])
    tables = {r[0] for r in team.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"team_windows", "developer_labels"}
    team.close()


# -- settings & quit ------------------------------------------------------------
def test_settings_first_run_and_pseudonym(client, paths):
    s = client.get("/api/settings").json()
    assert s["participant"] == "" and s["onboarded"] is False and s["desktop"] is False
    s = client.post("/api/settings", json={"participant": "P2", "onboarded": True}).json()
    assert s["participant"] == "P2" and s["onboarded"] is True
    assert (paths["dir"] / "settings.json").exists()
    assert client.get("/api/status").json()["settings"]["participant"] == "P2"
    assert client.post("/api/settings", json={"participant": "Real Name"}).status_code == 422
    assert client.get("/api/settings").json()["participant"] == "P2"


def test_autostart_only_in_desktop_build(client):
    r = client.post("/api/settings", json={"autostart": True})
    assert r.status_code == 422 and "desktop app" in r.json()["detail"]


def test_quit_stops_the_logger(client):
    client.post("/api/logger/start", json={"websites": False})
    assert client.post("/api/app/quit").json() == {"quitting": False}   # no server in tests
    assert client.get("/api/status").json()["logger"]["running"] is False


# -- browser context in the export --------------------------------------------------
def test_export_includes_categories_hostnames_and_video_titles(tmp_path):
    db = tmp_path / "a.db"
    store = EventStore(db)
    store.log(t(0), None, "agent_start")
    store.log(t(0), "Code.exe", "active")
    store.log(t(1), "chrome.exe", "switched", website="github.com", content_category="development")
    store.log(t(1, 30), "chrome.exe", "site_switched", website="youtube.com",
              content_category="educational", content_title="Graph Algorithms, Lecture 3")
    store.log(t(2), "chrome.exe", "content_changed", website="youtube.com",
              content_category="entertainment", content_title="Funny cats | compilation")
    store.log(t(2, 30), "Code.exe", "switched")
    store.log(t(6), None, "agent_stop")
    store.close()
    app = create_app(db, tmp_path / "team.db", clock=lambda: NOW, static_dir=None,
                     controller=LoggerController(db, agent_factory=fake_factory))
    with TestClient(app) as c:
        c.post("/api/labels", json={"window_starts": [ws(0), ws(5)], "label": "Distracted"})
        text = c.get("/api/export", params={"participant": "P1"}).text
        row, empty = list(csv.DictReader(io.StringIO(text)))
        assert row["browser_seconds"] == "90"
        assert row["hostnames"] == "youtube.com:60; github.com:30"
        assert row["video_titles"] == "Graph Algorithms, Lecture 3 | Funny cats | compilation"
        assert row["share_development"] == "0.3333" and row["share_educational"] == "0.3333"
        assert row["share_entertainment"] == "0.3333" and row["share_social"] == "0.0"
        assert row["distinct_hosts"] == "2" and float(row["site_switches_per_min"]) == 0.2
        # A window without browser time: shares are empty (unknown), never 0
        assert empty["browser_seconds"] == "0" and empty["share_development"] == ""
        assert empty["hostnames"] == "" and empty["video_titles"] == ""
        # Developer side keeps the new columns
        assert c.post("/api/dev/import", files={"file": ("e.csv", text, "text/csv")}).status_code == 200
        dev = {w["window_start"]: w for w in c.get("/api/dev/windows").json()}
        assert dev[ws(0)]["hostnames"] == "youtube.com:60; github.com:30"
        assert dev[ws(0)]["top_app"] == "Code.exe" and "chrome.exe:90" in dev[ws(0)]["app_seconds"]


def test_developer_accepts_exports_from_before_browser_columns(client):
    from focus_tracker.api.export import BASE_COLUMNS, LABEL_COLUMNS
    text = _export(client)
    old = list(csv.DictReader(io.StringIO(text)))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(BASE_COLUMNS + LABEL_COLUMNS), extrasaction="ignore")
    w.writeheader(); w.writerows(old)
    r = client.post("/api/dev/import", files={"file": ("old.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200 and r.json()["rows"] == 2


def test_old_team_db_is_upgraded(tmp_path):
    import sqlite3
    from focus_tracker.api.export import BASE_COLUMNS, LABEL_COLUMNS
    conn = sqlite3.connect(tmp_path / "team.db")
    conn.execute(f"CREATE TABLE team_windows ({', '.join(BASE_COLUMNS + LABEL_COLUMNS)},"
                 " PRIMARY KEY (participant, window_start, window_seconds))")
    conn.commit(); conn.close()
    team = TeamStore(tmp_path / "team.db")
    cols = {r[1] for r in team.conn.execute("PRAGMA table_info(team_windows)")}
    assert {"hostnames", "video_titles", "share_social"} <= cols
    team.close()


# -- predictions (model estimates, never labels) -----------------------------
@pytest.fixture(params=["rule", "gbt_z"])
def model_dir(request, tmp_path_factory):
    pytest.importorskip("lightgbm")
    import pandas as pd
    from test_ml import synth_rows
    from focus_tracker.models import train as T
    rows = synth_rows("P1", days=2, per_day=40, seed=1) + synth_rows("P2", days=2, per_day=40,
                                                                      seed=2)
    df = pd.DataFrame(rows)
    df["window_start"] = pd.to_datetime(df["window_start"])
    root = tmp_path_factory.mktemp("models")
    T.save(T.train_final(df, request.param), root, df)
    return root


@pytest.fixture
def pclient(paths, model_dir):
    ctl = LoggerController(paths["db"], agent_factory=fake_factory,
                           clock=lambda: NOW, poll_seconds=0.01)
    app = create_app(paths["db"], paths["team"], controller=ctl, clock=lambda: NOW,
                     static_dir=None, model_dir=model_dir)
    with TestClient(app) as c:
        yield c
    ctl.shutdown()


def _label_rows(db):
    store = EventStore(db)
    try:
        return store.conn.execute("SELECT * FROM window_labels ORDER BY window_start").fetchall()
    finally:
        store.close()


def test_predictions_cover_complete_windows_only(pclient, paths):
    r = pclient.get("/api/predictions", params={"date": "2026-11-23"})
    assert r.status_code == 200
    j = r.json()
    complete = [w["window_start"] for w in pclient.get(
        "/api/windows", params={"date": "2026-11-23"}).json() if w["complete"]]
    assert [w["window_start"] for w in j["windows"]] == complete and complete
    for w in j["windows"]:
        assert w["predicted"] in ("Focused", "Distracted", "Inactive")
        assert sum(w["probabilities"].values()) == pytest.approx(1, abs=0.01)
        assert w["user_label"] is None
    assert j["model"]["participants"] == 2
    assert sum(j["summary"]["minutes"].values()) > 0


def test_predictions_are_never_stored_or_exported(pclient, paths):
    pclient.post("/api/labels", json={"window_starts": [ws(0)], "label": "Focused"})
    before = _label_rows(paths["db"])
    counts = pclient.get("/api/status").json()["label_counts"]
    export = pclient.get("/api/export", params={"participant": "P9"}).text
    j = pclient.get("/api/predictions", params={"date": "2026-11-23"}).json()
    assert _label_rows(paths["db"]) == before
    assert pclient.get("/api/status").json()["label_counts"] == counts
    assert pclient.get("/api/export", params={"participant": "P9"}).text == export
    by_ws = {w["window_start"]: w for w in j["windows"]}
    assert by_ws[ws(0)]["user_label"] == "Focused" and by_ws[ws(5)]["user_label"] is None
    assert j["summary"]["agreement"]["n_labelled"] == 1


def test_predictions_without_model_are_unavailable(paths):
    app = create_app(paths["db"], paths["team"], clock=lambda: NOW, static_dir=None,
                     model_dir=None)
    with TestClient(app) as c:
        r = c.get("/api/predictions")
    assert r.status_code == 503 and "no trained model" in r.json()["detail"]


def test_predictions_filter_by_other_day(pclient):
    j = pclient.get("/api/predictions", params={"date": "2026-11-22"}).json()
    assert j["windows"] == [] and j["summary"]["agreement"]["n_labelled"] == 0


# -- label source and personal model ------------------------------------------
def _long_db(path, windows=30):
    """`windows` complete 5-minute windows: even ones coding, odd ones browsing."""
    store = EventStore(path)
    store.log(t(0), None, "agent_start")
    for k in range(windows):
        base = 5 * k
        if k % 2 == 0:
            store.log(t(base, 1), "Code.exe", "switched")
            for m in range(5):
                store.log(t(base + m, 30), "Code.exe", "keyboard_count", 60)
        else:
            store.log(t(base, 1), "chrome.exe", "switched", website="youtube.com")
            for m in range(5):
                store.log(t(base + m, 30), "chrome.exe", "mouse_count", 40)
    store.log(t(5 * windows), None, "agent_stop")
    store.close()


@pytest.fixture
def long_client(tmp_path, model_dir):
    db = tmp_path / "activity.db"
    _long_db(db)
    later = T0 + timedelta(hours=4)
    app = create_app(db, tmp_path / "team.db", clock=lambda: later, static_dir=None,
                     model_dir=model_dir)
    with TestClient(app) as c:
        c.app_ref = app
        yield c


def _label_all(c, source="user"):
    rows = c.get("/api/windows", params={"date": "2026-11-23"}).json()
    for w in rows:
        k = int((datetime.fromisoformat(w["window_start"]) - T0).total_seconds() // 300)
        c.post("/api/labels", json={"window_starts": [w["window_start"]], "source": source,
                                     "label": "Focused" if k % 2 == 0 else "Distracted"})
    return rows


def test_label_source_is_stored_exported_and_shown(client, paths):
    client.post("/api/labels", json={"window_starts": [ws(0)], "label": "Focused",
                                      "source": "accepted"})
    client.post("/api/labels", json={"window_starts": [ws(5)], "label": "Focused"})
    rows = {w["window_start"]: w for w in client.get("/api/windows",
                                                       params={"date": "2026-11-23"}).json()}
    assert rows[ws(0)]["label_source"] == "accepted" and rows[ws(5)]["label_source"] == "user"
    text = client.get("/api/export", params={"participant": "P1"}).text
    got = {r["window_start"]: r["label_source"] for r in csv.DictReader(io.StringIO(text))}
    assert got == {ws(0): "accepted", ws(5): "user"}
    bad = client.post("/api/labels", json={"window_starts": [ws(0)], "label": "Focused",
                                            "source": "model"})
    assert bad.status_code == 422


def test_old_label_table_gets_source_column(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE window_labels (window_start TEXT NOT NULL, window_seconds INTEGER"
                " NOT NULL, session_id INTEGER, label TEXT NOT NULL, labelled_at TEXT NOT NULL,"
                " updated_at TEXT NOT NULL, PRIMARY KEY (window_start, window_seconds))")
    con.execute("INSERT INTO window_labels VALUES ('2026-11-23T10:00:00', 300, NULL, 'Focused',"
                " 'x', 'x')")
    con.commit()
    con.close()
    store = EventStore(db)
    assert store.conn.execute("SELECT source FROM window_labels").fetchone() == ("user",)
    store.close()


def test_personal_model_takes_over_once_enough_own_labels(long_client, tmp_path):
    c = long_client
    j = c.get("/api/predictions", params={"date": "2026-11-23"}).json()
    assert j["model"]["source"] == "shared" and j["model"]["personal"]["state"] == "collecting"
    _label_all(c)
    c.get("/api/predictions", params={"date": "2026-11-23"})      # starts training
    c.app_ref.state.predictor.wait(60)
    j = c.get("/api/predictions", params={"date": "2026-11-23"}).json()
    assert j["model"]["source"] == "personal"
    p = j["model"]["personal"]
    assert p["state"] == "ready" and p["windows"] == 30 and p["estimate"]["n_test"] == 9
    assert all(len(w["reasons"]) == 3 for w in j["windows"])     # 2-label model explains too
    files = sorted(f.name for f in (tmp_path / "personal_model").iterdir())
    assert files == ["manifest.json", "model.joblib"]             # one copy, no leftovers
    assert (tmp_path / "personal_model" / "model.joblib").stat().st_size < 2_000_000
    # clearing the labels drops the personal model again
    c.request("DELETE", "/api/labels",
              json={"window_starts": [w["window_start"] for w in j["windows"]]})
    j = c.get("/api/predictions", params={"date": "2026-11-23"}).json()
    assert j["model"]["source"] == "shared"
    assert not (tmp_path / "personal_model" / "model.joblib").exists()


def test_accepted_guesses_never_train_the_personal_model(long_client):
    c = long_client
    _label_all(c, source="accepted")
    j = c.get("/api/predictions", params={"date": "2026-11-23"}).json()
    c.app_ref.state.predictor.wait(60)
    j = c.get("/api/predictions", params={"date": "2026-11-23"}).json()
    assert j["model"]["source"] == "shared"
    assert j["model"]["personal"]["state"] == "collecting" and j["model"]["personal"]["have"] == 0
