"""LLM explanations: only the allow-listed summary is sent; key never leaves the API.
No network: a fake HTTP client stands in for Groq."""
import json
from datetime import datetime, timedelta

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("lightgbm")
from fastapi.testclient import TestClient  # noqa: E402

from focus_tracker.api import llm as LLM  # noqa: E402
from focus_tracker.api.app import create_app  # noqa: E402
from focus_tracker.logger.storage import EventStore  # noqa: E402

T0 = datetime(2026, 11, 23, 10, 0, 0)
LATER = T0 + timedelta(hours=4)
SECRET = "gsk_test_secret_key_123"


def t(m, s=0):
    return T0 + timedelta(minutes=m, seconds=s)


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeGroq:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse(200, {"choices": [{"message": {"content": "A calm, focused morning."}}]})

    def close(self):
        pass


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "activity.db"
    s = EventStore(path)
    s.log(t(0), None, "agent_start")
    for k in range(6):
        s.log(t(5 * k, 1), "Code.exe" if k % 2 == 0 else "chrome.exe", "switched",
              website=None if k % 2 == 0 else "youtube.com",
              content_category=None if k % 2 == 0 else "entertainment",
              content_title=None if k % 2 == 0 else "Secret Video Title")
        s.log(t(5 * k + 2), "Code.exe" if k % 2 == 0 else "chrome.exe", "keyboard_count", 30)
    s.log(t(30), None, "agent_stop")
    s.close()
    return path


@pytest.fixture
def shared_model(tmp_path_factory):
    import pandas as pd
    from test_ml import synth_rows
    from focus_tracker.models import train as T
    df = pd.DataFrame(synth_rows("P1", days=2, per_day=30, seed=1)
                      + synth_rows("P2", days=2, per_day=30, seed=2))
    df["window_start"] = pd.to_datetime(df["window_start"])
    root = tmp_path_factory.mktemp("models")
    T.save(T.train_final(df, "gbt_raw"), root, df)
    return root


@pytest.fixture
def client(db, tmp_path, shared_model):
    app = create_app(db, tmp_path / "team.db", clock=lambda: LATER, static_dir=None,
                     model_dir=shared_model, personal_dir=None)
    app.state.llm_http = FakeGroq()
    with TestClient(app) as c:
        c.fake = app.state.llm_http
        c.app_ref = app
        yield c


def test_explain_needs_a_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/api/insights/explain", json={"date": "2026-11-23"})
    assert r.status_code == 400 and client.fake.calls == []


def test_key_is_saved_but_never_returned(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/api/settings", json={"llm_api_key": SECRET})
    assert r.status_code == 200 and SECRET not in r.text and r.json()["llm_key_set"]
    for path in ("/api/settings", "/api/status"):
        assert SECRET not in client.get(path).text
    r = client.post("/api/settings", json={"llm_api_key": ""})       # remove
    assert r.json()["llm_key_set"] is False


def test_only_the_allow_listed_summary_is_sent(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client.post("/api/settings", json={"llm_api_key": SECRET})
    r = client.post("/api/insights/explain", json={"date": "2026-11-23"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["text"] == "A calm, focused morning." and j["cached"] is False
    (call,) = client.fake.calls
    assert call["url"] == LLM.GROQ_URL
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    sent = call["json"]["messages"][1]["content"]
    assert json.loads(sent) == j["sent"]                         # UI shows exactly this
    for private in ("youtube", "Secret Video Title", "Code.exe", "chrome", ".exe"):
        assert private.lower() not in sent.lower()
    s = j["sent"]
    assert s["windows"] == 6 and set(s["minutes_by_estimated_state"]) == set(LLM.LABELS)
    assert "entertainment" in s["website_category_time_share"]
    assert "under 160 words" in call["json"]["messages"][0]["content"]


def test_answers_are_cached(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    first = client.post("/api/insights/explain", json={"date": "2026-11-23"}).json()
    again = client.post("/api/insights/explain", json={"date": "2026-11-23"}).json()
    assert len(client.fake.calls) == 1 and again["cached"] and again["text"] == first["text"]
    cache = json.loads((client.app_ref.state.llm_cache.path).read_text())
    assert len(cache) == 1 and SECRET not in json.dumps(cache)


def test_groq_errors_become_readable(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    client.fake.responses = [FakeResponse(401, {"error": {"message": "bad key"}})]
    r = client.post("/api/insights/explain", json={"date": "2026-11-23"})
    assert r.status_code == 502 and "rejected the API key" in r.json()["detail"]


def test_retired_model_falls_back(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    client.fake.responses = [FakeResponse(404, {"error": {"message": "The model does not exist"}})]
    r = client.post("/api/insights/explain", json={"date": "2026-11-23"})
    assert r.status_code == 200
    assert [c["json"]["model"] for c in client.fake.calls][:2] == list(LLM.FALLBACK_MODELS[:2])


def test_cache_is_capped(tmp_path):
    cache = LLM.ExplanationCache(tmp_path / "c.json")
    for i in range(LLM.CACHE_MAX + 5):
        cache.put(f"k{i}", {"text": "x", "generated_at": f"2026-01-01T00:00:{i:02d}"})
    data = json.loads((tmp_path / "c.json").read_text())
    assert len(data) == LLM.CACHE_MAX and "k0" not in data
