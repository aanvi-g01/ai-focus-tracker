"""Plain-language explanation of a day's predictions by an LLM (Groq).

Privacy: only a STRUCTURED SUMMARY built from an explicit allow-list leaves the
computer: minutes per predicted state, states per hour, confidence, the SHAP
reason phrases (template text, no names), and average app-CATEGORY and
site-CATEGORY shares. Never app names, hostnames, video titles, session task
text or raw events. Nothing is sent unless the user presses "Explain this day".

The API key is read from settings.json on this PC (or GROQ_API_KEY) and is never
returned to the browser. Answers are cached (at most CACHE_MAX small entries).
"""
import hashlib
import json
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b")
TIMEOUT = 30.0
CACHE_MAX = 30
LABELS = ("Focused", "Distracted", "Inactive")

# Window features that may be summarised (averaged) for the LLM.
APP_CATEGORY_FEATURES = ("app_dev", "app_browser", "app_docs", "app_comm", "app_media",
                         "app_system", "app_other")
ACTIVITY_FEATURES = ("active_share", "idle_share", "switches_per_min", "site_switches_per_min",
                     "keys_per_min", "mouse_per_min")

SYSTEM_PROMPT = """You explain a person's focus-tracking results for one day, in plain, friendly English.
You receive a JSON summary produced on their computer. The focus states are MODEL ESTIMATES
(a machine-learning model's guesses for each 5-minute window), not facts.

Write:
1. Three or four sentences on how the day went: when they seemed focused or distracted, and
   what the model's reasons suggest. Mention once that these are estimates. If
   "model.accuracy_estimate" or "agreement_with_your_labels" is low, say the guesses may be off.
2. Then "Try this:" followed by at most three short, concrete suggestions grounded in the numbers.

Rules: use only the numbers given; never invent apps, websites or activities; translate the
technical reason phrases and shares into everyday words (e.g. "you were idle more often than
usual", "most of your time was in the browser") instead of quoting them; times are 24-hour
local hours; no medical, psychological or diagnostic claims; no judgement of the person;
under 160 words; no headings."""


class LlmUnavailable(Exception):
    """Explanation could not be produced; the message is safe to show the user."""


# --------------------------------------------------------------------------
# Summary (the ONLY data that is sent)
# --------------------------------------------------------------------------
def _mean(values):
    vals = [v for v in values if v is not None and v == v]
    return round(sum(vals) / len(vals), 3) if vals else None


def day_summary(day: str, windows, model_info: dict, features=None) -> dict:
    """windows: the day's predictions (dicts from Predictor.predict_all plus
    user_label); features: {window_start: {feature: value}} from add_features."""
    minutes = {c: 0.0 for c in LABELS}
    per_hour = defaultdict(Counter)
    reasons = Counter()
    low_conf = 0
    for w in windows:
        p = w["predicted"]
        minutes[p] += (w.get("observed_seconds") or 0) / 60
        per_hour[w["window_start"][11:13]][p] += 1
        # "close to your usual" explains nothing; leave it out
        reasons.update(r for r in (w.get("reasons") or []) if "close to your usual" not in r)
        if w["probabilities"][p] < 0.6:
            low_conf += 1
    out = {
        "date": day,
        "windows": len(windows),
        "minutes_by_estimated_state": {c: round(v) for c, v in minutes.items()},
        "estimated_state_by_hour": {h: dict(c) for h, c in sorted(per_hour.items())},
        "low_confidence_windows": low_conf,
        "most_common_reasons": [{"reason": r, "windows": n} for r, n in reasons.most_common(8)],
    }
    if features:
        rows = [features[w["window_start"]] for w in windows if w["window_start"] in features]
        out["average_activity"] = {f: _mean(r.get(f) for r in rows) for f in ACTIVITY_FEATURES}
        out["app_category_time_share"] = {f[4:]: _mean(r.get(f) for r in rows)
                                          for f in APP_CATEGORY_FEATURES}
        site = sorted({k for r in rows for k in r if k.startswith("share_")})
        out["website_category_time_share"] = {k[6:]: _mean(r.get(k) for r in rows) for k in site}
    personal = (model_info or {}).get("personal") or {}
    if (model_info or {}).get("source") == "personal":
        est = personal.get("estimate") or {}
        out["model"] = {"type": "personal (trained on this person's own labels)",
                        "accuracy_estimate": est.get("accuracy"),
                        "accuracy_estimate_windows": est.get("n_test")}
    else:
        labelled = [w for w in windows if w.get("user_label")]
        out["model"] = {"type": "shared (trained on other study participants)"}
        if labelled:
            out["agreement_with_your_labels"] = {
                "labelled_windows": len(labelled),
                "matching": sum(w["predicted"] == w["user_label"] for w in labelled)}
    return out


# --------------------------------------------------------------------------
# Groq call
# --------------------------------------------------------------------------
def _request(client, key, model, summary):
    body = {"model": model, "temperature": 0.3, "max_completion_tokens": 1500,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": json.dumps(summary, ensure_ascii=False)}]}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    return client.post(GROQ_URL, json=body, timeout=TIMEOUT,
                       headers={"Authorization": f"Bearer {key}"})


def explain_day(summary: dict, api_key: str, model: str = "", http=None) -> tuple:
    """(text, model_used). `http` is an httpx.Client-like object (tests pass a fake)."""
    if not api_key:
        raise LlmUnavailable("no Groq API key set")
    import httpx
    client = http or httpx.Client()
    tried = []
    models = [model or DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != (model or DEFAULT_MODEL)]
    try:
        for m in models:
            tried.append(m)
            try:
                r = _request(client, api_key, m, summary)
            except httpx.TimeoutException as e:
                raise LlmUnavailable("Groq took too long to answer; try again") from e
            except httpx.RequestError as e:
                raise LlmUnavailable("can't reach Groq (no internet connection?)") from e
            if r.status_code in (400, 404) and "model" in r.text.lower():
                continue                                   # model retired: try the next
            if r.status_code == 401:
                raise LlmUnavailable("Groq rejected the API key; check it in Insights")
            if r.status_code == 429:
                raise LlmUnavailable("Groq rate limit reached; wait a minute and try again")
            if r.status_code >= 400:
                raise LlmUnavailable(f"Groq returned an error ({r.status_code})")
            choice = (r.json().get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "").strip()
            if not text:
                raise LlmUnavailable("Groq returned an empty answer; try again")
            return text, m
        raise LlmUnavailable(f"none of these Groq models are available: {', '.join(tried)}")
    finally:
        if http is None:
            client.close()


# --------------------------------------------------------------------------
# Small answer cache (memory + one JSON file, CACHE_MAX entries)
# --------------------------------------------------------------------------
class ExplanationCache:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data = None

    @staticmethod
    def key(summary, model):
        return hashlib.sha1(json.dumps([summary, model], sort_keys=True).encode()).hexdigest()[:20]

    def _load(self):
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8")) if self.path else {}
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def get(self, key):
        with self._lock:
            return self._load().get(key)

    def put(self, key, entry):
        with self._lock:
            data = self._load()
            data[key] = entry
            while len(data) > CACHE_MAX:                     # drop the oldest
                data.pop(min(data, key=lambda k: data[k].get("generated_at", "")))
            if self.path:
                try:
                    self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")
                except OSError:
                    pass


def now_iso():
    return datetime.now().isoformat(timespec="seconds")
