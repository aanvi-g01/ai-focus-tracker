"""HTTP endpoints of the local app. Each one is a thin wrapper over existing
logger / session / window / label code; nothing here infers a focus label.

The one exception is /api/predictions: model ESTIMATES computed on the fly
(see predictions.py). They are never stored as labels, never exported and
never counted in label totals."""
from contextlib import contextmanager
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..features.windows import DEFAULT_WINDOW_SECONDS, refresh_windows
from ..logger import labels as L
from ..logger.session import TASK_CATEGORIES, SessionError, SessionManager
from ..logger.storage import EventStore
from .developer import CsvImportError, TeamStore
from . import llm as LLM
from .export import export_filename, export_rows, feature_rows, to_csv, validate_pseudonym
from .logger_control import LoggerError
from .predictions import PredictionUnavailable, summarise
from .settings import set_autostart, shortcuts_supported

W = DEFAULT_WINDOW_SECONDS
router = APIRouter(prefix="/api")


# -- helpers -----------------------------------------------------------------
def _cfg(request: Request):
    return request.app.state


@contextmanager
def _store(request: Request):
    store = EventStore(_cfg(request).db_path)   # sqlite: one connection per request
    try:
        yield store
    finally:
        store.close()


@contextmanager
def _team(request: Request):
    team = TeamStore(_cfg(request).team_db_path)
    try:
        yield team
    finally:
        team.close()


def _now(request: Request) -> datetime:
    return _cfg(request).clock()


def _iso(dt):
    return dt.isoformat(timespec="seconds") if dt else None


def _session_json(s, now):
    if s is None:
        return None
    return {"id": s.id, "task": s.task, "task_category": s.task_category,
            "status": s.status, "started_at": _iso(s.started_at),
            "ended_at": _iso(s.ended_at), "active_seconds": s.active_seconds(now),
            "end_reason": s.end_reason}


def _window_json(w, label):
    return {"window_start": _iso(w.window_start), "window_end": _iso(w.window_end),
            "window_seconds": w.window_seconds, "session_id": w.session_id,
            "observed_seconds": w.observed_seconds, "coverage": w.coverage,
            "complete": w.complete, "active_seconds": w.active_seconds,
            "idle_seconds": w.idle_seconds, "idle_episodes": w.idle_episodes,
            "switch_count": w.switch_count, "site_switch_count": w.site_switch_count,
            "switches_per_min": w.switches_per_min, "distinct_apps": w.distinct_apps,
            "top_app": next(iter(w.app_seconds), None),
            "keyboard_count": w.keyboard_count, "mouse_count": w.mouse_count,
            "label": label}


def _parse_ts(text):
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise HTTPException(422, f"bad window_start {text!r}") from None


# -- status & logger ------------------------------------------------------------
@router.get("/status")
def status(request: Request):
    now = _now(request)
    with _store(request) as store:
        counts = dict(store.conn.execute(
            "SELECT label, COUNT(*) FROM window_labels WHERE window_seconds = ?"
            " GROUP BY label", (W,)))
        session = SessionManager(store).current()
        return {"logger": _cfg(request).controller.status(),
                "session": _session_json(session, now),
                "label_counts": {k: counts.get(k, 0) for k in L.LABELS},
                "labels": list(L.LABELS), "task_categories": list(TASK_CATEGORIES),
                "window_seconds": W, "now": _iso(now),
                "settings": _cfg(request).settings.public(),
                "desktop": shortcuts_supported()}


class LoggerStart(BaseModel):
    websites: bool = True
    read_titles: bool = False


@router.post("/logger/start")
def logger_start(body: LoggerStart, request: Request):
    try:
        _cfg(request).controller.start(body.websites, body.read_titles)
    except LoggerError as e:
        raise HTTPException(409, str(e)) from None
    return _cfg(request).controller.status()


@router.post("/logger/stop")
def logger_stop(request: Request):
    try:
        _cfg(request).controller.stop()
    except LoggerError as e:
        raise HTTPException(409, str(e)) from None
    return _cfg(request).controller.status()


# -- settings & quit ------------------------------------------------------------
class SettingsUpdate(BaseModel):
    participant: Optional[str] = None
    onboarded: Optional[bool] = None
    autostart: Optional[bool] = None
    llm_api_key: Optional[str] = None      # write-only; "" removes it
    llm_model: Optional[str] = None


@router.get("/settings")
def get_settings(request: Request):
    return {**_cfg(request).settings.public(), "desktop": shortcuts_supported()}


@router.post("/settings")
def update_settings(body: SettingsUpdate, request: Request):
    settings = _cfg(request).settings
    try:
        if body.autostart is not None and body.autostart != settings.load()["autostart"]:
            set_autostart(body.autostart)
        key = body.llm_api_key.strip() if body.llm_api_key is not None else None
        if key and not key.isascii():
            raise ValueError("that doesn't look like an API key")
        data = settings.update(participant=body.participant, onboarded=body.onboarded,
                               autostart=body.autostart, llm_api_key=key,
                               llm_model=body.llm_model)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e)) from None
    return {**settings.public(data), "desktop": shortcuts_supported()}


@router.post("/app/quit")
def quit_app(request: Request):
    """Stop logging cleanly, then stop the local server (desktop Quit button)."""
    _cfg(request).controller.shutdown()
    server = _cfg(request).server
    if server is not None:
        import threading
        threading.Timer(0.5, lambda: setattr(server, "should_exit", True)).start()
    return {"quitting": server is not None}


# -- sessions -------------------------------------------------------------------
class SessionStart(BaseModel):
    task: str
    category: Optional[str] = None


@router.post("/session/start")
def session_start(body: SessionStart, request: Request):
    now = _now(request)
    with _store(request) as store:
        try:
            s = SessionManager(store).start(body.task, now, category=body.category or None)
        except SessionError as e:
            raise HTTPException(409, str(e)) from None
        return _session_json(s, now)


@router.post("/session/{action}")
def session_action(action: str, request: Request):
    if action not in ("pause", "resume", "end"):
        raise HTTPException(404, "unknown session action")
    now = _now(request)
    with _store(request) as store:
        try:
            s = getattr(SessionManager(store), action)(now)
        except SessionError as e:
            raise HTTPException(409, str(e)) from None
        return _session_json(s, now)


# -- windows & labels -------------------------------------------------------------
@router.get("/windows")
def windows(request: Request, on: Optional[date] = Query(None, alias="date"),
            unlabelled: bool = False, session: Optional[int] = None):
    now = _now(request)
    with _store(request) as store:
        refresh_windows(store, W, now, live=_cfg(request).controller.running)
        day = None if session is not None else (on or now.date())
        rows = L.labelled_windows(store, W, session_id=session, on_date=day,
                                  unlabelled_only=unlabelled)
        sources = dict(store.conn.execute(
            "SELECT window_start, source FROM window_labels WHERE window_seconds = ?", (W,)))
        out = []
        for w, label in reversed(rows):
            j = _window_json(w, label)
            j["label_source"] = sources.get(j["window_start"]) if label else None
            out.append(j)
        return out


class LabelRequest(BaseModel):
    window_starts: List[str]
    label: str
    source: str = "user"         # "accepted" = the user confirmed the model's guess


class ClearRequest(BaseModel):
    window_starts: List[str]


@router.post("/labels")
def set_labels(body: LabelRequest, request: Request):
    try:
        L.normalize_label(body.label)
        if body.source not in L.LABEL_SOURCES:
            raise L.LabelError(f"source must be one of {', '.join(L.LABEL_SOURCES)}")
    except L.LabelError as e:
        raise HTTPException(422, str(e)) from None
    now = _now(request)
    results = []
    with _store(request) as store:
        # Label against current data, like `focus-label set` does.
        refresh_windows(store, W, now, live=_cfg(request).controller.running)
        for text in body.window_starts:
            try:
                w, prev = L.set_label(store, _parse_ts(text), body.label, now, W,
                                      source=body.source)
                results.append({"window_start": text, "ok": True, "previous": prev})
            except L.LabelError as e:
                results.append({"window_start": text, "ok": False, "error": str(e)})
    return {"results": results}


@router.delete("/labels")
def clear_labels(body: ClearRequest, request: Request):
    with _store(request) as store:
        refresh_windows(store, W, _now(request), live=_cfg(request).controller.running)
        cleared = sum(L.clear_label(store, _parse_ts(t), W) for t in body.window_starts)
    return {"cleared": cleared}


@router.get("/export")
def export(request: Request, participant: str):
    try:
        participant = validate_pseudonym(participant)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    with _store(request) as store:
        refresh_windows(store, W, _now(request), live=_cfg(request).controller.running)
        text = to_csv(export_rows(store, participant, W))
    name = export_filename(participant, _now(request))
    return Response(text, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})



# -- predictions (model estimates, never labels) ------------------------------
def _predictions_for(request, store, day, session=None):
    """(the requested windows oldest first, with user_label; all scored windows)."""
    cfg = _cfg(request)
    participant = cfg.settings.load().get("participant") or "me"
    refresh_windows(store, W, _now(request), live=cfg.controller.running)
    try:
        scored = cfg.predictor.predict_all(store, participant, W)
    except PredictionUnavailable as e:
        raise HTTPException(503, str(e))
    labels = {ws: label for ws, label in store.conn.execute(
        "SELECT window_start, label FROM window_labels WHERE window_seconds = ?", (W,))}
    rows = [{**w, "user_label": labels.get(w["window_start"])} for w in scored
            if (session is None or w["session_id"] == session)
            and (day is None or w["window_start"][:10] == day.isoformat())]
    return rows, scored


@router.get("/predictions")
def predictions(request: Request, on: Optional[date] = Query(None, alias="date"),
                session: Optional[int] = None):
    cfg = _cfg(request)
    day = None if session is not None else (on or _now(request).date())
    with _store(request) as store:
        rows, scored = _predictions_for(request, store, day, session)
    return {"model": cfg.predictor.info(), "date": day.isoformat() if day else None,
            "latest_date": scored[-1]["window_start"][:10] if scored else None,
            "session": session, "windows": rows[::-1],          # newest first
            "summary": summarise(rows)}


class ExplainRequest(BaseModel):
    date: date


@router.post("/insights/explain")
def explain(body: ExplainRequest, request: Request):
    """LLM wording of one day's predictions. Sends ONLY the structured summary
    from llm.day_summary (no app names, websites or titles), on request."""
    cfg = _cfg(request)
    key = cfg.settings.llm_key()
    if not key:
        raise HTTPException(400, "add a Groq API key first")
    with _store(request) as store:
        rows, _ = _predictions_for(request, store, body.date)
        if not rows:
            raise HTTPException(404, "no finished windows on this day")
        features = _window_features(store, {w["window_start"] for w in rows})
    summary = LLM.day_summary(body.date.isoformat(), rows, cfg.predictor.info(), features)
    model = cfg.settings.load().get("llm_model") or LLM.DEFAULT_MODEL
    ck = LLM.ExplanationCache.key(summary, model)
    hit = cfg.llm_cache.get(ck)
    if hit:
        return {**hit, "cached": True, "sent": summary}
    try:
        text, used = LLM.explain_day(summary, key, model, http=cfg.llm_http)
    except LLM.LlmUnavailable as e:
        raise HTTPException(502, str(e))
    entry = {"text": text, "model": used, "generated_at": LLM.now_iso(), "date": summary["date"]}
    cfg.llm_cache.put(ck, entry)
    return {**entry, "cached": False, "sent": summary}


def _window_features(store, starts):
    """{window_start: allow-listed numeric features} for the given windows."""
    import pandas as pd
    from ..models.data import add_features
    rows = [r for r in feature_rows(store, "me", W) if r["window_start"] in starts]
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = add_features(df)
    keep = list(LLM.ACTIVITY_FEATURES) + list(LLM.APP_CATEGORY_FEATURES) +         [c for c in df.columns if c.startswith("share_")]
    out = {}
    for _, r in df.iterrows():
        out[r["window_start"].isoformat(timespec="seconds")] = {
            k: (None if pd.isna(r[k]) else float(r[k])) for k in keep}
    return out


# -- developer --------------------------------------------------------------------
class DevKey(BaseModel):
    participant: str
    window_start: str
    window_seconds: int = W


class DevLabelRequest(BaseModel):
    items: List[DevKey]
    label: str
    annotator: str


class DevClearRequest(BaseModel):
    items: List[DevKey]


def _keys(items):
    return [(k.participant, k.window_start, k.window_seconds) for k in items]


@router.post("/dev/import")
async def dev_import(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "file too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(422, "file must be UTF-8 CSV") from None
    with _team(request) as team:
        try:
            return team.import_csv(text)
        except CsvImportError as e:
            raise HTTPException(422, str(e)) from None


@router.get("/dev/participants")
def dev_participants(request: Request):
    with _team(request) as team:
        return team.participants()


@router.get("/dev/windows")
def dev_windows(request: Request, participant: Optional[str] = None,
                unlabelled: bool = False, blind: bool = False):
    with _team(request) as team:
        return team.windows(participant, unlabelled, blind)


@router.post("/dev/labels")
def dev_set_labels(body: DevLabelRequest, request: Request):
    with _team(request) as team:
        try:
            res = team.set_labels(_keys(body.items), body.label, body.annotator,
                                  _now(request))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
    return {"results": [{**r, "key": list(r["key"])} for r in res]}


@router.delete("/dev/labels")
def dev_clear_labels(body: DevClearRequest, request: Request):
    with _team(request) as team:
        return {"cleared": team.clear_labels(_keys(body.items))}


@router.get("/dev/agreement")
def dev_agreement(request: Request):
    with _team(request) as team:
        return team.agreement()


@router.get("/dev/export")
def dev_export(request: Request):
    with _team(request) as team:
        text = team.export_csv()
    name = f"team_annotations_{_now(request):%Y%m%d_%H%M}.csv"
    return Response(text, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
