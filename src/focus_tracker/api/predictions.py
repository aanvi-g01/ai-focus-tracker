"""Live focus-state predictions for the local app (UC-18 / UC-19).

Predictions are MODEL ESTIMATES, computed on the fly from the window features
on this machine. They are never written to `window_labels`, never exported and
never counted as labels: the user's own labels stay the only ground truth.

Which model:
* the user's PERSONAL model (models/personal.py) once they have labelled enough
  windows themselves; it is retrained in a background thread when their labels
  change, and the previous one keeps serving meanwhile;
* otherwise the SHARED model trained on the study participants (models/).

The personal baseline (EWMA z-scores) needs the user's history in time order,
so every complete window is scored and the requested day / session is
filtered afterwards. Results are cached in memory until windows or model change.
"""
import hashlib
import json
import threading
from pathlib import Path

from ..features.windows import DEFAULT_WINDOW_SECONDS
from .export import feature_rows

LABEL_NAMES = ("Focused", "Distracted", "Inactive")


class PredictionUnavailable(Exception):
    """No model, or the ML libraries are not installed."""


def _ml_missing(e):
    return PredictionUnavailable(f"machine-learning libraries are not installed ({e.name})")


class Predictor:
    def __init__(self, model_roots, personal_dir=None,
                 window_seconds=DEFAULT_WINDOW_SECONDS, background=True):
        self.roots = [Path(r) for r in (model_roots or []) if r]
        self.personal_dir = Path(personal_dir) if personal_dir else None
        self.window_seconds = window_seconds
        self.background = background
        self._lock = threading.Lock()
        self._shared = None                 # (model, manifest, dir)
        self._personal = None               # (model, manifest)
        self._personal_loaded = False
        self._personal_status = {"state": "off"}
        self._thread = None
        self._cache_key = None
        self._cache = None
        self._active = None                 # "personal" | "shared"

    # -- shared model -----------------------------------------------------------
    def _load_shared(self):
        if self._shared is not None:
            return self._shared
        try:
            from ..models.train import latest_model_dir, load, read_manifest
        except ImportError as e:
            raise _ml_missing(e) from e
        d = latest_model_dir(*self.roots)
        if d is None:
            raise PredictionUnavailable("no trained model found; run focus-train first")
        try:
            self._shared = (load(d), read_manifest(d), d)
        except ImportError as e:
            raise _ml_missing(e) from e
        except (OSError, ValueError) as e:
            raise PredictionUnavailable(f"could not load the model in {d.name}: {e}") from e
        return self._shared

    # -- personal model ---------------------------------------------------------
    def _refresh_personal(self, store):
        """Use / (re)train the personal model. Called with self._lock held."""
        if self.personal_dir is None:
            return
        try:
            from ..models import personal as P
        except ImportError:
            return
        if not self._personal_loaded:
            self._personal = P.load(self.personal_dir)
            self._personal_loaded = True
        labels = P.user_labels(store, self.window_seconds)
        ready, counts = P.readiness(labels)
        fp = P.fingerprint(labels)
        status = {"have": len(labels), "needed": P.MIN_WINDOWS, "counts": counts}
        if not ready:
            if self._personal is not None:              # labels were cleared: drop it
                P.remove(self.personal_dir)
                self._personal = None
            self._personal_status = {"state": "collecting", **status}
            return
        if self._personal is not None and self._personal[1].get("fingerprint") == fp:
            self._personal_status = {"state": "ready", **status}
            return
        if self._thread is not None and self._thread.is_alive():
            self._personal_status = {"state": "training", **status}
            return
        df = P.training_frame(store, labels, self.window_seconds)
        self._personal_status = {"state": "training", **status}

        def work():
            try:
                model, manifest = P.train_personal(df, fp)
                P.save(model, manifest, self.personal_dir)
                with self._lock:
                    self._personal = (model, manifest)
                    self._personal_status = {"state": "ready", **status}
            except Exception as e:                       # keep serving the old model
                with self._lock:
                    self._personal_status = {"state": "error", "error": str(e), **status}

        if self.background:
            self._thread = threading.Thread(target=work, name="personal-model", daemon=True)
            self._thread.start()
        else:                                            # tests: train synchronously
            self._lock.release()
            try:
                work()
            finally:
                self._lock.acquire()

    def wait(self, timeout=None):
        """Block until a background retrain (if any) has finished."""
        t = self._thread
        if t is not None:
            t.join(timeout)

    # -- info -------------------------------------------------------------------
    def info(self):
        out = {"source": self._active, "personal": dict(self._personal_status)}
        if self._personal is not None:
            m = self._personal[1]
            out["personal"].update({"trained_at": m.get("trained_at"), "windows": m.get("windows"),
                                    "estimate": m.get("estimate")})
        if self._shared is not None:
            m, d = self._shared[1], self._shared[2]
            data = m.get("data", {})
            out["shared"] = {"id": d.name, "kind": m.get("kind"), "trained_at": m.get("trained_at"),
                             "participants": len(data.get("participants", {})),
                             "windows": data.get("windows")}
        # flat fields kept for older callers
        active = out.get("shared", {}) if self._active == "shared" else {}
        out.update({"id": active.get("id"), "kind": "gbt_z" if self._active == "personal"
                    else active.get("kind"), "participants": active.get("participants"),
                    "windows": active.get("windows"), "trained_at": active.get("trained_at")})
        return out

    # -- predictions -------------------------------------------------------------
    def predict_all(self, store, participant="me", window_seconds=None):
        """[{window_start, window_end, session_id, observed_seconds, predicted,
        probabilities, reasons}] for every complete window, oldest first."""
        window_seconds = window_seconds or self.window_seconds
        with self._lock:
            self._refresh_personal(store)
            shared_error = None
            try:
                self._load_shared()
            except PredictionUnavailable as e:
                shared_error = e
            if self._personal is not None:
                model, tag = self._personal[0], "personal:" + self._personal[1]["fingerprint"]
                self._active = "personal"
            elif shared_error is None:
                model, tag = self._shared[0], "shared:" + self._shared[2].name
                self._active = "shared"
            else:
                raise shared_error
            rows = feature_rows(store, participant or "me", window_seconds)
            key = (tag, hashlib.sha1(
                json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest())
            if key == self._cache_key:
                return self._cache
            out = self._score(model, rows) if rows else []
            self._cache_key, self._cache = key, out
            return out

    @staticmethod
    def _score(m, rows):
        import warnings
        import pandas as pd
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = m.predict_proba(df)
            pred = m.predict(df)
            reasons = None
            if m.kind.startswith("gbt"):
                expl, _ = m.explain(df)
                reasons = expl["reasons"].reindex(range(len(df)))
        out = []
        for i, r in enumerate(rows):
            out.append({
                "window_start": r["window_start"], "window_end": r["window_end"],
                "session_id": r["session_id"], "observed_seconds": r["observed_seconds"],
                "predicted": str(pred[i]),
                "probabilities": {c: round(float(proba.iloc[i][c]), 3) for c in LABEL_NAMES},
                "reasons": list(reasons.iloc[i]) if reasons is not None else []})
        return out


def summarise(windows):
    minutes = {c: 0.0 for c in LABEL_NAMES}
    for w in windows:
        minutes[w["predicted"]] += (w["observed_seconds"] or 0) / 60
    labelled = [w for w in windows if w.get("user_label")]
    return {"minutes": {c: round(v, 1) for c, v in minutes.items()},
            "agreement": {"n_labelled": len(labelled),
                          "matches": sum(w["predicted"] == w["user_label"] for w in labelled)}}
