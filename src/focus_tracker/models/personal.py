"""Personal model: trained on this computer from the user's OWN labels.

A check across the 10 participants (each person's earlier days -> their later
days) showed a model trained only on the person's own windows beats the shared
model (mean accuracy 0.73 vs 0.64), so the app switches to a personal model
as soon as there are enough labels, and falls back to the shared one before.

* Only labels the user chose themselves (source "user") are used. Model guesses
  they accepted are left out, so the model never learns from its own output.
* Storage: exactly one model on disk (model.joblib + manifest.json, about 1 MB),
  overwritten on each retrain; nothing is exported or uploaded.
* An honest accuracy estimate is computed at training time: train on the
  earlier part of the user's labels, test on the later part.
"""
import hashlib
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..api.export import feature_rows
from ..features.windows import DEFAULT_WINDOW_SECONDS
from ..logger.labels import LABELS
from .data import RAW_FEATURES

KIND = "gbt_z"
MIN_WINDOWS = 20            # own typed labels needed before a personal model is trained
MIN_PER_LABEL = 3           # ... covering at least two labels with this many windows each
PARTICIPANT = "me"


def user_labels(store, window_seconds=DEFAULT_WINDOW_SECONDS):
    """{window_start: label} for labels the user chose themselves."""
    return dict(store.conn.execute(
        "SELECT window_start, label FROM window_labels"
        " WHERE window_seconds = ? AND source = 'user' ORDER BY window_start",
        (window_seconds,)))


def fingerprint(labels: dict) -> str:
    return hashlib.sha1(json.dumps(sorted(labels.items())).encode()).hexdigest()[:16]


def readiness(labels: dict):
    """(ready, counts): enough own labels, covering at least two labels."""
    counts = {c: sum(1 for v in labels.values() if v == c) for c in LABELS}
    enough = sum(n >= MIN_PER_LABEL for n in counts.values()) >= 2
    return len(labels) >= MIN_WINDOWS and enough, counts


def training_frame(store, labels: dict, window_seconds=DEFAULT_WINDOW_SECONDS):
    rows = [r for r in feature_rows(store, PARTICIPANT, window_seconds)
            if r["window_start"] in labels]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["label"] = df["window_start"].map(labels)
    df["window_start"] = pd.to_datetime(df["window_start"])
    return df.sort_values("window_start").reset_index(drop=True)


def estimate(df, seed=0, test_frac=0.3):
    """Accuracy on the user's LATER labelled windows of a model trained on the
    earlier ones (None when there is too little data to say)."""
    from .train import train_final
    n_test = int(round(len(df) * test_frac))
    train, test = df.iloc[:len(df) - n_test], df.iloc[len(df) - n_test:]
    if n_test < 5 or train["label"].nunique() < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = train_final(train, KIND, seed=seed)
        pred = m.predict(df)[len(df) - n_test:]       # full history for the baseline
    return {"accuracy": round(float(np.mean(pred == test["label"].to_numpy())), 3),
            "n_test": int(n_test)}


def train_personal(df, fp, seed=0):
    """(FocusModel, manifest) trained on all of df."""
    from .train import train_final
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = train_final(df, KIND, seed=seed)
    manifest = {"kind": KIND, "personal": True, "fingerprint": fp,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "windows": int(len(df)),
                "labels": {c: int((df["label"] == c).sum()) for c in LABELS},
                "estimate": estimate(df, seed), "raw_features": list(RAW_FEATURES)}
    return model, manifest


def save(model, manifest, folder):
    """Replace the stored personal model (one copy only, written atomically)."""
    import joblib
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "model.joblib.tmp"
    joblib.dump(model, tmp, compress=3)
    os.replace(tmp, folder / "model.joblib")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load(folder):
    """(model, manifest) or None when there is no usable personal model."""
    folder = Path(folder)
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("raw_features") != list(RAW_FEATURES):
            return None                                 # features changed: retrain
        import joblib
        return joblib.load(folder / "model.joblib"), manifest
    except (OSError, ValueError, EOFError):
        return None


def remove(folder):
    for name in ("model.joblib", "manifest.json", "model.joblib.tmp"):
        try:
            (Path(folder) / name).unlink()
        except OSError:
            pass
