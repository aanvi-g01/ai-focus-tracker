"""Load participants' exported windows and derive model features.

Accepts the per-participant exports (`focus_windows_P*.csv`, column `label`)
and the Developer page's merged export (column `user_label`). Only the
participants' OWN labels are used; developer labels are dropped.
Missing values stay NaN (e.g. category shares in a window without browser
time, or columns absent from older exports).
"""
import glob
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..api.export import BASE_COLUMNS, CATEGORIES
from ..logger.labels import LABELS

REQUIRED = set(BASE_COLUMNS) | {"label"}
TASK_CATEGORIES = ("studying", "coding", "research", "writing", "other")

# Foreground app -> broad category (lower-case process name, ".exe" removed).
APP_CATEGORIES = {
    "dev": ("code", "pycharm64", "pycharm", "idea64", "devenv", "sublime_text", "notepad++",
            "eclipse", "studio64", "windowsterminal", "wt", "cmd", "powershell", "pwsh",
            "git-bash", "mintty", "jupyter-lab", "spyder", "rstudio", "matlab", "arduino ide",
            "cursor", "windsurf", "dbeaver", "postman"),
    "browser": ("chrome", "msedge", "firefox", "brave", "opera", "vivaldi"),
    "docs": ("winword", "excel", "powerpnt", "acrobat", "acrord32", "onenote", "notepad",
             "sumatrapdf", "foxitpdfreader", "obsidian", "notion", "zotero", "mendeley"),
    "comm": ("whatsapp", "whatsapp.root", "telegram", "slack", "discord", "teams", "ms-teams",
             "zoom", "outlook", "olk", "thunderbird", "skype"),
    "media": ("spotify", "vlc", "netflix", "steam", "epicgameslauncher", "wmplayer",
              "music.ui", "video.ui", "primevideo"),
    "system": ("explorer", "searchapp", "searchhost", "shellexperiencehost", "lockapp",
               "applicationframehost", "snippingtool", "taskmgr", "systemsettings",
               "startmenuexperiencehost", "asuskeyboardhost"),
}
_APP_LOOKUP = {name: cat for cat, names in APP_CATEGORIES.items() for name in names}
APP_GROUPS = tuple(APP_CATEGORIES) + ("other",)

SHARE_COLUMNS = tuple(f"share_{c}" for c in CATEGORIES) + ("share_unknown",)

# Continuous features that get a personal z-score (slide 9, step 2).
Z_FEATURES = ("idle_share", "active_share", "switches_per_min", "site_switches_per_min",
              "keys_per_min", "mouse_per_min", "distinct_apps", "idle_episodes",
              "browser_share", "distinct_hosts", "share_entertainment", "share_social",
              "app_dev", "app_media", "app_comm")

RAW_FEATURES = (("idle_share", "active_share", "coverage", "switches_per_min",
                 "site_switches_per_min", "keys_per_min", "mouse_per_min", "idle_episodes",
                 "distinct_apps", "browser_share", "distinct_hosts")
                + SHARE_COLUMNS
                + tuple(f"app_{g}" for g in APP_GROUPS)
                + tuple(f"task_{t}" for t in TASK_CATEGORIES + ("none",))
                + ("hour_sin", "hour_cos"))


def app_category(app: str) -> str:
    name = (app or "").strip().lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return _APP_LOOKUP.get(name, "other")


def parse_app_seconds(text):
    """'Code.exe:179; chrome.exe:120' -> {'Code.exe': 179.0, 'chrome.exe': 120.0}."""
    out = {}
    if not isinstance(text, str) or not text.strip():
        return out
    for part in text.split(";"):
        name, sep, secs = part.strip().rpartition(":")
        if sep and name:
            try:
                out[name] = out.get(name, 0.0) + float(secs)
            except ValueError:
                continue
    return out


def _files(paths):
    files = []
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.csv")))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError("no CSV files found in " + ", ".join(map(str, paths)))
    return files


def load_exports(paths) -> pd.DataFrame:
    """All labelled windows from export CSVs, one row per participant window."""
    frames = []
    for f in _files(paths if isinstance(paths, (list, tuple)) else [paths]):
        df = pd.read_csv(f, dtype={"hostnames": str, "video_titles": str,
                                   "app_seconds": str, "top_app": str})
        if "label" not in df.columns and "user_label" in df.columns:
            df = df.rename(columns={"user_label": "label"})      # merged developer export
        df = df.drop(columns=[c for c in ("developer_label", "annotator") if c in df.columns])
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{Path(f).name}: not an app export (missing "
                             f"{', '.join(sorted(missing))})")
        df["source_file"] = Path(f).name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Tolerate hand-edited files: " distracted " -> "Distracted".
    canon = {c.lower(): c for c in LABELS}
    df["label"] = df["label"].map(lambda v: canon.get(v.strip().lower(), v)
                                  if isinstance(v, str) else v)
    df = df[df["label"].isin(LABELS)].copy()
    # Model guesses the user only accepted are not training data (the model would
    # learn its own output); older exports have no label_source and are all typed.
    if "label_source" in df.columns:
        accepted = df["label_source"].astype(str).str.strip().eq("accepted")
        if accepted.any():
            warnings.warn(f"{int(accepted.sum())} windows labelled by accepting the model's "
                          "guess were left out of training")
        df = df[~accepted].copy()
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["participant"] = df["participant"].astype(str)
    before = len(df)
    df = df.drop_duplicates(["participant", "window_start", "window_seconds"], keep="last")
    if len(df) < before:
        warnings.warn(f"{before - len(df)} duplicate windows (same participant and time) "
                      "found in several files; kept the last one")
    return df.sort_values(["participant", "window_start"]).reset_index(drop=True)


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else \
        pd.Series(np.nan, index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with RAW_FEATURES added (NaN where not measurable)."""
    df = df.copy()
    obs = _num(df, "observed_seconds").where(lambda s: s > 0)
    mins = obs / 60
    df["idle_share"] = _num(df, "idle_seconds") / obs
    df["active_share"] = _num(df, "active_seconds") / obs
    df["coverage"] = _num(df, "coverage")
    df["switches_per_min"] = _num(df, "switches_per_min")
    site_rate = _num(df, "site_switches_per_min")
    df["site_switches_per_min"] = site_rate.fillna(_num(df, "site_switch_count") / mins)
    df["keys_per_min"] = _num(df, "keyboard_count") / mins
    df["mouse_per_min"] = _num(df, "mouse_count") / mins
    df["idle_episodes"] = _num(df, "idle_episodes")
    df["distinct_apps"] = _num(df, "distinct_apps")
    df["browser_share"] = (_num(df, "browser_seconds") / obs).clip(upper=1)
    df["distinct_hosts"] = _num(df, "distinct_hosts")
    for c in SHARE_COLUMNS:
        df[c] = _num(df, c)

    apps = df["app_seconds"] if "app_seconds" in df.columns else pd.Series(None, index=df.index)
    shares = []
    for text in apps:
        secs = parse_app_seconds(text)
        total = sum(secs.values())
        row = dict.fromkeys(APP_GROUPS, 0.0)
        for name, s in secs.items():
            row[app_category(name)] += s
        shares.append({f"app_{g}": (v / total if total else math.nan) for g, v in row.items()})
    df = pd.concat([df, pd.DataFrame(shares, index=df.index)], axis=1)

    task = df["session_task_category"] if "session_task_category" in df.columns else None
    for t in TASK_CATEGORIES:
        df[f"task_{t}"] = (task == t).astype(float) if task is not None else 0.0
    df["task_none"] = (task.isna() | ~task.isin(TASK_CATEGORIES)).astype(float) \
        if task is not None else 1.0
    hour = df["window_start"].dt.hour + df["window_start"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df
