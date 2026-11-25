"""Developer annotations of participants' exported windows (separate team.db).

* Participants' own labels (column `user_label`) are the ground truth and are
  never changed here. This module never opens activity.db.
* Developer labels are a separate annotation with the annotator's name. The
  developer only sees exported features, so these labels are feature-based:
  use them for agreement analysis, never to evaluate a model trained on the
  same features (that would be circular).
"""
import csv
import io
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from ..logger.labels import LABELS, normalize_label
from .export import (BASE_COLUMNS, EXPORT_COLUMNS, LABEL_COLUMNS, WEB_COLUMNS, to_csv,
                     validate_pseudonym)

_TEXT = {"participant", "window_start", "window_end", "session_task_category",
         "label", "label_source", "labelled_at", "updated_at", "hostnames", "video_titles",
         "top_app", "app_seconds"}
_FLOAT = {"coverage", "switches_per_min", "site_switches_per_min"} | {
    c for c in WEB_COLUMNS if c.startswith("share_")}
# Exports made before the browser columns existed are still accepted.
_REQUIRED = set(BASE_COLUMNS) | set(LABEL_COLUMNS)


def _sql_type(c):
    return "TEXT" if c in _TEXT else "REAL" if c in _FLOAT else "INTEGER"
_KEY = ("participant", "window_start", "window_seconds")

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_windows ({cols},
    PRIMARY KEY (participant, window_start, window_seconds));
CREATE TABLE IF NOT EXISTS developer_labels (
    participant    TEXT    NOT NULL,
    window_start   TEXT    NOT NULL,
    window_seconds INTEGER NOT NULL,
    label          TEXT    NOT NULL CHECK (label IN ('Focused', 'Distracted', 'Inactive')),
    annotator      TEXT    NOT NULL,
    labelled_at    TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL,
    PRIMARY KEY (participant, window_start, window_seconds)
);
""".format(cols=", ".join(f"{c} {_sql_type(c)}" for c in EXPORT_COLUMNS))

EXPORT_TEAM_COLUMNS = EXPORT_COLUMNS[:-4] + ("user_label", "developer_label",
                                             "annotator")


class CsvImportError(ValueError):
    """Rejected CSV upload."""


def _cast(col, value):
    if value is None or value == "":
        return None
    if col in _TEXT:
        return value
    try:
        return float(value) if col in _FLOAT else int(float(value))
    except ValueError:
        raise CsvImportError(f"column {col!r} has a non-numeric value {value!r}") from None


class TeamStore:
    def __init__(self, path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=10)
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(team_windows)")}
        for c in EXPORT_COLUMNS:                 # team.db from an older version
            if c not in have:
                self.conn.execute(f"ALTER TABLE team_windows ADD COLUMN {c} {_sql_type(c)}")
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -- import ------------------------------------------------------------
    def import_csv(self, text: str) -> dict:
        reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        header = set(reader.fieldnames or [])
        extra = header - set(EXPORT_COLUMNS)
        missing = _REQUIRED - header
        if extra:
            raise CsvImportError("unexpected columns (only app exports are accepted): "
                               + ", ".join(sorted(extra)))
        if missing:
            raise CsvImportError("missing columns: " + ", ".join(sorted(missing)))
        rows, participants = [], Counter()
        for i, r in enumerate(reader, start=2):
            try:
                validate_pseudonym(r["participant"])
                if r["label"]:
                    normalize_label(r["label"])
                row = tuple(_cast(c, r.get(c)) for c in EXPORT_COLUMNS)
            except ValueError as e:
                raise CsvImportError(f"line {i}: {e}") from None
            if not (r["window_start"] and r["window_seconds"]):
                raise CsvImportError(f"line {i}: window_start / window_seconds are required")
            rows.append(row)
            participants[r["participant"]] += 1
        placeholders = ", ".join("?" * len(EXPORT_COLUMNS))
        with self.conn:
            self.conn.executemany(
                f"INSERT OR REPLACE INTO team_windows ({', '.join(EXPORT_COLUMNS)})"
                f" VALUES ({placeholders})", rows)
        return {"rows": len(rows), "participants": dict(participants)}

    # -- reading -----------------------------------------------------------
    def participants(self):
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT participant FROM team_windows ORDER BY participant")]

    def windows(self, participant=None, unlabelled_only=False, blind=False):
        sql = (f"SELECT {', '.join('t.' + c for c in EXPORT_COLUMNS)},"
               " d.label, d.annotator FROM team_windows t LEFT JOIN developer_labels d"
               " USING (participant, window_start, window_seconds)")
        args, where = [], []
        if participant:
            where.append("t.participant = ?")
            args.append(participant)
        if unlabelled_only:
            where.append("d.label IS NULL")
        if where:
            sql += " WHERE " + " AND ".join(where)
        out = []
        for row in self.conn.execute(sql + " ORDER BY t.participant, t.window_start", args):
            d = dict(zip(EXPORT_COLUMNS, row[:len(EXPORT_COLUMNS)]))
            d["user_label"] = None if blind else d.pop("label")
            d.pop("label", None)
            d["developer_label"], d["annotator"] = row[len(EXPORT_COLUMNS):]
            out.append(d)
        return out

    # -- developer labels --------------------------------------------------
    def _exists(self, key):
        return self.conn.execute(
            "SELECT 1 FROM team_windows WHERE participant = ? AND window_start = ?"
            " AND window_seconds = ?", key).fetchone() is not None

    def set_labels(self, items, label, annotator, now: datetime):
        label = normalize_label(label)
        annotator = (annotator or "").strip()
        if not annotator or len(annotator) > 40:
            raise ValueError("annotator name is required (max 40 characters)")
        ts, results = now.isoformat(timespec="seconds"), []
        with self.conn:
            for key in items:
                key = tuple(key)
                if not self._exists(key):
                    results.append({"key": key, "ok": False, "error": "unknown window"})
                    continue
                prev = self.conn.execute(
                    "SELECT label FROM developer_labels WHERE participant = ?"
                    " AND window_start = ? AND window_seconds = ?", key).fetchone()
                self.conn.execute(
                    "INSERT INTO developer_labels VALUES (?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT (participant, window_start, window_seconds) DO UPDATE"
                    " SET label = excluded.label, annotator = excluded.annotator,"
                    " updated_at = excluded.updated_at", (*key, label, annotator, ts, ts))
                results.append({"key": key, "ok": True,
                                "previous": prev[0] if prev else None})
        return results

    def clear_labels(self, items):
        with self.conn:
            return sum(self.conn.execute(
                "DELETE FROM developer_labels WHERE participant = ? AND window_start = ?"
                " AND window_seconds = ?", tuple(k)).rowcount for k in items)

    # -- agreement & export ------------------------------------------------
    def agreement(self):
        pairs = {}
        for p, u, d in self.conn.execute(
                "SELECT t.participant, t.label, d.label FROM team_windows t"
                " JOIN developer_labels d USING (participant, window_start, window_seconds)"
                " WHERE t.label IS NOT NULL"):
            pairs.setdefault(p, []).append((u, d))
        result = {p: agreement_stats(v) for p, v in sorted(pairs.items())}
        result["ALL"] = agreement_stats([x for v in pairs.values() for x in v])
        return result

    def export_csv(self) -> str:
        return to_csv(self.windows(), EXPORT_TEAM_COLUMNS)


def agreement_stats(pairs):
    """Observed agreement, Cohen's kappa and a 3x3 confusion (user rows)."""
    n = len(pairs)
    confusion = {u: {d: 0 for d in LABELS} for u in LABELS}
    for u, d in pairs:
        confusion[u][d] += 1
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None, "confusion": confusion}
    po = sum(u == d for u, d in pairs) / n
    users, devs = Counter(u for u, _ in pairs), Counter(d for _, d in pairs)
    pe = sum(users[c] * devs[c] for c in LABELS) / (n * n)
    kappa = None if pe == 1 else (po - pe) / (1 - pe)
    return {"n": n, "agreement": round(po, 4),
            "kappa": None if kappa is None else round(kappa, 4), "confusion": confusion}
