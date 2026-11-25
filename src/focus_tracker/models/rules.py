"""Rule-based yardstick classifier (slide 9 step 3; RQ2 compares models to it).

    Inactive   if idle_share >= t_idle
    Distracted if (entertainment + social share of browser time) >= t_ent
               or app switches per minute >= t_switch
    Focused    otherwise

The three thresholds are chosen by grid search on the TRAINING windows only
(macro-F1), so the yardstick is as strong as fixed global rules can be.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..logger.labels import LABELS

GRID = {"t_idle": (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
        "t_ent": (0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
        "t_switch": (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)}


def _inputs(df):
    idle = pd.to_numeric(df["idle_share"], errors="coerce").fillna(0).to_numpy()
    ent = (pd.to_numeric(df["share_entertainment"], errors="coerce").fillna(0)
           + pd.to_numeric(df["share_social"], errors="coerce").fillna(0)).to_numpy()
    sw = pd.to_numeric(df["switches_per_min"], errors="coerce").fillna(0).to_numpy()
    return idle, ent, sw


def apply_rules(idle, ent, sw, t_idle, t_ent, t_switch):
    out = np.full(len(idle), "Focused", dtype=object)
    out[(ent >= t_ent) | (sw >= t_switch)] = "Distracted"
    out[idle >= t_idle] = "Inactive"
    return out


class RuleClassifier:
    def __init__(self):
        self.thresholds = None

    def fit(self, train: pd.DataFrame):
        idle, ent, sw = _inputs(train)
        y = train["label"].to_numpy()
        labels = [c for c in LABELS if c in set(y)]
        best = None
        for t in itertools.product(*GRID.values()):
            pred = apply_rules(idle, ent, sw, *t)
            score = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
            if best is None or score > best[0]:
                best = (score, t)
        self.thresholds = dict(zip(GRID, best[1]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("fit first")
        return apply_rules(*_inputs(df), **self.thresholds)
