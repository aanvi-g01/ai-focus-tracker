"""Gradient-boosted trees (LightGBM) per-window classifier (slide 9 step 3).

Small data, so: shallow trees, balanced class weights, and a tiny
hyperparameter grid chosen on the LAST part (in time) of each participant's
training windows (forward-chaining validation), then refit on all training.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..logger.labels import LABELS

GRID = ({"num_leaves": 7, "n_estimators": 150},
        {"num_leaves": 7, "n_estimators": 400},
        {"num_leaves": 15, "n_estimators": 150},
        {"num_leaves": 15, "n_estimators": 400})


def _lgbm(params, seed, n_classes=3):
    """Multiclass model; with only two labels present (common for one person's
    own windows) LightGBM needs the binary objective instead."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass" if n_classes > 2 else "binary",
                              learning_rate=0.05,
                              min_child_samples=5, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.9, class_weight="balanced",
                              random_state=seed, verbose=-1, **params)


def time_tail_split(df, frac=0.2):
    """Per participant, the latest `frac` of windows -> validation."""
    val = []
    for _, g in df.sort_values("window_start").groupby("participant"):
        n = int(round(len(g) * frac))
        if n:
            val.extend(g.index[-n:])
    return df.index.difference(val), pd.Index(val)


class GbtModel:
    def __init__(self, features, seed=0):
        self.features = list(features)
        self.seed = seed
        self.model = None
        self.params = None
        self.classes_ = None

    def _xy(self, df):
        X = df[self.features].apply(pd.to_numeric, errors="coerce").astype(float)
        return X, df["label"].to_numpy() if "label" in df else None

    def fit(self, train: pd.DataFrame):
        X, y = self._xy(train)
        tr, va = time_tail_split(train)
        best = (None, GRID[0])
        if len(va) and len(set(y[train.index.get_indexer(tr)])) > 1:
            labels = [c for c in LABELS if c in set(train.loc[va, "label"])]
            for params in GRID:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = _lgbm(params, self.seed, train.loc[tr, "label"].nunique())                         .fit(X.loc[tr], train.loc[tr, "label"])
                    s = f1_score(train.loc[va, "label"], m.predict(X.loc[va]),
                                 labels=labels, average="macro", zero_division=0)
                if best[0] is None or s > best[0]:
                    best = (s, params)
        self.params = best[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = _lgbm(self.params, self.seed, len(set(y))).fit(X, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, df) -> np.ndarray:
        X, _ = self._xy(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(X)

    def predict_proba(self, df) -> np.ndarray:
        X, _ = self._xy(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict_proba(X)
