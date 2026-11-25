"""Supervised Gaussian HMM with Viterbi decoding (slide 9 step 3).

Hidden states are the three focus labels, so the model is estimated directly
from labelled data (no Baum-Welch, no state-to-label mapping problem):
* pi, A: counts of first labels / label transitions between consecutive
  windows of a sequence, with add-one smoothing;
* emissions: diagonal Gaussian per state on standardized features
  (missing values imputed with the training mean, after standardizing = 0).
Sequences are split where consecutive windows are more than `max_gap`
apart (logger off, or unlabelled stretch). Viterbi runs in log space.
"""
import numpy as np
import pandas as pd

from ..logger.labels import LABELS

CORE_FEATURES = ("idle_share", "switches_per_min", "site_switches_per_min",
                 "keys_per_min", "mouse_per_min", "browser_share",
                 "share_entertainment", "share_social", "app_dev", "app_media", "app_comm")


def sequences(df, max_gap_seconds):
    """Lists of row indices: per participant, split at gaps > max_gap_seconds."""
    out = []
    for _, g in df.sort_values("window_start").groupby("participant", sort=False):
        cur, prev = [], None
        for idx, ts in zip(g.index, g["window_start"]):
            if prev is not None and (ts - prev).total_seconds() > max_gap_seconds:
                out.append(cur)
                cur = []
            cur.append(idx)
            prev = ts
        if cur:
            out.append(cur)
    return out


def viterbi(log_pi, log_A, log_B):
    """Most likely state path. log_B: (T, N) log emission likelihoods."""
    T, N = log_B.shape
    delta = log_pi + log_B[0]
    back = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + log_A          # (from, to)
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_B[t]
    path = [int(delta.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


class SupervisedHMM:
    def __init__(self, features=CORE_FEATURES, window_seconds=300, max_gap_seconds=600,
                 var_floor=1e-2):
        self.features = list(features)
        self.window_seconds = window_seconds
        self.max_gap_seconds = max_gap_seconds
        self.var_floor = var_floor
        self.states = list(LABELS)

    def _X(self, df):
        X = df[self.features].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()
        X = (X - self.mean_) / self.std_
        return np.nan_to_num(X, nan=0.0)          # NaN -> training mean

    def fit(self, train: pd.DataFrame):
        raw = train[self.features].apply(pd.to_numeric, errors="coerce").astype(float)
        self.mean_ = np.nan_to_num(raw.mean().to_numpy(), nan=0.0)
        std = raw.std().to_numpy()
        self.std_ = np.where(np.isnan(std) | (std <= 0), 1.0, std)
        X = self._X(train)
        y = train["label"].to_numpy()
        n = len(self.states)
        self.means_ = np.zeros((n, X.shape[1]))
        self.vars_ = np.ones((n, X.shape[1]))
        for i, s in enumerate(self.states):
            rows = X[y == s]
            if len(rows):
                self.means_[i] = rows.mean(axis=0)
                self.vars_[i] = np.maximum(rows.var(axis=0), self.var_floor) if len(rows) > 1 \
                    else 1.0
        pi = np.ones(n)
        A = np.ones((n, n))
        pos = {idx: k for k, idx in enumerate(train.index)}
        for seq in sequences(train, self.max_gap_seconds):
            labels = [self.states.index(y[pos[i]]) for i in seq]
            pi[labels[0]] += 1
            for a, b in zip(labels, labels[1:]):
                A[a, b] += 1
        self.log_pi_ = np.log(pi / pi.sum())
        self.log_A_ = np.log(A / A.sum(axis=1, keepdims=True))
        return self

    def _log_emissions(self, X):
        diff = X[:, None, :] - self.means_[None]                 # (T, N, F)
        return -0.5 * (np.log(2 * np.pi * self.vars_)[None] + diff ** 2 / self.vars_[None]).sum(-1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = pd.Series(index=df.index, dtype=object)
        X = pd.DataFrame(self._X(df), index=df.index)
        for seq in sequences(df, self.max_gap_seconds):
            path = viterbi(self.log_pi_, self.log_A_, self._log_emissions(X.loc[seq].to_numpy()))
            out.loc[seq] = [self.states[k] for k in path]
        return out.to_numpy()
