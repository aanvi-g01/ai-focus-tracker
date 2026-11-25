"""Personal baseline with habituation (UC-17, slide 9 step 2).

For each participant and feature, windows are processed in time order:

    z_t   = (x_t - mu_{t-1}) / max(sigma_{t-1}, floor)     # compare with the PAST
    mu_t  = (1 - a) mu_{t-1} + a x_t                        # then update
    var_t = (1 - a) (var_{t-1} + a (x_t - mu_{t-1})^2)

So the baseline follows the user as their habits change (old behaviour fades
with weight (1-a)^k), and a window is never compared with itself or the future.

Cold start: a new user has little history, so their statistics are blended
with a population prior (computed from TRAINING windows only):
    mu_eff = w mu_user + (1 - w) mu_prior,  w = n / (n + prior_weight)
NaN inputs give a NaN z-score and do not update the baseline.
"""
import math

import numpy as np
import pandas as pd


class EwmaBaseline:
    def __init__(self, alpha=0.05, prior_weight=10, floor_fraction=0.1):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.floor_fraction = floor_fraction
        self.prior = {}

    def fit_prior(self, train: pd.DataFrame, features):
        """Population mean / variance per feature from the training windows."""
        self.prior = {}
        for f in features:
            x = pd.to_numeric(train[f], errors="coerce").to_numpy(float)
            x = x[~np.isnan(x)]
            mu = float(x.mean()) if len(x) else 0.0
            var = float(x.var()) if len(x) > 1 else 1.0
            self.prior[f] = (mu, var if var > 0 else 1.0)
        return self

    def _floor(self, f):
        return max(1e-6, self.floor_fraction * math.sqrt(self.prior[f][1]))

    def z_scores(self, x, feature) -> np.ndarray:
        """z-scores of one participant's chronological values of one feature."""
        a, k = self.alpha, self.prior_weight
        p_mu, p_var = self.prior[feature]
        floor = self._floor(feature)
        mu = var = 0.0
        n = 0
        out = np.full(len(x), np.nan)
        for i, v in enumerate(x):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            w = n / (n + k) if (n + k) > 0 else 0.0
            m_eff = w * mu + (1 - w) * p_mu
            v_eff = w * var + (1 - w) * p_var
            out[i] = (v - m_eff) / max(math.sqrt(max(v_eff, 0.0)), floor)
            if n == 0:
                mu, var = float(v), 0.0
            else:
                d = v - mu
                mu += a * d
                var = (1 - a) * (var + a * d * d)
            n += 1
        return out

    def transform(self, df: pd.DataFrame, features) -> pd.DataFrame:
        """`z_<feature>` columns for every row of df (all participants).

        Rows are processed per participant in window_start order, so test
        windows see the training history that precedes them, as in real use.
        """
        if not self.prior:
            raise RuntimeError("call fit_prior() first")
        out = pd.DataFrame(index=df.index, columns=[f"z_{f}" for f in features], dtype=float)
        for _, g in df.sort_values("window_start").groupby("participant", sort=False):
            for f in features:
                vals = pd.to_numeric(g[f], errors="coerce").to_numpy(float)
                out.loc[g.index, f"z_{f}"] = self.z_scores(vals, f)
        return out
