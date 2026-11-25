"""Does SMOTE-style synthetic data improve focus labelling? Synthetic rows are
generated from TRAINING windows only and every score is on real held-out windows.

usage: python scripts/synth_experiment.py data/participants <an activity.db with own labels>
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.simplefilter("ignore")
from focus_tracker.api.export import export_rows  # noqa: E402
from focus_tracker.logger.labels import LABELS  # noqa: E402
from focus_tracker.logger.storage import EventStore  # noqa: E402
from focus_tracker.models.data import RAW_FEATURES, Z_FEATURES, add_features, load_exports  # noqa: E402
from focus_tracker.models.evaluate import _with_z, time_split  # noqa: E402
from focus_tracker.models.gbt import GbtModel, _lgbm  # noqa: E402

Z_COLUMNS = tuple(f"z_{f}" for f in Z_FEATURES)
FEATS = {"gbt_raw": list(RAW_FEATURES), "gbt_z": list(RAW_FEATURES) + list(Z_COLUMNS)}


def smote(X, y, rng, k=5, target=None):
    """Interpolate new rows within each class up to `target` rows per class."""
    target = target or max(pd.Series(y).value_counts())
    binary = [j for j in range(X.shape[1])
              if set(np.unique(X[~np.isnan(X[:, j]), j])) <= {0.0, 1.0}]
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0, med)
    sd = np.nanstd(X, axis=0)
    sd = np.where((sd == 0) | np.isnan(sd), 1, sd)
    Z = (np.where(np.isnan(X), med, X) - med) / sd
    new_X, new_y = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        need = target - len(idx)
        if need <= 0 or len(idx) < 2:
            continue
        d = ((Z[idx, None, :] - Z[None, idx, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, :min(k, len(idx) - 1)]
        for _ in range(need):
            a = rng.integers(len(idx))
            b = nn[a, rng.integers(nn.shape[1])]
            xa, xb = X[idx[a]], X[idx[b]]
            u = rng.random()
            x = np.where(np.isnan(xa) | np.isnan(xb), xa, xa + u * (xb - xa))
            for j in binary:                       # one-hot / flags: copy, don't blend
                x[j] = xa[j] if u < 0.5 else xb[j]
            new_X.append(x)
            new_y.append(c)
    if not new_X:
        return X, y
    return np.vstack([X, new_X]), np.concatenate([y, new_y])


def fit_predict(train, test, kind, synth, seed=0):
    feats = FEATS[kind]
    m = GbtModel(feats, seed).fit(train)          # hyper-params chosen on REAL data only
    if synth:
        X = train[feats].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()
        Xa, ya = smote(X, train["label"].to_numpy(), np.random.default_rng(seed))
        m.model = _lgbm(m.params, seed).fit(pd.DataFrame(Xa, columns=feats), ya)
    return m.predict(test)


def macro(y, p):
    labs = [c for c in LABELS if c in set(y)]
    return f1_score(y, p, labels=labs, average="macro", zero_division=0)


def main(parts_dir, my_db):
    df = add_features(load_exports([parts_dir]))
    res = {}
    # 1. time split (later days of the same people)
    tr, te = time_split(df)
    dz = _with_z(df, tr, 0.05)
    for kind in FEATS:
        for synth in (False, True):
            p = fit_predict(dz.loc[tr], dz.loc[te], kind, synth)
            res[("time split", kind, synth)] = macro(df.loc[te, "label"].to_numpy(), p)
    # 2. leave one participant out (new people)
    for kind in FEATS:
        for synth in (False, True):
            vals = []
            for part in sorted(df["participant"].unique()):
                tr_i = df.index[df["participant"] != part]
                te_i = df.index[df["participant"] == part]
                dz = _with_z(df, tr_i, 0.05)
                vals.append(macro(df.loc[te_i, "label"].to_numpy(),
                                  fit_predict(dz.loc[tr_i], dz.loc[te_i], kind, synth)))
            res[("LOPO mean", kind, synth)] = float(np.mean(vals))
    # 3. a person outside the training data (the local app user)
    me = pd.DataFrame(export_rows(EventStore(my_db), "me"))
    me["window_start"] = pd.to_datetime(me["window_start"])
    both = add_features(pd.concat([load_exports([parts_dir]), me], ignore_index=True))
    tr_i = both.index[both["participant"] != "me"]
    te_i = both.index[both["participant"] == "me"]
    dz = _with_z(both, tr_i, 0.05)
    for kind in FEATS:
        for synth in (False, True):
            y = both.loc[te_i, "label"].to_numpy()
            p = fit_predict(dz.loc[tr_i], dz.loc[te_i], kind, synth)
            res[("new user (you)", kind, synth)] = macro(y, p)
            res[("new user acc", kind, synth)] = float(np.mean(y == p))
    counts = df["label"].value_counts().to_dict()
    print("real training windows:", counts)
    print(f"\n{'test':<16} {'model':<8} {'real only':>10} {'+ synthetic':>12}")
    for (t, k, s), v in res.items():
        if s:
            continue
        print(f"{t:<16} {k:<8} {v:>10.3f} {res[(t, k, True)]:>12.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
