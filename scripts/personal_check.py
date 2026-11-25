"""Personal vs shared models. For every person: train on their EARLIER windows
(time split as in evaluate.py), test on their LATER windows. Nothing is tuned on
test windows.

  shared_new   other people's windows only (a brand-new user)
  shared_own   everyone's earlier windows pooled (the current evaluation setup)
  personal     only the person's own earlier windows
  blend        other people + own earlier windows, own windows up-weighted
               (repeated) so they count as much as everyone else together

usage: python scripts/personal_check.py data/participants <an activity.db with own labels> results/run2/personal_check.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.simplefilter("ignore")
from focus_tracker.api.export import export_rows  # noqa: E402
from focus_tracker.baseline.ewma import EwmaBaseline  # noqa: E402
from focus_tracker.logger.labels import LABELS  # noqa: E402
from focus_tracker.logger.storage import EventStore  # noqa: E402
from focus_tracker.models.data import RAW_FEATURES, Z_FEATURES, add_features, load_exports  # noqa: E402
from focus_tracker.models.evaluate import time_split  # noqa: E402
from focus_tracker.models.gbt import GbtModel  # noqa: E402
from focus_tracker.models.rules import RuleClassifier  # noqa: E402

Z_COLUMNS = tuple(f"z_{f}" for f in Z_FEATURES)
KINDS = ("rule", "gbt_raw", "gbt_z")
SETUPS = ("shared_new", "shared_own", "personal", "blend")


def fit_predict(kind, df, own_idx, other_idx, test_idx, reps=1, seed=0):
    """Train on other_idx + own_idx (own rows repeated `reps` times); predict test_idx.
    z-scores: prior from the (unrepeated) training rows; each person's history in
    time order, past windows only."""
    train_idx = other_idx.union(own_idx)
    if len(set(df.loc[train_idx, "label"])) < 2:
        return None
    if kind == "gbt_z":
        base = EwmaBaseline(0.05).fit_prior(df.loc[train_idx], Z_FEATURES)
        df = pd.concat([df, base.transform(df, Z_FEATURES)], axis=1)
    train = pd.concat([df.loc[other_idx]] + [df.loc[own_idx]] * reps, ignore_index=True)
    if kind == "rule":
        m = RuleClassifier().fit(train)
    elif kind == "gbt_raw":
        m = GbtModel(RAW_FEATURES, seed).fit(train)
    else:
        m = GbtModel(tuple(RAW_FEATURES) + Z_COLUMNS, seed).fit(train)
    return np.asarray(m.predict(df.loc[test_idx]))


def score(y, p):
    labs = [c for c in LABELS if c in set(y)]
    return (f1_score(y, p, labels=labs, average="macro", zero_division=0), float(np.mean(y == p)))


def main(parts_dir, my_db, out_csv):
    pop = load_exports([parts_dir])
    me = pd.DataFrame(export_rows(EventStore(my_db), "you"))
    me["window_start"] = pd.to_datetime(me["window_start"])
    df = add_features(pd.concat([pop, me], ignore_index=True)
                      .sort_values(["participant", "window_start"]).reset_index(drop=True))
    tr_idx, te_idx = time_split(df)
    is_tr = df.index.isin(tr_idx)
    rows = []
    for person in sorted(df["participant"].unique()):
        mine = (df["participant"] == person).to_numpy()
        test = df.index[mine & df.index.isin(te_idx)]
        own_tr = df.index[mine & is_tr]
        others = df.index[~mine & (df["participant"] != "you").to_numpy()]   # 'you' never trains others
        others_tr = df.index[~mine & is_tr & (df["participant"] != "you").to_numpy()]
        if len(test) == 0:
            continue
        y = df.loc[test, "label"].to_numpy()
        reps = max(1, round(len(others) / max(1, len(own_tr))))
        empty = df.index[:0]
        trains = {                                   # (others, own, repeats)
            "shared_new": (others, empty, 1),
            "shared_own": (others_tr, own_tr, 1),
            "personal": (empty, own_tr, 1),
            "blend": (others, own_tr, reps),
        }
        own_classes = sorted(set(df.loc[own_tr, "label"]))
        for kind in KINDS:
            for setup, (oth, own, r) in trains.items():
                p = fit_predict(kind, df, own, oth, test, r)
                f1, acc = score(y, p) if p is not None else (np.nan, np.nan)
                rows.append({"person": person, "kind": kind, "setup": setup, "macro_f1": f1,
                             "accuracy": acc, "n_test": len(test), "n_own_train": len(own_tr),
                             "own_train_classes": "/".join(c[0] for c in own_classes),
                             "test_classes": "/".join(c[0] for c in sorted(set(y)))})
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)

    # personal needs >= 2 labels in the person's own training windows: fall back
    # to shared_new otherwise (what the app would do)
    piv = res.pivot_table(index=["kind", "person"], columns="setup", values="accuracy")
    pf1 = res.pivot_table(index=["kind", "person"], columns="setup", values="macro_f1")
    for t in (piv, pf1):
        t["personal*"] = t["personal"].fillna(t["shared_new"])
    print("Accuracy on each person's later windows")
    for kind in KINDS:
        t = piv.loc[kind][list(SETUPS[:2]) + ["personal*", "blend"]]
        print(f"\n== {kind} ==")
        info = res[(res.kind == kind) & (res.setup == "personal")].set_index("person")
        t = t.assign(n_test=info["n_test"], own_train=info["n_own_train"],
                     own_labels=info["own_train_classes"], test_labels=info["test_classes"])
        print(t.round(2).to_string())
        f = pf1.loc[kind][list(SETUPS[:2]) + ["personal*", "blend"]]
        print("mean accuracy :", t[list(SETUPS[:2]) + ["personal*", "blend"]].mean().round(3).to_dict())
        print("mean macro-F1 :", f.mean().round(3).to_dict())
        w = (t["personal*"] > t["shared_own"]).sum(), (t["personal*"] < t["shared_own"]).sum()
        print(f"personal* vs shared_own: better for {w[0]}, worse for {w[1]} of {len(t)} people")


if __name__ == "__main__":
    main(*sys.argv[1:])
