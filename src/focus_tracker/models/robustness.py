"""Robustness checks for the final evaluation (run after focus-evaluate / focus-train).

    python -m focus_tracker.models.robustness --data data/participants --out results/run2 \
        [--relabel old_export.csv new_export.csv]

Same protocol as evaluate.py (per-participant time split, leave-one-participant-out,
pooled training, nothing tuned on test windows). Checks:
1. Time-of-day ablation: LightGBM without hour_sin / hour_cos.
2. Scores without single-class participants (their perfect scores are trivial).
3. Seed stability of LightGBM raw vs + z-scores (RQ1).
4. Focused vs Distracted only (secondary, 2-class) from the same predictions.
5. Inactive breakdown (where the Inactive labels are, and how they are predicted).
6. Label consistency between two exports of the same windows (optional).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from ..logger.labels import LABELS
from .data import RAW_FEATURES, add_features, load_exports
from .evaluate import (MODEL_NAMES, Z_COLUMNS, _with_z, bootstrap_ci, fit_predict_all,
                       mcnemar, scores, time_split)
from .gbt import GbtModel

TIME_FEATURES = ("hour_sin", "hour_cos")
MODELS = ("rule", "gbt_raw", "gbt_z", "hmm")
SEEDS = (0, 1, 2, 3, 4)
TWO_CLASS = ("Focused", "Distracted")


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
def gbt_no_time(df, train_idx, test_idx, alpha, seed):
    """LightGBM raw / + z predictions without the time-of-day features."""
    dz = _with_z(df, train_idx, alpha)
    tr, te = dz.loc[train_idx], dz.loc[test_idx]
    raw = [f for f in RAW_FEATURES if f not in TIME_FEATURES]
    return {"gbt_raw": GbtModel(raw, seed).fit(tr).predict(te),
            "gbt_z": GbtModel(tuple(raw) + Z_COLUMNS, seed).fit(tr).predict(te)}


def lopo_predictions(df, alpha, seed, no_time=False):
    """Leave-one-participant-out predictions for every window, one column per model."""
    out = pd.DataFrame(index=df.index, columns=MODELS if not no_time else ("gbt_raw", "gbt_z"),
                       dtype=object)
    for p in sorted(df["participant"].unique()):
        tr = df.index[df["participant"] != p]
        te = df.index[df["participant"] == p]
        if len(set(df.loc[tr, "label"])) < 2:
            continue
        preds = gbt_no_time(df, tr, te, alpha, seed) if no_time else \
            fit_predict_all(df, tr, te, alpha, seed)[0]
        for m, pr in preds.items():
            out.loc[te, m] = pr
    return out


def per_user_mean(y, pred, parts, exclude=()):
    """Mean over participants of their own macro-F1 (as in the LOPO table)."""
    vals = [scores(y[parts == p], pred[parts == p])["macro_f1"]
            for p in sorted(set(parts)) if p not in exclude]
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def two_class_f1(y, pred):
    """Macro-F1 over Focused/Distracted on windows labelled one of them."""
    keep = np.isin(y, TWO_CLASS)
    if not keep.any():
        return None
    labels = [c for c in TWO_CLASS if c in set(y[keep])]
    return round(float(f1_score(y[keep], pred[keep], labels=labels, average="macro",
                                zero_division=0)), 4)


def single_class_participants(df):
    return sorted(p for p, g in df.groupby("participant") if g["label"].nunique() < 2)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def label_consistency(old_path, new_path):
    """Agreement between two exports of the same person's windows."""
    a = load_exports(old_path)[["window_start", "label"]]
    b = load_exports(new_path)[["window_start", "label"]]
    m = a.merge(b, on="window_start", suffixes=("_old", "_new"))
    if m.empty:
        return {"n_common": 0}
    same = (m["label_old"] == m["label_new"])
    both = sorted(set(m["label_old"]) | set(m["label_new"]))
    kappa = cohen_kappa_score(m["label_old"], m["label_new"], labels=both) \
        if len(both) > 1 else 1.0
    return {"n_common": int(len(m)), "agreement": round(float(same.mean()), 4),
            "kappa": round(float(kappa), 4) if not np.isnan(kappa) else 1.0,
            "changes": {f"{o} -> {n}": int(c) for (o, n), c in
                        m[~same].groupby(["label_old", "label_new"]).size().items()}}


def run(df, alpha=0.05, seed=0, seeds=SEEDS, relabel=None):
    df = add_features(df)
    tr_idx, te_idx = time_split(df)
    y = df.loc[te_idx, "label"].to_numpy()
    parts_te = df.loc[te_idx, "participant"].to_numpy()
    single = single_class_participants(df)
    res = {"config": {"alpha": alpha, "seed": seed, "seeds": list(seeds)},
           "single_class_participants": single}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full, _, _ = fit_predict_all(df, tr_idx, te_idx, alpha, seed)
        no_time = gbt_no_time(df, tr_idx, te_idx, alpha, seed)
        lopo = lopo_predictions(df, alpha, seed)
        lopo_nt = lopo_predictions(df, alpha, seed, no_time=True)

        seed_runs = {"gbt_raw": [], "gbt_z": []}
        for s in seeds:
            pr = full if s == seed else gbt_full_features(df, tr_idx, te_idx, alpha, s)
            for m in seed_runs:
                seed_runs[m].append(scores(y, pr[m])["macro_f1"])

    ly = df["label"].to_numpy()
    lparts = df["participant"].to_numpy()
    scored = lopo.notna().all(axis=1).to_numpy()

    def lopo_mean(pred, exclude=()):
        return per_user_mean(ly[scored], pred[scored], lparts[scored], exclude)

    # 1. time-of-day ablation
    res["time_ablation"] = {}
    for m in ("gbt_raw", "gbt_z"):
        f, n = full[m], no_time[m]
        res["time_ablation"][m] = {
            "with_time": {"macro_f1": scores(y, f)["macro_f1"],
                          "lopo_mean": lopo_mean(lopo[m].to_numpy())},
            "without_time": {"macro_f1": scores(y, n)["macro_f1"],
                             "ci95": bootstrap_ci(y, n, seed=seed),
                             "lopo_mean": lopo_mean(lopo_nt[m].to_numpy())},
            "mcnemar_without_vs_with": mcnemar(y, n, f)}

    # 2. without single-class participants
    keep = ~np.isin(parts_te, single)
    res["without_single_class"] = {m: {
        "time_split_macro_f1": scores(y[keep], full[m][keep])["macro_f1"],
        "lopo_mean": lopo_mean(lopo[m].to_numpy(), exclude=single),
        "lopo_mean_all": lopo_mean(lopo[m].to_numpy())} for m in MODELS}

    # 3. seed stability
    dz = [z - r for z, r in zip(seed_runs["gbt_z"], seed_runs["gbt_raw"])]
    res["seed_stability"] = {
        **{m: {"values": v, "mean": round(float(np.mean(v)), 4),
               "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0}
           for m, v in seed_runs.items()},
        "delta_z_minus_raw": {"values": [round(d, 4) for d in dz],
                              "mean": round(float(np.mean(dz)), 4),
                              "min": round(float(np.min(dz)), 4),
                              "positive_in": f"{sum(d > 0 for d in dz)}/{len(dz)}"}}

    # 4. Focused vs Distracted
    res["focused_vs_distracted"] = {m: {
        "time_split_macro_f1": two_class_f1(y, full[m]),
        "lopo_macro_f1": two_class_f1(ly[scored], lopo[m].to_numpy()[scored])}
        for m in MODELS}

    # 5. Inactive breakdown
    inact = {}
    for p, g in df.groupby("participant"):
        n = int((g["label"] == "Inactive").sum())
        if n:
            inact[p] = {"total": n,
                        "train": int(((g["label"] == "Inactive") & g.index.isin(tr_idx)).sum()),
                        "test": int(((g["label"] == "Inactive") & g.index.isin(te_idx)).sum())}
    is_in = y == "Inactive"
    lopo_in = (ly == "Inactive") & scored
    res["inactive"] = {
        "per_participant": inact,
        "test_windows": int(is_in.sum()),
        "test_predicted_as": {m: {c: int((full[m][is_in] == c).sum()) for c in LABELS}
                              for m in MODELS},
        "lopo_recall": {m: (round(float((lopo[m].to_numpy()[lopo_in] == "Inactive").mean()), 4)
                            if lopo_in.any() else None) for m in MODELS},
        "lopo_recall_per_participant": {
            p: {m: round(float((lopo.loc[g.index, m] == "Inactive").mean()), 4) for m in MODELS}
            for p, g in df[(df["label"] == "Inactive") & scored].groupby("participant")}}

    if relabel:
        res["label_consistency"] = label_consistency(*relabel)
    return res


def gbt_full_features(df, train_idx, test_idx, alpha, seed):
    """Full-feature LightGBM raw / + z predictions (for the seed runs)."""
    dz = _with_z(df, train_idx, alpha)
    tr, te = dz.loc[train_idx], dz.loc[test_idx]
    return {"gbt_raw": GbtModel(RAW_FEATURES, seed).fit(tr).predict(te),
            "gbt_z": GbtModel(tuple(RAW_FEATURES) + Z_COLUMNS, seed).fit(tr).predict(te)}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def markdown(res):
    L = ["# Robustness checks", ""]
    L += ["## 1. Time-of-day ablation (without hour_sin / hour_cos)", "",
          "| Model | Macro-F1 with time | without time | 95% CI (without) | LOPO mean with | "
          "LOPO mean without | McNemar p |", "|---|---|---|---|---|---|---|"]
    for m, r in res["time_ablation"].items():
        ci = r["without_time"]["ci95"]
        L.append(f"| {MODEL_NAMES[m]} | {_f(r['with_time']['macro_f1'])} | "
                 f"{_f(r['without_time']['macro_f1'])} | "
                 f"{'n/a' if not ci else f'{ci[0]:.3f}-{ci[1]:.3f}'} | "
                 f"{_f(r['with_time']['lopo_mean'])} | {_f(r['without_time']['lopo_mean'])} | "
                 f"{r['mcnemar_without_vs_with']['p_value']:.4f} |")
    single = res["single_class_participants"]
    L += ["", f"## 2. Without single-class participants ({', '.join(single) or 'none'})", "",
          "Still used for training; only removed from scoring.", "",
          "| Model | Time-split macro-F1 | LOPO mean (all) | LOPO mean (excluded) |",
          "|---|---|---|---|"]
    for m, r in res["without_single_class"].items():
        L.append(f"| {MODEL_NAMES[m]} | {_f(r['time_split_macro_f1'])} | "
                 f"{_f(r['lopo_mean_all'])} | {_f(r['lopo_mean'])} |")
    s = res["seed_stability"]
    d = s["delta_z_minus_raw"]
    L += ["", f"## 3. Seed stability (seeds {res['config']['seeds']})", "",
          "| Model | Mean macro-F1 | SD |", "|---|---|---|"]
    for m in ("gbt_raw", "gbt_z"):
        L.append(f"| {MODEL_NAMES[m]} | {_f(s[m]['mean'])} | {_f(s[m]['sd'])} |")
    L += ["", f"ΔF1 (z minus raw): mean {d['mean']:+.3f}, smallest {d['min']:+.3f}, "
          f"positive in {d['positive_in']} seeds.", ""]
    L += ["## 4. Focused vs Distracted only (secondary, 2-class macro-F1)", "",
          "| Model | Time split | LOPO |", "|---|---|---|"]
    for m, r in res["focused_vs_distracted"].items():
        L.append(f"| {MODEL_NAMES[m]} | {_f(r['time_split_macro_f1'])} | "
                 f"{_f(r['lopo_macro_f1'])} |")
    ina = res["inactive"]
    L += ["", "## 5. Inactive", "", "| Participant | Inactive windows | Train | Test |",
          "|---|---|---|---|"]
    for p, c in ina["per_participant"].items():
        L.append(f"| {p} | {c['total']} | {c['train']} | {c['test']} |")
    L += ["", f"Inactive test windows: {ina['test_windows']}. Predicted as:", "",
          "| Model | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for m, c in ina["test_predicted_as"].items():
        L.append(f"| {MODEL_NAMES[m]} | " + " | ".join(str(c[k]) for k in LABELS) + " |")
    L += ["", "Leave-one-participant-out Inactive recall (held-out user's Inactive windows):", "",
          "| Held-out | " + " | ".join(MODEL_NAMES[m] for m in MODELS) + " |",
          "|---|" + "---|" * len(MODELS)]
    for p, r in ina["lopo_recall_per_participant"].items():
        L.append(f"| {p} | " + " | ".join(_f(r[m]) for m in MODELS) + " |")
    L.append("| **All** | " + " | ".join(_f(ina["lopo_recall"][m]) for m in MODELS) + " |")
    if "label_consistency" in res:
        c = res["label_consistency"]
        L += ["", "## 6. Label consistency (same windows, two exports)", ""]
        if c["n_common"]:
            L += [f"{c['n_common']} shared windows: agreement {c['agreement']:.1%}, "
                  f"Cohen's kappa {c['kappa']:.3f}.", "",
                  "Changes: " + (", ".join(f"{k}: {v}" for k, v in c["changes"].items())
                                 or "none")]
        else:
            L.append("No shared windows.")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(prog="focus-robustness", description=__doc__.split("\n")[0])
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--relabel", nargs=2, metavar=("OLD_CSV", "NEW_CSV"))
    args = p.parse_args(argv)
    try:
        res = run(load_exports(args.data), args.alpha, args.seed, relabel=args.relabel)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "robustness.md").write_text(markdown(res), encoding="utf-8")
    print(f"Robustness report: {out / 'robustness.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
