"""Train and evaluate all models on participants' exported windows.

    focus-evaluate --data exports/ --out results/

Experiments (slides 3, 9, 10):
* Main split: per participant, earliest days train / latest days test;
  models are trained on all participants' training windows (pooled) and
  scored on each participant's later windows.
* RQ2  learned vs rule:   dF1 = F1(model) - F1(rule), exact McNemar test.
* RQ1  personal baseline: LightGBM raw features vs raw + EWMA z-scores
       (+ sensitivity to alpha), same split and seed.
* RQ3  low activity:      recall of Focused on windows with idle share >= 0.6.
* Generalisation:         leave-one-participant-out macro-F1.
Macro-F1 is primary (classes are imbalanced). Nothing is tuned on test windows.
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ..baseline.ewma import EwmaBaseline
from ..logger.labels import LABELS
from .data import RAW_FEATURES, Z_FEATURES, add_features, load_exports
from .gbt import GbtModel
from .hmm import SupervisedHMM
from .rules import RuleClassifier

MIN_PER_CLASS = 5
LOW_ACTIVITY_IDLE_SHARE = 0.6
TEST_FRACTION = 0.3


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def time_split(df, test_frac=TEST_FRACTION):
    """(train_index, test_index): per participant, the latest days (or, with
    one day only, the latest windows after a one-window gap) are test."""
    train, test = [], []
    for _, g in df.sort_values("window_start").groupby("participant"):
        days = sorted(g["window_start"].dt.date.unique())
        if len(days) >= 2:
            n_test = max(1, int(round(len(days) * test_frac)))
            is_test = g["window_start"].dt.date.isin(days[-n_test:])
            train += list(g.index[~is_test])
            test += list(g.index[is_test])
        else:
            cut = int(math.floor(len(g) * (1 - test_frac)))
            train += list(g.index[:cut])
            test += list(g.index[cut + 1:])       # embargo: drop one window at the cut
    return pd.Index(train), pd.Index(test)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def present_labels(*ys):
    seen = set().union(*[set(y) for y in ys])
    return [c for c in LABELS if c in seen]


def scores(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = [c for c in LABELS if c in set(y_true)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)) if len(y_true) else None,
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else None,
        "classes_in_test": labels,
        "per_class": {c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                          "f1": round(float(f[i]), 4), "support": int(s[i])}
                      for i, c in enumerate(LABELS)},
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def bootstrap_ci(y_true, y_pred, n=1000, seed=0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) < 2:
        return None
    rng = np.random.default_rng(seed)
    labels = [c for c in LABELS if c in set(y_true)]
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[i], y_pred[i], labels=labels, average="macro",
                             zero_division=0))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test: b = only A correct, c = only B correct."""
    y, a, b_ = map(np.asarray, (y_true, pred_a, pred_b))
    b = int(((a == y) & (b_ != y)).sum())
    c = int(((a != y) & (b_ == y)).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_value": min(1.0, p)}


def class_counts(df):
    return {c: int((df["label"] == c).sum()) for c in LABELS}


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def _with_z(df, train_idx, alpha):
    base = EwmaBaseline(alpha=alpha).fit_prior(df.loc[train_idx], Z_FEATURES)
    return pd.concat([df, base.transform(df, Z_FEATURES)], axis=1)


Z_COLUMNS = tuple(f"z_{f}" for f in Z_FEATURES)


def fit_predict_all(df, train_idx, test_idx, alpha, seed):
    """Predictions of every model on test_idx, plus fitted models."""
    dz = _with_z(df, train_idx, alpha)
    tr, te = dz.loc[train_idx], dz.loc[test_idx]
    rule = RuleClassifier().fit(tr)
    gbt_raw = GbtModel(RAW_FEATURES, seed).fit(tr)
    gbt_z = GbtModel(RAW_FEATURES + Z_COLUMNS, seed).fit(tr)
    hmm = SupervisedHMM().fit(tr)
    preds = {"rule": rule.predict(te), "gbt_raw": gbt_raw.predict(te),
             "gbt_z": gbt_z.predict(te), "hmm": hmm.predict(te)}
    return preds, {"rule": rule, "gbt_raw": gbt_raw, "gbt_z": gbt_z, "hmm": hmm}, dz


MODEL_NAMES = {"rule": "Rule baseline", "gbt_raw": "LightGBM (raw features)",
               "gbt_z": "LightGBM (+ personal z-scores)", "hmm": "HMM (Viterbi)"}


def run(df, alpha=0.05, alphas=(0.02, 0.05, 0.1), seed=0, test_frac=TEST_FRACTION):
    df = add_features(df)
    train_idx, test_idx = time_split(df, test_frac)
    warnings_ = []
    train_counts = class_counts(df.loc[train_idx])
    for c, n in train_counts.items():
        if n < MIN_PER_CLASS:
            warnings_.append(f"only {n} training windows labelled {c} (need >= "
                             f"{MIN_PER_CLASS}); results for {c} are not reliable")
    if len(test_idx) == 0 or len(set(df.loc[train_idx, "label"])) < 2:
        raise ValueError("not enough labelled data to train and test "
                         f"(train classes {train_counts}, test windows {len(test_idx)})")

    preds, models, dz = fit_predict_all(df, train_idx, test_idx, alpha, seed)
    y = df.loc[test_idx, "label"].to_numpy()
    te = dz.loc[test_idx]

    res = {"config": {"alpha": alpha, "seed": seed, "test_fraction": test_frac,
                      "low_activity_idle_share": LOW_ACTIVITY_IDLE_SHARE,
                      "rule_thresholds": models["rule"].thresholds,
                      "gbt_params": models["gbt_z"].params},
           "warnings": warnings_,
           "data": data_card(df, train_idx, test_idx),
           "models": {}}
    for m, p in preds.items():
        res["models"][m] = {**scores(y, p), "macro_f1_ci95": bootstrap_ci(y, p, seed=seed),
                            "per_participant": {
                                part: scores(y[mask], p[mask])["macro_f1"]
                                for part in sorted(te["participant"].unique())
                                for mask in [(te["participant"] == part).to_numpy()]}}

    def delta(a, b):
        return {"delta_f1": round(res["models"][a]["macro_f1"] - res["models"][b]["macro_f1"], 4),
                "mcnemar": mcnemar(y, preds[a], preds[b])}
    res["rq2_learned_vs_rule"] = {"gbt_z_vs_rule": delta("gbt_z", "rule"),
                                  "hmm_vs_rule": delta("hmm", "rule")}

    sens = {}
    for a in alphas:
        if a == alpha:
            sens[str(a)] = res["models"]["gbt_z"]["macro_f1"]
            continue
        dza = _with_z(df, train_idx, a)
        m = GbtModel(RAW_FEATURES + Z_COLUMNS, seed).fit(dza.loc[train_idx])
        sens[str(a)] = scores(y, m.predict(dza.loc[test_idx]))["macro_f1"]
    res["rq1_personal_baseline"] = {"z_vs_raw": delta("gbt_z", "gbt_raw"),
                                    "alpha_sensitivity_macro_f1": sens}

    low = (te["idle_share"] >= LOW_ACTIVITY_IDLE_SHARE).to_numpy() & (y == "Focused")
    res["rq3_low_activity"] = {
        "n_windows": int(low.sum()),
        "recall_focused": {m: (round(float((p[low] == "Focused").mean()), 4) if low.any() else None)
                           for m, p in preds.items()}}

    res["lopo"] = lopo(df, alpha, seed)

    from ..explain.shap_explain import explain
    expl, importance = explain(models["gbt_z"], te)
    pred_df = pd.DataFrame({
        "participant": te["participant"], "window_start": te["window_start"],
        "label": y, **{f"pred_{m}": p for m, p in preds.items()},
        "explanation": expl["explanation"].reindex(te.index)}, index=te.index)
    return res, pred_df, expl, importance


def lopo(df, alpha, seed):
    """Leave-one-participant-out macro-F1 per held-out participant."""
    parts = sorted(df["participant"].unique())
    if len(parts) < 2:
        return {"skipped": "needs at least 2 participants"}
    out = {}
    for p in parts:
        tr = df.index[df["participant"] != p]
        te = df.index[df["participant"] == p]
        if len(set(df.loc[tr, "label"])) < 2:
            out[p] = {"skipped": "training data has fewer than 2 classes"}
            continue
        preds, _, _ = fit_predict_all(df, tr, te, alpha, seed)
        y = df.loc[te, "label"].to_numpy()
        out[p] = {m: scores(y, pr)["macro_f1"] for m, pr in preds.items()}
    return out


def data_card(df, train_idx, test_idx):
    card = {"participants": {}, "total_windows": int(len(df)),
            "train_windows": int(len(train_idx)), "test_windows": int(len(test_idx))}
    for p, g in df.groupby("participant"):
        card["participants"][p] = {
            "windows": int(len(g)), "days": int(g["window_start"].dt.date.nunique()),
            "first": str(g["window_start"].min()), "last": str(g["window_start"].max()),
            "labels": class_counts(g),
            "train": int(g.index.isin(train_idx).sum()), "test": int(g.index.isin(test_idx).sum())}
    return card


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def summary_markdown(res):
    L = ["# Focus-state classification results", ""]
    if res["warnings"]:
        L += ["**Warnings:**", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    d = res["data"]
    L += ["## Data", "", f"{d['total_windows']} labelled windows "
          f"({d['train_windows']} train / {d['test_windows']} test).", "",
          "| Participant | Days | Windows | Focused | Distracted | Inactive | Train | Test |",
          "|---|---|---|---|---|---|---|---|"]
    for p, c in d["participants"].items():
        lab = c["labels"]
        L.append(f"| {p} | {c['days']} | {c['windows']} | {lab['Focused']} | "
                 f"{lab['Distracted']} | {lab['Inactive']} | {c['train']} | {c['test']} |")
    L += ["", "## Models (test windows)", "",
          "| Model | Macro-F1 | 95% CI | Accuracy | F1 Focused | F1 Distracted | F1 Inactive |",
          "|---|---|---|---|---|---|---|"]
    for m, r in res["models"].items():
        ci = r["macro_f1_ci95"]
        pc = r["per_class"]
        L.append(f"| {MODEL_NAMES[m]} | {_f(r['macro_f1'])} | "
                 f"{'n/a' if not ci else f'{ci[0]:.3f}-{ci[1]:.3f}'} | {_f(r['accuracy'])} | "
                 f"{pc['Focused']['f1']:.3f} | {pc['Distracted']['f1']:.3f} | "
                 f"{pc['Inactive']['f1']:.3f} |")
    L += ["", "## RQ2: learned models vs rule baseline", "",
          "| Comparison | ΔF1 | McNemar b / c | p-value |", "|---|---|---|---|"]
    for k, v in res["rq2_learned_vs_rule"].items():
        mc = v["mcnemar"]
        L.append(f"| {k} | {v['delta_f1']:+.3f} | {mc['b']} / {mc['c']} | {mc['p_value']:.4f} |")
    r1 = res["rq1_personal_baseline"]
    mc = r1["z_vs_raw"]["mcnemar"]
    L += ["", "## RQ1: personal baseline (z-scores) vs raw features", "",
          f"ΔF1 (LightGBM + z minus raw) = {r1['z_vs_raw']['delta_f1']:+.3f}, "
          f"McNemar b/c = {mc['b']}/{mc['c']}, p = {mc['p_value']:.4f}.", "",
          "Alpha sensitivity (macro-F1): " + ", ".join(
              f"α={a}: {_f(v)}" for a, v in r1["alpha_sensitivity_macro_f1"].items()), "",
          "Note: exports contain labelled windows only, so the baseline is built from "
          "labelled history.", ""]
    r3 = res["rq3_low_activity"]
    L += ["## RQ3: low activity is not distraction", "",
          f"Focused test windows with idle share ≥ {res['config']['low_activity_idle_share']}: "
          f"n = {r3['n_windows']}.", "", "| Model | Recall (Focused) |", "|---|---|"]
    for m, v in r3["recall_focused"].items():
        L.append(f"| {MODEL_NAMES[m]} | {_f(v)} |")
    L += ["", "## Per participant (macro-F1, test windows)", "",
          "| Participant | " + " | ".join(MODEL_NAMES[m] for m in res["models"]) + " |",
          "|---|" + "---|" * len(res["models"])]
    for p in res["data"]["participants"]:
        L.append(f"| {p} | " + " | ".join(_f(res["models"][m]["per_participant"].get(p))
                                         for m in res["models"]) + " |")
    if "skipped" not in res["lopo"]:
        L += ["", "## Leave-one-participant-out (new user, macro-F1)", "",
              "| Held-out | " + " | ".join(MODEL_NAMES[m] for m in res["models"]) + " |",
              "|---|" + "---|" * len(res["models"])]
        for p, v in res["lopo"].items():
            if "skipped" in v:
                L.append(f"| {p} | {v['skipped']} |")
            else:
                L.append(f"| {p} | " + " | ".join(_f(v[m]) for m in res["models"]) + " |")
    L += ["", f"Rule thresholds (fit on training windows): {res['config']['rule_thresholds']}",
          f"LightGBM parameters: {res['config']['gbt_params']}", ""]
    return "\n".join(L)


def plots(res, importance, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(res["models"]), figsize=(4.2 * len(res["models"]), 4))
    for ax, (m, r) in zip(np.atleast_1d(axes), res["models"].items()):
        cm = np.array(r["confusion"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), LABELS, rotation=30)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{MODEL_NAMES[m]}\nmacro-F1 {_f(r['macro_f1'])}", fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    parts = list(res["data"]["participants"])
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(parts)), 3.6))
    width = 0.8 / len(res["models"])
    for k, (m, r) in enumerate(res["models"].items()):
        vals = [r["per_participant"].get(p) or 0 for p in parts]
        ax.bar(np.arange(len(parts)) + k * width, vals, width, label=MODEL_NAMES[m])
    ax.set_xticks(np.arange(len(parts)) + width * (len(res["models"]) - 1) / 2, parts)
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "per_participant_f1.png", dpi=150)
    plt.close(fig)

    from ..explain.shap_explain import nice
    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([nice(f) for f in top.index], top.values, color="#3730a3")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Most influential features (LightGBM + z-scores)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "shap_importance.png", dpi=150)
    plt.close(fig)


def evaluate(data_paths, out_dir, alpha=0.05, seed=0, test_frac=TEST_FRACTION):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_exports(data_paths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        res, pred_df, expl, importance = run(df, alpha=alpha, seed=seed, test_frac=test_frac)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(res), encoding="utf-8")
    pred_df.to_csv(out / "predictions.csv", index=False)
    expl.to_csv(out / "explanations.csv", index=False)
    importance.rename("mean_abs_shap").to_csv(out / "shap_importance.csv")
    plots(res, importance, out)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(prog="focus-evaluate", description=__doc__.split("\n")[0])
    p.add_argument("--data", nargs="+", required=True,
                   help="export CSV files and/or folders containing them")
    p.add_argument("--out", required=True, help="folder for results (created if missing)")
    p.add_argument("--alpha", type=float, default=0.05, help="EWMA baseline speed")
    p.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    try:
        res = evaluate(args.data, args.out, args.alpha, args.seed, args.test_fraction)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    for w in res["warnings"]:
        print("WARNING:", w)
    for m, r in res["models"].items():
        print(f"{MODEL_NAMES[m]:<34} macro-F1 {_f(r['macro_f1'])}")
    print(f"Full report: {Path(args.out) / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
