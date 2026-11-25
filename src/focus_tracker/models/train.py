"""Train the final focus-state model on ALL labelled windows and save it.

    focus-train --data data/participants --out models --eval-out results/run1

1. Runs the full evaluation (same as focus-evaluate) so the saved model comes
   with honest numbers: time split per participant + leave-one-participant-out.
2. Picks the model kind (default: best mean LOPO macro-F1, i.e. the best
   estimate for a NEW user; ties go to the simpler model).
3. Refits that kind on every labelled window and writes
   models/<timestamp>/{model.joblib, manifest.json[, lgbm_booster.txt]}.
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..baseline.ewma import EwmaBaseline
from ..logger.labels import LABELS
from .data import RAW_FEATURES, Z_FEATURES, add_features, load_exports
from .gbt import GbtModel
from .hmm import SupervisedHMM
from .rules import RuleClassifier

KINDS = ("rule", "gbt_raw", "gbt_z", "hmm")          # simplest first (tie-break order)
Z_COLUMNS = tuple(f"z_{f}" for f in Z_FEATURES)
FORMAT_VERSION = 1


class FocusModel:
    """A trained model plus everything needed to turn export rows into predictions."""

    def __init__(self, kind, estimator, baseline=None, alpha=0.05, seed=0):
        self.kind = kind
        self.estimator = estimator
        self.baseline = baseline
        self.alpha = alpha
        self.seed = seed
        self.raw_features = list(RAW_FEATURES)

    def _prepare(self, df):
        """Export rows -> feature rows. For gbt_z the personal z-scores are computed
        over each participant's chronological history in `df` (past windows only)."""
        if "label" not in df.columns:
            df = df.assign(label=None)
        if not pd.api.types.is_datetime64_any_dtype(df["window_start"]):
            df = df.assign(window_start=pd.to_datetime(df["window_start"]))
        df = add_features(df)
        if self.kind == "gbt_z":
            df = pd.concat([df, self.baseline.transform(df, Z_FEATURES)], axis=1)
        return df

    def predict(self, df) -> np.ndarray:
        return np.asarray(self.estimator.predict(self._prepare(df)))

    def predict_proba(self, df) -> pd.DataFrame:
        """Class probabilities (columns = LABELS). Rule/HMM give one-hot rows."""
        d = self._prepare(df)
        if self.kind.startswith("gbt"):
            p = pd.DataFrame(self.estimator.predict_proba(d), index=d.index,
                             columns=self.estimator.classes_)
        else:
            pred = self.estimator.predict(d)
            p = pd.DataFrame({c: (pred == c).astype(float) for c in LABELS}, index=d.index)
        return p.reindex(columns=list(LABELS), fill_value=0.0)

    def explain(self, df, top=3):
        if not self.kind.startswith("gbt"):
            raise ValueError(f"explanations need a LightGBM model, not {self.kind}")
        from ..explain.shap_explain import explain
        return explain(self.estimator, self._prepare(df), top=top)


def train_final(df, kind="gbt_z", alpha=0.05, seed=0) -> FocusModel:
    """Fit `kind` on every labelled window (no split): same constructors as evaluation."""
    if kind not in KINDS:
        raise ValueError(f"model must be one of {', '.join(KINDS)}")
    df = add_features(df)
    if len(set(df["label"])) < 2:
        raise ValueError("need at least 2 different labels to train")
    baseline = None
    if kind == "gbt_z":
        baseline = EwmaBaseline(alpha=alpha).fit_prior(df, Z_FEATURES)
        df = pd.concat([df, baseline.transform(df, Z_FEATURES)], axis=1)
    if kind == "rule":
        est = RuleClassifier().fit(df)
    elif kind == "gbt_raw":
        est = GbtModel(RAW_FEATURES, seed).fit(df)
    elif kind == "gbt_z":
        est = GbtModel(tuple(RAW_FEATURES) + Z_COLUMNS, seed).fit(df)
    else:
        est = SupervisedHMM().fit(df)
    return FocusModel(kind, est, baseline, alpha, seed)


def lopo_means(res):
    """Mean leave-one-participant-out macro-F1 per model kind (skipped users ignored)."""
    rows = [v for v in (res or {}).get("lopo", {}).values() if isinstance(v, dict)
            and "skipped" not in v]
    out = {}
    for k in KINDS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    return out


def choose_kind(res):
    """Best mean LOPO macro-F1; falls back to the time-split test macro-F1."""
    means = lopo_means(res)
    if not any(v is not None for v in means.values()):
        means = {k: res["models"][k]["macro_f1"] for k in KINDS}
    best = max(KINDS, key=lambda k: (means[k] if means[k] is not None else -1,
                                     -KINDS.index(k)))
    return best, means


def _versions():
    out = {}
    for mod in ("lightgbm", "sklearn", "pandas", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                       # pragma: no cover
            out[mod] = None
    return out


def save(model: FocusModel, out_dir, df, res=None, lopo=None) -> Path:
    import joblib
    out = Path(out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, out / "model.joblib")
    est = model.estimator
    params = getattr(est, "params", None) or getattr(est, "thresholds", None)
    manifest = {
        "format_version": FORMAT_VERSION, "kind": model.kind,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "alpha": model.alpha, "seed": model.seed, "params": params,
        "labels": list(LABELS), "raw_features": model.raw_features,
        "model_features": list(getattr(est, "features", [])),
        "data": {"windows": int(len(df)),
                 "first": str(df["window_start"].min()), "last": str(df["window_start"].max()),
                 "participants": {p: {c: int((g["label"] == c).sum()) for c in LABELS}
                                  for p, g in df.groupby("participant")}},
        "evaluation": None if res is None else {
            "lopo_mean_macro_f1": lopo,
            "time_split_macro_f1": {k: res["models"][k]["macro_f1"] for k in res["models"]},
            "lopo": res.get("lopo"), "warnings": res.get("warnings")},
        "versions": _versions()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                       encoding="utf-8")
    if model.kind.startswith("gbt"):
        est.model.booster_.save_model(str(out / "lgbm_booster.txt"))
    return out


def latest_model_dir(*roots):
    """Newest saved-model folder (one with a manifest.json) under any of `roots`;
    a root may itself be a model folder. None when there is none."""
    found = []
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        candidates = [root] + (sorted(root.iterdir()) if root.is_dir() else [])
        for d in candidates:
            m = d / "manifest.json"
            if m.is_file():
                try:
                    trained = json.loads(m.read_text(encoding="utf-8")).get("trained_at", "")
                except (OSError, ValueError):
                    continue
                found.append((trained, d))
    return max(found, key=lambda t: t[0])[1] if found else None


def read_manifest(path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))


def load(path) -> FocusModel:
    """Load a saved model folder (or its model.joblib)."""
    import joblib
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("raw_features") != list(RAW_FEATURES):
        raise ValueError("saved model was trained with a different feature set; retrain it")
    return joblib.load(folder / "model.joblib")


def main(argv=None):
    p = argparse.ArgumentParser(prog="focus-train", description=__doc__.split("\n")[0])
    p.add_argument("--data", nargs="+", required=True,
                   help="export CSV files and/or folders containing them")
    p.add_argument("--out", default="models", help="folder for saved models")
    p.add_argument("--eval-out", default="results", help="folder for the evaluation report")
    p.add_argument("--model", default="auto", choices=("auto",) + KINDS)
    p.add_argument("--alpha", type=float, default=0.05, help="EWMA baseline speed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-eval", action="store_true",
                   help="train without evaluating (requires --model)")
    args = p.parse_args(argv)
    if args.skip_eval and args.model == "auto":
        p.error("--skip-eval needs an explicit --model")
    try:
        res = None
        if not args.skip_eval:
            from .evaluate import evaluate
            res = evaluate(args.data, args.eval_out, args.alpha, args.seed)
        kind, means = (args.model, lopo_means(res)) if args.model != "auto" else choose_kind(res)
        df = load_exports(args.data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = train_final(df, kind, args.alpha, args.seed)
        folder = save(model, args.out, df, res, means)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if res:
        for w in res["warnings"]:
            print("WARNING:", w)
        print("Mean leave-one-participant-out macro-F1:")
        for k, v in means.items():
            print(f"  {k:<8} {'n/a' if v is None else f'{v:.3f}'}"
                  f"   (time-split test {res['models'][k]['macro_f1']:.3f})")
        print(f"Evaluation report: {Path(args.eval_out) / 'summary.md'}")
    print(f"Trained '{kind}' on {len(df)} windows from {df['participant'].nunique()} "
          f"participants -> {folder}")
    return 0


if __name__ == "__main__":
    # Run through the importable module so pickled classes are not saved as __main__.
    from focus_tracker.models.train import main as _main
    sys.exit(_main())
