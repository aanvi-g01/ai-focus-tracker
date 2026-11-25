"""Make a shareable copy of an evaluation run with pseudonyms instead of names.

    python scripts/anonymise_results.py results/run2 docs/results/run2

* Participant names -> P01, P02, ... (sorted, case-insensitive); the person
  outside the study in personal_check.csv ("you") -> X1.
* Exact window times are dropped (predictions get a per-person day number);
  first/last timestamps are removed from the data card.
* Markdown reports and plots are regenerated from the renamed data, so no name
  survives in text or images. The name -> pseudonym mapping is NOT written.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from focus_tracker.models import evaluate as E
from focus_tracker.models import robustness as R

OUTSIDE = {"you": "X1"}


def rename(obj, mapping):
    if isinstance(obj, dict):
        out = {mapping.get(k, k): rename(v, mapping) for k, v in obj.items()}
        pseud = set(mapping.values())
        if out and all(k in pseud for k in out):          # participant tables: P01, P02, ...
            out = dict(sorted(out.items()))
        return out
    if isinstance(obj, list):
        return [rename(v, mapping) for v in obj]
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    return obj


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    res = json.loads((src / "results.json").read_text(encoding="utf-8"))
    names = sorted(res["data"]["participants"], key=str.lower)
    mapping = {n: f"P{i + 1:02d}" for i, n in enumerate(names)}
    mapping.update(OUTSIDE)

    # main evaluation
    for card in res["data"]["participants"].values():
        card.pop("first", None)
        card.pop("last", None)
    res = rename(res, mapping)
    (dst / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (dst / "summary.md").write_text(E.summary_markdown(res), encoding="utf-8")
    importance = pd.read_csv(src / "shap_importance.csv", index_col=0).iloc[:, 0]
    E.plots(res, importance, dst)
    shutil.copy(src / "shap_importance.csv", dst / "shap_importance.csv")

    pred = pd.read_csv(src / "predictions.csv")
    pred["window_start"] = pd.to_datetime(pred["window_start"])
    pred["day"] = pred.groupby("participant")["window_start"].transform(
        lambda s: s.dt.date.rank(method="dense").astype(int))
    pred["participant"] = pred["participant"].map(mapping)
    pred = pred.drop(columns=["window_start"])
    pred[["participant", "day"] + [c for c in pred.columns if c not in ("participant", "day")]] \
        .to_csv(dst / "predictions.csv", index=False)
    shutil.copy(src / "explanations.csv", dst / "explanations.csv")   # no names or times

    # robustness checks
    if (src / "robustness.json").exists():
        rob = rename(json.loads((src / "robustness.json").read_text(encoding="utf-8")), mapping)
        (dst / "robustness.json").write_text(json.dumps(rob, indent=2), encoding="utf-8")
        (dst / "robustness.md").write_text(R.markdown(rob), encoding="utf-8")

    # personal-model check
    if (src / "personal_check.csv").exists():
        pc = pd.read_csv(src / "personal_check.csv")
        pc["person"] = pc["person"].map(lambda p: mapping.get(p, p))
        pc.to_csv(dst / "personal_check.csv", index=False)

    # nothing may still contain a name
    for f in dst.iterdir():
        if f.suffix in (".md", ".json", ".csv"):
            text = f.read_text(encoding="utf-8").lower()
            leaked = [n for n in names if (len(n) > 2 and n.lower() in text)
                      or f'"{n.lower()}"' in text or f"| {n.lower()} |" in text]
            if leaked:
                raise SystemExit(f"{f.name} still mentions {len(leaked)} participant name(s)")
    print(f"anonymised {len(names)} participants -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
