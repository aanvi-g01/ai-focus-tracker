"""ML pipeline tests. SYNTHETIC data only - never used for reported results."""
import csv
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("shap")

from focus_tracker.api.export import EXPORT_COLUMNS  # noqa: E402
from focus_tracker.baseline.ewma import EwmaBaseline  # noqa: E402
from focus_tracker.models import evaluate as E  # noqa: E402
from focus_tracker.models.data import (add_features, app_category, load_exports,  # noqa: E402
                                       parse_app_seconds)
from focus_tracker.models.hmm import SupervisedHMM, viterbi  # noqa: E402
from focus_tracker.models.rules import RuleClassifier  # noqa: E402

# Per label: (idle share, app switches/min, keys/min). Entertainment share is
# deliberately the same for all labels, so Focused vs Distracted is separable
# only by switching/typing RELATIVE to the person (what z-scores capture).
PROFILE = {"Focused": (0.15, 0.6, 40.0), "Distracted": (0.10, 2.0, 8.0),
           "Inactive": (0.95, 0.05, 0.5)}
STAY = 0.8          # label persistence between consecutive windows


def synth_rows(participant, days=4, per_day=60, scale=1.0, seed=0, start=datetime(2026, 11, 20, 9)):
    """Export-format rows. `scale` is a personal habit: switches `scale` times more
    and types `scale` times less than others, for every label."""
    rng = np.random.default_rng(seed)
    rows, label = [], "Focused"
    for d in range(days):
        t = start + timedelta(days=d)
        for _ in range(per_day):
            if rng.random() > STAY:
                label = rng.choice(list(PROFILE))
            idle, sw, keys = PROFILE[label]
            idle = float(np.clip(idle + rng.normal(0, 0.05), 0, 1))
            sw = max(0.0, (sw + rng.normal(0, 0.2)) * scale)
            keys = max(0.0, (keys + rng.normal(0, 4)) / scale)
            ent = float(np.clip(rng.normal(0.3, 0.15), 0, 1))
            browser = int(rng.uniform(30, 200))
            row = dict.fromkeys(EXPORT_COLUMNS, "")
            row.update({
                "participant": participant, "window_start": t.isoformat(timespec="seconds"),
                "window_end": (t + timedelta(seconds=300)).isoformat(timespec="seconds"),
                "window_seconds": 300, "session_id": d + 1, "session_task_category": "studying",
                "observed_seconds": 300, "coverage": 1.0,
                "idle_seconds": int(idle * 300), "active_seconds": 300 - int(idle * 300),
                "idle_episodes": int(idle > 0.5), "switch_count": round(sw * 5),
                "site_switch_count": round(sw * 2), "switches_per_min": round(round(sw * 5) / 5, 3),
                "distinct_apps": 2, "keyboard_count": round(keys * 5), "mouse_count": round(keys * 20),
                "top_app": "Code.exe", "app_seconds": f"Code.exe:{300 - browser}; chrome.exe:{browser}",
                "browser_seconds": browser, "distinct_hosts": 2, "site_switches_per_min": round(sw * 0.4, 3),
                "share_entertainment": round(ent, 4), "share_educational": round(1 - ent, 4),
                "share_social": 0.0, "hostnames": "youtube.com:100; github.com:50",
                "video_titles": "Some lecture", "label": label, "label_source": "user",
                "labelled_at": "2026-11-23T10:00:00", "updated_at": "2026-11-23T10:00:00"})
            rows.append(row)
            t += timedelta(seconds=300)
    return rows


def write_csv(path, rows, columns=EXPORT_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "exports"
    d.mkdir()
    write_csv(d / "focus_windows_P1.csv", synth_rows("P1", seed=1))
    write_csv(d / "focus_windows_P2.csv", synth_rows("P2", seed=2))
    write_csv(d / "focus_windows_P3.csv", synth_rows("P3", seed=3, scale=3.0))  # shifted habit
    return d


# -- loading & features ----------------------------------------------------------
def test_load_folder_and_merged_export(tmp_path, folder):
    df = load_exports(folder)
    assert set(df["participant"]) == {"P1", "P2", "P3"} and len(df) == 720
    assert df.groupby("participant")["window_start"].apply(lambda s: s.is_monotonic_increasing).all()
    # merged developer export: user_label used, developer label ignored
    rows = synth_rows("P9", days=1, per_day=5)
    merged = [{**{k: v for k, v in r.items() if k not in ("label", "label_source",
                                                         "labelled_at", "updated_at")},
               "user_label": r["label"], "developer_label": "Inactive", "annotator": "dev"}
              for r in rows]
    cols = [c for c in EXPORT_COLUMNS if c not in ("label", "label_source", "labelled_at",
                                                    "updated_at")] + ["user_label",
                                                                       "developer_label", "annotator"]
    m = load_exports(write_csv(tmp_path / "team.csv", merged, cols))
    assert list(m["label"]) == [r["label"] for r in rows]
    assert "developer_label" not in m.columns


def test_load_rejects_other_files(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not an app export"):
        load_exports(p)


def test_features_keep_missing_as_nan(folder):
    df = load_exports(folder).head(3).copy()
    df.loc[df.index[0], ["browser_seconds", "share_entertainment", "app_seconds"]] = np.nan
    f = add_features(df)
    row = f.iloc[0]
    assert math.isnan(row["share_entertainment"]) and math.isnan(row["browser_share"])
    assert math.isnan(row["app_dev"])
    assert f.iloc[1]["app_dev"] > 0 and f.iloc[1]["idle_share"] >= 0


def test_app_parsing_and_categories():
    assert parse_app_seconds("Code.exe:179; chrome.exe:120") == {"Code.exe": 179.0, "chrome.exe": 120.0}
    assert parse_app_seconds(None) == {}
    assert app_category("Code.exe") == "dev" and app_category("Spotify.exe") == "media"
    assert app_category("WhatsApp.Root.exe") == "comm" and app_category("weird.exe") == "other"


# -- personal baseline ---------------------------------------------------------------
def test_ewma_z_uses_only_the_past_and_matches_hand_calculation():
    b = EwmaBaseline(alpha=0.5, prior_weight=0)
    b.prior = {"x": (0.0, 4.0)}                # prior sigma 2 -> floor 0.2
    z = b.z_scores([2.0, 4.0, 4.0], "x")
    # w=0 for the first window -> prior: (2-0)/2 = 1
    # then mu=2, var=0 -> sigma floor 0.2: (4-2)/0.2 = 10
    # update: d=2, mu=3, var=0.5*(0+0.5*4)=1 -> (4-3)/1 = 1
    assert np.allclose(z, [1.0, 10.0, 1.0])


def test_ewma_future_values_never_change_past_scores():
    b = EwmaBaseline(alpha=0.1)
    b.prior = {"x": (1.0, 1.0)}
    a = b.z_scores([1, 2, 3, 4], "x")
    c = b.z_scores([1, 2, 3, 100], "x")
    assert np.allclose(a[:3], c[:3])


def test_ewma_nan_gives_nan_and_skips_update():
    b = EwmaBaseline(alpha=0.5, prior_weight=0)
    b.prior = {"x": (0.0, 1.0)}
    with_nan = b.z_scores([1.0, float("nan"), 3.0], "x")
    without = b.z_scores([1.0, 3.0], "x")
    assert math.isnan(with_nan[1]) and with_nan[2] == pytest.approx(without[1])


def test_ewma_cold_start_blends_with_prior():
    b = EwmaBaseline(alpha=0.5, prior_weight=10)
    b.prior = {"x": (0.0, 1.0)}
    z = b.z_scores([5.0, 5.0], "x")
    # second window: w = 1/11 -> mean 5/11, mostly the prior
    assert z[1] == pytest.approx((5 - 5 / 11) / math.sqrt(10 / 11))


def test_transform_is_per_participant(folder):
    df = add_features(load_exports(folder))
    b = EwmaBaseline().fit_prior(df, ["switches_per_min"])
    z = b.transform(df, ["switches_per_min"])["z_switches_per_min"]
    p3 = df["participant"] == "P3"
    # after warm-up, P3's own z-scores are centred near 0 despite 3x switching
    assert abs(z[p3].iloc[50:].mean()) < 0.6


# -- models --------------------------------------------------------------------------
def test_rules_fit_on_training_only(folder):
    df = add_features(load_exports(folder))
    tr, te = E.time_split(df)
    r = RuleClassifier().fit(df.loc[tr])
    shuffled = df.copy()
    shuffled.loc[te, "label"] = "Inactive"         # changing TEST labels changes nothing
    assert RuleClassifier().fit(shuffled.loc[tr]).thresholds == r.thresholds
    assert set(r.predict(df.loc[te])) <= {"Focused", "Distracted", "Inactive"}


def test_viterbi_recovers_known_path():
    log_pi = np.log([0.9, 0.1])
    log_A = np.log([[0.9, 0.1], [0.1, 0.9]])
    # emissions strongly favour 0,0,1,1 but a noisy middle observation
    B = np.log([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9]])
    assert viterbi(log_pi, log_A, B) == [0, 0, 1, 1]


def test_hmm_learns_sticky_transitions(folder):
    df = add_features(load_exports(folder))
    hmm = SupervisedHMM().fit(df)
    A = np.exp(hmm.log_A_)
    assert np.allclose(A.sum(axis=1), 1) and (np.diag(A) > 0.6).all()
    assert set(hmm.predict(df.head(30))) <= {"Focused", "Distracted", "Inactive"}


# -- evaluation ----------------------------------------------------------------------
def test_time_split_has_no_overlap_and_test_is_later(folder):
    df = add_features(load_exports(folder))
    tr, te = E.time_split(df)
    assert len(tr.intersection(te)) == 0
    for p, g in df.groupby("participant"):
        assert g.loc[g.index.intersection(tr), "window_start"].max() < \
            g.loc[g.index.intersection(te), "window_start"].min()


def test_single_day_split_drops_one_window_at_the_cut(tmp_path):
    df = add_features(load_exports(write_csv(tmp_path / "a.csv",
                                             synth_rows("P1", days=1, per_day=20))))
    tr, te = E.time_split(df)
    assert len(tr) == 14 and len(te) == 5


def test_mcnemar_exact():
    y = np.array(["F"] * 10)
    a = np.array(["F"] * 10)
    b = np.array(["F"] * 2 + ["D"] * 8)            # b=8 (only A right), c=0
    m = E.mcnemar(y, a, b)
    assert (m["b"], m["c"]) == (8, 0) and m["p_value"] == pytest.approx(2 / 2 ** 8)
    assert E.mcnemar(y, a, a)["p_value"] == 1.0


def test_personal_baseline_helps_a_new_user_with_a_different_habit(folder):
    # P3's focused windows look like other people's distracted ones in raw
    # numbers; relative to P3's own history they don't. Leave-one-participant-out
    # = a brand-new user the model has never seen.
    df = add_features(load_exports(folder))
    others = df["participant"] != "P3"
    focused = (df["label"] == "Focused") & (df["window_start"].dt.day > 20)  # after warm-up

    def effect_size(s):     # gap between P3 and others, in units of spread
        return abs(s[focused & ~others].mean() - s[focused & others].mean()) / s[focused].std()

    raw, rel = [], []
    for f in ("switches_per_min", "keys_per_min"):
        z = EwmaBaseline().fit_prior(df[others], [f]).transform(df, [f])[f"z_{f}"]
        raw.append(effect_size(df[f]))
        rel.append(effect_size(z))
    # In raw units P3's focus looks very different from others'; relative to each
    # person's own history the difference at least halves.
    assert np.mean(raw) > 1.5 and np.mean(rel) < 0.5 * np.mean(raw)
    res, *_ = E.run(load_exports(folder))
    assert res["lopo"]["P3"]["gbt_z"] > res["lopo"]["P3"]["gbt_raw"]


def test_low_data_guard(tmp_path):
    rows = synth_rows("P1", days=3, per_day=30, seed=5)
    for r in rows:
        if r["label"] == "Inactive":
            r["label"] = "Focused"
    rows[0]["label"] = "Inactive"                   # 1 Inactive window only
    res, *_ = E.run(load_exports(write_csv(tmp_path / "a.csv", rows)))
    assert any("Inactive" in w for w in res["warnings"])


def test_run_refuses_without_two_classes(tmp_path):
    rows = synth_rows("P1", days=2, per_day=10)
    for r in rows:
        r["label"] = "Focused"
    with pytest.raises(ValueError, match="not enough labelled data"):
        E.run(load_exports(write_csv(tmp_path / "a.csv", rows)))


def test_cli_end_to_end(folder, tmp_path):
    out = tmp_path / "results"
    assert E.main(["--data", str(folder), "--out", str(out)]) == 0
    for name in ("results.json", "summary.md", "predictions.csv", "explanations.csv",
                 "shap_importance.csv", "confusion_matrices.png", "per_participant_f1.png",
                 "shap_importance.png"):
        assert (out / name).exists(), name
    res = json.loads((out / "results.json").read_text())
    assert set(res["models"]) == {"rule", "gbt_raw", "gbt_z", "hmm"}
    assert set(res["lopo"]) == {"P1", "P2", "P3"}
    assert "rq3_low_activity" in res and "hmm_vs_rule" in res["rq2_learned_vs_rule"]
    preds = pd.read_csv(out / "predictions.csv")
    expl = pd.read_csv(out / "explanations.csv")
    assert len(preds) == len(expl) == res["data"]["test_windows"]
    assert expl[["top1_feature", "top2_feature", "top3_feature"]].notna().all().all()
    assert preds["explanation"].str.match(r"^(Focused|Distracted|Inactive): ").all()


# --------------------------------------------------------------------------
# Final model: train on everything, save, load
# --------------------------------------------------------------------------
from focus_tracker.logger.labels import LABELS  # noqa: E402
from focus_tracker.models import train as T  # noqa: E402


def test_labels_are_normalised_on_load(tmp_path):
    rows = synth_rows("P1", days=1, per_day=10)
    rows[0]["label"], rows[1]["label"] = "Distracted ", " focused"
    write_csv(tmp_path / "a.csv", rows)
    df = load_exports(tmp_path / "a.csv")
    assert len(df) == 10
    assert list(df["label"][:2]) == ["Distracted", "Focused"]


def test_old_exports_without_app_seconds_still_train(tmp_path):
    cols = [c for c in EXPORT_COLUMNS if c not in ("top_app", "app_seconds")]
    write_csv(tmp_path / "old.csv", synth_rows("P1", seed=1), columns=cols)
    write_csv(tmp_path / "new.csv", synth_rows("P2", seed=2))
    df = load_exports(tmp_path)
    feats = add_features(df)
    assert feats.loc[feats["participant"] == "P1", "app_dev"].isna().all()
    for kind in T.KINDS:
        assert set(T.train_final(df, kind).predict(df)) <= set(LABELS)


@pytest.mark.parametrize("kind", T.KINDS)
def test_save_load_round_trip(folder, tmp_path, kind):
    df = load_exports(folder)
    m = T.train_final(df, kind)
    path = T.save(m, tmp_path / "models", df)
    assert (path / "manifest.json").exists()
    assert (path / "lgbm_booster.txt").exists() == kind.startswith("gbt")
    m2 = T.load(path)
    assert (m2.predict(df) == m.predict(df)).all()
    proba = m2.predict_proba(df)
    assert list(proba.columns) == list(LABELS)
    assert np.allclose(proba.sum(axis=1), 1)


def test_gbt_z_prediction_uses_only_the_past(folder):
    df = load_exports(folder)
    m = T.train_final(df, "gbt_z")
    one = df[df["participant"] == "P1"].reset_index(drop=True)
    full = m.predict(one)
    prefix = m.predict(one.iloc[:50])
    assert (full[:50] == prefix).all()


def test_load_refuses_changed_features(folder, tmp_path):
    df = load_exports(folder)
    path = T.save(T.train_final(df, "rule"), tmp_path / "models", df)
    man = json.loads((path / "manifest.json").read_text())
    man["raw_features"] = man["raw_features"][:-1]
    (path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError):
        T.load(path)


def test_train_cli_end_to_end(folder, tmp_path):
    out, ev = tmp_path / "models", tmp_path / "eval"
    assert T.main(["--data", str(folder), "--out", str(out), "--eval-out", str(ev)]) == 0
    (saved,) = list(out.iterdir())
    man = json.loads((saved / "manifest.json").read_text())
    res = json.loads((ev / "results.json").read_text())
    assert man["kind"] == T.choose_kind(res)[0]
    assert (ev / "summary.md").exists()
    assert set(T.load(saved).predict(load_exports(folder))) <= set(LABELS)


# --------------------------------------------------------------------------
# Robustness checks
# --------------------------------------------------------------------------
from focus_tracker.models import robustness as R  # noqa: E402


def test_robustness_helpers():
    y = np.array(["Focused", "Distracted", "Inactive", "Focused"])
    pred = np.array(["Focused", "Distracted", "Focused", "Focused"])
    assert R.two_class_f1(y, pred) == 1.0               # Inactive rows ignored
    parts = np.array(["A", "A", "B", "B"])
    assert R.per_user_mean(y, pred, parts, exclude=("B",)) == 1.0


def test_label_consistency(tmp_path):
    rows = synth_rows("P1", days=1, per_day=30, seed=3)
    write_csv(tmp_path / "old.csv", rows)
    write_csv(tmp_path / "same.csv", rows)
    assert R.label_consistency(tmp_path / "old.csv", tmp_path / "same.csv")["kappa"] == 1.0
    flipped = [dict(r) for r in rows]
    for r in flipped[:5]:
        r["label"] = "Inactive" if r["label"] != "Inactive" else "Focused"
    write_csv(tmp_path / "new.csv", flipped)
    c = R.label_consistency(tmp_path / "old.csv", tmp_path / "new.csv")
    assert c["n_common"] == 30 and c["agreement"] == pytest.approx(25 / 30, abs=1e-4)


def test_robustness_end_to_end(folder, tmp_path):
    write_csv(folder / "focus_windows_P9.csv",
              [{**r, "label": "Focused"} for r in synth_rows("P9", days=2, per_day=20, seed=9)])
    out = tmp_path / "rob"
    assert R.main(["--data", str(folder), "--out", str(out)]) == 0
    res = json.loads((out / "robustness.json").read_text())
    assert res["single_class_participants"] == ["P9"]
    assert len(res["seed_stability"]["gbt_z"]["values"]) == len(R.SEEDS)
    # the full-feature numbers match the main evaluation protocol
    main_res, *_ = E.run(load_exports(folder))
    for m in ("gbt_raw", "gbt_z"):
        assert res["time_ablation"][m]["with_time"]["macro_f1"] == \
            pytest.approx(main_res["models"][m]["macro_f1"])
    assert (out / "robustness.md").read_text(encoding="utf-8").startswith("# Robustness")


def test_ablation_drops_time_features(folder):
    df = add_features(load_exports(folder))
    tr, te = E.time_split(df)
    import focus_tracker.models.robustness as mod
    seen = []
    orig = mod.GbtModel

    class Spy(orig):
        def __init__(self, features, seed=0):
            seen.append(list(features))
            super().__init__(features, seed)
    mod.GbtModel = Spy
    try:
        mod.gbt_no_time(df, tr, te, 0.05, 0)
    finally:
        mod.GbtModel = orig
    assert seen and all(not any(f.startswith("hour_") for f in fs) for fs in seen)


def test_latest_model_dir(folder, tmp_path):
    df = load_exports(folder)
    assert T.latest_model_dir(tmp_path / "none") is None
    a = T.save(T.train_final(df, "rule"), tmp_path / "m", df)
    man = json.loads((a / "manifest.json").read_text())
    man["trained_at"] = "2000-01-01T00:00:00"
    (a / "manifest.json").write_text(json.dumps(man))
    b = T.save(T.train_final(df, "rule"), tmp_path / "m2", df)
    assert T.latest_model_dir(tmp_path / "m", tmp_path / "m2") == b
    assert T.latest_model_dir(a) == a                       # a model folder itself


def test_builtin_treeshap_matches_shap_package(folder):
    shap = pytest.importorskip("shap")
    from focus_tracker.explain.shap_explain import _class_values, contributions
    df = load_exports(folder)
    m = T.train_final(df, "gbt_raw")
    X = m._prepare(df)[m.estimator.features].astype(float)
    ours = contributions(m.estimator, X)
    ref = _class_values(shap.TreeExplainer(m.estimator.model).shap_values(X), len(X),
                        len(m.estimator.classes_))
    assert np.allclose(ours, ref, atol=1e-6)
    expl, _ = m.explain(df.head(5))
    assert all(len(r) == 3 and all(isinstance(x, str) for x in r) for r in expl["reasons"])


def test_accepted_labels_are_not_training_data(tmp_path):
    rows = synth_rows("P1", days=1, per_day=10)
    for r in rows[:4]:
        r["label_source"] = "accepted"
    write_csv(tmp_path / "a.csv", rows)
    with pytest.warns(UserWarning, match="accepting"):
        df = load_exports(tmp_path / "a.csv")
    assert len(df) == 6


def test_gbt_trains_on_two_labels(folder):
    df = load_exports(folder)
    two = df[df["label"] != "Inactive"]
    m = T.train_final(two, "gbt_z")
    proba = m.predict_proba(two)
    assert proba["Inactive"].eq(0).all() and set(m.predict(two)) <= {"Focused", "Distracted"}
    expl, _ = m.explain(two.head(3))
    assert all(len(r) == 3 for r in expl["reasons"])
