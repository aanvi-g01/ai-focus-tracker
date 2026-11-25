"""Report figures from the ANONYMISED run (docs/results/run2) -> docs/results/run2/figures.

    python scripts/make_report_figures.py docs/results/run2

Static PNGs for a printed report: light surface, the validated reference
categorical palette in fixed slot order, thin bars with a surface gap, one
y-axis per chart, legends for >= 2 series, recessive grid, ink-coloured text.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# reference palette (light), fixed order; never cycled
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = {"rule": "Rule baseline", "gbt_raw": "LightGBM (raw)",
          "gbt_z": "LightGBM + personal z-scores", "hmm": "HMM (Viterbi)"}
SHORT = ["Rule\nbaseline", "LightGBM\n(raw)", "LightGBM +\nz-scores", "HMM\n(Viterbi)"]
LABELS = ("Focused", "Distracted", "Inactive")
DPI = 200

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, groups, series, values, colors, ylabel, ylim=(0, 1), fmt="{:.2f}", label_all=True):
    """Grouped bars: values[s][g]. 2px surface gap between neighbouring bars."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (s, c) in enumerate(zip(series, colors)):
        v = np.asarray(values[i], dtype=float)
        b = ax.bar(x + (i - (n - 1) / 2) * width, np.nan_to_num(v), width, color=c, label=s,
                   edgecolor=SURFACE, linewidth=1.5)
        if label_all:
            for rect, val in zip(b, v):
                if not np.isnan(val):
                    ax.annotate(fmt.format(val), (rect.get_x() + rect.get_width() / 2, val),
                                xytext=(0, 2), textcoords="offset points", ha="center",
                                va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    style(ax)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)
    print("  ", name)


# --------------------------------------------------------------------------
def fig_label_distribution(res, out):
    parts = res["data"]["participants"]
    names = list(parts)
    fig, ax = plt.subplots(figsize=(8, 4))
    left = np.zeros(len(names))
    for lab, c in zip(LABELS, SLOT):
        vals = np.array([parts[p]["labels"][lab] for p in names], dtype=float)
        ax.barh(names, vals, left=left, color=c, label=lab, edgecolor=SURFACE, linewidth=1.5)
        left += vals
    for y, total in enumerate(left):
        ax.annotate(f"{int(total)}", (total, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Self-labelled 5-minute windows")
    ax.set_title("Labels per participant", pad=24)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_xlim(0, max(left) * 1.08)
    style(ax, "x")
    save(fig, out, "fig_label_distribution.png")


def fig_main_models(res, out):
    m = res["models"]
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = ["Macro-F1"] + [f"F1 {c}" for c in LABELS]
    vals = [[m[k]["macro_f1"]] + [m[k]["per_class"][c]["f1"] for c in LABELS] for k in MODELS]
    bars(ax, groups, list(MODELS.values()), vals, SLOT, "Score on later-day test windows")
    ax.set_title("Model comparison (per-participant time split)")
    ax.legend(ncol=2, loc="upper right", fontsize=8)
    save(fig, out, "fig_model_comparison.png")


def fig_lopo(res, out):
    lopo = {p: v for p, v in res["lopo"].items() if "skipped" not in v}
    parts = list(lopo)
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = [[lopo[p][k] for p in parts] for k in MODELS]
    bars(ax, parts, list(MODELS.values()), vals, SLOT, "Macro-F1 (held-out participant)",
         label_all=False)
    for i, k in enumerate(MODELS):
        mean = np.mean(vals[i])
        ax.axhline(mean, color=SLOT[i], linewidth=1, linestyle=(0, (3, 3)))
    ax.set_title("Leave-one-participant-out: a brand-new user (dashed = model mean)", pad=24)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    save(fig, out, "fig_lopo.png")


def fig_rq3(res, out):
    r = res["rq3_low_activity"]["recall_focused"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars(ax, SHORT, ["Recall"], [[r[k] for k in MODELS]], [SLOT[0]], "Recall of Focused")
    ax.text(0, 1.01, f"Focused test windows with idle share ≥ 60% "
            f"(n = {res['rq3_low_activity']['n_windows']}) labelled Focused by each model",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.set_title("Low activity is not distraction", pad=18)
    save(fig, out, "fig_rq3_low_activity.png")


def fig_time_ablation(rob, out):
    t = rob["time_ablation"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    groups = ["LightGBM raw\n(later days)", "LightGBM + z\n(later days)",
              "LightGBM raw\n(new user)", "LightGBM + z\n(new user)"]
    w = [t["gbt_raw"]["with_time"]["macro_f1"], t["gbt_z"]["with_time"]["macro_f1"],
         t["gbt_raw"]["with_time"]["lopo_mean"], t["gbt_z"]["with_time"]["lopo_mean"]]
    wo = [t["gbt_raw"]["without_time"]["macro_f1"], t["gbt_z"]["without_time"]["macro_f1"],
          t["gbt_raw"]["without_time"]["lopo_mean"], t["gbt_z"]["without_time"]["lopo_mean"]]
    bars(ax, groups, ["With time of day", "Without time of day"], [w, wo], SLOT[:2], "Macro-F1")
    ax.set_title("Time-of-day ablation")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out, "fig_time_ablation.png")


def fig_seed_stability(rob, out):
    s = rob["seed_stability"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, k in enumerate(("gbt_raw", "gbt_z")):
        v = s[k]["values"]
        jitter = np.linspace(-0.12, 0.12, len(v))            # keep close dots visible
        ax.scatter(v, i + jitter, s=55, color=SLOT[i], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=MODELS[k])
        ax.plot([s[k]["mean"]] * 2, [i - 0.25, i + 0.25], color=INK, linewidth=2)
        ax.annotate(f"mean {s[k]['mean']:.3f} ± {s[k]['sd']:.3f}", (s[k]["mean"], i + 0.28),
                    xytext=(4, 0), textcoords="offset points", ha="left", fontsize=8,
                    color=INK2)
    ax.set_yticks([0, 1], ["LightGBM (raw)", "LightGBM + z-scores"])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(0.3, 0.6)
    ax.set_xlabel("Macro-F1 on later-day test windows (one dot per random seed)")
    d = s["delta_z_minus_raw"]
    ax.set_title(f"Seed stability: z-scores help in {d['positive_in']} seeds "
                 f"(mean ΔF1 {d['mean']:+.3f})")
    style(ax, "x")
    save(fig, out, "fig_seed_stability.png")


def fig_two_class(rob, out):
    f = rob["focused_vs_distracted"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars(ax, SHORT,
         ["Later days (same people)", "New user (LOPO)"],
         [[f[k]["time_split_macro_f1"] for k in MODELS], [f[k]["lopo_macro_f1"] for k in MODELS]],
         SLOT[:2], "Macro-F1 (Focused vs Distracted)")
    ax.set_title("Focused vs Distracted only (two classes)")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out, "fig_focused_vs_distracted.png")


def fig_personal(pc, out):
    z = pc[pc["kind"] == "gbt_z"].pivot(index="person", columns="setup", values="accuracy")
    z["personal"] = z["personal"].fillna(z["shared_new"])          # fallback, as in the app
    order = ["shared_new", "shared_own", "blend", "personal"]
    names = {"shared_new": "Shared,\nnew user", "shared_own": "Shared +\nown pooled",
             "blend": "Shared + own\nup-weighted", "personal": "Personal\n(own only)"}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    means = [z[s].mean() for s in order]
    b = ax.bar([names[s] for s in order], means, width=0.55, color=SLOT[0],
               edgecolor=SURFACE, linewidth=1.5)
    for rect, v in zip(b, means):
        ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean accuracy on own later windows")
    ax.set_title(f"Personal vs shared models (LightGBM + z, {len(z)} people)")
    style(ax)
    save(fig, out, "fig_personal_vs_shared.png")

    # per-person dumbbell: pooled shared vs personal
    z = z.sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(z))
    for i, (p, r) in enumerate(z.iterrows()):
        ax.plot([r["shared_own"], r["personal"]], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter(z["shared_own"], y, s=60, color=SLOT[1], edgecolor=SURFACE, linewidth=2,
               zorder=3, label="Shared (everyone pooled)")
    ax.scatter(z["personal"], y, s=60, color=SLOT[0], edgecolor=SURFACE, linewidth=2,
               zorder=3, label="Personal (own windows only)")
    ax.set_yticks(y, z.index)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Accuracy on the person's later windows")
    ax.set_title("Per person: shared vs personal model", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    style(ax, "x")
    save(fig, out, "fig_personal_per_person.png")


def fig_synthetic(syn, out):
    tests = list(dict.fromkeys(syn["test"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, model in zip(axes, ["LightGBM raw", "LightGBM + z-scores"]):
        d = syn[syn["model"] == model].set_index("test").loc[tests]
        bars(ax, [t.replace(" (", "\n(") for t in tests], ["Real data only", "+ synthetic (SMOTE)"],
             [d["real_only"].tolist(), d["with_synthetic"].tolist()], SLOT[:2],
             "Macro-F1" if ax is axes[0] else "", ylim=(0, 0.6))
        ax.set_title(model)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Synthetic training data did not help (tested on real windows only)",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, out, "fig_synthetic_data.png")


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=SLOT[0] + "22" if accent else SURFACE,
                                    edgecolor=SLOT[0] if accent else INK2, linewidth=1.2))
        ax.text(x + 0.12, y + h - 0.22, title, fontsize=9.5, fontweight="bold", va="top")
        ax.text(x + 0.12, y + h - 0.55, body, fontsize=7.8, color=INK2, va="top", linespacing=1.4)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color=INK2, linewidth=1.2))

    box(0.1, 2.3, 2.0, 1.7, "Activity logger", "app switches, idle time,\nkey/mouse counts,\n"
        "website names (opt-in)\n→ SQLite on the PC")
    box(2.5, 2.3, 2.0, 1.7, "5-minute windows", "activity rates, app &\nwebsite category\n"
        "shares, time of day")
    box(4.9, 2.3, 2.0, 1.7, "Self-labels", "Focused / Distracted /\nInactive, chosen by the\n"
        "user (accepted guesses\nkept apart)")
    box(7.3, 2.3, 3.6, 1.7, "Models", "shared: LightGBM + personal\nEWMA z-scores, trained on\n"
        "10 participants (focus-train)\npersonal: same model, user's own\nlabels (≥ 20), retrained locally",
        accent=True)
    box(0.1, 0.15, 3.3, 1.6, "Local app (FastAPI + React)", "Collect & label: model guess\nnext to "
        "your label, Accept model\nInsights: day summary, per-window\nreasons")
    box(3.8, 0.15, 3.3, 1.6, "Explanations", "SHAP (LightGBM TreeSHAP):\ntop 3 reasons per window\n"
        "Optional Groq LLM: day summary\nfrom numbers & categories only")
    box(7.5, 0.15, 3.4, 1.6, "Evaluation", "time split, leave-one-participant-\nout, McNemar, "
        "bootstrap CI,\nrobustness & personal-model\nchecks")
    arrow(2.1, 3.15, 2.5, 3.15)
    arrow(4.5, 3.15, 4.9, 3.15)
    arrow(6.9, 3.15, 7.3, 3.15)
    arrow(8.6, 2.3, 5.5, 1.75)
    arrow(9.1, 2.3, 9.2, 1.75)
    arrow(3.8, 0.95, 3.4, 0.95)
    ax.text(0.1, 4.12, "AI Focus Tracker: data flow (everything runs on the user's PC; "
            "only the optional LLM summary leaves it)", fontsize=10.5, fontweight="bold")
    save(fig, out, "fig_architecture.png")


def main(run_dir):
    run = Path(run_dir)
    out = run / "figures"
    out.mkdir(exist_ok=True)
    res = json.loads((run / "results.json").read_text(encoding="utf-8"))
    rob = json.loads((run / "robustness.json").read_text(encoding="utf-8"))
    print("writing", out)
    fig_architecture(out)
    fig_label_distribution(res, out)
    fig_main_models(res, out)
    fig_lopo(res, out)
    fig_rq3(res, out)
    fig_time_ablation(rob, out)
    fig_seed_stability(rob, out)
    fig_two_class(rob, out)
    fig_personal(pd.read_csv(run / "personal_check.csv"), out)
    fig_synthetic(pd.read_csv(run / "synthetic_experiment.csv"), out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/results/run2")
