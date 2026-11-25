"""SHAP explanations for the LightGBM model (UC-19, slide 9 step 4).

f(x) = phi_0 + sum_i phi_i : for each window, the three features that pushed
the PREDICTED class's score the most are reported, with the feature's value
and (when available) its personal z-score.

SHAP values come from LightGBM's built-in TreeSHAP (`pred_contrib=True`), the
same algorithm as shap.TreeExplainer, so the desktop app needs no `shap` package.
"""
import numpy as np
import pandas as pd

NICE = {
    "idle_share": "idle share", "active_share": "active share",
    "switches_per_min": "app switches/min", "site_switches_per_min": "site switches/min",
    "keys_per_min": "keys/min", "mouse_per_min": "mouse events/min",
    "browser_share": "browser time share", "distinct_hosts": "distinct sites",
    "distinct_apps": "distinct apps", "idle_episodes": "idle episodes",
    "share_entertainment": "entertainment share", "share_social": "social share",
    "share_educational": "educational share", "share_development": "development share",
    "share_documentation": "documentation share", "share_search": "search share",
    "share_communication": "communication share", "share_news": "news share",
    "share_shopping": "shopping share", "share_other": "other-sites share",
    "share_unknown": "unknown-sites share", "app_dev": "coding/terminal app time",
    "app_browser": "browser app time", "app_docs": "documents app time",
    "app_comm": "messaging app time", "app_media": "media app time",
    "app_system": "system app time", "app_other": "other app time",
    "hour_sin": "time of day", "hour_cos": "time of day", "coverage": "logger coverage",
}


def nice(feature):
    if feature.startswith("z_"):
        return NICE.get(feature[2:], feature[2:]) + " vs your usual"
    if feature.startswith("task_"):
        return f"session task: {feature[5:]}"
    return NICE.get(feature, feature)


def _class_values(raw, n_rows, n_classes):
    """Normalise shap output to an array (rows, features, classes)."""
    if isinstance(raw, list):                     # older shap: list per class
        return np.stack(raw, axis=-1)
    arr = np.asarray(raw)
    if arr.ndim == 2:                             # binary / single output
        return arr[:, :, None]
    if arr.shape[0] == n_classes and arr.shape[1] == n_rows:
        return np.moveaxis(arr, 0, -1)
    return arr


def contributions(gbt, X) -> np.ndarray:
    """TreeSHAP values as (rows, features, classes), bias terms dropped."""
    raw = np.asarray(gbt.model.predict(X, pred_contrib=True))
    k, f = len(gbt.classes_), X.shape[1]
    if raw.shape[1] == f + 1:                     # binary: one block, for classes_[1]
        phi = raw[:, :f]
        return np.stack([-phi, phi], axis=-1)
    return raw.reshape(len(X), k, f + 1)[:, :, :f].transpose(0, 2, 1)


def plain_reason(feature, value):
    """Short user-facing wording, e.g. 'keys/min higher than your usual'."""
    if pd.isna(value):
        return f"no data for {nice(feature).replace(' vs your usual', '')}"
    if feature.startswith("z_"):
        base = NICE.get(feature[2:], feature[2:])
        if abs(value) < 0.5:
            return f"{base} close to your usual"
        return f"{base} {'higher' if value > 0 else 'lower'} than your usual"
    if feature.startswith("task_"):
        if feature == "task_none":
            return "no session running" if value else "in a session"
        return f"session task: {feature[5:]}" if value else f"not a {feature[5:]} session"
    if feature.startswith("hour_"):
        return "time of day"
    if feature.startswith(("share_", "app_")) or feature.endswith("_share") \
            or feature == "coverage":
        return f"{nice(feature)} {value:.0%}"
    return f"{nice(feature)} {value:.1f}"


def explain(gbt, df: pd.DataFrame, top=3):
    """(per-window explanations DataFrame, global importance Series)."""
    import warnings
    X = df[gbt.features].apply(pd.to_numeric, errors="coerce").astype(float)
    pred = gbt.predict(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = contributions(gbt, X)
    rows = []
    for r, (idx, label) in enumerate(zip(df.index, pred)):
        phi = values[r, :, gbt.classes_.index(label)]
        order = np.argsort(-phi)[:top]            # features pushing TOWARDS the prediction
        parts, feats, reasons = [], [], []
        for j in order:
            f = gbt.features[j]
            v = X.iloc[r, j]
            text = f"{nice(f)} {'n/a' if pd.isna(v) else f'{v:.2f}'}"
            if f.startswith("z_"):
                text = f"{nice(f)} {'n/a' if pd.isna(v) else f'z {v:+.1f}'}"
            parts.append(text)
            reasons.append(plain_reason(f, v))
            feats.append((f, float(phi[j])))
        rows.append({"index": idx, "predicted": label,
                     "explanation": f"{label}: " + "; ".join(parts),
                     "reasons": reasons,
                     **{f"top{k + 1}_feature": feats[k][0] for k in range(len(feats))},
                     **{f"top{k + 1}_shap": round(feats[k][1], 4) for k in range(len(feats))}})
    importance = pd.Series(np.abs(values).mean(axis=(0, 2)), index=gbt.features) \
        .sort_values(ascending=False)
    return pd.DataFrame(rows).set_index("index"), importance
