# Focus-state classification results

## Data

654 labelled windows (432 train / 218 test).

| Participant | Days | Windows | Focused | Distracted | Inactive | Train | Test |
|---|---|---|---|---|---|---|---|
| P01 | 1 | 43 | 33 | 9 | 1 | 30 | 12 |
| P02 | 3 | 88 | 3 | 85 | 0 | 23 | 65 |
| P03 | 2 | 26 | 18 | 8 | 0 | 12 | 14 |
| P04 | 2 | 23 | 18 | 5 | 0 | 11 | 12 |
| P05 | 2 | 24 | 19 | 5 | 0 | 1 | 23 |
| P06 | 1 | 20 | 20 | 0 | 0 | 14 | 5 |
| P07 | 1 | 16 | 10 | 3 | 3 | 11 | 4 |
| P08 | 1 | 25 | 18 | 7 | 0 | 17 | 7 |
| P09 | 3 | 93 | 52 | 25 | 16 | 72 | 21 |
| P10 | 3 | 296 | 90 | 13 | 193 | 241 | 55 |

## Models (test windows)

| Model | Macro-F1 | 95% CI | Accuracy | F1 Focused | F1 Distracted | F1 Inactive |
|---|---|---|---|---|---|---|
| Rule baseline | 0.251 | 0.209-0.300 | 0.450 | 0.642 | 0.021 | 0.091 |
| LightGBM (raw features) | 0.350 | 0.281-0.424 | 0.514 | 0.653 | 0.259 | 0.138 |
| LightGBM (+ personal z-scores) | 0.446 | 0.404-0.492 | 0.651 | 0.785 | 0.554 | 0.000 |
| HMM (Viterbi) | 0.224 | 0.176-0.268 | 0.339 | 0.521 | 0.094 | 0.055 |

## RQ2: learned models vs rule baseline

| Comparison | ΔF1 | McNemar b / c | p-value |
|---|---|---|---|
| gbt_z_vs_rule | +0.195 | 49 / 5 | 0.0000 |
| hmm_vs_rule | -0.028 | 4 / 28 | 0.0000 |

## RQ1: personal baseline (z-scores) vs raw features

ΔF1 (LightGBM + z minus raw) = +0.097, McNemar b/c = 38/8, p = 0.0000.

Alpha sensitivity (macro-F1): α=0.02: 0.381, α=0.05: 0.446, α=0.1: 0.430

Note: exports contain labelled windows only, so the baseline is built from labelled history.

## RQ3: low activity is not distraction

Focused test windows with idle share ≥ 0.6: n = 51.

| Model | Recall (Focused) |
|---|---|
| Rule baseline | 0.588 |
| LightGBM (raw features) | 0.647 |
| LightGBM (+ personal z-scores) | 0.784 |
| HMM (Viterbi) | 0.176 |

## Per participant (macro-F1, test windows)

| Participant | Rule baseline | LightGBM (raw features) | LightGBM (+ personal z-scores) | HMM (Viterbi) |
|---|---|---|---|---|
| P01 | 0.433 | 0.294 | 0.556 | 0.482 |
| P02 | 0.000 | 0.375 | 0.687 | 0.000 |
| P03 | 0.400 | 0.391 | 0.391 | 0.321 |
| P04 | 0.455 | 0.400 | 0.476 | 0.455 |
| P05 | 0.250 | 0.182 | 0.345 | 0.182 |
| P06 | 1.000 | 1.000 | 1.000 | 1.000 |
| P07 | 1.000 | 1.000 | 0.333 | 1.000 |
| P08 | 0.727 | 0.833 | 0.923 | 0.000 |
| P09 | 0.198 | 0.215 | 0.229 | 0.121 |
| P10 | 0.466 | 0.461 | 0.466 | 0.538 |

## Leave-one-participant-out (new user, macro-F1)

| Held-out | Rule baseline | LightGBM (raw features) | LightGBM (+ personal z-scores) | HMM (Viterbi) |
|---|---|---|---|---|
| P01 | 0.503 | 0.354 | 0.505 | 0.315 |
| P02 | 0.054 | 0.273 | 0.345 | 0.214 |
| P03 | 0.333 | 0.314 | 0.409 | 0.302 |
| P04 | 0.394 | 0.628 | 0.395 | 0.333 |
| P05 | 0.269 | 0.269 | 0.269 | 0.174 |
| P06 | 0.095 | 1.000 | 1.000 | 1.000 |
| P07 | 0.421 | 0.242 | 0.337 | 0.381 |
| P08 | 0.377 | 0.138 | 0.363 | 0.404 |
| P09 | 0.351 | 0.291 | 0.172 | 0.319 |
| P10 | 0.599 | 0.178 | 0.256 | 0.444 |

Rule thresholds (fit on training windows): {'t_idle': 0.95, 't_ent': 0.6, 't_switch': 8.0}
LightGBM parameters: {'num_leaves': 15, 'n_estimators': 400}
