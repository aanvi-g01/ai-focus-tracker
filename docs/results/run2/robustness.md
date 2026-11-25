# Robustness checks

## 1. Time-of-day ablation (without hour_sin / hour_cos)

| Model | Macro-F1 with time | without time | 95% CI (without) | LOPO mean with | LOPO mean without | McNemar p |
|---|---|---|---|---|---|---|
| LightGBM (raw features) | 0.350 | 0.460 | 0.416-0.500 | 0.369 | 0.384 | 0.0002 |
| LightGBM (+ personal z-scores) | 0.446 | 0.532 | 0.497-0.566 | 0.405 | 0.405 | 0.0145 |

## 2. Without single-class participants (P06)

Still used for training; only removed from scoring.

| Model | Time-split macro-F1 | LOPO mean (all) | LOPO mean (excluded) |
|---|---|---|---|
| Rule baseline | 0.247 | 0.340 | 0.367 |
| LightGBM (raw features) | 0.346 | 0.369 | 0.298 |
| LightGBM (+ personal z-scores) | 0.444 | 0.405 | 0.339 |
| HMM (Viterbi) | 0.217 | 0.389 | 0.321 |

## 3. Seed stability (seeds [0, 1, 2, 3, 4])

| Model | Mean macro-F1 | SD |
|---|---|---|
| LightGBM (raw features) | 0.351 | 0.005 |
| LightGBM (+ personal z-scores) | 0.523 | 0.043 |

ΔF1 (z minus raw): mean +0.172, smallest +0.097, positive in 5/5 seeds.

## 4. Focused vs Distracted only (secondary, 2-class macro-F1)

| Model | Time split | LOPO |
|---|---|---|
| Rule baseline | 0.337 | 0.371 |
| LightGBM (raw features) | 0.462 | 0.507 |
| LightGBM (+ personal z-scores) | 0.680 | 0.563 |
| HMM (Viterbi) | 0.313 | 0.339 |

## 5. Inactive

| Participant | Inactive windows | Train | Test |
|---|---|---|---|
| P01 | 1 | 1 | 0 |
| P07 | 3 | 1 | 2 |
| P09 | 16 | 11 | 5 |
| P10 | 193 | 193 | 0 |

Inactive test windows: 7. Predicted as:

| Model | Focused | Distracted | Inactive |
|---|---|---|---|
| Rule baseline | 5 | 0 | 2 |
| LightGBM (raw features) | 5 | 0 | 2 |
| LightGBM (+ personal z-scores) | 7 | 0 | 0 |
| HMM (Viterbi) | 5 | 0 | 2 |

Leave-one-participant-out Inactive recall (held-out user's Inactive windows):

| Held-out | Rule baseline | LightGBM (raw features) | LightGBM (+ personal z-scores) | HMM (Viterbi) |
|---|---|---|---|---|
| P01 | 1.000 | 1.000 | 1.000 | 1.000 |
| P07 | 0.667 | 0.000 | 0.000 | 0.667 |
| P09 | 0.500 | 0.250 | 0.000 | 0.500 |
| P10 | 0.964 | 0.036 | 0.000 | 0.969 |
| **All** | 0.925 | 0.056 | 0.005 | 0.930 |

## 6. Label consistency (same windows, two exports)

140 shared windows: agreement 78.6%, Cohen's kappa 0.566.

Changes: Distracted -> Focused: 4, Distracted -> Inactive: 4, Focused -> Inactive: 21, Inactive -> Focused: 1
