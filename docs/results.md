# Focus-state classification: results

Final run: [`docs/results/run2/`](results/run2/) holds the full outputs (summary, robustness checks, per-window predictions, SHAP, personal-model check, plots). It is an anonymised copy made by `scripts/anonymise_results.py`: participants are P01–P10, the tester outside the study is X1, and exact window times are removed. The original `results/run2/` stays on the team's machine.
Final model: `models/20261126_003136/` (LightGBM with personal z-scores, trained on all windows).
Participants are not named here; per-participant tables are in the run folder.

## Data

| | |
|---|---|
| Participants | 10 |
| Labelled 5-minute windows | 654 (Focused 281, Distracted 160, Inactive 213) |
| Days per participant | 1–3 |
| Split | per participant, latest days (or latest windows) as test: 432 train / 218 test |
| Labels | self-reported by each participant; never inferred |

## Protocol

- **Time split.** Each participant's earliest windows are used for training and their latest for testing. Models are trained on the pooled training windows of all participants.
- **Leave-one-participant-out (LOPO).** Each participant in turn is held out completely, which simulates a new user.
- **Metric.** Macro-F1 is the primary metric, because the classes are imbalanced. Significance uses a bootstrap 95% CI and an exact McNemar test.
- **No test-set tuning.** Hyperparameters are chosen on the tail of the training data only. Nothing is tuned on test windows.

Models compared:
- **Rule baseline:** idle, entertainment and switching thresholds fitted on training data.
- **LightGBM** on raw features.
- **LightGBM + personal z-scores:** features compared with the user's own EWMA baseline.
- **Supervised HMM** with Viterbi decoding.

## Main results (time split, test windows)

| Model | Macro-F1 | 95% CI | Accuracy | F1 Focused | F1 Distracted | F1 Inactive |
|---|---|---|---|---|---|---|
| Rule baseline | 0.251 | 0.209–0.300 | 0.450 | 0.642 | 0.021 | 0.091 |
| LightGBM (raw) | 0.350 | 0.281–0.424 | 0.514 | 0.653 | 0.259 | 0.138 |
| **LightGBM + z-scores** | **0.446** | 0.404–0.492 | 0.651 | 0.785 | 0.554 | 0.000 |
| HMM (Viterbi) | 0.224 | 0.176–0.268 | 0.339 | 0.521 | 0.094 | 0.055 |

Mean LOPO macro-F1 (new user): rule 0.340, LightGBM raw 0.369, **LightGBM + z 0.405**, HMM 0.389.
LightGBM + z-scores is the final model because it has the best mean LOPO macro-F1.

## Research questions

**RQ1: Does a personal baseline help?** Yes.
- Adding z-scores improves LightGBM by ΔF1 = +0.097 (McNemar p < 0.001).
- Across 5 random seeds, the gain averages +0.172. It is positive in 5 of 5 seeds, with the smallest gain +0.097.
- Macro-F1 over seeds: z-scores 0.523 ± 0.043, raw 0.351 ± 0.005.
- Alpha sensitivity (EWMA speed): α = 0.02 gives 0.381, α = 0.05 gives 0.446 (used), α = 0.1 gives 0.430.

**RQ2: Do learned models beat rules?** Partly.
- LightGBM + z-scores beats the rule baseline on the time split (ΔF1 +0.195, p < 0.001).
- The HMM does not; it is worse than the rules (ΔF1 −0.028, p < 0.001).
- For completely new users, see the limitations below.

**RQ3: Is low activity misread as distraction?** Less often with the personal model.
- The test set has 51 Focused windows with idle share ≥ 0.6.
- Recall of Focused on these windows:

| Model | Recall (Focused) |
|---|---|
| Rule | 0.588 |
| LightGBM raw | 0.647 |
| LightGBM + z | 0.784 |
| HMM | 0.176 |

## Robustness checks (`results/run2/robustness.md`)

| Check | Result |
|---|---|
| Time-of-day features removed | LOPO unchanged (0.405 → 0.405). The time split rises to 0.532, which is within seed variation. The model does not depend on *when* people logged. |
| Single-label participant excluded from scoring | Mean LOPO: LightGBM + z 0.339, rule 0.367, raw 0.298, HMM 0.321 |
| Focused vs Distracted only (2-class) | LightGBM + z: time split 0.680, LOPO 0.563. Rule: 0.337 / 0.371 |
| Inactive, new user (LOPO recall) | Rule 0.925, HMM 0.930, LightGBM raw 0.056, LightGBM + z 0.005 |
| Self-label consistency | One participant re-labelled 140 windows: 78.6 % agreement, Cohen's κ = 0.566. Most changes were Focused → Inactive (21). |

## Personal models (`scripts/personal_check.py`, `results/run2/personal_check.csv`)

Testing on a person outside the study showed the shared model was far off for them (32 % of their 60 self-labelled windows correct). So for each of the 10 participants plus that person, models were trained on the person's **earlier** windows and tested on their **later** windows (same split as above; a person whose own training windows have fewer than two labels falls back to the shared model).

| Training data (LightGBM + z-scores) | Mean accuracy | Mean macro-F1 |
|---|---|---|
| Other people only (new user) | 0.47 | 0.39 |
| Everyone's earlier days pooled (current evaluation) | 0.64 | 0.50 |
| Other people + own windows up-weighted | 0.63 | 0.50 |
| **Own earlier windows only (personal)** | **0.73** | **0.66** |

- The personal model was better for 7 of 11 people and worse for 3. With LightGBM on raw features the split was 8 better, 1 worse (mean accuracy 0.71).
- Test sets are small (4–65 windows), and some are single-label: one participant's test windows are all Distracted, another's all Focused.
- The app therefore trains a personal model on the user's own computer once they have 20 self-chosen labels. Guesses accepted with "Accept model" are never used for training.

## Synthetic data (`scripts/synth_experiment.py`)

Two kinds of synthetic data were considered. **SMOTE-style windows** interpolated between real windows of the same label (training folds only, about +120 Distracted and +70 Inactive) were tested on real held-out windows only:

| Test | Model | Real only | + synthetic |
|---|---|---|---|
| Later days, same people | LightGBM raw | 0.350 | 0.368 |
| | LightGBM + z | 0.446 | 0.361 |
| New person (LOPO mean) | LightGBM raw | 0.369 | 0.418 |
| | LightGBM + z | 0.405 | 0.400 |
| Person outside the study | LightGBM raw | 0.488 | 0.503 |
| | LightGBM + z | 0.196 | 0.213 |

- The changes are within the seed variation (±0.04), except that synthetic data **hurt** the final model on the time split. Accuracy for the person outside the study was unchanged (42 % / 33 %).
- Synthetic data was therefore not used.
- **Rule-generated data** (synthetic windows not derived from the collected data) was rejected without training: its labels would come from hand-written rules, which contradicts the self-label principle. The rule baseline already shows how such rules perform (0.25–0.34).

## Explanations

- **Per window:** the three features with the largest SHAP values toward the predicted state (LightGBM TreeSHAP), worded by a fixed template.
- **Per day (optional):** an LLM on Groq (`openai/gpt-oss-120b`) rewrites the day's structured summary in plain words, on request. It receives only numbers, reason phrases and app/website categories (see `docs/privacy.md`). It rewords the model's estimates and does not change them.

## Limitations

- **Small dataset.** Ten participants contributed 654 self-labelled 5-minute windows, mostly over 1–3 days. Confidence intervals are wide, and per-participant scores vary considerably between runs.
- **Noisy labels.** Self-labels were only moderately consistent (κ = 0.57 when one participant re-labelled the same windows), especially between Focused and Inactive. This limits the accuracy any model can reach.
- **Inactive.** Inactive labels came almost entirely from one participant (193 of 213). The learned models therefore did not learn to detect Inactive for other users, whereas a simple idle-time rule did (recall 0.93).
- **New users.** For a user with no history, the learned models did not outperform the rule baseline once a participant who used only one label was excluded (macro-F1 0.34 vs 0.37). The personal model's advantage appears once a user has history: time split 0.45 vs 0.25, and Focused vs Distracted 0.68 vs 0.34.
- **Imbalanced participants.** One participant labelled almost everything Distracted and another everything Focused. Both affect the per-participant averages.

## Reproduce

```
pip install -e .[ml]
focus-train --data data/participants --out models --eval-out results/run2
python -m focus_tracker.models.robustness --data data/participants --out results/run2 \
    --relabel <old_export.csv> <new_export.csv>
```
