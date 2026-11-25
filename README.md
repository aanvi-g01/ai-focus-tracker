# AI Focus Tracker

Results of the final evaluation: [docs/results.md](docs/results.md).

## Training the focus model

1. Each participant labels their windows in the app, then exports a CSV from the
   Collect page (`focus_windows_<participant>_<time>.csv`).
2. Put all exports in `data/participants/` (git-ignored: personal data). Files in
   sub-folders (e.g. `data/participants/superseded/`) are ignored.
3. Install the ML extras and train:

   ```
   pip install -e .[ml]
   focus-train --data data/participants --out models --eval-out results/run2
   ```

   This runs the full evaluation (per-participant time split, leave-one-participant-out,
   RQ1–RQ3, SHAP) into `results/run2/summary.md`, picks the model with the best mean
   leave-one-participant-out macro-F1 (`--model` to override), refits it on all windows
   and saves `models/<timestamp>/{model.joblib, manifest.json, lgbm_booster.txt}`.

4. Robustness checks (time-of-day ablation, single-label participants excluded, seed
   stability, Focused-vs-Distracted, Inactive breakdown, optional label consistency):

   ```
   python -m focus_tracker.models.robustness --data data/participants --out results/run2 \
       [--relabel old_export.csv new_export.csv]
   ```

   Writes `results/run2/robustness.{md,json}`.

`focus-evaluate` runs only the evaluation, without saving a model.

## Live predictions (Insights page)

`focus-app` (or the desktop exe) shows the newest model in `models/` on the **Insights**
tab: a predicted state, confidence and the top reasons for every finished window, next
to your own label. Predictions are computed on the fly and never stored or exported.
Once a user has labelled about 20 windows themselves (two or more labels), the app
trains a **personal model** from their own labels in the background (about a second,
one ~150 KB file in `personal_model/` next to the database) and uses it instead of the
shared model; the page shows an honest accuracy estimate on their most recent labels.
Guesses accepted with *Accept model* are stored as `accepted` and never used for training.
The exe bundles the shared model it was built with; a newer model folder copied to
`%LOCALAPPDATA%\FocusTracker\models\` is used instead without rebuilding.
Build the exe with `python packaging/build_exe.py`.

## LLM explanations (optional)

On the Insights tab, **Explain this day** asks Groq (`openai/gpt-oss-120b` by default) to
describe the day's estimates in plain words. It needs a Groq API key, entered on that page
(stored only in the local `settings.json`) or set as `GROQ_API_KEY`. Only a structured
summary is sent: numbers, SHAP reason phrases and app/website categories, never app names,
websites or titles (see `docs/privacy.md`). Answers are cached in `llm_cache.json`.

## Testing the model

- **Code tests** (synthetic data only): `pytest tests/test_ml.py`
- **Model quality**: the evaluation above *is* the test. `summary.md` scores every model
  on windows it never trained on (later days, and whole held-out participants).
- **Try the saved model on an export**:

  ```python
  from focus_tracker.models.train import load
  from focus_tracker.models.data import load_exports

  model = load("models/20261126_003136")
  df = load_exports("path/to/focus_windows_X.csv")
  df["predicted"] = model.predict(df)                     # Focused / Distracted / Inactive
  print((df["predicted"] == df["label"]).mean())          # accuracy vs the user's labels
  print(model.predict_proba(df).head())                   # class probabilities
  expl, _ = model.explain(df); print(expl["explanation"].head())   # why (SHAP)
  ```

  The final model was trained on every participant in `data/participants/`, so scoring it
  on those same files is optimistic. Use an export from a **new** participant for a fair test.
