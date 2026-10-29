# AI Focus Tracker — GP055

A personalized, explainable focus-state classifier (Focused / Distracted / Inactive)
built from privacy-preserving desktop activity statistics.

## Quick start
**macOS**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
python -m focus_tracker.logger.agent --db data/activity.db
pytest
```
macOS needs two permissions for your terminal / VS Code (System Settings -> Privacy & Security):
**Input Monitoring** and **Accessibility**. Restart the terminal after granting.

**Windows**
```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -e . -r requirements.txt
python -m focus_tracker.logger.agent --db data/activity.db
pytest
```

## Layout
```
src/focus_tracker/
  logger/     W1-2  agent, probes, input counter, SQLite store   <- implemented
  features/   W3    2-5 min window features
  baseline/   W3    per-user baseline + rule-based classifier
  models/     W4    HMM / gradient-boosted trees
  explain/    W5    SHAP / state-transition explanations
  api/        W5    FastAPI backend
  insights/   W5    LLM insights (structured summaries only)
dashboard/    W5    React + Recharts
docs/privacy.md     privacy rules
```

## Roadmap
- [x] W1-2 logger scaffold (app, switches, idle, input counts)
- [ ] W3 features + personal baseline + rule-based comparison
- [ ] W4 self-report labels + HMM/GBT training
- [ ] W5 explanations, API, dashboard, LLM insights
- [ ] W6 evaluation (accuracy/precision/recall/F1/confusion matrix), report, demo

Privacy: see [docs/privacy.md](docs/privacy.md). No keystrokes, no screen content.
