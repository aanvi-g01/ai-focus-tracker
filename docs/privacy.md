# Privacy rules (fixed in Week 1)

The logger records only:

| Recorded | Not recorded |
|---|---|
| foreground process name (e.g. `Code.exe`) | window titles, URLs, file names |
| app-switch events | which keys were pressed (only a count) |
| idle start / resume | screen content, screenshots |
| keyboard / mouse event **counts** per 5 s | clipboard, network traffic |
| browser **hostname** only (e.g. `github.com`) | path, query, search terms, page title |
| one-word browser **content category** (e.g. `educational`) | titles of non-video sites, page text |
| **video title** on video-streaming sites, opt-in (`--read-titles`) | URLs / video ids, comments, search terms |
| focus sessions: the task **you typed**, its category, start/pause/end times | anything inferred about you |

Enforced in code: whitelisted event names (`storage.py`), no content columns in
the schema, and input callbacks that ignore their arguments (`input_counter.py`).
Tests in `tests/test_logger.py` and `tests/test_sessions.py` guard these rules.
Raw logs stay on the machine; only structured summaries may go to the LLM API.

## Content categories (`logger/content.py`)

* Computed locally from the hostname; the stored value is validated against a
  fixed list (`CONTENT_CATEGORIES`).
* Page titles are read **only** with `focus-logger --read-titles` and **only**
  on `TITLE_HOSTS` (youtube.com, vimeo.com, medium.com, ...) and `VIDEO_HOSTS`
  (youtube.com, netflix.com, twitch.tv, primevideo.com, hotstar.com, ...).
  Titles of mail, search, banking or any other site are never read.
* On **video-streaming sites** the cleaned video title is stored in
  `events.content_title` (browser name, " - YouTube", notification counts and
  control characters removed; max 200 characters). Storage rejects a title on
  any other website.
* On the other `TITLE_HOSTS` (e.g. medium.com) the title is only matched
  against keyword lists in memory and discarded.
* Video titles can reveal personal interests. They stay in the local database
  and are not part of the session context features.

## Focus sessions (`logger/session.py`)

* `focus-session start "studying DSA"` / `pause` / `resume` / `end` / `status`.
* The task text is whatever you type (max 80 characters) - don't type
  anything you would not want in the local database.
* Session context (`features/session_context.py`) relates the declared task to
  content categories as `related` / `unrelated` / `unknown`. These are model
  inputs, never verdicts: nothing labels you focused or distracted, and the
  free-text task is not included in the derived features.

## Windows and self-labels (`features/windows.py`, `logger/labels.py`)

* `windows` holds per-window aggregates (seconds, counts, app names) computed
  from `events`; it adds no new information and can be rebuilt at any time.
* `window_labels` holds only what **you** enter with `focus-label`
  (`Focused` / `Distracted` / `Inactive`). No label is ever inferred, and
  session names or categories never influence labels.

## Predictions (`api/predictions.py`, Insights page)

* The app can show the trained model's **estimate** for each finished window.
  It is computed on this computer from the window features above, each time
  the Insights page asks for it.
* Estimates are **never stored** (not in `window_labels` or anywhere else),
  **never exported** and never counted as labels. Your own labels remain the
  only ground truth, and the Insights page shows the two side by side.
* The model file contains only learned parameters and feature statistics from
  the training participants (no raw logs, hostnames or titles).
* **Personal model.** Once you have labelled about 20 windows yourself (at least
  two different labels), the app trains a model from *your* labels on this
  computer and uses it instead of the shared one. It is one file of roughly
  0.1–1 MB in `personal_model/` next to your database, replaced on every
  retrain (no history is kept), and it is never exported or uploaded. Clearing
  your labels below the threshold deletes it.
* **Accepted guesses.** If you press *Accept model*, the label is saved with
  `source = accepted` (shown with a ✓). It is your decision and is exported
  with `label_source = accepted`, but it is never used to train any model, so
  a model never learns from its own guesses.
* Per-window explanations ("why") use the model's own SHAP values (LightGBM
  TreeSHAP) and a fixed wording template, computed on this computer.

## LLM explanations (Groq, optional)

* Off until you add a Groq API key on the Insights page (or set `GROQ_API_KEY`).
  The key is stored in `settings.json` on this computer only and is never shown
  again or sent anywhere except to Groq.
* Nothing is sent until you press **Explain this day**. Then only a structured
  summary of that day goes to Groq (`api/llm.py`, `day_summary`): minutes per
  estimated state, estimated states per hour, how many windows were low-confidence,
  the most common SHAP reason phrases, average activity rates (idle share,
  switches, key/mouse counts per minute), and average time shares per app
  **category** (coding, browser, documents, messaging, media, system, other) and
  per website **category** (entertainment, educational, ...).
* Never sent: app names, website names, video titles, session task text, raw
  events or labels' timestamps. The page can show the exact JSON that was sent.
* Answers are cached in `llm_cache.json` next to the database (at most 30, a few KB).
