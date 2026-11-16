# AI Focus Tracker: Use Cases

This document describes how people and system components interact with the AI
Focus Tracker. It covers what is **implemented** today (activity logging,
focus sessions, browser context, 5-minute windows and self-labelling) and,
separately, what is **planned** (personal baseline, classifier, explanations,
insights, dashboard).

## Actors

| Actor | Description |
|---|---|
| **User** | The person whose focus is being tracked (e.g. a student). Starts the logger, manages sessions, labels windows. |
| **Logger agent** | Background process (`focus-logger`) that records privacy-safe activity signals once per second. |
| **Aggregation pipeline** | Turns raw events into 5-minute feature windows (`features/windows.py`). |
| **Focus classifier** *(planned)* | Future model that predicts Focused / Distracted / Inactive from window features, trained on the user's own labels. |

## Summary

| ID | Use case | Primary actor | Status |
|---|---|---|---|
| UC-01 | Start and stop activity logging | User | Implemented |
| UC-02 | Record foreground application and switches | Logger agent | Implemented |
| UC-03 | Record idle episodes | Logger agent | Implemented |
| UC-04 | Count keyboard and mouse activity | Logger agent | Implemented |
| UC-05 | Record the website hostname | Logger agent | Implemented |
| UC-06 | Classify browser content | Logger agent | Implemented |
| UC-07 | Record video titles (opt-in) | Logger agent | Implemented |
| UC-08 | Start a focus session | User | Implemented |
| UC-09 | Pause and resume a session | User | Implemented |
| UC-10 | End a session and check its status | User | Implemented |
| UC-11 | Recover an unfinished session | Logger agent | Implemented |
| UC-12 | Aggregate activity into 5-minute windows | Aggregation pipeline | Implemented |
| UC-13 | Review activity windows | User | Implemented |
| UC-14 | Label a window | User | Implemented |
| UC-15 | Correct or remove a label | User | Implemented |
| UC-16 | Relate the session task to browser content | Aggregation pipeline | Implemented |
| UC-17 | Learn a personal baseline | Focus classifier | Planned |
| UC-18 | Classify focus state per window | Focus classifier | Planned |
| UC-19 | Explain a classification | User | Planned |
| UC-20 | View focus insights and dashboard | User | Planned |

---

## A. Activity logging

### UC-01 Start and stop activity logging
- **Actor:** User
- **Goal:** Record privacy-safe activity signals while working.
- **Preconditions:** Package installed (`pip install -e .`).
- **Main flow:**
  1. User runs `focus-logger` (optionally `--no-websites`, `--read-titles`, `--idle-threshold`).
  2. Agent prints exactly what it records and logs `agent_start`.
  3. Agent polls once per second until the user presses Ctrl+C.
  4. Agent closes any idle episode in progress and logs `agent_stop`.
- **Alternate flows:**
  - `uiautomation` not installed → warning; website fields stay empty, everything else works.
  - An open session was left over from a long gap → see UC-11.
- **Postconditions:** Events are stored locally in `data/activity.db`.

### UC-02 Record foreground application and switches
- **Actor:** Logger agent
- **Goal:** Know which application is in front and when it changes.
- **Main flow:** Each tick the agent reads the foreground **process name** (e.g. `Code.exe`). The first app seen is logged as `active`; every change is logged as `switched`.
- **Privacy:** Window titles, file names and screen content are never read.
- **Used by:** switch count/rate, distinct apps and per-app time (UC-12).

### UC-03 Record idle episodes
- **Actor:** Logger agent
- **Goal:** Measure periods without input.
- **Main flow:**
  1. When no input has occurred for the threshold (default 60 s), log `idle`.
  2. When input resumes, log `idle_duration` (total seconds) and `active`.
- **Note:** Idle time is a raw measurement, **not** a judgement. Watching a lecture and being away from the desk both look idle; the label comes from the user (UC-14).

### UC-04 Count keyboard and mouse activity
- **Actor:** Logger agent
- **Goal:** Measure input intensity without capturing what was typed.
- **Main flow:** Every 5 s the agent stores the **number** of key and mouse events (`keyboard_count`, `mouse_count`).
- **Privacy:** Key identity, typed text and mouse coordinates are never read. The input callbacks ignore their arguments by design.

### UC-05 Record the website hostname
- **Actor:** Logger agent
- **Preconditions:** A supported browser is in front; website tracking not disabled.
- **Main flow:** The address bar is read in memory and immediately reduced to the bare hostname (e.g. `github.com`). A change of site inside the same browser is logged as `site_switched`.
- **Alternate flow:** Address bar unreadable (typing, new tab, internal page) → hostname unknown, not counted as a switch.
- **Privacy:** Paths, queries, search terms and URLs are never stored; storage rejects anything that isn't a hostname.

### UC-06 Classify browser content
- **Actor:** Logger agent
- **Goal:** Describe *what kind* of page is open, in one word.
- **Main flow:** The hostname is mapped locally to a category: `educational`, `documentation`, `development`, `entertainment`, `social`, `news`, `shopping`, `communication`, `search` or `other`.
- **Alternate flow:** Mixed-content platforms (e.g. youtube.com): the category is decided from the page title, only if the user opted in (UC-07); otherwise it stays unknown.
- **Note:** The category describes the page, not the person.

### UC-07 Record video titles (opt-in)
- **Actor:** Logger agent
- **Preconditions:** Logger started with `--read-titles`.
- **Main flow:**
  1. On video-streaming sites only (YouTube, Netflix, Twitch, Prime Video, Hotstar, …) the window title is read.
  2. It is cleaned (browser name, " - YouTube", notification counts removed; max 200 characters) and stored as `content_title`.
  3. Opening a different video on the same site logs `content_changed`.
- **Alternate flows:**
  - On medium.com and dev.to, the title is used for the category only and discarded.
  - Mail, search, banking and all other sites: the title is never read.

---

## B. Focus sessions

### UC-08 Start a focus session
- **Actor:** User
- **Goal:** Declare what they intend to work on.
- **Main flow:**
  1. User runs `focus-session start "Studying DSA"` (optionally `--category`).
  2. The system cleans the name, derives a broad task category (`studying`, `coding`, `research`, `writing`, `other`) and assigns a unique session ID.
  3. From now on, every event the logger writes is tagged with this session ID.
- **Alternate flow:** A session is already open → error; the user must end it first. Only one session can be open at a time, even from two terminals.
- **Note:** A session records an **intention**, never a focus state.

### UC-09 Pause and resume a session
- **Actor:** User
- **Main flow:** `focus-session pause` stops tagging events and starts counting paused time; `focus-session resume` continues tagging.
- **Postconditions:** Paused time is excluded from the session's active time.

### UC-10 End a session and check its status
- **Actor:** User
- **Main flow:** `focus-session end` closes the session and records its active time. `focus-session status` shows the open session, its state and its active time.
- **Postconditions:** The session has a start time, an end time and `end_reason = user`.

### UC-11 Recover an unfinished session
- **Actor:** Logger agent
- **Trigger:** The logger starts.
- **Main flow:**
  1. If a session is still open and nothing has been recorded for more than 2 hours (`--stale-after`), the session is closed at its **last recorded activity**.
  2. It is marked `end_reason = recovered`, and the user is told.
- **Alternate flow:** Short gap (e.g. a reboot) → the session simply continues across the restart.
- **Why:** Forgotten sessions or crashes must not tag hours of unrelated activity or inflate session time.

---

## C. Windows and self-labelling

### UC-12 Aggregate activity into 5-minute windows
- **Actor:** Aggregation pipeline (run automatically by `focus-label`)
- **Goal:** Turn the raw event stream into one structured record per clock-aligned window (10:00–10:05, …; length configurable).
- **Features per window:**
  - active seconds, idle seconds and idle episodes
  - app switches, website switches and switches per minute
  - number of distinct apps and seconds per app
  - keyboard and mouse counts
  - the session the window belongs to
- **Missing and incomplete data:**
  - Time when the logger was off is reported as `observed_seconds` / `coverage`, never as active or idle.
  - Windows with no data are omitted.
  - The window still in progress is flagged `complete = false`.
  - A window belongs to a session only if the session covers most of its observed time.
- **Postconditions:** The `windows` table is rebuilt. It can be regenerated at any time and never affects labels.

### UC-13 Review activity windows
- **Actor:** User
- **Main flow:** `focus-label list` (today by default; `--date`, `--session`, `--unlabelled`, `--all`) shows each window's time, session number, active/idle seconds, switches, input counts, top app and current label.
- **Note:** Only the session number is shown, not its name, so the name doesn't nudge the user's label.

### UC-14 Label a window
- **Actor:** User
- **Goal:** Provide ground truth for the future classifier.
- **Main flow:**
  1. User runs `focus-label set 10:05 Focused` (`Focused`, `Distracted` or `Inactive`; any case; any time inside the window works).
  2. The label is stored against the window and its session.
- **Alternate flows:**
  - Window has no recorded activity → error.
  - Window still in progress → error; label it once it is complete.
  - Unknown label → error listing the valid ones.
- **Rule:** Labels come **only** from the user. A session named "Studying DSA" never makes a window Focused, and the user may label such a window Distracted.

### UC-15 Correct or remove a label
- **Actor:** User
- **Main flow:** Running `focus-label set` again overwrites the label (the system reports "Corrected Focused -> Distracted"); `focus-label clear 10:05` removes it.
- **Postconditions:** The first-labelled time is kept; the last-changed time is updated. Only the latest label is stored.

### UC-16 Relate the session task to browser content
- **Actor:** Aggregation pipeline
- **Goal:** Give the future classifier context about whether the content fits the declared intention.
- **Main flow:** For a session, the task category and each content category are combined into `related`, `unrelated` or `unknown` (e.g. studying + educational → related). Counts per category and relation are summarised per session.
- **Rule:** This is a model **input**, never a verdict. Music while coding or a lecture used to procrastinate are both possible, so no focus label is derived from it. The free-text session name is not part of the features.

---

## D. Planned use cases

The repository has placeholder packages for these (`baseline/`, `models/`,
`explain/`, `insights/`, `api/`, `dashboard/`).

### UC-17 Learn a personal baseline *(planned)*
- **Actor:** Focus classifier
- **Goal:** Learn what *normal* looks like for this user (typical switch rate, input intensity, idle patterns), so that features are judged relative to the user rather than a global average.

### UC-18 Classify focus state per window *(planned)*
- **Actor:** Focus classifier
- **Main flow:** Train on windows the user labelled (UC-14), then predict Focused / Distracted / Inactive for unlabelled windows, using window features (UC-12), session context (UC-16) and the personal baseline (UC-17).
- **Alternate flow:** Too few labels → the system asks the user to label more windows instead of guessing.

### UC-19 Explain a classification *(planned)*
- **Actor:** User
- **Goal:** Understand *why* a window was classified a certain way (e.g. "high switch rate and mostly entertainment content compared with your usual coding sessions").
- **Privacy:** Only structured summaries, never raw logs, may be sent to an external LLM for wording.

### UC-20 View focus insights and dashboard *(planned)*
- **Actor:** User
- **Goal:** See trends: focused time per session or day, typical distraction sources and the times of day the user focuses best.

---

## Privacy rules (apply to all use cases)

* Never collected: keystrokes, typed text, mouse coordinates, screenshots,
  screen content, URLs, search queries, or page titles outside the opt-in
  video case (UC-07).
* All data stays in the local SQLite database.
* Focus labels come only from the user. No component infers them from session
  names, categories or single signals.
* Details: [privacy.md](privacy.md).
