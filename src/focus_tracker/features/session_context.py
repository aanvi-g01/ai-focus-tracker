"""Session context features: declared task x browser content category.

These are INPUTS for the future focus-state classifier, not outputs of it.
`relation()` only says whether a page category is one that commonly serves a
declared task ("related"), commonly does not ("unrelated"), or cannot be judged
("unknown"). It deliberately never says focused/distracted: a lecture on
youtube.com can be procrastination, and music on spotify.com can accompany
deep work. The classifier weighs this together with idle time, switching
rate, input counts and the user's personal baseline.

Nothing here is stored; it is derived on demand from the raw event rows.
"""
from collections import Counter
from datetime import datetime

from ..logger.session import SessionManager

RELATIONS = ("related", "unrelated", "unknown")

# Content categories that commonly serve each declared task category.
_RELATED = {
    "studying": {"educational", "documentation", "search"},
    "coding": {"development", "documentation", "educational", "search"},
    "research": {"educational", "documentation", "search", "news"},
    "writing": {"documentation", "educational", "search"},
}
# Categories that are informative enough to call "unrelated" when not related.
_JUDGEABLE = {"educational", "documentation", "development", "entertainment",
              "social", "news", "shopping", "search"}


def relation(task_category, content_category):
    """'related' | 'unrelated' | 'unknown' - a context feature, never a label."""
    related = _RELATED.get(task_category)
    if related is None or content_category not in _JUDGEABLE:
        return "unknown"   # task 'other', no category, 'other', 'communication'
    return "related" if content_category in related else "unrelated"


def session_context(store, session_id, now=None):
    """Context summary for one session, from its own event rows.

    Counts are numbers of logged browser rows (transitions and periodic input
    counts), a cheap proxy for exposure - not durations and not judgements.
    """
    s = SessionManager(store).get(session_id)
    if s is None:
        raise KeyError(f"no session {session_id}")
    categories, relations = Counter(), Counter()
    for (category,) in store.conn.execute(
            "SELECT content_category FROM events WHERE session_id = ?"
            " AND content_category IS NOT NULL ORDER BY id", (session_id,)):
        categories[category] += 1
        relations[relation(s.task_category, category)] += 1
    return {
        "session_id": s.id,
        "task_category": s.task_category,
        "status": s.status,
        "active_seconds": s.active_seconds(now or datetime.now()),
        "content_category_counts": dict(categories),
        "relation_counts": {r: relations.get(r, 0) for r in RELATIONS},
    }
