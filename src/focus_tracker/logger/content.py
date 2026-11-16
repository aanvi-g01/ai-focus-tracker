"""Broad content category of the foreground browser page, computed LOCALLY.

Privacy:
- The category is one word from CONTENT_CATEGORIES, derived from the hostname
  alone, except for a short list of mixed-content platforms (TITLE_HOSTS, e.g.
  youtube.com) where the hostname says nothing about the content.
- Page titles are read only when the user opted in (--read-titles) and only on
  TITLE_HOSTS and VIDEO_HOSTS. On video-streaming platforms (VIDEO_HOSTS) the
  cleaned video title is stored; on the other TITLE_HOSTS (medium.com, dev.to)
  it is used for the category in memory and dropped. Titles of mail, search,
  banking or any unlisted site are never read.
- Nothing here labels the user as focused or distracted: "entertainment" is a
  description of the page, not of the person.
"""
import re

CONTENT_CATEGORIES = frozenset({
    "educational", "documentation", "development", "entertainment", "social",
    "news", "shopping", "communication", "search", "other",
})

# hostname suffix -> category (matches the host itself and any subdomain)
_HOST_CATEGORIES = {
    "development": (
        "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
        "stackexchange.com", "leetcode.com", "hackerrank.com", "codeforces.com",
        "codechef.com", "replit.com", "codepen.io", "localhost",
    ),
    "documentation": (
        "docs.python.org", "developer.mozilla.org", "readthedocs.io",
        "learn.microsoft.com", "devdocs.io", "cppreference.com",
        "docs.oracle.com", "pkg.go.dev", "docs.rs",
    ),
    "educational": (
        "wikipedia.org", "coursera.org", "edx.org", "khanacademy.org",
        "udemy.com", "nptel.ac.in", "geeksforgeeks.org", "w3schools.com",
        "arxiv.org", "scholar.google.com", "brilliant.org", "edu", "ac.in", "ac.uk",
    ),
    "entertainment": (
        "netflix.com", "primevideo.com", "hotstar.com", "disneyplus.com",
        "twitch.tv", "spotify.com", "9gag.com", "crunchyroll.com",
    ),
    "social": (
        "facebook.com", "instagram.com", "twitter.com", "x.com", "reddit.com",
        "linkedin.com", "tiktok.com", "pinterest.com", "snapchat.com", "threads.net",
    ),
    "news": (
        "news.google.com", "news.ycombinator.com", "bbc.com", "bbc.co.uk",
        "cnn.com", "nytimes.com", "theguardian.com", "reuters.com", "ndtv.com",
        "indiatimes.com", "thehindu.com",
    ),
    "shopping": (
        "amazon.com", "amazon.in", "flipkart.com", "ebay.com", "myntra.com",
        "aliexpress.com",
    ),
    "communication": (
        "mail.google.com", "outlook.live.com", "outlook.office.com",
        "web.whatsapp.com", "slack.com", "discord.com", "teams.microsoft.com",
        "meet.google.com", "zoom.us", "web.telegram.org",
    ),
    "search": (
        "google.com", "bing.com", "duckduckgo.com", "search.yahoo.com",
        "search.brave.com", "ecosia.org",
    ),
}

# Mixed-content platforms: the category is decided by the page title.
TITLE_HOSTS = ("youtube.com", "vimeo.com", "medium.com", "dev.to", "dailymotion.com")

# Video-streaming platforms: the (cleaned) video title is STORED in
# events.content_title when the user opted in with --read-titles.
VIDEO_HOSTS = (
    "youtube.com", "vimeo.com", "dailymotion.com", "twitch.tv", "netflix.com",
    "primevideo.com", "hotstar.com", "disneyplus.com", "crunchyroll.com",
)
MAX_TITLE_CHARS = 200

_BROWSER_SUFFIX = re.compile(
    r"\s+[-–—]\s+(?:google chrome|microsoft edge|mozilla firefox|brave|"
    r"opera|vivaldi)$", re.I)
_MORE_TABS = re.compile(r"\s+and \d+ more pages?$", re.I)   # Edge tab groups
_PLATFORM_SUFFIX = re.compile(
    r"\s+[-–—|]\s+(?:youtube|vimeo|dailymotion|twitch|netflix|prime video|"
    r"disney\+ hotstar|hotstar|disney\+|crunchyroll)$", re.I)
_NOTIFICATIONS = re.compile(r"^\(\d+\+?\)\s*")                # "(3) Video title"
_PLATFORM_NAMES = {"youtube", "vimeo", "dailymotion", "twitch", "netflix",
                   "prime video", "hotstar", "disney+ hotstar", "disney+", "crunchyroll"}

_TITLE_KEYWORDS = {
    "educational": (
        "lecture", "tutorial", "course", "lesson", "explained", "explanation",
        "algorithm", "algorithms", "data structure", "data structures", "dsa",
        "leetcode", "introduction to", "intro to", "how to", "learn", "learning",
        "crash course", "full course", "chapter", "exam", "interview prep",
        "programming", "python", "java", "javascript", "c\\+\\+", "math", "maths",
        "physics", "chemistry", "calculus", "statistics", "machine learning",
        "nptel", "opencourseware", "documentation", "guide",
    ),
    "entertainment": (
        "official video", "music video", "official audio", "trailer", "vlog",
        "funny", "prank", "gameplay", "highlights", "reaction", "movie",
        "episode", "meme", "memes", "lyrics", "song", "comedy", "shorts",
        "full match", "live stream", "unboxing", "asmr",
    ),
}
_TITLE_PATTERNS = {
    cat: re.compile(r"(?<![a-z0-9])(?:" + "|".join(words) + r")(?![a-z0-9])")
    for cat, words in _TITLE_KEYWORDS.items()
}


def _matches(host, suffix):
    return host == suffix or host.endswith("." + suffix)


def _title_decides_category(host):
    return bool(host) and any(_matches(host, h) for h in TITLE_HOSTS)


def is_video_host(host):
    """True for video-streaming platforms, whose video title may be stored."""
    return bool(host) and any(_matches(host, h) for h in VIDEO_HOSTS)


def needs_title(host):
    """True only for hosts whose title is used at all (category or storage)."""
    return _title_decides_category(host) or is_video_host(host)


def clean_title(raw):
    """Reduce a browser window title to the video/page title itself.

    Drops the browser name, Edge's "and N more pages", the platform suffix
    (" - YouTube"), unread-notification counts and control characters; caps
    the length. Returns None for an empty title or a bare platform page
    (e.g. the YouTube home page, titled just "YouTube").
    """
    if not raw or not isinstance(raw, str):
        return None
    t = "".join(ch for ch in raw if ch.isprintable() or ch.isspace())
    t = " ".join(t.split())
    t = _BROWSER_SUFFIX.sub("", t)
    t = _MORE_TABS.sub("", t)
    t = _PLATFORM_SUFFIX.sub("", t)
    t = _NOTIFICATIONS.sub("", t)
    t = t[:MAX_TITLE_CHARS].strip()
    if not t or t.lower() in _PLATFORM_NAMES:
        return None
    return t


def is_clean_title(value):
    """Storage-side check: a single printable line within the length cap."""
    return (isinstance(value, str) and 0 < len(value) <= MAX_TITLE_CHARS
            and value == value.strip() and all(ch.isprintable() for ch in value))


def category_from_title(title):
    """Keyword vote over a title held in memory only. Returns a category."""
    if not title or not isinstance(title, str):
        return "other"
    text = title.lower()
    hits = {cat: len(p.findall(text)) for cat, p in _TITLE_PATTERNS.items()}
    best = max(hits.values())
    winners = [cat for cat, n in hits.items() if n == best]
    return winners[0] if best and len(winners) == 1 else "other"


def classify_content(host, title=None):
    """Category for a hostname (+ title for TITLE_HOSTS only).

    Returns None when unknown: no hostname, or a mixed-content platform whose
    title was not available (not read, or reading failed).
    """
    if not host:
        return None
    if _title_decides_category(host):
        return category_from_title(title) if title else None
    # Most specific rule wins: 'mail.google.com' before 'google.com'.
    best, best_len = "other", -1
    for cat, suffixes in _HOST_CATEGORIES.items():
        for s in suffixes:
            if len(s) > best_len and _matches(host, s):
                best, best_len = cat, len(s)
    return best
