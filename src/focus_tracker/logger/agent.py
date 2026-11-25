"""Activity-logging agent (Week 1-2).

Records ONLY: active app, window switches, idle time, input-event counts and,
when the foreground app is a browser, the website HOSTNAME (never the path,
query, page title or anything done on the page) and a one-word content
CATEGORY derived locally from it (see content.py).  Every row is also stamped
with the id of the focus session active at that moment (see session.py).

While it runs, the agent also rebuilds the `windows` table every window length
(default 5 min) counted from its own start, and once more when it stops, so
windows anchored at the start time appear automatically
(see features/windows.py).
"""
import argparse
import signal
import sys
import time
from datetime import datetime, timedelta

from ..features.windows import (DEFAULT_WINDOW_SECONDS, refresh_windows,
                                validate_window_seconds)
from .browser import (get_active_site, get_active_title, is_browser,
                      normalize_site, uia_available)
from .content import classify_content, clean_title, is_video_host, needs_title
from .idle import IdleTracker
from .input_counter import InputCounter
from .probes import get_active_app, get_idle_seconds
from .session import STALE_AFTER_SECONDS, SessionManager
from .storage import EventStore

POLL_SECONDS = 1.0
FLUSH_SECONDS = 5.0       # how often input counts are written
IDLE_THRESHOLD = 60.0     # seconds without input => 'idle'
SITE_POLL_SECONDS = 2.0   # how often the browser address bar is read


class Agent:
    def __init__(self, store, counter, get_app=get_active_app,
                 get_idle=None, idle_threshold=IDLE_THRESHOLD,
                 flush_seconds=FLUSH_SECONDS,
                 get_site=None, site_poll_seconds=SITE_POLL_SECONDS,
                 get_title=None, window_seconds=None):
        """`get_site=None` disables website tracking (the default, so unit
        tests never touch the real desktop); main() passes get_active_site.
        `get_title=None` (default) never reads a page title; with it, titles of
        mixed-content / video platforms only are read: used for the category,
        and stored as content_title on video-streaming platforms.
        `window_seconds=None` (default) never touches the `windows` table;
        with it, begin() schedules a refresh every window_seconds."""
        self.store, self.counter = store, counter
        self.get_app = get_app
        self.get_idle = get_idle or (lambda: get_idle_seconds(counter.last_input))
        self.idle_threshold = idle_threshold
        self.idle = IdleTracker(idle_threshold)
        self.flush_seconds = flush_seconds
        self.get_site = get_site
        self.site_poll_seconds = site_poll_seconds
        self.get_title = get_title
        self.current_app = None
        self.current_site = None      # hostname as read last time (None = unknown)
        self._last_site = None        # last *readable* hostname in this browser stay
        self.current_category = None  # content category of current_site
        self._last_category = None    # last known category of _last_site
        self.current_title = None     # video title (video platforms only)
        self._last_title = None
        self._site_checked_at = None
        self._last_flush = None
        self.window_seconds = (validate_window_seconds(window_seconds)
                               if window_seconds is not None else None)
        self._next_refresh = None     # next window boundary (set by begin())

    @property
    def is_idle(self):
        return self.idle.is_idle

    # -- logging: every row is stamped with the current site while in a browser --
    def _log(self, now, event, value=1):
        in_browser = is_browser(self.current_app)
        self.store.log(now, self.current_app, event, value,
                       website=self.current_site if in_browser else None,
                       content_category=self.current_category if in_browser else None,
                       content_title=self.current_title if in_browser else None,
                       session_id=self.store.active_session_id())

    def _categorize(self, site):
        """(category, title_to_store) for `site`. A title is kept only for
        video-streaming platforms; for other hosts it never leaves here."""
        if site is None:
            return None, None
        title = None
        if self.get_title is not None and needs_title(site):
            try:
                title = clean_title(self.get_title())
            except Exception:
                title = None
        return classify_content(site, title), (title if is_video_host(site) else None)

    def _update_site(self, now, app, app_changed):
        """Refresh the current site and category. Returns the event to log for
        a change inside the same browser ('site_switched' / 'content_changed')
        or None."""
        if self.get_site is None or not is_browser(app):
            self.current_site = self._last_site = self._site_checked_at = None
            self.current_category = self._last_category = None
            self.current_title = self._last_title = None
            return None
        due = (app_changed or self._site_checked_at is None or
               (now - self._site_checked_at).total_seconds() >= self.site_poll_seconds)
        if not due:
            return None
        self._site_checked_at = now
        try:
            site = normalize_site(self.get_site())   # re-sanitise: hostname only
        except Exception:
            site = None
        category, title = self._categorize(site)
        self.current_site, self.current_category, self.current_title = site, category, title
        if app_changed:
            self._last_site, self._last_category, self._last_title = site, category, title
            return None
        # An unreadable address bar (typing, new tab, internal page) is "unknown",
        # not a switch: only a *different readable* site counts as a switch.
        if site is None:
            return None
        event = None
        if self._last_site is not None and site != self._last_site:
            event = "site_switched"
        elif site == self._last_site and (
                (category is not None and self._last_category is not None
                 and category != self._last_category) or
                (title is not None and self._last_title is not None
                 and title != self._last_title)):
            event = "content_changed"   # e.g. a different video on the same site
        # Unreadable values (None) on the same site keep the last known ones.
        if site != self._last_site or category is not None:
            self._last_category = category
        if site != self._last_site or title is not None:
            self._last_title = title
        self._last_site = site
        return event

    def step(self, now: datetime):
        """One polling tick. Pure logic -> easy to unit test."""
        if self._last_flush is None:
            self._last_flush = now

        app = self.get_app()
        if app and app != self.current_app:
            event = "active" if self.current_app is None else "switched"
            self.current_app = app
            self._update_site(now, app, app_changed=True)
            self._log(now, event)
        elif app:
            event = self._update_site(now, app, app_changed=False)
            if event:
                self._log(now, event)

        # Idle time is a raw signal only: we record that an idle episode began
        # and how long it lasted. Nothing here labels the user (no 'distracted').
        change = self.idle.update(self.get_idle())
        if change is not None:
            if change.kind == "started":
                self._log(now, "idle")
            else:
                self._log(now, "idle_duration", change.seconds)
                self._log(now, "active")

        if (now - self._last_flush).total_seconds() >= self.flush_seconds:
            keys, mouse = self.counter.drain()
            if keys:
                self._log(now, "keyboard_count", keys)
            if mouse:
                self._log(now, "mouse_count", mouse)
            self._last_flush = now

        if self._next_refresh is not None and now >= self._next_refresh:
            self.refresh_windows(now, live=True)
            step = timedelta(seconds=self.window_seconds)
            while self._next_refresh <= now:
                self._next_refresh += step

    def refresh_windows(self, now: datetime, live: bool):
        """Rebuild the windows table. A failure is reported, never fatal:
        raw events keep being logged and the next refresh rebuilds all."""
        if self.window_seconds is None:
            return
        try:
            refresh_windows(self.store, self.window_seconds, now, live=live)
        except Exception as e:
            print(f"WARNING: could not refresh windows: {e}", file=sys.stderr)

    def begin(self, now: datetime):
        """Mark the start of this run; windows are anchored at this moment."""
        self.store.log(now, None, "agent_start",
                       session_id=self.store.active_session_id())
        if self.window_seconds is not None:
            self._next_refresh = now + timedelta(seconds=self.window_seconds)

    def end(self, now: datetime):
        """Close the run, then rebuild windows so its last one is complete."""
        self.finish(now)
        self.store.log(now, None, "agent_stop",
                       session_id=self.store.active_session_id())
        self._next_refresh = None
        self.refresh_windows(now, live=False)

    def finish(self, now: datetime):
        """Record an idle episode still in progress (e.g. on shutdown)."""
        change = self.idle.close()
        if change is not None:
            self._log(now, "idle_duration", change.seconds)

    def run(self):
        self.begin(datetime.now())
        stop = {"flag": False}
        signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
        try:
            while not stop["flag"]:
                self.step(datetime.now())
                time.sleep(POLL_SECONDS)
        finally:
            self.end(datetime.now())


def main():
    p = argparse.ArgumentParser(description="AI Focus Tracker activity logger")
    p.add_argument("--db", default="data/activity.db")
    p.add_argument("--idle-threshold", type=float, default=IDLE_THRESHOLD)
    p.add_argument("--no-websites", action="store_true",
                   help="do not read the browser address bar (website column stays NULL)")
    p.add_argument("--site-poll", type=float, default=SITE_POLL_SECONDS,
                   help="seconds between address-bar reads (default 2)")
    p.add_argument("--read-titles", action="store_true",
                   help="opt-in: on video/blog platforms only, read the page title to "
                        "classify content; on video-streaming sites (youtube.com, "
                        "netflix.com, ...) the video title is also stored")
    p.add_argument("--stale-after", type=float, default=STALE_AFTER_SECONDS / 3600,
                   help="hours without any recorded activity after which a session "
                        "left open is auto-closed at its last activity (default 2)")
    p.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS,
                   help="length of the activity windows, built automatically from the "
                        "moment the logger starts (default 300 = 5 min; minimum 60)")
    args = p.parse_args()
    try:
        validate_window_seconds(args.window_seconds)
    except ValueError as e:
        p.error(str(e))

    use_sites = not args.no_websites
    if use_sites and not uia_available():
        print("WARNING: 'uiautomation' is not installed, so websites cannot be read "
              "(pip install uiautomation). The website column will stay NULL.")

    store = EventStore(args.db)
    recovered = SessionManager(store).recover_stale(
        datetime.now(), stale_after=args.stale_after * 3600)
    if recovered:
        print(f"Unfinished session #{recovered.id} {recovered.task!r} was auto-closed "
              f"at its last activity ({recovered.ended_at:%Y-%m-%d %H:%M}).")
    counter = InputCounter()
    counter.start()
    what = "app names, switches, idle, input COUNTS" + (
        ", browser site HOSTNAME + content category" if use_sites else "") + " only"
    print(f"Logging to {args.db} ({what}). Windows of {args.window_seconds}s are "
          f"built every {args.window_seconds}s from now. Ctrl+C to stop.")
    try:
        Agent(store, counter, idle_threshold=args.idle_threshold,
              get_site=get_active_site if use_sites else None,
              site_poll_seconds=args.site_poll,
              get_title=get_active_title if use_sites and args.read_titles else None,
              window_seconds=args.window_seconds).run()
    finally:
        counter.stop()
        store.close()


if __name__ == "__main__":
    main()
