"""Activity-logging agent (Week 1-2).

Records ONLY: active app, window switches, idle time, input-event counts and,
when the foreground app is a browser, the website HOSTNAME (never the path,
query, page title or anything done on the page).
"""
import argparse
import signal
import time
from datetime import datetime

from .browser import get_active_site, is_browser, normalize_site, uia_available
from .idle import IdleTracker
from .input_counter import InputCounter
from .probes import get_active_app, get_idle_seconds
from .storage import EventStore

POLL_SECONDS = 1.0
FLUSH_SECONDS = 5.0       # how often input counts are written
IDLE_THRESHOLD = 60.0     # seconds without input => 'idle'
SITE_POLL_SECONDS = 2.0   # how often the browser address bar is read


class Agent:
    def __init__(self, store, counter, get_app=get_active_app,
                 get_idle=None, idle_threshold=IDLE_THRESHOLD,
                 flush_seconds=FLUSH_SECONDS,
                 get_site=None, site_poll_seconds=SITE_POLL_SECONDS):
        """`get_site=None` disables website tracking (the default, so unit
        tests never touch the real desktop); main() passes get_active_site."""
        self.store, self.counter = store, counter
        self.get_app = get_app
        self.get_idle = get_idle or (lambda: get_idle_seconds(counter.last_input))
        self.idle_threshold = idle_threshold
        self.idle = IdleTracker(idle_threshold)
        self.flush_seconds = flush_seconds
        self.get_site = get_site
        self.site_poll_seconds = site_poll_seconds
        self.current_app = None
        self.current_site = None      # hostname as read last time (None = unknown)
        self._last_site = None        # last *readable* hostname in this browser stay
        self._site_checked_at = None
        self._last_flush = None

    @property
    def is_idle(self):
        return self.idle.is_idle

    # -- logging: every row is stamped with the current site while in a browser --
    def _log(self, now, event, value=1):
        site = self.current_site if is_browser(self.current_app) else None
        self.store.log(now, self.current_app, event, value, website=site)

    def _update_site(self, now, app, app_changed):
        """Refresh the current site. Returns True if the site changed inside
        the same browser (the caller logs 'site_switched')."""
        if self.get_site is None or not is_browser(app):
            self.current_site = self._last_site = self._site_checked_at = None
            return False
        due = (app_changed or self._site_checked_at is None or
               (now - self._site_checked_at).total_seconds() >= self.site_poll_seconds)
        if not due:
            return False
        self._site_checked_at = now
        try:
            site = normalize_site(self.get_site())   # re-sanitise: hostname only
        except Exception:
            site = None
        self.current_site = site
        if app_changed:
            self._last_site = site
            return False
        # An unreadable address bar (typing, new tab, internal page) is "unknown",
        # not a switch: only a *different readable* site counts as a switch.
        changed = (site is not None and self._last_site is not None
                   and site != self._last_site)
        if site is not None:
            self._last_site = site
        return changed

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
        elif app and self._update_site(now, app, app_changed=False):
            self._log(now, "site_switched")

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

    def finish(self, now: datetime):
        """Record an idle episode still in progress (e.g. on shutdown)."""
        change = self.idle.close()
        if change is not None:
            self._log(now, "idle_duration", change.seconds)

    def run(self):
        self.store.log(datetime.now(), None, "agent_start")
        stop = {"flag": False}
        signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
        try:
            while not stop["flag"]:
                self.step(datetime.now())
                time.sleep(POLL_SECONDS)
        finally:
            self.finish(datetime.now())
            self.store.log(datetime.now(), None, "agent_stop")


def main():
    p = argparse.ArgumentParser(description="AI Focus Tracker activity logger")
    p.add_argument("--db", default="data/activity.db")
    p.add_argument("--idle-threshold", type=float, default=IDLE_THRESHOLD)
    p.add_argument("--no-websites", action="store_true",
                   help="do not read the browser address bar (website column stays NULL)")
    p.add_argument("--site-poll", type=float, default=SITE_POLL_SECONDS,
                   help="seconds between address-bar reads (default 2)")
    args = p.parse_args()

    use_sites = not args.no_websites
    if use_sites and not uia_available():
        print("WARNING: 'uiautomation' is not installed, so websites cannot be read "
              "(pip install uiautomation). The website column will stay NULL.")

    store = EventStore(args.db)
    counter = InputCounter()
    counter.start()
    what = "app names, switches, idle, input COUNTS" + (
        ", browser site HOSTNAME" if use_sites else "") + " only"
    print(f"Logging to {args.db} ({what}). Ctrl+C to stop.")
    try:
        Agent(store, counter, idle_threshold=args.idle_threshold,
              get_site=get_active_site if use_sites else None,
              site_poll_seconds=args.site_poll).run()
    finally:
        counter.stop()
        store.close()


if __name__ == "__main__":
    main()
