"""Activity-logging agent (Week 1-2).

Records ONLY: active app, window switches, idle time, input-event counts.
"""
import argparse
import signal
import time
from datetime import datetime

from .idle import IdleTracker
from .input_counter import InputCounter
from .probes import get_active_app, get_idle_seconds
from .storage import EventStore

POLL_SECONDS = 1.0
FLUSH_SECONDS = 5.0       # how often input counts are written
IDLE_THRESHOLD = 20.0     # seconds without input => 'idle'


class Agent:
    def __init__(self, store, counter, get_app=get_active_app,
                 get_idle=None, idle_threshold=IDLE_THRESHOLD,
                 flush_seconds=FLUSH_SECONDS):
        self.store, self.counter = store, counter
        self.get_app = get_app
        self.get_idle = get_idle or (lambda: get_idle_seconds(counter.last_input))
        self.idle_threshold = idle_threshold
        self.idle = IdleTracker(idle_threshold)
        self.flush_seconds = flush_seconds
        self.current_app = None
        self._last_flush = None

    @property
    def is_idle(self):
        return self.idle.is_idle

    def step(self, now: datetime):
        """One polling tick. Pure logic -> easy to unit test."""
        if self._last_flush is None:
            self._last_flush = now

        app = self.get_app()
        if app and app != self.current_app:
            event = "active" if self.current_app is None else "switched"
            self.store.log(now, app, event)
            self.current_app = app

        # Idle time is a raw signal only: we record that an idle episode began
        # and how long it lasted. Nothing here labels the user (no 'distracted').
        change = self.idle.update(self.get_idle())
        if change is not None:
            if change.kind == "started":
                self.store.log(now, self.current_app, "idle")
            else:
                self.store.log(now, self.current_app, "idle_duration", change.seconds)
                self.store.log(now, self.current_app, "active")

        if (now - self._last_flush).total_seconds() >= self.flush_seconds:
            keys, mouse = self.counter.drain()
            if keys:
                self.store.log(now, self.current_app, "keyboard_count", keys)
            if mouse:
                self.store.log(now, self.current_app, "mouse_count", mouse)
            self._last_flush = now

    def finish(self, now: datetime):
        """Record an idle episode still in progress (e.g. on shutdown)."""
        change = self.idle.close()
        if change is not None:
            self.store.log(now, self.current_app, "idle_duration", change.seconds)

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
    args = p.parse_args()

    store = EventStore(args.db)
    counter = InputCounter()
    counter.start()
    print(f"Logging to {args.db} (app names, switches, idle, input COUNTS only). Ctrl+C to stop.")
    try:
        Agent(store, counter, idle_threshold=args.idle_threshold).run()
    finally:
        counter.stop()
        store.close()


if __name__ == "__main__":
    main()
