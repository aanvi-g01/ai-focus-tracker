"""Idle-episode tracking: a RAW SIGNAL, not a judgement.

IDLE TIME vs INACTIVITY CLASSIFICATION
--------------------------------------
* Idle time (this module) is a measurement: "N seconds since the last
  keyboard/mouse event". It says nothing about *why* there was no input.
* Inactivity classification (the 'Inactive' class in the synopsis) is a model
  output made later, per time window, from several features plus the user's
  personal baseline.  Long idle time can be reading a PDF, watching a lecture,
  thinking, or being away from the desk.  Short idle time can still be
  distracted (scrolling a feed).  So idle time is only ONE input feature.

This module therefore never produces a label such as 'distracted', 'focused'
or 'inactive'.  It only reports when an idle episode started and how long it
lasted, and it never sees what was typed or clicked - only a number of seconds.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IdleTransition:
    kind: str       # "started" | "ended"
    seconds: int    # total idle length for "ended"; 0 for "started"


class IdleTracker:
    """Turns a stream of 'seconds since last input' readings into episodes.

    An episode begins when the reading reaches `threshold` and ends when it
    drops back below it (i.e. new input arrived).  The reported duration is
    the longest reading seen during the episode, so it includes the threshold
    period itself and is accurate to within one polling interval.
    """

    def __init__(self, threshold: float = 60.0):
        self.threshold = threshold
        self.is_idle = False
        self._peak = 0.0

    def update(self, idle_seconds: float) -> Optional[IdleTransition]:
        idle = idle_seconds >= self.threshold
        if idle and not self.is_idle:
            self.is_idle = True
            self._peak = idle_seconds
            return IdleTransition("started", 0)
        if idle:
            self._peak = max(self._peak, idle_seconds)
            return None
        if self.is_idle:                      # input resumed
            return self._end()
        return None

    def close(self) -> Optional[IdleTransition]:
        """Call on shutdown so an in-progress episode is not lost."""
        return self._end() if self.is_idle else None

    def _end(self) -> IdleTransition:
        duration = int(round(self._peak))
        self.is_idle = False
        self._peak = 0.0
        return IdleTransition("ended", duration)
