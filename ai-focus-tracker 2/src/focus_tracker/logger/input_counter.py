"""Counts input events. It NEVER inspects which key was pressed.

The callbacks deliberately ignore their arguments, so key identity is never
read, held in memory or written anywhere.
"""
import threading
import time


class InputCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._keys = 0
        self._mouse = 0
        self.last_input = time.monotonic()
        self._listeners = []

    # -- callbacks: arguments intentionally unused --
    def _on_key(self, *_args, **_kw):
        with self._lock:
            self._keys += 1
            self.last_input = time.monotonic()

    def _on_mouse(self, *_args, **_kw):
        with self._lock:
            self._mouse += 1
            self.last_input = time.monotonic()

    def start(self):
        from pynput import keyboard, mouse  # lazy: keeps tests headless-safe
        self._listeners = [
            keyboard.Listener(on_press=self._on_key),
            mouse.Listener(on_click=self._on_mouse, on_scroll=self._on_mouse,
                           on_move=self._on_mouse),
        ]
        for l in self._listeners:
            l.start()

    def stop(self):
        for l in self._listeners:
            l.stop()

    def drain(self):
        """Return (key_count, mouse_count) since last call and reset."""
        with self._lock:
            k, m = self._keys, self._mouse
            self._keys = self._mouse = 0
        return k, m
