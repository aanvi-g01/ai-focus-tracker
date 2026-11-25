"""Run the activity logger in a background thread of the local app.

Uses only the public Agent API (begin / step / end), built the same way as
`focus-logger` (logger/agent.py main()). Agent.run() is not used because its
Ctrl+C handler can only be installed on the main thread; a threading.Event
stops the loop instead, so agent_stop and the final window refresh still
happen exactly as with the CLI.

A lock file next to the database (pid of the owning process) prevents two
loggers in this app from writing the same database. It cannot stop someone
also running the `focus-logger` CLI by hand - the UI warns about that.
"""
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..features.windows import DEFAULT_WINDOW_SECONDS
from ..logger.agent import POLL_SECONDS
from ..logger.session import SessionManager
from ..logger.storage import EventStore


class LoggerError(RuntimeError):
    """Cannot start / stop the logger in the requested way."""


def default_agent_factory(store, websites: bool, read_titles: bool,
                          window_seconds: int = DEFAULT_WINDOW_SECONDS):
    """(agent, cleanup) built like `focus-logger` does. Imports lazily so tests
    never touch the real keyboard, mouse or browser."""
    from ..logger.agent import Agent
    from ..logger.browser import get_active_site, get_active_title, uia_available
    from ..logger.input_counter import InputCounter

    use_sites = websites and uia_available()
    counter = InputCounter()
    counter.start()
    agent = Agent(store, counter,
                  get_site=get_active_site if use_sites else None,
                  get_title=get_active_title if use_sites and read_titles else None,
                  window_seconds=window_seconds)
    return agent, counter.stop


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True   # cannot tell: be safe, treat the lock as held


class LoggerController:
    def __init__(self, db_path, agent_factory: Callable = default_agent_factory,
                 lock_path=None, clock: Callable[[], datetime] = datetime.now,
                 poll_seconds: float = POLL_SECONDS):
        self.db_path = str(db_path)
        self.lock_path = Path(lock_path) if lock_path else \
            Path(self.db_path).with_name("logger.lock")
        self.agent_factory = agent_factory
        self.clock = clock
        self.poll_seconds = poll_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._mutex = threading.Lock()
        self.started_at: Optional[datetime] = None
        self.options = {"websites": False, "read_titles": False}
        self.last_error: Optional[str] = None
        self.recovered: Optional[str] = None

    # -- lock file -------------------------------------------------------
    def _acquire_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                pid = int(self.lock_path.read_text().strip())
            except ValueError:
                pid = None
            if pid is not None and pid != os.getpid() and _pid_alive(pid):
                raise LoggerError(f"another logger (process {pid}) is using this database")
        self.lock_path.write_text(str(os.getpid()))

    def _release_lock(self):
        try:
            if self.lock_path.exists() and \
                    self.lock_path.read_text().strip() == str(os.getpid()):
                self.lock_path.unlink()
        except OSError:
            pass

    # -- lifecycle -------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, websites: bool = True, read_titles: bool = False):
        with self._mutex:
            if self.running:
                raise LoggerError("the logger is already running")
            self._acquire_lock()
            self._stop.clear()
            self.options = {"websites": bool(websites), "read_titles": bool(read_titles)}
            self.last_error = self.recovered = None
            ready = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(ready,),
                                            name="focus-logger", daemon=True)
            self._thread.start()
            ready.wait(timeout=10)
            if self.last_error and not self.running:
                raise LoggerError(self.last_error)

    def stop(self, timeout: float = 15):
        with self._mutex:
            if not self.running:
                raise LoggerError("the logger is not running")
            self._stop.set()
            self._thread.join(timeout)

    def shutdown(self):
        """Stop if running (server exit); never raises."""
        if self.running:
            try:
                self.stop()
            except LoggerError:
                pass

    def status(self) -> dict:
        return {"running": self.running,
                "started_at": self.started_at.isoformat(timespec="seconds")
                if self.running and self.started_at else None,
                **self.options, "last_error": self.last_error,
                "recovered_session": self.recovered}

    def _run(self, ready: threading.Event):
        store = agent = cleanup = None
        try:
            store = EventStore(self.db_path)   # own connection: sqlite is per-thread
            now = self.clock()
            rec = SessionManager(store).recover_stale(now)
            if rec:
                self.recovered = (f"Unfinished session #{rec.id} was auto-closed at its "
                                  f"last activity ({rec.ended_at:%Y-%m-%d %H:%M}).")
            agent, cleanup = self.agent_factory(store, **self.options)
            agent.begin(now)
            self.started_at = now
            ready.set()
            while not self._stop.is_set():
                agent.step(self.clock())
                self._stop.wait(self.poll_seconds)
        except Exception as e:                       # report, never crash the server
            self.last_error = f"{type(e).__name__}: {e}"
        finally:
            try:
                if agent is not None:
                    agent.end(self.clock())
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
            if cleanup:
                cleanup()
            if store is not None:
                store.close()
            self._release_lock()
            self.started_at = None
            ready.set()
