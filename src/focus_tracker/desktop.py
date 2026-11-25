"""Desktop launcher: the entry point of FocusTracker.exe.

Double-click behaviour, for non-technical users:
* Data lives in %LOCALAPPDATA%\\FocusTracker (activity.db, team.db,
  settings.json, app.log) - not next to the .exe, which may sit in Downloads.
* If the app is already running, it just opens the page again.
* First launch creates a desktop shortcut.
* No console window: output goes to app.log. Errors show a message box.
"""
import argparse
import os
import sys
import threading
import traceback
import urllib.request
import webbrowser
from pathlib import Path

PORT = 8765


def home_dir() -> Path:
    override = os.environ.get("FOCUS_TRACKER_HOME")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "FocusTracker"


def static_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "dashboard_dist"
    return Path(__file__).resolve().parents[2] / "dashboard" / "dist"


def model_roots(home: Path):
    """Where to look for the prediction model, newest wins: a model dropped into
    <home>/models overrides the one bundled in the exe (no rebuild needed)."""
    roots = [home / "models"]
    if getattr(sys, "frozen", False):
        roots.append(Path(sys._MEIPASS) / "model")
    else:
        roots.append(Path(__file__).resolve().parents[2] / "models")
    return roots


def message_box(text, title="Focus Tracker"):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, str(text), title, 0x10)
    except Exception:
        print(text, file=sys.stderr)


def already_running(port) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def port_free(port) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _redirect_output(home: Path):
    """Windowed .exe has no console: send prints and tracebacks to app.log."""
    if sys.stdout is None or sys.stderr is None or getattr(sys, "frozen", False):
        log = open(home / "app.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log


def main(argv=None):
    p = argparse.ArgumentParser(prog="FocusTracker")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-shortcut", action="store_true", help="for testing builds")
    args = p.parse_args(argv)
    url = f"http://127.0.0.1:{args.port}"

    home = home_dir()
    home.mkdir(parents=True, exist_ok=True)
    _redirect_output(home)

    if already_running(args.port):
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    if not port_free(args.port):
        message_box(f"Focus Tracker cannot start: port {args.port} is used by another "
                    "program. Restart the computer and try again.")
        return 1

    try:
        from .api.app import create_app, run_server
        from .api.settings import ensure_desktop_shortcut

        if not args.no_shortcut:
            ensure_desktop_shortcut()
        app = create_app(home / "activity.db", home / "team.db",
                         static_dir=static_dir(), settings_path=home / "settings.json",
                         model_dir=model_roots(home))
        print(f"Focus Tracker starting at {url}; data in {home}")
        if not args.no_browser:
            threading.Timer(1.5, webbrowser.open, args=(url,)).start()
        run_server(app, args.port, log_config=None)
        return 0
    except Exception:
        traceback.print_exc()
        message_box("Focus Tracker stopped because of an error. Please send the file\n"
                    f"{home / 'app.log'}\nto your team.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
