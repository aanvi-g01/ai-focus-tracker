
"""
Browser awareness: hostname of the foreground browser tab.

Privacy:
- Stores only the hostname.
- Never stores scheme, port, credentials, path, query or fragment.
- Never reads page content, form fields or keystrokes.
"""

import ctypes
import importlib.util
import re
import sys
from urllib.parse import urlsplit


BROWSER_PROCESSES = frozenset({
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "vivaldi.exe",
})

ADDRESS_BAR_NAME = r"(?i).*address.*"

# Correct hostname validation.
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST_RE = re.compile(
    rf"^(?:localhost|{_LABEL}(?:\.{_LABEL})+)$"
)


def is_browser(app):
    if not app:
        return False

    return app.lower().strip() in BROWSER_PROCESSES


def is_hostname(value):
    """Check whether value is a valid bare hostname."""
    return (
        isinstance(value, str)
        and len(value) <= 253
        and bool(_HOST_RE.fullmatch(value))
    )


def normalize_site(raw):
    """
    Convert a browser address into a bare hostname.

    Returns None for search queries, internal browser pages,
    local files and invalid addresses.
    """

    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()

    if not raw or any(ch.isspace() for ch in raw):
        return None

    if "://" not in raw:
        raw = "//" + raw

    try:
        host = urlsplit(raw).hostname
    except ValueError:
        return None

    if not host:
        return None

    host = host.rstrip(".").lower()

    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None

    return host if is_hostname(host) else None


def uia_available():
    return importlib.util.find_spec("uiautomation") is not None


def _foreground_hwnd():
    return ctypes.windll.user32.GetForegroundWindow()


def _read_control_value(control):
    """Read only the accessibility ValuePattern of a control."""

    if not control.Exists(0, 0):
        return None

    try:
        pattern = control.GetValuePattern()
        return pattern.Value
    except Exception:
        return None


def _read_address_bar(hwnd):
    """
    Read the address bar through Windows UI Automation.
    Raw address exists only temporarily in memory.
    """

    import uiautomation as auto

    window = auto.ControlFromHandle(hwnd)

    # Different browsers may expose the address bar differently.
    control_types = [
        window.EditControl,
        window.ComboBoxControl,
    ]

    for control_type in control_types:
        try:
            bar = control_type(
                RegexName=ADDRESS_BAR_NAME,
                searchDepth=10
            )

            value = _read_control_value(bar)

            if value:
                return value

        except Exception:
            continue

    return None


def get_active_site():
    """
    Return the hostname of the foreground browser tab.
    Return None if unavailable.
    """

    if sys.platform != "win32":
        return None

    if not uia_available():
        return None

    try:
        hwnd = _foreground_hwnd()

        if not hwnd:
            return None

        raw_address = _read_address_bar(hwnd)

        # Immediately reduce the address to its hostname.
        return normalize_site(raw_address)

    except Exception:
        return None


def _probe():
    """Manually test browser hostname detection."""

    import time
    from .probes import get_active_app

    print("UI Automation installed:", uia_available())
    print("Click your browser window within 5 seconds...")

    time.sleep(5)

    app = get_active_app()

    print("Foreground app:", app)
    print("Is browser:", is_browser(app))

    if is_browser(app):
        site = get_active_site()
        print("Detected hostname:", site or "NULL")
    else:
        print("Not a supported browser.")


if __name__ == "__main__":
    _probe()
