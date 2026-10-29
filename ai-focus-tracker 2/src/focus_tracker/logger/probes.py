"""OS probes: foreground application and idle time (Windows + macOS).

Only the *process/app name* of the foreground window is read - never the window
title (titles often contain document names / URLs, i.e. screen content).
"""
import sys
import time


def get_active_app():
    """Name of the foreground application, or None if unavailable."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            import psutil
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return psutil.Process(pid.value).name()
        if sys.platform == "darwin":
            from AppKit import NSWorkspace  # pip: pyobjc-framework-Cocoa
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return str(app.localizedName()) if app else None
    except Exception:
        return None
    return None


def get_idle_seconds(last_input_monotonic=None):
    """Seconds since last keyboard/mouse input."""
    try:
        if sys.platform == "win32":
            import ctypes

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

            info = LASTINPUTINFO()
            info.cbSize = ctypes.sizeof(info)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
                millis = (ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF
                return max(0.0, millis / 1000.0)
        elif sys.platform == "darwin":
            import Quartz  # pip: pyobjc-framework-Quartz
            return float(Quartz.CGEventSourceSecondsSinceLastEventType(
                Quartz.kCGEventSourceStateCombinedSessionState,
                Quartz.kCGAnyInputEventType))
    except Exception:
        pass
    # Fallback: derive from our own input counter timestamps
    if last_input_monotonic is not None:
        return max(0.0, time.monotonic() - last_input_monotonic)
    return 0.0
