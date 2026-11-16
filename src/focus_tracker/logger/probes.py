"""OS probes: foreground application and idle time (Windows + macOS).

Only the *process/app name* of the foreground window is read - never the window
title (titles often contain document names / URLs, i.e. screen content).
The single exception is opt-in and lives in browser.get_active_title(): video /
blog platforms only, reduced to a category in memory and never stored.
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


_TICK_MASK = 0xFFFFFFFF


def idle_seconds_from_ticks(now_tick, last_input_tick):
    """Idle seconds from two 32-bit millisecond tick counts.

    GetLastInputInfo().dwTime and GetTickCount() are both 32-bit and wrap
    every ~49.7 days; masking the difference keeps the result correct across
    a wrap.  Pure function so it can be tested on any OS.
    """
    return ((now_tick - last_input_tick) & _TICK_MASK) / 1000.0


def _windows_idle_seconds():
    """Seconds since last keyboard/mouse input via Win32 GetLastInputInfo.

    Returns None if the call fails (caller then falls back).  Only a tick
    count is read - no key, no coordinates, no window content.  Note: this is
    per-session, and synthetic input (remote-desktop, automation tools) also
    resets it.
    """
    import ctypes
    from ctypes import wintypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
    user32.GetLastInputInfo.restype = wintypes.BOOL
    kernel32.GetTickCount.argtypes = []
    kernel32.GetTickCount.restype = wintypes.DWORD

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    return idle_seconds_from_ticks(kernel32.GetTickCount(), info.dwTime)


def get_idle_seconds(last_input_monotonic=None):
    """Seconds since last keyboard/mouse input (OS-level, content-free)."""
    try:
        if sys.platform == "win32":
            secs = _windows_idle_seconds()
            if secs is not None:
                return max(0.0, secs)
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
