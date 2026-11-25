"""Small per-user app settings (settings.json next to the database) and the
Windows shortcuts used by the desktop build.

Settings: the participant pseudonym (asked once on first run), whether the
first-run screen was completed, and whether to start with Windows.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from .export import validate_pseudonym

DEFAULTS = {"participant": "", "onboarded": False, "autostart": False,
            # LLM explanations (Groq). The key stays in this file on this PC and is
            # never returned by the API; GROQ_API_KEY in the environment also works.
            "llm_api_key": "", "llm_model": ""}
SECRET_KEYS = ("llm_api_key",)
APP_NAME = "Focus Tracker"


class SettingsStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return {k: data.get(k, v) for k, v in DEFAULTS.items()}

    def llm_key(self) -> str:
        return (self.load().get("llm_api_key") or os.environ.get("GROQ_API_KEY") or "").strip()

    def public(self, data=None) -> dict:
        """Settings safe to send to the browser: secrets replaced by flags."""
        data = dict(self.load() if data is None else data)
        for k in SECRET_KEYS:
            data.pop(k, None)
        data["llm_key_set"] = bool(self.llm_key())
        data["llm_key_from_env"] = bool(os.environ.get("GROQ_API_KEY")) and             not (self.load().get("llm_api_key") or "").strip()
        return data

    def update(self, **changes) -> dict:
        data = self.load()
        if "participant" in changes and changes["participant"] is not None:
            changes["participant"] = validate_pseudonym(changes["participant"])
        data.update({k: v for k, v in changes.items() if k in DEFAULTS and v is not None})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data


# --------------------------------------------------------------------------
# Windows shortcuts (desktop build only)
# --------------------------------------------------------------------------
def frozen_exe():
    """Path of the running .exe when packaged with PyInstaller, else None."""
    return Path(sys.executable) if getattr(sys, "frozen", False) else None


def shortcuts_supported() -> bool:
    return sys.platform == "win32" and frozen_exe() is not None


def _known_folder(name):
    """'Desktop' / 'Startup' folder via PowerShell (handles OneDrive desktops)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"[Environment]::GetFolderPath('{name}')"],
        capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    path = out.stdout.strip()
    return Path(path) if path else None


def _make_shortcut(link: Path, target: Path):
    ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:FT_LINK);"
          "$s.TargetPath = $env:FT_TARGET; $s.WorkingDirectory = $env:FT_DIR;"
          "$s.Description = 'AI Focus Tracker'; $s.Save()")
    env = {**os.environ, "FT_LINK": str(link), "FT_TARGET": str(target),
           "FT_DIR": str(target.parent)}
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env,
                   capture_output=True, timeout=20, check=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def ensure_desktop_shortcut() -> bool:
    """Create the desktop shortcut once (desktop build only). Never raises."""
    exe = frozen_exe()
    if not shortcuts_supported():
        return False
    try:
        desktop = _known_folder("Desktop")
        if not desktop:
            return False
        link = desktop / f"{APP_NAME}.lnk"
        if not link.exists():
            _make_shortcut(link, exe)
        return True
    except Exception:
        return False


def set_autostart(enabled: bool) -> bool:
    """Add / remove the Startup-folder shortcut. Returns the resulting state."""
    if not shortcuts_supported():
        raise RuntimeError("start with Windows is only available in the desktop app")
    startup = _known_folder("Startup")
    if not startup:
        raise RuntimeError("could not find the Windows Startup folder")
    link = startup / f"{APP_NAME}.lnk"
    if enabled:
        _make_shortcut(link, frozen_exe())
    elif link.exists():
        link.unlink()
    return link.exists()
