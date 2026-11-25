"""PyInstaller entry point for FocusTracker.exe (see build_exe.py)."""
import sys

from focus_tracker.desktop import main

if __name__ == "__main__":
    sys.exit(main())
