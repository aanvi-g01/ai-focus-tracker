"""Build dist/FocusTracker.exe (Windows).

    python packaging/build_exe.py [--dist DIR]      # default: dist/

Steps: build the React page (npm), make the icon, then run PyInstaller in
one-file, windowed mode with the page and the newest trained model (models/)
bundled inside. Requires: pip install -e ".[app,ml]" pyinstaller uiautomation,
and Node.js. Explanations use LightGBM's built-in TreeSHAP, so shap (and its
numba/llvmlite) and matplotlib are left out to keep the exe small.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging"
DASH = ROOT / "dashboard"


def run(cmd, cwd=ROOT):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, shell=(sys.platform == "win32" and cmd[0] == "npm"))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dist", default=str(ROOT / "dist"),
                   help="output folder (use another one while FocusTracker.exe is running)")
    dist = Path(p.parse_args(argv).dist).resolve()
    if sys.platform != "win32":
        sys.exit("FocusTracker.exe must be built on Windows")
    if not (DASH / "node_modules").exists():
        run(["npm", "install", "--no-audit", "--no-fund"], cwd=DASH)
    run(["npm", "run", "build"], cwd=DASH)

    sys.path.insert(0, str(PKG))
    from make_icon import write_ico
    icon = PKG / "build" / "focus_tracker.ico"
    icon.parent.mkdir(exist_ok=True)
    write_ico(icon)

    sys.path.insert(0, str(ROOT / "src"))
    from focus_tracker.models.train import latest_model_dir
    model = latest_model_dir(ROOT / "models")
    if model is None:
        sys.exit("no trained model in models/; run focus-train first")
    print(f"Bundling model {model.name}")

    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--onefile", "--windowed", "--name", "FocusTracker",
         "--icon", str(icon),
         "--distpath", str(dist), "--workpath", str(PKG / "build" / "work"),
         "--specpath", str(PKG / "build"),
         "--paths", str(ROOT / "src"),
         "--add-data", f"{DASH / 'dist'};dashboard_dist",
         "--add-data", f"{model};model/{model.name}",
         # Imported lazily or by name at runtime, so PyInstaller can't see them:
         "--collect-submodules", "uvicorn",
         "--collect-submodules", "pynput",
         "--collect-all", "uiautomation",
         "--collect-submodules", "focus_tracker",
         "--hidden-import", "comtypes.stream",
         # Prediction model: unpickled by joblib, so its classes are loaded by name.
         "--collect-all", "lightgbm",
         "--hidden-import", "joblib",
         "--hidden-import", "sklearn.utils._typedefs",
         "--hidden-import", "sklearn.neighbors._partition_nodes",
         "--exclude-module", "shap", "--exclude-module", "numba",
         "--exclude-module", "llvmlite", "--exclude-module", "matplotlib",
         "--exclude-module", "IPython", "--exclude-module", "pytest",
         str(PKG / "launch.py")])
    exe = dist / "FocusTracker.exe"
    print(f"\nBuilt {exe} ({exe.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
