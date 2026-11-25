"""Local data-collection app: FastAPI backend + the React page in dashboard/.

    focus-app                       # http://127.0.0.1:8765, opens the browser
    focus-app --db data/activity.db --team-db data/team.db --no-browser

Binds to 127.0.0.1 only: nothing is reachable from other machines and nothing
is uploaded anywhere. The logger runs inside this process (Start/Stop button);
do not run the `focus-logger` CLI on the same database at the same time.
"""
import argparse
import threading
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .llm import ExplanationCache
from .logger_control import LoggerController
from .predictions import Predictor
from .routes import router
from .settings import SettingsStore

HOST, PORT = "127.0.0.1", 8765
DEFAULT_STATIC = Path(__file__).resolve().parents[3] / "dashboard" / "dist"
DEFAULT_MODELS = Path(__file__).resolve().parents[3] / "models"

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>Focus Tracker</title>
<body style="font-family:sans-serif;max-width:40em;margin:3em auto">
<h1>Frontend not built</h1><p>Run <code>npm install</code> and <code>npm run build</code>
in the <code>dashboard</code> folder, then restart <code>focus-app</code>.
The API is running at <a href="/docs">/docs</a>.</p>"""


def create_app(db_path="data/activity.db", team_db_path="data/team.db",
               controller=None, clock=datetime.now, static_dir=DEFAULT_STATIC,
               settings_path=None, model_dir=DEFAULT_MODELS, personal_dir="default"):
    """`model_dir`: a saved-model folder, a folder of them, or a list of either;
    the newest model found is the SHARED model (None disables it).
    `personal_dir`: where the user's own model is kept (default: next to the
    database; None disables personal models)."""
    @asynccontextmanager
    async def lifespan(app):
        yield
        app.state.controller.shutdown()     # stop logging cleanly on server exit

    app = FastAPI(title="AI Focus Tracker (local)", lifespan=lifespan)
    app.state.db_path = str(db_path)
    app.state.team_db_path = str(team_db_path)
    app.state.clock = clock
    app.state.controller = controller or LoggerController(db_path, clock=clock)
    app.state.settings = SettingsStore(settings_path or
                                       Path(db_path).with_name("settings.json"))
    app.state.server = None      # set by run_server(); lets /api/app/quit stop it
    app.state.llm_cache = ExplanationCache(Path(db_path).with_name("llm_cache.json")
                                           if str(db_path) != ":memory:" else None)
    app.state.llm_http = None        # tests inject a fake HTTP client
    if personal_dir == "default":
        personal_dir = Path(db_path).with_name("personal_model")
    app.state.predictor = Predictor(model_dir if isinstance(model_dir, (list, tuple))
                                    else [model_dir], personal_dir=personal_dir)
    app.include_router(router)
    static_dir = Path(static_dir) if static_dir else None
    if static_dir and (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def not_built():
            return _NOT_BUILT
    return app


def main(argv=None):
    p = argparse.ArgumentParser(prog="focus-app", description=__doc__.split("\n")[0])
    p.add_argument("--db", default="data/activity.db")
    p.add_argument("--team-db", default="data/team.db",
                   help="developer annotations of exported windows (separate file)")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    app = create_app(args.db, args.team_db)
    url = f"http://{HOST}:{args.port}"
    print(f"AI Focus Tracker app at {url} (local only). Ctrl+C to quit.")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    run_server(app, args.port)
    return 0


def run_server(app, port=PORT, log_config="default"):
    """Run uvicorn; keeps the Server on app.state so the Quit button can stop it.
    `log_config=None` for the windowed desktop build (no console streams)."""
    import uvicorn
    kwargs = {} if log_config == "default" else {"log_config": log_config}
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port,
                                           log_level="warning", **kwargs))
    app.state.server = server
    server.run()


if __name__ == "__main__":
    raise SystemExit(main())
