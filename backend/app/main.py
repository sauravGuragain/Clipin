"""Application entry point.

Single process: FastAPI serves the API, the UI, and runs the job workers.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.environment import check_environment, format_report
from app.core.jobs import runner
from app.models.db import init_db
from app.services.audio import handle_extract_audio

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()

    report = check_environment()
    print("\nEnvironment check:")
    print(format_report(report))

    if not report.ok:
        print(
            "\nRefusing to start: a required capability is missing. "
            "Fix the items marked FAIL above.\n",
            file=sys.stderr,
        )
        raise RuntimeError("environment check failed")

    runner.register("extract_audio", handle_extract_audio)
    await runner.start()

    print(f"\nReady on http://{settings.host}:{settings.port}")
    print(f"Job workers: {runner.concurrency}\n")
    try:
        yield
    finally:
        await runner.stop()


app = FastAPI(title="Clipper AI", version="0.2.0", lifespan=lifespan)
app.include_router(router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
