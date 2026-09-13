"""FastAPI application.

Startup refuses to serve on any condition that would make later evidence untrustworthy:
sqlite without loadable extensions, drifted migrations, or model files that disagree with
models.lock (invariant 8).

The frontend build is served from the same process (no Node in production, D7).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app import audit, models_lock, runtime_config
from app.api import audit as audit_router
from app.api import capture as capture_router
from app.api import cases as cases_router
from app.api import config as config_router
from app.api import health as health_router
from app.api import identifications as identifications_router
from app.api import jobs as jobs_router
from app.api import live as live_router
from app.api import media as media_router
from app.api import models as models_router
from app.api import persons as persons_router
from app.api import review as review_router
from app.api import thresholds as thresholds_router
from app.api import tracks as tracks_router
from app.api import watch as watch_router
from app.config import Settings, get_settings
from app.db.conn import assert_extension_loading_available, connect, transaction
from app.db.migrate import migrate
from app.thresholds import seed_default_threshold_set

log = logging.getLogger("app.main")


def sync_models_table(conn: sqlite3.Connection, lock: models_lock.ModelsLock, actor: str) -> int:
    """Mirror models.lock into the `models` table so match rows can reference it.

    Returns the number of rows inserted or updated.
    """
    changed = 0
    with transaction(conn):
        for entry in lock.models:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (entry.id,)).fetchone()
            values = (
                entry.name,
                entry.version,
                entry.kind,
                entry.sha256,
                entry.license,
                int(entry.commercial_use),
                entry.dim,
            )
            if row is None:
                conn.execute(
                    "INSERT INTO models (id, name, version, kind, sha256, license, "
                    "commercial_use, dim) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry.id, *values),
                )
                changed += 1
                audit.append(
                    conn,
                    actor=actor,
                    action="model.register",
                    object_type="model",
                    object_id=entry.id,
                    payload={
                        "sha256": entry.sha256,
                        "license": entry.license,
                        "commercial_use": entry.commercial_use,
                    },
                )
            elif str(row["sha256"]) != entry.sha256:
                # models.lock is the source of truth; a digest change is a different model.
                conn.execute(
                    "UPDATE models SET name = ?, version = ?, kind = ?, sha256 = ?, "
                    "license = ?, commercial_use = ?, dim = ? WHERE id = ?",
                    (*values, entry.id),
                )
                changed += 1
                audit.append(
                    conn,
                    actor=actor,
                    action="model.update",
                    object_type="model",
                    object_id=entry.id,
                    payload={"sha256": entry.sha256, "previous_sha256": str(row["sha256"])},
                )
    return changed


def startup_checks(settings: Settings) -> models_lock.ModelsLock:
    settings.ensure_dirs()
    assert_extension_loading_available()

    conn = connect(settings.db_path)
    try:
        applied = migrate(conn)
        lock = models_lock.verify(settings.models_dir)
        if applied:
            with transaction(conn):
                audit.append(
                    conn,
                    actor=settings.operator_name,
                    action="db.migrate",
                    object_type="schema",
                    object_id=str(applied[-1].version),
                    payload={"applied": [f"{m.version:04d}_{m.name}" for m in applied]},
                )
        sync_models_table(conn, lock, settings.operator_name)
        # The effective embedder, not the environment's: a model switch through
        # PATCH /api/config is durable, so a restart must bootstrap the model that is
        # actually active rather than the one the .env still names.
        effective = runtime_config.effective(conn, settings)
        with transaction(conn):
            seed_default_threshold_set(
                conn, model_id=effective.embedder_model, actor=settings.operator_name
            )
    finally:
        conn.close()
    return lock


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    app.state.models_lock = startup_checks(settings)
    log.info("ready: db=%s frontend=%s", settings.db_path, settings.frontend_dist)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings if settings is not None else get_settings()
    app = FastAPI(
        title="Local Face Match System",
        version=health_router.VERSION,
        lifespan=lifespan,
    )
    app.state.settings = resolved

    app.include_router(health_router.router)
    app.include_router(audit_router.router)
    app.include_router(cases_router.router)
    app.include_router(jobs_router.router)
    app.include_router(media_router.router)
    app.include_router(capture_router.router)
    app.include_router(live_router.router)
    app.include_router(watch_router.router)
    app.include_router(tracks_router.router)
    app.include_router(identifications_router.router)
    app.include_router(persons_router.router)
    app.include_router(review_router.router)
    app.include_router(thresholds_router.router)
    app.include_router(models_router.router)
    app.include_router(config_router.router)

    # The bundle is one 300 KB chunk and the API returns JSON lists; both compress by
    # roughly 3.5x. Loopback is not slow, but the browser still parses what it is sent, and
    # nothing here leaves the machine, so there is no traffic-analysis argument against it.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    # Mounted last so /api never collides with a built asset path.
    if resolved.frontend_dist.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=resolved.frontend_dist, html=True),
            name="frontend",
        )
    else:
        log.warning(
            "frontend build not found at %s; serving API only (run `bun run build`)",
            resolved.frontend_dist,
        )
    return app


app = create_app()
