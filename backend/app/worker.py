"""Single job worker process (D10).

Handlers registered: `audit_verify`, `process`, `rematch`, `reembed`. An unregistered kind
fails its job loudly rather than silently succeeding.

Every job runs against the *effective* settings — the environment overlaid with the
durable overrides in `runtime_config` — read fresh from the database as the job is claimed.
That is what makes `PATCH /api/config` take effect on the next job instead of the next
restart, and it is why a config change is refused while a re-embed is in flight: the job
that is half-committed must keep the parameters it started with.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

from app import audit, models_lock, runtime_config
from app.config import Settings, get_settings
from app.core.registry import get_active_models
from app.db.conn import connect, transaction
from app.db.migrate import migrate
from app.jobs import Job, claim_next, finish, requeue_running
from app.pipeline.matching import rematch
from app.pipeline.process import (
    MEDIA_KIND_VIDEO,
    MediaNotFoundError,
    mark_failed,
    process_image,
    process_video,
)
from app.pipeline.reembed import reembed

log = logging.getLogger("app.worker")

Handler = Callable[[sqlite3.Connection, Job, Settings], dict[str, Any]]


class UnknownJobKindError(RuntimeError):
    """No handler is registered for this job kind."""


_LOCK_CACHE: dict[tuple[str, str, str], models_lock.ModelsLock] = {}


def _verified_lock(settings: Settings) -> models_lock.ModelsLock:
    """Verify models.lock once per (models_dir, detector, embedder), not once per job.

    Invariant 8 is about refusing to RUN a model whose bytes moved; re-hashing the same
    files between two jobs of the same model proves nothing the first verify did not.
    A config switch changes the key, so the new model is verified before it is used.
    """
    key = (str(settings.models_dir), settings.detector_model, settings.embedder_model)
    cached = _LOCK_CACHE.get(key)
    if cached is None:
        cached = models_lock.verify(settings.models_dir)
        _LOCK_CACHE[key] = cached
    return cached


def handle_audit_verify(
    conn: sqlite3.Connection, job: Job, settings: Settings
) -> dict[str, Any]:
    result = audit.verify(conn)
    outcome: dict[str, Any] = {
        "ok": result.ok,
        "checked": result.checked,
        "head_seq": result.head_seq,
        "head_hash": result.head_hash,
        "bad_seq": result.bad_seq,
        "reason": result.reason,
    }
    with transaction(conn):
        audit.append(
            conn,
            actor=settings.operator_name,
            action="audit.verify",
            object_type="audit_log",
            object_id=str(result.head_seq),
            payload=outcome,
        )
    if not result.ok:
        raise audit.AuditChainError(
            f"chain verification failed at seq {result.bad_seq}: {result.reason}"
        )
    return outcome

def handle_process(
    conn: sqlite3.Connection, job: Job, settings: Settings
) -> dict[str, Any]:
    media_id = job.params.get("media_id")
    if not isinstance(media_id, str) or not media_id:
        raise ValueError("process job requires a string media_id")
    lock = _verified_lock(settings)
    active = get_active_models(settings, lock)
    row = conn.execute("SELECT kind FROM media WHERE id = ?", (media_id,)).fetchone()
    if row is None:
        raise MediaNotFoundError(f"no media {media_id!r}")
    if str(row["kind"]) == MEDIA_KIND_VIDEO:
        # The job id goes in because a video checkpoints into its own `jobs.progress` and
        # resumes from it after a kill (spec 6.2, "Job resume"). A still has nothing to
        # resume: it is one frame and one transaction.
        return process_video(
            conn,
            settings,
            active,
            media_id=media_id,
            actor=settings.operator_name,
            job_id=job.id,
        ).as_progress()
    return process_image(
        conn, settings, active, media_id=media_id, actor=settings.operator_name
    ).as_progress()


def handle_rematch(
    conn: sqlite3.Connection, job: Job, settings: Settings
) -> dict[str, Any]:
    lock = _verified_lock(settings)
    active = get_active_models(settings, lock)
    return rematch(
        conn,
        settings,
        embedder_model_id=active.embedder_model_id,
        execution_provider=active.execution_provider,
        actor=settings.operator_name,
    ).as_progress()


def handle_reembed(
    conn: sqlite3.Connection, job: Job, settings: Settings
) -> dict[str, Any]:
    """Re-embed stored crops, track means and templates under the active embedder.

    The target model is pinned in the job's params at enqueue time. If the configuration
    has moved since, the job fails rather than re-embedding into a model nobody asked for:
    a half-finished switch that silently retargets is how two models end up mixed in one
    gallery (invariant 2).
    """
    lock = _verified_lock(settings)
    active = get_active_models(settings, lock)
    target = job.params.get("embedder_model_id")
    if isinstance(target, str) and target != active.embedder_model_id:
        raise ValueError(
            f"reembed job targets {target!r} but the active embedder is now "
            f"{active.embedder_model_id!r}; re-enqueue the job for the active model"
        )
    return reembed(
        conn, settings, active, job_id=job.id, actor=settings.operator_name
    ).as_progress()



HANDLERS: dict[str, Handler] = {
    "audit_verify": handle_audit_verify,
    "process": handle_process,
    "rematch": handle_rematch,
    "reembed": handle_reembed,
}


def run_job(conn: sqlite3.Connection, job: Job, settings: Settings) -> dict[str, Any]:
    handler = HANDLERS.get(job.kind)
    if handler is None:
        raise UnknownJobKindError(f"no handler registered for job kind {job.kind!r}")
    return handler(conn, job, settings)


def run_once(conn: sqlite3.Connection, settings: Settings) -> Job | None:
    """Claim and run at most one job. Returns the job it ran, or None if the queue was empty.

    `settings` is the process's base configuration; the job runs against that overlaid with
    the durable overrides, re-read here so a change made through `PATCH /api/config` is
    picked up by the very next job without restarting the worker.
    """
    job = claim_next(conn, actor=settings.operator_name)
    if job is None:
        return None
    settings = runtime_config.effective(conn, settings)
    try:
        progress = run_job(conn, job, settings)
    except Exception as exc:
        log.exception("job %s (%s) failed", job.id, job.kind)
        finish(
            conn,
            job.id,
            actor=settings.operator_name,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        if job.kind == "process":
            media_id = job.params.get("media_id")
            if isinstance(media_id, str):
                mark_failed(
                    conn,
                    media_id=media_id,
                    actor=settings.operator_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return job
    finish(
        conn,
        job.id,
        actor=settings.operator_name,
        status="done",
        progress=progress,
    )
    return job


class Worker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else get_settings()
        self._stop = False

    def request_stop(self, signum: int, frame: FrameType | None) -> None:
        log.info("worker received signal %s, finishing current job", signum)
        self._stop = True

    def run(self) -> None:
        self.settings.ensure_dirs()
        conn = connect(self.settings.db_path)
        try:
            migrate(conn)
            abandoned = requeue_running(conn, actor=self.settings.operator_name)
            if abandoned:
                log.info("requeued %d job(s) left running by a previous worker", len(abandoned))
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
            log.info("worker ready, polling %s", self.settings.db_path)
            while not self._stop:
                if run_once(conn, self.settings) is None:
                    time.sleep(self.settings.worker_poll_seconds)
        finally:
            conn.close()


def main() -> None:  # pragma: no cover - process entry point
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    Worker().run()


if __name__ == "__main__":  # pragma: no cover
    main()
