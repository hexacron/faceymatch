"""Single job worker process (D10).

M0 registers one handler: `audit_verify`, which recomputes the hash chain and records the
outcome both in `jobs.progress` and as an audit entry, so verification is itself auditable.
Pipeline handlers (ingest, process, rematch, reembed, cluster, export) arrive with their
milestones; an unregistered kind fails its job loudly rather than silently succeeding.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

from app import audit
from app.config import Settings, get_settings
from app.db.conn import connect, transaction
from app.db.migrate import migrate
from app.jobs import Job, claim_next, finish

log = logging.getLogger("app.worker")

Handler = Callable[[sqlite3.Connection, Job, Settings], dict[str, Any]]


class UnknownJobKindError(RuntimeError):
    """No handler is registered for this job kind."""


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


HANDLERS: dict[str, Handler] = {
    "audit_verify": handle_audit_verify,
}


def run_job(conn: sqlite3.Connection, job: Job, settings: Settings) -> dict[str, Any]:
    handler = HANDLERS.get(job.kind)
    if handler is None:
        raise UnknownJobKindError(f"no handler registered for job kind {job.kind!r}")
    return handler(conn, job, settings)


def run_once(conn: sqlite3.Connection, settings: Settings) -> Job | None:
    """Claim and run at most one job. Returns the job it ran, or None if the queue was empty."""
    job = claim_next(conn, actor=settings.operator_name)
    if job is None:
        return None
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
