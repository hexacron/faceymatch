"""Job queue over the `jobs` table (D10). One worker process, no Redis.

Claiming uses BEGIN IMMEDIATE plus a status guard, so two workers cannot take the same job.
`progress` is a JSON object: the structured resume checkpoint for pipeline jobs, the
verification outcome for audit_verify.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal, get_args

from app import audit
from app.db.conn import transaction
from app.ids import new_id

JobKind = Literal["ingest", "process", "rematch", "reembed", "cluster", "export", "audit_verify"]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]

JOB_KINDS: frozenset[str] = frozenset(get_args(JobKind))


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    kind: JobKind
    status: JobStatus
    params: dict[str, Any]
    progress: dict[str, Any]
    error: str | None
    created_at: str
    updated_at: str


def row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=str(row["id"]),
        kind=row["kind"],
        status=row["status"],
        params=json.loads(row["params_json"]),
        progress=json.loads(row["progress"]),
        error=row["error"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def enqueue(
    conn: sqlite3.Connection,
    *,
    kind: JobKind,
    actor: str,
    params: dict[str, Any] | None = None,
    case_id: str | None = None,
) -> Job:
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind {kind!r}")
    job_id = new_id()
    now = audit.now_ts()
    payload = params if params is not None else {}
    with transaction(conn):
        conn.execute(
            "INSERT INTO jobs (id, kind, status, params_json, progress, error, "
            "created_at, updated_at) VALUES (?, ?, 'queued', ?, '{}', NULL, ?, ?)",
            (job_id, kind, audit.canonical_json(payload).decode("utf-8"), now, now),
        )
        audit.append(
            conn,
            actor=actor,
            action="job.enqueue",
            object_type="job",
            object_id=job_id,
            case_id=case_id,
            payload={"kind": kind, "params": payload},
        )
    job = get(conn, job_id)
    if job is None:  # pragma: no cover - the insert above just committed
        raise RuntimeError(f"job {job_id} vanished after insert")
    return job


def get(conn: sqlite3.Connection, job_id: str) -> Job | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return None if row is None else row_to_job(row)


def claim_next(conn: sqlite3.Connection, *, actor: str) -> Job | None:
    """Atomically take the oldest queued job. Returns None when the queue is empty."""
    with transaction(conn):
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        job_id = str(row["id"])
        now = audit.now_ts()
        updated = conn.execute(
            "UPDATE jobs SET status = 'running', updated_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (now, job_id),
        )
        if updated.rowcount != 1:  # pragma: no cover - IMMEDIATE lock makes this unreachable
            return None
        audit.append(
            conn,
            actor=actor,
            action="job.start",
            object_type="job",
            object_id=job_id,
            payload={"kind": str(row["kind"])},
        )
    return get(conn, job_id)


def set_progress(conn: sqlite3.Connection, job_id: str, progress: dict[str, Any]) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
            (audit.canonical_json(progress).decode("utf-8"), audit.now_ts(), job_id),
        )


def finish(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    actor: str,
    status: Literal["done", "failed", "cancelled"],
    progress: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    now = audit.now_ts()
    with transaction(conn):
        if progress is None:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, error, now, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, progress = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    status,
                    error,
                    audit.canonical_json(progress).decode("utf-8"),
                    now,
                    job_id,
                ),
            )
        audit.append(
            conn,
            actor=actor,
            action=f"job.{status}",
            object_type="job",
            object_id=job_id,
            payload={"error": error} if error is not None else {},
        )
