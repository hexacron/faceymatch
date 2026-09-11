"""Job worker behaviour (D10) and the audit_verify handler."""

from __future__ import annotations

import sqlite3

from app import audit, jobs
from app.config import Settings
from app.db.conn import transaction
from app.worker import run_once


def test_audit_verify_job_runs_and_records_its_outcome(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    for i in range(10):
        with transaction(conn):
            audit.append(
                conn, actor="tester", action="probe", object_type="t", object_id=str(i)
            )

    queued = jobs.enqueue(conn, kind="audit_verify", actor=settings.operator_name)
    assert queued.status == "queued"

    ran = run_once(conn, settings)
    assert ran is not None
    assert ran.id == queued.id

    done = jobs.get(conn, queued.id)
    assert done is not None
    assert done.status == "done"
    assert done.progress["ok"] is True
    # 10 probes + enqueue + start entries, all verified before the result was recorded.
    assert done.progress["checked"] >= 12

    actions = [
        row["action"]
        for row in conn.execute("SELECT action FROM audit_log ORDER BY seq").fetchall()
    ]
    assert actions[-3:] == ["job.start", "audit.verify", "job.done"]

    # Verification writes to the chain, so the chain must still verify afterwards.
    assert audit.verify(conn).ok


def test_run_once_returns_none_on_an_empty_queue(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    assert run_once(conn, settings) is None


def test_unregistered_job_kind_fails_the_job(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    job = jobs.enqueue(conn, kind="export", actor=settings.operator_name)
    run_once(conn, settings)

    failed = jobs.get(conn, job.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error is not None
    assert "no handler registered" in failed.error
    assert audit.verify(conn).ok


def test_a_job_is_claimed_only_once(conn: sqlite3.Connection, settings: Settings) -> None:
    jobs.enqueue(conn, kind="audit_verify", actor=settings.operator_name)
    first = jobs.claim_next(conn, actor=settings.operator_name)
    second = jobs.claim_next(conn, actor=settings.operator_name)
    assert first is not None
    assert second is None
    assert first.status == "running"


def test_failed_verification_fails_the_job(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    with transaction(conn):
        audit.append(conn, actor="tester", action="probe", object_type="t", payload={"v": 1})
    conn.execute("DROP TRIGGER audit_log_no_update")
    conn.execute("UPDATE audit_log SET payload_json = '{\"v\":2}' WHERE seq = 1")

    job = jobs.enqueue(conn, kind="audit_verify", actor=settings.operator_name)
    run_once(conn, settings)

    failed = jobs.get(conn, job.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error is not None
    assert "chain verification failed at seq 1" in failed.error
