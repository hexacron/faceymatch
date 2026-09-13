"""Job worker behaviour (D10) and the audit_verify handler."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app import audit, jobs, models_lock, worker
from app.config import Settings
from app.core.registry import ActiveModels
from app.core.types import Detection
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


class _UnusedDetector:
    model_id = "detector"

    def detect(self, image: np.ndarray) -> list[Detection]:
        raise AssertionError("the rematch handler must not detect")


class _UnusedEmbedder:
    model_id = "embedder"
    dim = 2

    def embed(self, crops: np.ndarray) -> np.ndarray:
        raise AssertionError("the rematch handler must not embed")


def test_two_jobs_verify_the_weights_once(
    conn: sqlite3.Connection, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 8 is enforced per model, not per job: 203 MB is not re-hashed each time.

    A second `process` or `rematch` job on the same configuration reads the same bytes the
    first one already proved, so the second verify is pure cost.
    """
    worker._LOCK_CACHE.clear()
    calls: list[Path] = []

    def counting_verify(
        models_dir: Path, lock: models_lock.ModelsLock | None = None
    ) -> models_lock.ModelsLock:
        calls.append(models_dir)
        return models_lock.load(models_dir)

    monkeypatch.setattr(worker.models_lock, "verify", counting_verify)
    monkeypatch.setattr(
        worker,
        "get_active_models",
        lambda settings, lock: ActiveModels(
            detector=_UnusedDetector(),
            embedder=_UnusedEmbedder(),
            detector_model_id="detector",
            embedder_model_id="embedder",
            execution_provider="CPUExecutionProvider",
        ),
    )

    for _ in range(2):
        jobs.enqueue(conn, kind="rematch", actor=settings.operator_name)
        ran = run_once(conn, settings)
        assert ran is not None
        done = jobs.get(conn, ran.id)
        assert done is not None
        assert done.status == "done", done.error

    assert calls == [settings.models_dir]


def test_a_different_model_is_verified_before_it_runs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache key is the models it would load, so a switch re-verifies rather than trusts."""
    worker._LOCK_CACHE.clear()
    calls: list[str] = []

    def counting_verify(
        models_dir: Path, lock: models_lock.ModelsLock | None = None
    ) -> models_lock.ModelsLock:
        calls.append(str(models_dir))
        return models_lock.load(models_dir)

    monkeypatch.setattr(worker.models_lock, "verify", counting_verify)

    worker._verified_lock(settings)
    worker._verified_lock(settings.model_copy(update={"embedder_model": "another-model"}))
    assert len(calls) == 2
