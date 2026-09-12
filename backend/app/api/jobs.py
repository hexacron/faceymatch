"""Job API (spec section 8).

M0 exposes the one kind that has a handler. Pipeline kinds get their endpoints with the
milestones that implement them.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.api.deps import ConnDep, SettingsDep
from app.jobs import JobKind, JobStatus, enqueue, get

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobOut(BaseModel):
    id: str
    kind: JobKind
    status: JobStatus
    params: dict[str, Any]
    progress: dict[str, Any]
    error: str | None
    created_at: str
    updated_at: str


@router.post("/audit_verify", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def enqueue_audit_verify(conn: ConnDep, settings: SettingsDep) -> JobOut:
    """Queue a full hash-chain verification. The worker records the result in the chain."""
    job = enqueue(conn, kind="audit_verify", actor=settings.operator_name)
    return JobOut(**asdict(job))

@router.post("/rematch", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def enqueue_rematch(conn: ConnDep, settings: SettingsDep) -> JobOut:
    """Queue block-matrix re-matching from stored track means; no media re-decode."""
    job = enqueue(
        conn,
        kind="rematch",
        actor=settings.operator_name,
        params={"reason": "operator_request"},
    )
    return JobOut(**asdict(job))


@router.post("/reembed", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def enqueue_reembed(conn: ConnDep, settings: SettingsDep) -> JobOut:
    """Queue a re-embed of every stored crop, track mean and template under the active embedder.

    `PATCH /api/config` already enqueues this when the embedder changes. This endpoint is
    for re-running a switch that did not finish — a killed worker, a crop store that was
    still being restored — without pretending to change a setting that is already set. Every
    write is guarded on the row it would create, so it skips what a previous run already
    committed instead of duplicating it, and it never re-decodes original media
    (spec 6.2 step 6, 6.3).
    """
    job = enqueue(
        conn,
        kind="reembed",
        actor=settings.operator_name,
        params={
            "embedder_model_id": settings.embedder_model,
            "reason": "operator_request",
        },
    )
    return JobOut(**asdict(job))


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, conn: ConnDep) -> JobOut:
    job = get(conn, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return JobOut(**asdict(job))
