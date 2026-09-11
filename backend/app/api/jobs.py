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
from app.jobs import enqueue, get

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
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


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, conn: ConnDep) -> JobOut:
    job = get(conn, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return JobOut(**asdict(job))
