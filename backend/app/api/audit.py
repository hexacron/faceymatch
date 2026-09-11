"""Audit log read API (spec section 8)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app import audit
from app.api.deps import ConnDep

router = APIRouter(prefix="/api", tags=["audit"])

MAX_LIMIT = 1000


class AuditEntryOut(BaseModel):
    seq: int
    ts: str
    actor: str
    case_id: str | None
    action: str
    object_type: str
    object_id: str | None
    payload: dict[str, Any]
    prev_hash: str
    hash: str


class AuditPage(BaseModel):
    entries: list[AuditEntryOut]
    next_seq: int | None
    head_seq: int


@router.get("/audit", response_model=AuditPage)
def read_audit(
    conn: ConnDep,
    from_seq: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 100,
) -> AuditPage:
    """Entries with seq >= from_seq, oldest first."""
    head_seq, _ = audit.head(conn)
    rows = conn.execute(
        "SELECT seq, ts, actor, case_id, action, object_type, object_id, payload_json, "
        "prev_hash, hash FROM audit_log WHERE seq >= ? ORDER BY seq LIMIT ?",
        (from_seq, limit),
    ).fetchall()
    entries = [AuditEntryOut(**asdict(audit.row_to_entry(row))) for row in rows]
    next_seq = entries[-1].seq + 1 if entries and entries[-1].seq < head_seq else None
    return AuditPage(entries=entries, next_seq=next_seq, head_seq=head_seq)
