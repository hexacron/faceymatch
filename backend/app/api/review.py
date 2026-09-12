"""Candidate review queue and bulk operator decisions."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, ValidationError

from app.api.deps import ConnDep, SettingsDep
from app.api.identifications import IdentificationIn, apply_decision
from app.core.types import Band, Decision
from app.enrollment import EnrollmentError
from app.jobs import enqueue

router = APIRouter(prefix="/api/review", tags=["review"])


class ReviewItemOut(BaseModel):
    track_id: str
    media_id: str
    case_id: str
    band: Band
    score: float
    person_id: str
    name: str
    crop_sha256: str | None
    t_ms: int


class ReviewListOut(BaseModel):
    items: list[ReviewItemOut]


class BulkDecisionIn(BaseModel):
    track_id: str = Field(min_length=1)
    decision: Decision
    person_id: str | None = None
    # `new` creates a person, so it needs the same display name the single-decision
    # endpoint requires. Without it a bulk `new` could only ever fail validation.
    new_name: str | None = Field(default=None, min_length=1, max_length=200)
    # Opt-in enrollment, same semantics as IdentificationIn.enroll (D17).
    enroll: bool = False


class BulkRequestIn(BaseModel):
    decisions: list[BulkDecisionIn]


class BulkErrorOut(BaseModel):
    track_id: str
    error: str


class BulkResultOut(BaseModel):
    applied: int
    errors: list[BulkErrorOut]


@router.get("", response_model=ReviewListOut)
def list_review(
    conn: ConnDep,
    band: Annotated[Literal["possible", "ambiguous"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> ReviewListOut:
    # The band filter belongs in SQL: applied after LIMIT it would drop the requested band
    # behind higher-scoring rows of the other one. Bound, not interpolated.
    rows = conn.execute(
        "SELECT t.id AS track_id, t.media_id, med.case_id, m.band, m.score, m.person_id, "
        "p.display_name, d.crop_sha256, t.start_ms AS t_ms FROM matches m "
        "JOIN tracks t ON t.id = m.track_id JOIN media med ON med.id = t.media_id "
        "JOIN persons p ON p.id = m.person_id "
        "LEFT JOIN detections d ON d.id = t.best_detection_id "
        "LEFT JOIN identities i ON i.track_id = t.id "
        "WHERE m.rank = 1 AND m.band IN ('possible', 'ambiguous') "
        "AND (? IS NULL OR m.band = ?) "
        "AND (i.source IS NULL OR i.source <> 'operator') "
        "ORDER BY m.score DESC, t.id LIMIT ?",
        (band, band, limit),
    ).fetchall()
    return ReviewListOut(
        items=[
            ReviewItemOut(
                track_id=str(row["track_id"]),
                media_id=str(row["media_id"]),
                case_id=str(row["case_id"]),
                band=row["band"],
                score=float(row["score"]),
                person_id=str(row["person_id"]),
                name=str(row["display_name"]),
                crop_sha256=None if row["crop_sha256"] is None else str(row["crop_sha256"]),
                t_ms=int(row["t_ms"]),
            )
            for row in rows
        ]
    )


@router.post("/bulk", response_model=BulkResultOut)
def bulk_review(body: BulkRequestIn, conn: ConnDep, settings: SettingsDep) -> BulkResultOut:
    applied = 0
    errors: list[BulkErrorOut] = []
    enrolled = False
    for item in body.decisions:
        try:
            request = IdentificationIn(
                track_id=item.track_id,
                decision=item.decision,
                person_id=item.person_id,
                new_name=item.new_name,
                enroll=item.enroll,
            )
            result = apply_decision(conn, settings, request)
            enrolled = enrolled or result.template_created
            applied += 1
        except (LookupError, EnrollmentError, ValidationError, ValueError) as exc:
            errors.append(BulkErrorOut(track_id=item.track_id, error=str(exc)))
    if enrolled:
        enqueue(
            conn,
            kind="rematch",
            actor=settings.operator_name,
            params={"reason": "bulk_operator_enrollment"},
        )
    return BulkResultOut(applied=applied, errors=errors)
