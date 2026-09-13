"""Operator identification decisions (spec 6.5-6.6, 8)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app import audit
from app.api.deps import ConnDep, SettingsDep
from app.config import Settings
from app.core.types import Decision
from app.db.conn import transaction
from app.enrollment import EnrollmentError, create_person, create_template_from_detection
from app.ids import new_id
from app.jobs import enqueue

router = APIRouter(prefix="/api/identifications", tags=["identifications"])


class IdentificationIn(BaseModel):
    track_id: str = Field(min_length=1)
    decision: Decision
    person_id: str | None = None
    new_name: str | None = Field(default=None, min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=2000)
    # D17/spec 6.6: tagging and enrolling are separate operator actions. Confirming a track
    # says who it is; it does not donate that crop to the gallery unless asked, because an
    # arbitrary-quality crop per confirm is exactly the drift D17 exists to prevent.
    # `new` ignores this: a person born with no template could never match anything.
    enroll: bool = False

    @model_validator(mode="after")
    def validate_decision_fields(self) -> IdentificationIn:
        if self.decision in {"confirm", "reassign"} and not self.person_id:
            raise ValueError(f"person_id is required for {self.decision}")
        if self.decision == "new" and not self.new_name:
            raise ValueError("new_name is required for new")
        if self.decision == "reject" and (self.person_id or self.new_name):
            raise ValueError("reject does not accept person_id or new_name")
        if self.decision == "reject" and self.enroll:
            raise ValueError("reject does not accept enroll")
        return self


class IdentificationOut(BaseModel):
    id: str
    track_id: str
    person_id: str | None
    decision: Decision
    operator: str
    note: str | None
    created_at: str
    # Whether THIS request created a template. Tagging is not enrolling (D17), and `new`
    # bootstraps one only when the track has a quality-passing crop to enrol from, so a
    # saved decision does not imply the person can ever be matched. The client cannot
    # infer this from a follow-up fetch: for confirm/reassign the person already existed,
    # so an unchanged template_count is indistinguishable from a stale read. Reporting it
    # here is what lets the UI say "tagged" and "enrolled" as the different things they are.
    template_created: bool


@router.post("", response_model=IdentificationOut, status_code=status.HTTP_201_CREATED)
def identify(body: IdentificationIn, conn: ConnDep, settings: SettingsDep) -> IdentificationOut:
    try:
        applied = apply_decision(conn, settings, body)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EnrollmentError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if applied.template_created:
        enqueue(
            conn,
            kind="rematch",
            actor=settings.operator_name,
            params={"reason": "operator_enrollment", "track_id": body.track_id},
        )
    return applied


def apply_decision(
    conn: sqlite3.Connection, settings: Settings, body: IdentificationIn
) -> IdentificationOut:
    """Apply one decision atomically; used by both single and bulk endpoints."""
    track = conn.execute(
        "SELECT t.id, t.best_detection_id, m.case_id FROM tracks t "
        "JOIN media m ON m.id = t.media_id WHERE t.id = ?",
        (body.track_id,),
    ).fetchone()
    if track is None:
        raise LookupError("track not found")

    person_id = body.person_id
    identification_id = new_id()
    now = audit.now_ts()
    template_created = False
    with transaction(conn):
        if body.decision == "new":
            person_id = create_person(
                conn,
                display_name=str(body.new_name),
                case_id=str(track["case_id"]),
                actor=settings.operator_name,
                audit_payload={
                    "display_name": body.new_name,
                    "from_track_id": body.track_id,
                },
            )
        elif person_id is not None:
            person = conn.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone()
            if person is None:
                raise LookupError("person not found")

        if body.decision == "reject":
            conn.execute("DELETE FROM identities WHERE track_id = ?", (body.track_id,))
            person_id = None
        else:
            if person_id is None:  # guarded by pydantic, retained for direct internal calls
                raise ValueError("person_id is required")
            match = conn.execute(
                "SELECT id, threshold_set_id FROM matches WHERE track_id = ? AND person_id = ? "
                "ORDER BY rank LIMIT 1",
                (body.track_id, person_id),
            ).fetchone()
            conn.execute(
                "INSERT INTO identities (track_id, person_id, source, match_id, "
                "threshold_set_id, updated_at) VALUES (?, ?, 'operator', ?, ?, ?) "
                "ON CONFLICT(track_id) DO UPDATE SET person_id = excluded.person_id, "
                "source = 'operator', match_id = excluded.match_id, "
                "threshold_set_id = excluded.threshold_set_id, updated_at = excluded.updated_at",
                (
                    body.track_id,
                    person_id,
                    None if match is None else str(match["id"]),
                    None if match is None else str(match["threshold_set_id"]),
                    now,
                ),
            )
            # `new` bootstraps its one template; confirm and reassign only when asked.
            detection_id = track["best_detection_id"]
            if detection_id is not None and (body.decision == "new" or body.enroll):
                try:
                    created = create_template_from_detection(
                        conn,
                        person_id=person_id,
                        detection_id=str(detection_id),
                        embedder_model_id=settings.embedder_model,
                        actor=settings.operator_name,
                    )
                    template_created = created is not None
                except EnrollmentError as exc:
                    # Tagging a low-quality track is valid even when it cannot be enrolled.
                    if "no quality-passing embedding" not in str(exc):
                        raise

        conn.execute(
            "INSERT INTO identifications (id, track_id, person_id, decision, operator, note, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                identification_id,
                body.track_id,
                person_id,
                body.decision,
                settings.operator_name,
                body.note,
                now,
            ),
        )
        audit.append(
            conn,
            actor=settings.operator_name,
            case_id=str(track["case_id"]),
            action=f"identification.{body.decision}",
            object_type="track",
            object_id=body.track_id,
            payload={
                "identification_id": identification_id,
                "person_id": person_id,
                "note": body.note,
                "template_created": template_created,
            },
        )

    return IdentificationOut(
        id=identification_id,
        track_id=body.track_id,
        person_id=person_id,
        decision=body.decision,
        operator=settings.operator_name,
        note=body.note,
        created_at=now,
        template_created=template_created,
    )
