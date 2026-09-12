"""Persons, templates and appearances (spec 6.6, 6.8, 8)."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app import audit
from app.api.deps import ConnDep, SettingsDep
from app.core.types import Band, IdentitySource, PersonStatus, TemplateStatus
from app.db.conn import transaction
from app.enrollment import EnrollmentError, create_template_from_detection
from app.ids import new_id
from app.jobs import enqueue

router = APIRouter(prefix="/api/persons", tags=["persons"])


class PersonCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)


class PersonOut(BaseModel):
    id: str
    display_name: str
    notes: str | None
    do_not_enroll: bool
    status: PersonStatus
    template_count: int
    # The person's representative face: the crop of their best active template. Null when
    # they have no active template with a stored crop, which is exactly the unenrolled case.
    crop_sha256: str | None
    created_at: str
    created_by: str


class PersonListOut(BaseModel):
    items: list[PersonOut]


class TemplateCreate(BaseModel):
    detection_id: str = Field(min_length=1)


class TemplateOut(BaseModel):
    id: str
    detection_id: str | None
    source_case_id: str | None
    embedder_model_id: str
    quality: float | None
    status: TemplateStatus
    crop_sha256: str | None
    created_at: str
    created_by: str


class AppearanceOut(BaseModel):
    track_id: str
    media_id: str
    case_id: str
    source: IdentitySource
    score: float | None
    band: Band | None
    t_ms: int
    crop_sha256: str | None
    ingested_at: str


class PersonDetailOut(BaseModel):
    person: PersonOut
    templates: list[TemplateOut]
    appearances: list[AppearanceOut]


# The representative face for a person, as one correlated scalar subquery so the list
# endpoint answers for every person in a single statement instead of an N+1 fan-out.
#
# Correlated subquery rather than a window function: the pick is "best active template with
# a crop, per person", and a windowed ROW_NUMBER() would need its own nested SELECT joined
# back onto the existing GROUP BY aggregate, i.e. a second scan plus an extra nesting level
# for no gain. This form is an index seek on templates_person (person_id, status) with an
# early LIMIT 1, and it drops verbatim into both the list and the detail query, so the two
# payloads cannot disagree about which face is the person's.
#
# Ordering: highest quality wins; SQLite sorts NULLs last under DESC, so an unscored
# template never outranks a scored one. Ties break on the newest template.
_REPRESENTATIVE_CROP = (
    "SELECT d.crop_sha256 FROM templates rt "
    "JOIN detections d ON d.id = rt.detection_id "
    "WHERE rt.person_id = p.id AND rt.status = 'active' AND d.crop_sha256 IS NOT NULL "
    "ORDER BY rt.quality DESC, rt.created_at DESC, rt.id DESC LIMIT 1"
)

# One row shape for both endpoints, so the list and the detail cannot disagree.
_PERSON_SELECT = (
    "SELECT p.*, COUNT(t.id) AS template_count, "
    "(" + _REPRESENTATIVE_CROP + ") AS crop_sha256 "
    "FROM persons p "
    "LEFT JOIN templates t ON t.person_id = p.id AND t.status = 'active' "
)
_LIST_PERSONS = _PERSON_SELECT + "GROUP BY p.id ORDER BY p.display_name COLLATE NOCASE, p.id"
_GET_PERSON = _PERSON_SELECT + "WHERE p.id = ? GROUP BY p.id"


@router.get("", response_model=PersonListOut)
def list_persons(
    conn: ConnDep,
    q: Annotated[str | None, Query()] = None,
    status_filter: Annotated[
        Literal["enrolled", "unenrolled"] | None, Query(alias="status")
    ] = None,
) -> PersonListOut:
    rows = conn.execute(_LIST_PERSONS).fetchall()
    items = [_person_out(row) for row in rows]
    if q:
        needle = q.casefold()
        items = [item for item in items if needle in item.display_name.casefold()]
    if status_filter is not None:
        items = [item for item in items if item.status == status_filter]
    return PersonListOut(items=items)


@router.post("", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
def create_person(body: PersonCreate, conn: ConnDep, settings: SettingsDep) -> PersonOut:
    person_id = new_id()
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT INTO persons (id, display_name, notes, status, created_at, created_by) "
            "VALUES (?, ?, ?, 'unenrolled', ?, ?)",
            (person_id, body.display_name, body.notes, now, settings.operator_name),
        )
        audit.append(
            conn,
            actor=settings.operator_name,
            action="person.create",
            object_type="person",
            object_id=person_id,
            payload={"display_name": body.display_name, "notes": body.notes},
        )
    return PersonOut(
        id=person_id,
        display_name=body.display_name,
        notes=body.notes,
        do_not_enroll=False,
        status="unenrolled",
        template_count=0,
        crop_sha256=None,
        created_at=now,
        created_by=settings.operator_name,
    )


@router.get("/{person_id}", response_model=PersonDetailOut)
def get_person(person_id: str, conn: ConnDep) -> PersonDetailOut:
    row = conn.execute(_GET_PERSON, (person_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="person not found")
    template_rows = conn.execute(
        "SELECT t.*, d.crop_sha256 FROM templates t "
        "LEFT JOIN detections d ON d.id = t.detection_id "
        "WHERE t.person_id = ? ORDER BY t.created_at DESC, t.id DESC",
        (person_id,),
    ).fetchall()
    appearance_rows = conn.execute(
        "SELECT tr.id AS track_id, tr.media_id, m.case_id, i.source, mt.score, mt.band, "
        "tr.start_ms AS t_ms, d.crop_sha256, m.ingested_at FROM identities i "
        "JOIN tracks tr ON tr.id = i.track_id JOIN media m ON m.id = tr.media_id "
        "LEFT JOIN matches mt ON mt.track_id = tr.id AND mt.rank = 1 "
        "LEFT JOIN detections d ON d.id = tr.best_detection_id "
        "WHERE i.person_id = ? ORDER BY m.ingested_at DESC, tr.id DESC",
        (person_id,),
    ).fetchall()
    return PersonDetailOut(
        person=_person_out(row),
        templates=[_template_out(item) for item in template_rows],
        appearances=[
            AppearanceOut(
                track_id=str(item["track_id"]),
                media_id=str(item["media_id"]),
                case_id=str(item["case_id"]),
                source=item["source"],
                score=None if item["score"] is None else float(item["score"]),
                band=item["band"],
                t_ms=int(item["t_ms"]),
                crop_sha256=(
                    None if item["crop_sha256"] is None else str(item["crop_sha256"])
                ),
                ingested_at=str(item["ingested_at"]),
            )
            for item in appearance_rows
        ],
    )


@router.post(
    "/{person_id}/templates",
    response_model=TemplateOut,
    status_code=status.HTTP_201_CREATED,
)
def create_template(
    person_id: str, body: TemplateCreate, conn: ConnDep, settings: SettingsDep
) -> TemplateOut:
    with transaction(conn):
        try:
            template_id = create_template_from_detection(
                conn,
                person_id=person_id,
                detection_id=body.detection_id,
                embedder_model_id=settings.embedder_model,
                actor=settings.operator_name,
            )
        except EnrollmentError as exc:
            message = str(exc)
            code = (
                status.HTTP_404_NOT_FOUND
                if message == "person not found"
                else status.HTTP_409_CONFLICT
            )
            raise HTTPException(status_code=code, detail=message) from exc
    if template_id is None:
        row = conn.execute(
            "SELECT t.*, d.crop_sha256 FROM templates t "
            "LEFT JOIN detections d ON d.id = t.detection_id "
            "WHERE t.person_id = ? AND t.detection_id = ? AND t.embedder_model_id = ? "
            "AND t.status = 'active'",
            (person_id, body.detection_id, settings.embedder_model),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT t.*, d.crop_sha256 FROM templates t "
            "LEFT JOIN detections d ON d.id = t.detection_id WHERE t.id = ?",
            (template_id,),
        ).fetchone()
        enqueue(
            conn,
            kind="rematch",
            actor=settings.operator_name,
            params={"reason": "template_created", "template_id": template_id},
        )
    if row is None:
        raise HTTPException(status_code=500, detail="template write failed")
    return _template_out(row)


def _person_out(row: sqlite3.Row) -> PersonOut:
    return PersonOut(
        id=str(row["id"]),
        display_name=str(row["display_name"]),
        notes=None if row["notes"] is None else str(row["notes"]),
        do_not_enroll=bool(row["do_not_enroll"]),
        status=row["status"],
        template_count=int(row["template_count"]),
        crop_sha256=None if row["crop_sha256"] is None else str(row["crop_sha256"]),
        created_at=str(row["created_at"]),
        created_by=str(row["created_by"]),
    )


def _template_out(row: sqlite3.Row) -> TemplateOut:
    return TemplateOut(
        id=str(row["id"]),
        detection_id=None if row["detection_id"] is None else str(row["detection_id"]),
        source_case_id=(
            None if row["source_case_id"] is None else str(row["source_case_id"])
        ),
        embedder_model_id=str(row["embedder_model_id"]),
        quality=None if row["quality"] is None else float(row["quality"]),
        status=row["status"],
        crop_sha256=None if row["crop_sha256"] is None else str(row["crop_sha256"]),
        created_at=str(row["created_at"]),
        created_by=str(row["created_by"]),
    )
