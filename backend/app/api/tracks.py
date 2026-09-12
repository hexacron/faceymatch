"""Track detail and aligned-crop retrieval (spec section 8)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.deps import ConnDep, SettingsDep
from app.core import storage
from app.core.types import Band, IdentitySource, RecordedDecision

router = APIRouter(prefix="/api", tags=["tracks"])


class CandidateOut(BaseModel):
    person_id: str
    name: str
    rank: int
    score: float
    band: Band
    best_template_id: str


class TrackIdentityOut(BaseModel):
    person_id: str
    name: str
    source: IdentitySource
    updated_at: str


class TrackHistoryOut(BaseModel):
    id: str
    decision: RecordedDecision
    person_id: str | None
    name: str | None
    operator: str
    note: str | None
    created_at: str


class TrackDetailOut(BaseModel):
    track_id: str
    media_id: str
    case_id: str
    start_ms: int
    end_ms: int
    identity: TrackIdentityOut | None
    candidates: list[CandidateOut]
    crops: list[str]
    detection_ids: list[str]
    history: list[TrackHistoryOut]


@router.get("/tracks/{track_id}", response_model=TrackDetailOut)
def get_track(track_id: str, conn: ConnDep) -> TrackDetailOut:
    row = conn.execute(
        "SELECT t.*, m.case_id, i.person_id, i.source, i.updated_at, "
        "p.display_name AS identity_name FROM tracks t "
        "JOIN media m ON m.id = t.media_id "
        "LEFT JOIN identities i ON i.track_id = t.id "
        "LEFT JOIN persons p ON p.id = i.person_id WHERE t.id = ?",
        (track_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="track not found")
    candidate_rows = conn.execute(
        "SELECT m.person_id, p.display_name, m.rank, m.score, m.band, m.best_template_id "
        "FROM matches m JOIN persons p ON p.id = m.person_id "
        "WHERE m.track_id = ? ORDER BY m.rank",
        (track_id,),
    ).fetchall()
    crop_rows = conn.execute(
        "SELECT id, crop_sha256 FROM detections WHERE track_id = ? "
        "AND crop_sha256 IS NOT NULL "
        "ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END, det_score DESC, t_ms, id",
        (track_id, row["best_detection_id"]),
    ).fetchall()
    history_rows = conn.execute(
        "SELECT h.*, p.display_name FROM identifications h "
        "LEFT JOIN persons p ON p.id = h.person_id WHERE h.track_id = ? "
        "ORDER BY h.created_at DESC, h.id DESC",
        (track_id,),
    ).fetchall()
    identity = None
    if row["person_id"] is not None:
        identity = TrackIdentityOut(
            person_id=str(row["person_id"]),
            name=str(row["identity_name"]),
            source=row["source"],
            updated_at=str(row["updated_at"]),
        )
    return TrackDetailOut(
        track_id=str(row["id"]),
        media_id=str(row["media_id"]),
        case_id=str(row["case_id"]),
        start_ms=int(row["start_ms"]),
        end_ms=int(row["end_ms"]),
        identity=identity,
        candidates=[
            CandidateOut(
                person_id=str(item["person_id"]),
                name=str(item["display_name"]),
                rank=int(item["rank"]),
                score=float(item["score"]),
                band=item["band"],
                best_template_id=str(item["best_template_id"]),
            )
            for item in candidate_rows
        ],
        crops=[str(item["crop_sha256"]) for item in crop_rows],
        detection_ids=[str(item["id"]) for item in crop_rows],
        history=[_history_out(item) for item in history_rows],
    )


@router.get("/crops/{sha256}")
def get_crop(sha256: str, settings: SettingsDep) -> FileResponse:
    try:
        path = storage.path_for(settings.crops_dir, sha256)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="crop not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="crop not found")
    return FileResponse(path, media_type="image/png", headers={"ETag": f'"{sha256}"'})


def _history_out(row: sqlite3.Row) -> TrackHistoryOut:
    return TrackHistoryOut(
        id=str(row["id"]),
        decision=row["decision"],
        person_id=None if row["person_id"] is None else str(row["person_id"]),
        name=None if row["display_name"] is None else str(row["display_name"]),
        operator=str(row["operator"]),
        note=None if row["note"] is None else str(row["note"]),
        created_at=str(row["created_at"]),
    )
