"""Live match (tier 2): advisory, match-only, stores nothing.

`POST /api/live/match` takes one frame and answers "who might these faces be", running the
same detector, quality gate, alignment, embedder and band logic as the stored pipeline. It
writes no row and appends no audit entry, which is exactly what makes it safe to call
repeatedly while the operator browses.

It is therefore also *not* an enrolment path. Nothing here can be tagged or confirmed:
to act on a face the operator sees, capture it first through `POST /api/capture` (tier 1),
which makes it hashed, content-addressed evidence, then decide on the stored detection.
See `app/pipeline/live.py` for the full rationale.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.deps import ConnDep, LockDep, SettingsDep
from app.core.registry import get_active_models
from app.core.types import Band
from app.pipeline import decode, live
from app.pipeline.ingest import CaseNotFoundError, require_case

router = APIRouter(prefix="/api/live", tags=["live"])


class LiveCandidateOut(BaseModel):
    person_id: str
    name: str
    rank: int
    score: float
    band: Band
    best_template_id: str


class LiveFaceOut(BaseModel):
    x: float
    y: float
    w: float
    h: float
    det_score: float
    quality_passed: bool
    quality_reasons: list[str]
    candidates: list[LiveCandidateOut]


class LiveTimingsOut(BaseModel):
    """Per-stage wall time in ms, so the UI can size its own sampling interval."""

    decode: float
    detect: float
    quality_align: float
    embed: float
    match: float


class LiveMatchOut(BaseModel):
    width: int
    height: int
    faces: list[LiveFaceOut]
    # Echoes the request, so a client can tell an empty candidate list from a frame nobody
    # was asked to identify.
    identified: bool
    threshold_set_id: str | None
    # The gate for the *stored* path. No live face is ever accepted: this only lets the UI
    # say why a stored match would not self-confirm either (invariant 4).
    auto_accept_allowed: bool
    auto_accept_reason: str | None
    gallery_persons: int
    elapsed_ms: int
    timings: LiveTimingsOut


@router.post("/match", response_model=LiveMatchOut)
def match_live_frame(
    conn: ConnDep,
    settings: SettingsDep,
    lock: LockDep,
    frame: Annotated[UploadFile, File()],
    case_id: Annotated[str | None, Form()] = None,
    identify: Annotated[bool, Form()] = True,
) -> LiveMatchOut:
    """Detect and score one frame. Nothing is persisted.

    `case_id` is optional and does not scope the gallery — persons and templates are global
    (spec 7, 12). It is validated when present so a stale case id in the UI surfaces as a
    404 here rather than silently mattering nowhere.

    `identify=false` returns boxes and the quality verdict only: no crop is warped, nothing
    is embedded, no gallery row is scored. It is for a client that wants boxes to track a
    moving face at detector latency and can ask for names less often.
    """
    if case_id:
        try:
            require_case(conn, case_id)
        except CaseNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    data = frame.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"frame exceeds max_upload_bytes ({settings.max_upload_bytes})",
        )

    # Cached per process: ONNX sessions are built once, never per request.
    models = get_active_models(settings, lock)
    try:
        result = live.match_frame(conn, settings, models, frame=data, identify=identify)
    except decode.ImageDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return LiveMatchOut(
        width=result.width,
        height=result.height,
        faces=[
            LiveFaceOut(
                x=face.x,
                y=face.y,
                w=face.w,
                h=face.h,
                det_score=face.det_score,
                quality_passed=face.quality_passed,
                quality_reasons=face.quality_reasons,
                candidates=[
                    LiveCandidateOut(
                        person_id=candidate.person_id,
                        name=candidate.name,
                        rank=candidate.rank,
                        score=candidate.score,
                        band=candidate.band,
                        best_template_id=candidate.best_template_id,
                    )
                    for candidate in face.candidates
                ],
            )
            for face in result.faces
        ],
        identified=result.identified,
        threshold_set_id=result.threshold_set_id,
        auto_accept_allowed=result.auto_accept_allowed,
        auto_accept_reason=result.auto_accept_reason,
        gallery_persons=result.gallery_persons,
        elapsed_ms=result.elapsed_ms,
        timings=LiveTimingsOut(**result.timings.as_dict()),
    )
