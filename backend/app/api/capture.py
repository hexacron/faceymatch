"""Screen capture ingest (tier 1: evidence).

The operator has a face on screen in some other application. This endpoint grabs those
pixels and hands them to the *same* `ingest_file` used by `POST /api/media`: hashed,
content-addressed under `media_dir`, one `media` row in the case, one `process` job for the
existing worker, one audit entry (spec 6.1, 9, invariant 7). There is no second detection
path here, and nothing ephemeral: a capture that reaches 201 is evidence.

The transient, match-only tier lives in `app/api/live.py` and stores nothing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import ConnDep, SettingsDep
from app.api.media import MediaUploadOut
from app.pipeline import capture, decode
from app.pipeline.capture import CaptureMode
from app.pipeline.ingest import Acquisition, CaseNotFoundError, ingest_file, require_case

router = APIRouter(prefix="/api", tags=["capture"])


class CaptureIn(BaseModel):
    case_id: str = Field(min_length=1)
    # Region is what the UI offers first: crosshair selection over whatever is on screen.
    mode: CaptureMode = "region"
    source_url: str | None = None


@router.post("/capture", response_model=MediaUploadOut, status_code=status.HTTP_201_CREATED)
def capture_screen(
    body: CaptureIn, conn: ConnDep, settings: SettingsDep
) -> MediaUploadOut:
    """Capture the screen (macOS) and ingest it as evidence in `case_id`.

    400 when the operator cancelled the selection, 404 for an unknown case, 413 over
    `max_upload_bytes`, 503 when capture cannot work on this host (not macOS, binary
    missing, or Screen Recording permission not granted).
    """
    # Checked before the capture: an unknown case must not cost the operator a selection.
    try:
        require_case(conn, body.case_id)
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="capture-") as tmpdir:
        # Temp file, never the clipboard: a pasteboard round-trip would clobber whatever
        # the operator had copied. The directory (and the file) go away on exit, so the
        # raw frame never outlives the ingest that hashed it.
        staged = Path(tmpdir) / f"capture{capture.CAPTURE_SUFFIX}"
        try:
            size = capture.capture_image(
                dest=staged, mode=body.mode, timeout=settings.capture_timeout_seconds
            )
        except capture.CaptureCancelledError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except capture.CaptureUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

        if size > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"capture exceeds max_upload_bytes ({settings.max_upload_bytes})",
            )

        try:
            result = ingest_file(
                conn,
                settings,
                case_id=body.case_id,
                src=staged,
                source_url=body.source_url or None,
                actor=settings.operator_name,
                acquisition=Acquisition("screen_capture", body.mode),
            )
        except CaseNotFoundError as exc:  # pragma: no cover - case deleted mid-capture
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except decode.ImageDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"capture produced unusable bytes: {exc}",
            ) from exc

    return MediaUploadOut(
        media_id=result.media_id,
        sha256=result.sha256,
        job_id=None if result.reused else result.job_id,
        reused=result.reused,
    )
