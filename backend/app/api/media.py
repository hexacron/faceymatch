"""Media ingest, listing and byte serving (spec 6.1, 8).

Wire shapes are the locked contract in `frontend/src/api/types.ts`: `MediaList`,
`MediaUpload`, `MediaImport` and `MediaTracks` are envelopes, never bare arrays.

`GET /api/media/{id}/file` implements byte ranges. Stills do not need them, but the video
player in M2 does, and there is one reader for both kinds rather than two.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from app.api.deps import ConnDep, SettingsDep
from app.config import Settings
from app.core import storage
from app.core.types import Band, IdentitySource, MediaKind, MediaStatus
from app.jobs import JobStatus
from app.pipeline import decode
from app.pipeline.ingest import CaseNotFoundError, ingest_file, ingest_folder

router = APIRouter(prefix="/api/media", tags=["media"])

OCTET_STREAM = "application/octet-stream"
_RANGE_UNIT = "bytes="

# `_MEDIA_SELECT` only ever joins jobs whose params carry a `media_id`, and those are the
# per-file pipeline kinds. A gallery-wide `rematch` has no media_id and never lands here.
MediaJobKind = Literal["process", "rematch"]


class MediaJobOut(BaseModel):
    id: str
    kind: MediaJobKind
    status: JobStatus
    error: str | None
    progress: dict[str, Any]
    updated_at: str


class MediaOut(BaseModel):
    id: str
    case_id: str
    sha256: str
    kind: MediaKind
    source_url: str | None
    acquired_at: str | None
    width: int | None
    height: int | None
    duration_ms: int | None
    fps: float | None
    ingested_at: str
    status: MediaStatus
    job: MediaJobOut | None
    detection_count: int


class MediaListOut(BaseModel):
    items: list[MediaOut]


class MediaUploadOut(BaseModel):
    media_id: str
    sha256: str
    # Null exactly when `reused`: identical bytes in this case already have a job.
    job_id: str | None
    reused: bool


class MediaImportIn(BaseModel):
    case_id: str = Field(min_length=1)
    folder_path: str = Field(min_length=1)


class MediaImportOut(BaseModel):
    job_ids: list[str]
    media_ids: list[str]
    reused: int


class TrackSampleOut(BaseModel):
    t_ms: int
    x: float
    y: float
    w: float
    h: float
    detection_id: str
    crop_sha256: str | None


class TrackOverlayOut(BaseModel):
    track_id: str
    person_id: str | None
    name: str | None
    band: Band | None
    score: float | None
    source: IdentitySource | None
    samples: list[TrackSampleOut]
    crop_sha256: str | None


class MediaTracksOut(BaseModel):
    media_id: str
    width: int | None
    height: int | None
    tracks: list[TrackOverlayOut]


@router.post("", response_model=MediaUploadOut, status_code=status.HTTP_201_CREATED)
def upload_media(
    conn: ConnDep,
    settings: SettingsDep,
    case_id: Annotated[str, Form(min_length=1)],
    file: Annotated[UploadFile, File()],
    source_url: Annotated[str | None, Form()] = None,
) -> MediaUploadOut:
    """Multipart upload of one still image."""
    filename = file.filename or ""
    try:
        suffix = decode.require_supported_image(filename)
    except decode.UnsupportedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc

    with tempfile.TemporaryDirectory(prefix="upload-") as tmpdir:
        staged = Path(tmpdir) / f"upload{suffix}"
        _spool(file, staged, limit=settings.max_upload_bytes)
        result = _ingest(
            conn, settings, case_id=case_id, src=staged, source_url=source_url or None
        )
    return MediaUploadOut(
        media_id=result.media_id,
        sha256=result.sha256,
        job_id=None if result.reused else result.job_id,
        reused=result.reused,
    )


@router.post("/import", response_model=MediaImportOut, status_code=status.HTTP_202_ACCEPTED)
def import_folder(body: MediaImportIn, conn: ConnDep, settings: SettingsDep) -> MediaImportOut:
    """Recursive folder import: one `process` job per newly registered file (spec 6.1)."""
    folder = Path(body.folder_path).expanduser()
    try:
        results = ingest_folder(
            conn,
            settings,
            case_id=body.case_id,
            folder=folder,
            actor=settings.operator_name,
        )
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (NotADirectoryError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"not a readable folder: {folder}"
        ) from exc
    return MediaImportOut(
        job_ids=[r.job_id for r in results if not r.reused],
        media_ids=[r.media_id for r in results],
        reused=sum(1 for r in results if r.reused),
    )


@router.get("", response_model=MediaListOut)
def list_media(
    conn: ConnDep,
    status_filter: Annotated[
        Literal["new", "processing", "done", "failed"] | None, Query(alias="status")
    ] = None,
    case_id: Annotated[str | None, Query()] = None,
) -> MediaListOut:
    """Library listing. Latest job and detection count come from SQL, never an N+1 fetch."""
    where: list[str] = []
    params: list[Any] = []
    if status_filter is not None:
        where.append("m.status = ?")
        params.append(status_filter)
    if case_id is not None:
        where.append("m.case_id = ?")
        params.append(case_id)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"{_MEDIA_SELECT} {clause} ORDER BY m.ingested_at DESC, m.id DESC",
        params,
    ).fetchall()
    return MediaListOut(items=[_media_out(row) for row in rows])


@router.get("/{media_id}", response_model=MediaOut)
def get_media(media_id: str, conn: ConnDep) -> MediaOut:
    row = conn.execute(f"{_MEDIA_SELECT} WHERE m.id = ?", (media_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="media not found")
    return _media_out(row)


@router.get("/{media_id}/file")
def get_media_file(media_id: str, request: Request, conn: ConnDep, settings: SettingsDep) -> (
    StreamingResponse
):
    """Serve the stored original, with byte-range support for the M2 video player."""
    row = conn.execute("SELECT path, sha256 FROM media WHERE id = ?", (media_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="media not found")
    path = _resolve(settings, str(row["path"]))
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_410_GONE, detail="stored object is missing from the store"
        )
    size = path.stat().st_size
    media_type = _content_type(path)
    headers = {"Accept-Ranges": "bytes", "ETag": f'"{row["sha256"]}"'}

    span = _parse_range(request.headers.get("range"), size)
    if span is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            _iter_file(path, 0, size), media_type=media_type, headers=headers
        )
    start, end = span
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        _iter_file(path, start, end - start + 1),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=media_type,
        headers=headers,
    )


@router.get("/{media_id}/tracks", response_model=MediaTracksOut)
def get_media_tracks(
    media_id: str,
    conn: ConnDep,
    from_ms: Annotated[int | None, Query(ge=0)] = None,
    to_ms: Annotated[int | None, Query(ge=0)] = None,
) -> MediaTracksOut:
    """Overlay payload: every track on this media, identified or not.

    An unidentified track reports null `person_id`/`name`/`source` rather than being
    omitted, because the overlay still has to draw its box.
    """
    media = conn.execute(
        "SELECT id, width, height FROM media WHERE id = ?", (media_id,)
    ).fetchone()
    if media is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="media not found")

    samples: dict[str, list[TrackSampleOut]] = {}
    for det in conn.execute(
        "SELECT d.track_id, d.id AS detection_id, d.t_ms, d.x, d.y, d.w, d.h, d.crop_sha256 "
        "FROM detections d JOIN tracks t ON t.id = d.track_id "
        "WHERE t.media_id = ? ORDER BY d.t_ms, d.det_idx",
        (media_id,),
    ):
        t_ms = int(det["t_ms"])
        if (from_ms is not None and t_ms < from_ms) or (to_ms is not None and t_ms > to_ms):
            continue
        samples.setdefault(str(det["track_id"]), []).append(
            TrackSampleOut(
                t_ms=t_ms,
                x=float(det["x"]),
                y=float(det["y"]),
                w=float(det["w"]),
                h=float(det["h"]),
                detection_id=str(det["detection_id"]),
                crop_sha256=_opt_str(det["crop_sha256"]),
            )
        )

    tracks: list[TrackOverlayOut] = []
    for row in conn.execute(
        "SELECT t.id AS track_id, t.start_ms, t.end_ms, "
        "i.person_id, i.source, p.display_name AS name, "
        "mt.band, mt.score, best.crop_sha256 AS crop_sha256 "
        "FROM tracks t "
        "LEFT JOIN identities i ON i.track_id = t.id "
        "LEFT JOIN persons p ON p.id = i.person_id "
        "LEFT JOIN matches mt ON mt.track_id = t.id AND mt.rank = 1 "
        "LEFT JOIN detections best ON best.id = t.best_detection_id "
        "WHERE t.media_id = ? ORDER BY t.start_ms, t.id",
        (media_id,),
    ):
        if from_ms is not None and int(row["end_ms"]) < from_ms:
            continue
        if to_ms is not None and int(row["start_ms"]) > to_ms:
            continue
        track_id = str(row["track_id"])
        tracks.append(
            TrackOverlayOut(
                track_id=track_id,
                person_id=_opt_str(row["person_id"]),
                name=_opt_str(row["name"]),
                band=row["band"],
                score=None if row["score"] is None else float(row["score"]),
                source=row["source"],
                samples=samples.get(track_id, []),
                crop_sha256=_opt_str(row["crop_sha256"]),
            )
        )
    return MediaTracksOut(
        media_id=str(media["id"]),
        width=_opt_int(media["width"]),
        height=_opt_int(media["height"]),
        tracks=tracks,
    )


# One statement for both list and detail: the latest pipeline job per media comes from a
# windowed join, and the detection count from a correlated aggregate. No per-row queries.
_MEDIA_SELECT = """
SELECT m.id, m.case_id, m.sha256, m.kind, m.source_url, m.acquired_at, m.width, m.height,
       m.duration_ms, m.fps, m.ingested_at, m.status,
       j.id AS job_id, j.kind AS job_kind, j.status AS job_status, j.error AS job_error,
       j.progress AS job_progress, j.updated_at AS job_updated_at,
       (SELECT COUNT(*) FROM detections d WHERE d.media_id = m.id) AS detection_count
FROM media m
LEFT JOIN (
    SELECT id, kind, status, error, progress, updated_at,
           json_extract(params_json, '$.media_id') AS media_id,
           ROW_NUMBER() OVER (
               PARTITION BY json_extract(params_json, '$.media_id')
               ORDER BY created_at DESC, id DESC
           ) AS rn
    FROM jobs
    WHERE json_extract(params_json, '$.media_id') IS NOT NULL
) j ON j.media_id = m.id AND j.rn = 1
"""


def _media_out(row: sqlite3.Row) -> MediaOut:
    job = (
        None
        if row["job_id"] is None
        else MediaJobOut(
            id=str(row["job_id"]),
            kind=row["job_kind"],
            status=row["job_status"],
            error=_opt_str(row["job_error"]),
            progress=_progress(row["job_progress"]),
            updated_at=str(row["job_updated_at"]),
        )
    )
    return MediaOut(
        id=str(row["id"]),
        case_id=str(row["case_id"]),
        sha256=str(row["sha256"]),
        kind=row["kind"],
        source_url=_opt_str(row["source_url"]),
        acquired_at=_opt_str(row["acquired_at"]),
        width=_opt_int(row["width"]),
        height=_opt_int(row["height"]),
        duration_ms=_opt_int(row["duration_ms"]),
        fps=None if row["fps"] is None else float(row["fps"]),
        ingested_at=str(row["ingested_at"]),
        status=row["status"],
        job=job,
        detection_count=int(row["detection_count"]),
    )


def _progress(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        return {}
    parsed: Any = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _opt_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[call-overload]


def _spool(file: UploadFile, dest: Path, *, limit: int) -> None:
    """Copy the upload to `dest`, refusing anything over `limit` bytes (413).

    The size is counted while streaming rather than trusted from `Content-Length`, so a
    lying header cannot fill the disk.
    """
    written = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(storage.CHUNK_BYTES):
            written += len(chunk)
            if written > limit:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"upload exceeds max_upload_bytes ({limit})",
                )
            out.write(chunk)


def _ingest(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    case_id: str,
    src: Path,
    source_url: str | None,
) -> Any:
    try:
        return ingest_file(
            conn,
            settings,
            case_id=case_id,
            src=src,
            source_url=source_url,
            actor=settings.operator_name,
        )
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except decode.UnsupportedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except decode.ImageDecodeError as exc:
        # Corrupt bytes are the client's problem, not a server fault: 400, never a 500.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _resolve(settings: Settings, stored_path: str) -> Path:
    """`media.path` is store-relative; absolute values (legacy imports) are used as-is."""
    path = Path(stored_path)
    return path if path.is_absolute() else settings.media_dir / path


def _content_type(path: Path) -> str:
    """Content type from the bytes: stored objects have no extension to read it from."""
    fmt = decode.probe_format(path)
    if fmt is None:
        return OCTET_STREAM
    Image.init()
    return Image.MIME.get(fmt, OCTET_STREAM)


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single byte range to inclusive (start, end), or None to serve the whole body.

    Malformed and multi-range headers fall back to a 200 full body, which RFC 9110 allows.
    A syntactically valid but unsatisfiable range is a 416 with `Content-Range: bytes */n`.
    """
    if header is None or not header.startswith(_RANGE_UNIT):
        return None
    spec = header[len(_RANGE_UNIT) :].strip()
    if not spec or "," in spec:
        return None
    first, sep, last = spec.partition("-")
    if not sep:
        return None
    try:
        if not first:
            suffix = int(last)
            if suffix <= 0:
                raise _unsatisfiable(size)
            return max(0, size - suffix), size - 1
        start = int(first)
        end = size - 1 if not last else min(int(last), size - 1)
    except ValueError:
        return None
    if start >= size or end < start:
        raise _unsatisfiable(size)
    return start, end


def _unsatisfiable(size: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
        detail="requested range not satisfiable",
        headers={"Content-Range": f"bytes */{size}"},
    )


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(storage.CHUNK_BYTES, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk
