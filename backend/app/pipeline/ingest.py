"""Ingest: hash, store, register, enqueue (spec 6.1, invariant 7).

Order is fixed and load-bearing: the file is hashed and copied into the content-addressed
store *before* anything decodes it (invariant 7), so the digest in `media.sha256` always
describes the bytes the operator handed us and never a re-encoded derivative.

Dedupe has two shapes (spec 6.1):

- Same bytes, same case: `media_case_sha256` is unique, so the existing row is returned
  with `reused=True` and no second `process` job is queued.
- Same bytes, different case: a new `media` row is written and it points at the object
  already on disk. One blob, many cases.

Every acquisition path lands here: multipart upload, folder import, and macOS screen
capture. They differ only in the `Acquisition` recorded on the audit entry (see below), so
there is exactly one hash-store-register-enqueue path and captured pixels are evidence on
the same terms as an uploaded file.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app import audit, jobs
from app.config import Settings
from app.core import storage
from app.db.conn import transaction
from app.ids import new_id
from app.pipeline import decode

MEDIA_KIND_IMAGE = "image"
MEDIA_STATUS_NEW = "new"

AcquisitionMode = Literal["upload", "folder_import", "screen_capture"]


class CaseNotFoundError(LookupError):
    """Media cannot be ingested into a case that does not exist."""


@dataclass(frozen=True, slots=True)
class Acquisition:
    """How these bytes reached us, recorded on the `media.ingest` audit entry.

    Why the payload and not a separate audit action: `media.ingest` already covers two
    distinct acquisition paths (multipart upload and folder import) and distinguishes them
    only by payload, while the sibling action `media.ingest_failed` denotes an *outcome*.
    Keeping one action preserves the property audit readers and the section 9 export bundle
    rely on — exactly one `media.ingest` entry per `media` row, whatever the source — and a
    payload field carries strictly more than a bare action would: mode *and* the
    screencapture selection mode.
    """

    mode: AcquisitionMode
    capture_mode: str | None = None

    def audit_fields(self) -> dict[str, str | None]:
        return {"acquisition": self.mode, "capture_mode": self.capture_mode}


UPLOAD = Acquisition("upload")
FOLDER_IMPORT = Acquisition("folder_import")


@dataclass(frozen=True, slots=True)
class IngestResult:
    media_id: str
    sha256: str
    job_id: str
    reused: bool


def ingest_file(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    case_id: str,
    src: Path,
    source_url: str | None = None,
    actor: str,
    acquisition: Acquisition = UPLOAD,
) -> IngestResult:
    """Register one still image in `case_id` and queue its processing job.

    Raises `UnsupportedImageError` for a suffix we do not decode, `CorruptImageError` for
    bytes that are not a readable image, and `CaseNotFoundError` for an unknown case.
    """
    decode.require_supported_image(src)
    require_case(conn, case_id)

    # Invariant 7: hash (and store) before any processing touches the pixels.
    digest, size_bytes = storage.store_file(settings.media_dir, src)

    existing = _find_media(conn, case_id=case_id, sha256=digest)
    if existing is not None:
        return IngestResult(
            media_id=existing,
            sha256=digest,
            job_id=_process_job_for(conn, media_id=existing, actor=actor),
            reused=True,
        )

    image = decode.decode_image(src)
    height, width = int(image.shape[0]), int(image.shape[1])
    rel_path = storage.relative_path_for(digest)
    media_id = new_id()
    now = audit.now_ts()

    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO media (id, case_id, sha256, kind, path, source_url, "
                "acquired_at, width, height, duration_ms, fps, ingested_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, ?, ?)",
                (
                    media_id,
                    case_id,
                    digest,
                    MEDIA_KIND_IMAGE,
                    rel_path,
                    source_url,
                    width,
                    height,
                    now,
                    MEDIA_STATUS_NEW,
                ),
            )
            audit.append(
                conn,
                actor=actor,
                action="media.ingest",
                object_type="media",
                object_id=media_id,
                case_id=case_id,
                payload={
                    "sha256": digest,
                    "kind": MEDIA_KIND_IMAGE,
                    "path": rel_path,
                    "size_bytes": size_bytes,
                    "width": width,
                    "height": height,
                    "source_url": source_url,
                    "filename": src.name,
                    **acquisition.audit_fields(),
                },
            )
    except sqlite3.IntegrityError:
        # Lost a race on media_case_sha256: the other writer's row is the canonical one.
        raced = _find_media(conn, case_id=case_id, sha256=digest)
        if raced is None:  # pragma: no cover - integrity error came from something else
            raise
        return IngestResult(
            media_id=raced,
            sha256=digest,
            job_id=_process_job_for(conn, media_id=raced, actor=actor),
            reused=True,
        )

    job = jobs.enqueue(
        conn,
        kind="process",
        actor=actor,
        params={"media_id": media_id},
        case_id=case_id,
    )
    return IngestResult(media_id=media_id, sha256=digest, job_id=job.id, reused=False)


def ingest_folder(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    case_id: str,
    folder: Path,
    actor: str,
) -> list[IngestResult]:
    """Recursive folder import: one job per decodable file, unsupported suffixes skipped.

    A single unreadable file does not abort the import. It is recorded as
    `media.ingest_failed` in the audit chain so the skip is evidence, not silence.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    require_case(conn, case_id)

    results: list[IngestResult] = []
    for path in _walk_supported(folder):
        try:
            results.append(
                ingest_file(
                    conn,
                    settings,
                    case_id=case_id,
                    src=path,
                    source_url=None,
                    actor=actor,
                    acquisition=FOLDER_IMPORT,
                )
            )
        except decode.ImageDecodeError as exc:
            with transaction(conn):
                audit.append(
                    conn,
                    actor=actor,
                    action="media.ingest_failed",
                    object_type="media",
                    case_id=case_id,
                    payload={"filename": path.name, "reason": str(exc)},
                )
    return results


def _walk_supported(folder: Path) -> Iterator[Path]:
    """Yield decodable files in a stable order, so a resumed import is reproducible."""
    for path in sorted(folder.rglob("*")):
        if path.is_file() and decode.is_supported_image(path):
            yield path


def require_case(conn: sqlite3.Connection, case_id: str) -> None:
    """Raise `CaseNotFoundError` unless the case exists.

    Public so a caller with a side effect to spend — the interactive screen capture — can
    reject an unknown case *before* asking the operator to select a region.
    """
    row = conn.execute("SELECT 1 FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        raise CaseNotFoundError(f"no case {case_id!r}")


def _find_media(conn: sqlite3.Connection, *, case_id: str, sha256: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM media WHERE case_id = ? AND sha256 = ?", (case_id, sha256)
    ).fetchone()
    return None if row is None else str(row["id"])


def _process_job_for(conn: sqlite3.Connection, *, media_id: str, actor: str) -> str:
    """The newest `process` job for this media, queueing one only if none exists.

    Re-ingesting identical bytes into the same case must not queue a second job, so the
    existing job id is reported back. The enqueue fallback covers a media row whose job
    was purged: without it that row would never be processed again.
    """
    row = conn.execute(
        "SELECT id FROM jobs WHERE kind = 'process' "
        "AND json_extract(params_json, '$.media_id') = ? "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (media_id,),
    ).fetchone()
    if row is not None:
        return str(row["id"])
    case_row = conn.execute("SELECT case_id FROM media WHERE id = ?", (media_id,)).fetchone()
    job = jobs.enqueue(
        conn,
        kind="process",
        actor=actor,
        params={"media_id": media_id},
        case_id=None if case_row is None else str(case_row["case_id"]),
    )
    return job.id
