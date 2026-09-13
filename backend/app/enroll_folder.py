"""Bulk enrolment from a curated `Person Name/*.jpg` tree (spec 6.6, invariants 3 and 13).

One operator request with a required reason, exactly like every other path that puts faces in
the gallery: nothing here runs automatically and nothing here is a side effect of matching.

It never decodes a pixel and never embeds. Each file is hashed, the `media` row those bytes
were imported as is looked up, and the template is built from the *stored* detection — which
is what keeps invariant 13 intact: the template derives from hashed, audited evidence, not
from whatever is on disk at the moment of the request. A file the import never registered is
reported as a skip rather than ingested here, so `app.pipeline.ingest` stays the one home for
turning bytes into evidence.

What it refuses to guess is as important as what it does: a file loose in the root, an image
with more than one embeddable face, a folder name two persons already answer to. Guessing any
of those writes a gallery error that is far harder to undo than a skip line is to read.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app import audit
from app.config import Settings
from app.core import storage
from app.db.conn import transaction
from app.enrollment import (
    EnrollmentError,
    create_person,
    create_template_from_detection,
    sync_person_status,
)
from app.pipeline.ingest import require_case, walk_supported

# The response is a report an operator reads, not a log. Capped for the same reason the
# reembed job reports a sample rather than every crop it touched.
MAX_SKIPS_REPORTED = 200

SKIP_NOT_IN_PERSON_FOLDER = "file is not inside a person folder"
SKIP_NOT_IMPORTED = "not imported: no media row for these bytes in this case"
SKIP_STILL_PROCESSING = "still processing"
SKIP_PROCESS_FAILED = "processing failed"
SKIP_NO_FACE = "no quality-passing face for the active embedder"
SKIP_MANY_FACES = "more than one face: enrol this one by hand"
SKIP_DO_NOT_ENROLL = "person is marked do_not_enroll"
SKIP_ALREADY = "already enrolled from this face"


class AmbiguousPersonError(LookupError):
    """Two or more persons already share a folder's name, so the target is not decidable."""


@dataclass(frozen=True, slots=True)
class Skip:
    file: str  # path relative to the folder root
    reason: str


@dataclass(frozen=True, slots=True)
class PersonOutcome:
    display_name: str
    person_id: str
    created: bool  # this request inserted the person row
    templates_created: int


@dataclass(frozen=True, slots=True)
class FolderEnrollResult:
    persons: list[PersonOutcome]
    templates_created: int
    skipped: list[Skip]
    files_seen: int


@dataclass(slots=True)
class _Target:
    """One folder's person, resolved once and reused for every file under it."""

    person_id: str
    created: bool
    do_not_enroll: bool
    templates_created: int = 0


def enroll_folder(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    case_id: str,
    folder: Path,
    actor: str,
    reason: str,
) -> FolderEnrollResult:
    """Enrol one person per immediate subfolder of `folder` from already-imported media.

    One transaction per file, the shape `POST /api/review/bulk` uses: one unusable file must
    not roll back the rest of the folder.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    require_case(conn, case_id)

    targets: dict[str, _Target] = {}
    skipped: list[Skip] = []
    files_seen = 0

    for path in walk_supported(folder):
        files_seen += 1
        rel = path.relative_to(folder)
        if len(rel.parts) < 2:
            # A file loose in the root has no person to belong to, and the only name
            # available is the root's own — which names the batch, not a face.
            skipped.append(Skip(str(rel), SKIP_NOT_IN_PERSON_FOLDER))
            continue
        name = rel.parts[0]

        # Hash only, never store: if these bytes are in this case they are already in the
        # object store, and if they are not, this request is not the path that puts them there.
        digest, _size = storage.sha256_file(path)
        media = conn.execute(
            "SELECT id, status FROM media WHERE case_id = ? AND sha256 = ?",
            (case_id, digest),
        ).fetchone()
        if media is None:
            skipped.append(Skip(str(rel), SKIP_NOT_IMPORTED))
            continue
        status = str(media["status"])
        if status in ("new", "processing"):
            skipped.append(Skip(str(rel), SKIP_STILL_PROCESSING))
            continue
        if status == "failed":
            skipped.append(Skip(str(rel), SKIP_PROCESS_FAILED))
            continue

        # Exactly the detections that can become a template: `process` only embeds faces
        # that passed the quality gate, so the join is the gate.
        detections = conn.execute(
            "SELECT d.id FROM detections d "
            "JOIN detection_embeddings de "
            "  ON de.detection_id = d.id AND de.embedder_model_id = ? "
            "WHERE d.media_id = ? ORDER BY d.id",
            (settings.embedder_model, str(media["id"])),
        ).fetchall()
        if not detections:
            skipped.append(Skip(str(rel), SKIP_NO_FACE))
            continue
        if len(detections) > 1:
            skipped.append(Skip(str(rel), SKIP_MANY_FACES))
            continue

        target = targets.get(name)
        if target is None:
            target = _resolve_person(
                conn, name=name, case_id=case_id, folder=str(rel.parent), actor=actor
            )
            targets[name] = target
        if target.do_not_enroll:
            # Checked here rather than left to the `templates_respect_do_not_enroll`
            # trigger, whose ABORT would take the whole statement with it.
            skipped.append(Skip(str(rel), SKIP_DO_NOT_ENROLL))
            continue

        try:
            with transaction(conn):
                template_id = create_template_from_detection(
                    conn,
                    person_id=target.person_id,
                    detection_id=str(detections[0]["id"]),
                    embedder_model_id=settings.embedder_model,
                    actor=actor,
                )
        except EnrollmentError:
            # The join above already filtered for an embedding, so this is the race with a
            # re-embed, not the ordinary path.
            skipped.append(Skip(str(rel), SKIP_NO_FACE))
            continue
        if template_id is None:
            # That exact person/detection/model template is already active, which is what
            # makes a second run over the same folder a no-op.
            skipped.append(Skip(str(rel), SKIP_ALREADY))
            continue
        target.templates_created += 1

    persons = [
        PersonOutcome(
            display_name=name,
            person_id=target.person_id,
            created=target.created,
            templates_created=target.templates_created,
        )
        for name, target in targets.items()
    ]
    templates_created = sum(person.templates_created for person in persons)

    for person in persons:
        if person.templates_created:
            with transaction(conn):
                sync_person_status(
                    conn, person_id=person.person_id, actor=actor, case_id=case_id
                )

    by_reason: dict[str, int] = {}
    for skip in skipped:
        by_reason[skip.reason] = by_reason.get(skip.reason, 0) + 1
    with transaction(conn):
        audit.append(
            conn,
            actor=actor,
            case_id=case_id,
            action="enrollment.folder",
            object_type="case",
            object_id=case_id,
            payload={
                "folder": str(folder),
                "reason": reason,
                "files_seen": files_seen,
                "persons_created": sum(1 for person in persons if person.created),
                "templates_created": templates_created,
                # Counts by reason, not the file list: this payload is hashed into the
                # chain and must not grow with the size of the folder.
                "skipped": by_reason,
            },
        )

    return FolderEnrollResult(
        persons=persons,
        templates_created=templates_created,
        skipped=skipped[:MAX_SKIPS_REPORTED],
        files_seen=files_seen,
    )


def _resolve_person(
    conn: sqlite3.Connection, *, name: str, case_id: str, folder: str, actor: str
) -> _Target:
    """The person this folder enrols into, creating them when the name is new.

    Matched case-insensitively and ordered oldest first, the way the gallery lists names.
    Two persons answering to one folder name is refused outright: merging into an arbitrary
    one of them is a gallery error that is hard to undo and impossible to spot later.
    """
    rows = conn.execute(
        "SELECT id, do_not_enroll FROM persons WHERE display_name = ? COLLATE NOCASE "
        "ORDER BY created_at, id",
        (name,),
    ).fetchall()
    if len(rows) > 1:
        raise AmbiguousPersonError(name)
    if len(rows) == 1:
        return _Target(
            person_id=str(rows[0]["id"]),
            created=False,
            do_not_enroll=bool(rows[0]["do_not_enroll"]),
        )
    with transaction(conn):
        person_id = create_person(
            conn,
            display_name=name,
            case_id=case_id,
            actor=actor,
            audit_payload={"display_name": name, "from_folder": folder},
        )
    return _Target(person_id=person_id, created=True, do_not_enroll=False)
