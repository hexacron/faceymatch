"""Section 12 deletion.

A purge is the one operation in this system that destroys rather than records. Everything
else appends: an enrolment adds a template, a revoke marks one withdrawn, a re-match
rewrites derived scores. A purge exists because section 12 requires that biometric material
can actually be removed, and "removed" cannot mean "flagged".

Person purge removes the person, every template of theirs, every identity claiming them and
every match scoring them, in every case. It does not touch the evidence those claims were
made from: the media, the detections and the stored crops stay, because they are the record
of what was in the picture, not a claim about who it was. Re-ingesting the same file would
produce the same detections and no identity at all, which is the correct end state.

Media purge is the other direction, and the one section 12 describes for a case: the file
goes, and with it every row derived from it — detections, crops, tracks, and the templates
enrolled from its faces. A person left at zero active templates becomes `unenrolled` (spec
7, 12) rather than disappearing, because the history of what was claimed about them is not
the operator's to lose. The bytes are content-addressed and can be shared with another
case, so the object and each crop are only unlinked once no row anywhere still points at
them: one case deleting its copy must not blind another.

The audit entry carries the SHA-256 of the display name rather than the name, because
section 12 says a purge entry keeps hashes only. A log that reprints the personal data it
just deleted has not deleted it. The id and the counts stay in the clear, so a reviewer can
still prove what was removed and when, and can still tie it to the identifications the
chain already recorded.

Irreversible, and deliberately not ceremonious. The operator confirms once in the UI; the
backend does not ask for the name retyped or a written justification, because a form that
has to be argued with gets pattern-matched through rather than read. The audit entry is
the record that it happened.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import audit
from app.config import Settings
from app.core import storage
from app.db.conn import transaction
from app.enrollment import sync_person_status


class PersonNotFoundError(LookupError):
    """No person with that id."""


@dataclass(frozen=True, slots=True)
class PersonPurgeResult:
    """What the purge actually removed. Reported to the caller and to the audit log."""

    person_id: str
    templates: int
    identities: int
    identifications: int
    matches: int
    clusters_unlabelled: int

    def as_json(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "templates": self.templates,
            "identities": self.identities,
            "identifications": self.identifications,
            "matches": self.matches,
            "clusters_unlabelled": self.clusters_unlabelled,
        }


def _count(conn: sqlite3.Connection, sql: str, person_id: str) -> int:
    row = conn.execute(sql, (person_id,)).fetchone()
    return 0 if row is None else int(row["count"])


def purge_person(
    conn: sqlite3.Connection, *, person_id: str, actor: str
) -> PersonPurgeResult:
    """Delete one person and everything that claims them. Inside the caller's transaction.

    Raises `PersonNotFoundError` before anything is written, so a purge of a person who is
    already gone leaves no trace but the caller's 404.
    """
    row = conn.execute(
        "SELECT display_name, enrolled_in_case_id FROM persons WHERE id = ?", (person_id,)
    ).fetchone()
    if row is None:
        raise PersonNotFoundError("person not found")
    display_name = str(row["display_name"])
    case_id = None if row["enrolled_in_case_id"] is None else str(row["enrolled_in_case_id"])

    result = PersonPurgeResult(
        person_id=person_id,
        templates=_count(
            conn, "SELECT COUNT(*) AS count FROM templates WHERE person_id = ?", person_id
        ),
        identities=_count(
            conn, "SELECT COUNT(*) AS count FROM identities WHERE person_id = ?", person_id
        ),
        identifications=_count(
            conn,
            "SELECT COUNT(*) AS count FROM identifications WHERE person_id = ?",
            person_id,
        ),
        matches=_count(
            conn, "SELECT COUNT(*) AS count FROM matches WHERE person_id = ?", person_id
        ),
        clusters_unlabelled=_count(
            conn,
            "SELECT COUNT(*) AS count FROM clusters WHERE label_person_id = ?",
            person_id,
        ),
    )

    # Order is the foreign keys read backwards. `identities.match_id` and
    # `matches.best_template_id` only ever point inside the same person — an operator
    # decision records the match row for the person it names, and a match's best template
    # is that person's template — so removing this person's identities before their matches,
    # and their matches before their templates, cannot strand another person's row.
    conn.execute("DELETE FROM identifications WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM identities WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM matches WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM templates WHERE person_id = ?", (person_id,))
    # A cluster labelled with this person keeps its membership and loses the name.
    conn.execute(
        "UPDATE clusters SET label_person_id = NULL WHERE label_person_id = ?", (person_id,)
    )
    conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))

    audit.append(
        conn,
        actor=actor,
        case_id=case_id,
        action="person.purge",
        object_type="person",
        object_id=person_id,
        payload={
            # Hashes only (section 12): the entry proves which row went without reprinting
            # the name that was the point of deleting it.
            "display_name_sha256": hashlib.sha256(display_name.encode("utf-8")).hexdigest(),
            **result.as_json(),
        },
    )
    return result


class MediaNotFoundError(LookupError):
    """No media with that id."""


@dataclass(frozen=True, slots=True)
class MediaPurgeResult:
    """What the purge actually removed. Reported to the caller and to the audit log."""

    media_id: str
    detections: int
    tracks: int
    templates: int
    identities: int
    identifications: int
    matches: int
    persons_unenrolled: int
    object_removed: bool
    crops_removed: int

    def as_json(self) -> dict[str, Any]:
        return {
            "media_id": self.media_id,
            "detections": self.detections,
            "tracks": self.tracks,
            "templates": self.templates,
            "identities": self.identities,
            "identifications": self.identifications,
            "matches": self.matches,
            "persons_unenrolled": self.persons_unenrolled,
            "object_removed": self.object_removed,
            "crops_removed": self.crops_removed,
        }


# Everything one media file owns, as literal SQL bound by name on `:media`. Spelled out
# statement by statement rather than composed from fragments, because the order is the
# foreign keys read backwards — a claim before the thing it claims about — and that order is
# the whole correctness argument: it should be readable in one screen.
#
# What the caller and the audit entry are told went. Counted before the deletes, with the
# same predicates the deletes use, so the report cannot drift from the act.
_MEDIA_COUNTS: tuple[tuple[str, str], ...] = (
    (
        "detections",
        "SELECT COUNT(*) AS count FROM detections WHERE media_id = :media",
    ),
    (
        "tracks",
        "SELECT COUNT(*) AS count FROM tracks WHERE media_id = :media",
    ),
    (
        "templates",
        "SELECT COUNT(*) AS count FROM templates t "
        "JOIN detections d ON d.id = t.detection_id WHERE d.media_id = :media",
    ),
    (
        "identifications",
        "SELECT COUNT(*) AS count FROM identifications "
        "WHERE track_id IN (SELECT id FROM tracks WHERE media_id = :media)",
    ),
    (
        "identities",
        "SELECT COUNT(*) AS count FROM identities "
        "WHERE track_id IN (SELECT id FROM tracks WHERE media_id = :media) "
        "OR (source = 'auto' AND match_id IN (SELECT m.id FROM matches m "
        "JOIN templates t ON t.id = m.best_template_id "
        "JOIN detections d ON d.id = t.detection_id WHERE d.media_id = :media))",
    ),
    (
        "matches",
        "SELECT COUNT(*) AS count FROM matches "
        "WHERE track_id IN (SELECT id FROM tracks WHERE media_id = :media) "
        "OR best_template_id IN (SELECT t.id FROM templates t "
        "JOIN detections d ON d.id = t.detection_id WHERE d.media_id = :media)",
    ),
)

_MEDIA_PURGE_STEPS: tuple[str, ...] = (
    "DELETE FROM identifications "
    "WHERE track_id IN (SELECT id FROM tracks WHERE media_id = :media)",
    "DELETE FROM identities "
    "WHERE track_id IN (SELECT id FROM tracks WHERE media_id = :media)",
    # A claim on another file that rests on a template about to go. An auto identity was
    # that template's score and nothing else, so it goes with it; an operator decision
    # stands (invariant 5) and only loses the match it was recorded against, which is what
    # the follow-up re-match replaces.
    "DELETE FROM identities WHERE source = 'auto' AND match_id IN "
    "(SELECT m.id FROM matches m JOIN templates t ON t.id = m.best_template_id "
    "JOIN detections d ON d.id = t.detection_id WHERE d.media_id = :media)",
    "UPDATE identities SET match_id = NULL WHERE match_id IN "
    "(SELECT m.id FROM matches m JOIN templates t ON t.id = m.best_template_id "
    "JOIN detections d ON d.id = t.detection_id WHERE d.media_id = :media)",
    "DELETE FROM matches WHERE track_id IN (SELECT id FROM tracks WHERE media_id = :media)",
    "DELETE FROM matches WHERE best_template_id IN (SELECT t.id FROM templates t "
    "JOIN detections d ON d.id = t.detection_id WHERE d.media_id = :media)",
    "DELETE FROM templates "
    "WHERE detection_id IN (SELECT id FROM detections WHERE media_id = :media)",
    "DELETE FROM detection_embeddings "
    "WHERE detection_id IN (SELECT id FROM detections WHERE media_id = :media)",
    # `tracks.best_detection_id` points into the rows about to go, and the track row
    # outlives them by one statement.
    "UPDATE tracks SET best_detection_id = NULL WHERE media_id = :media",
    "DELETE FROM detections WHERE media_id = :media",
    "DELETE FROM tracks WHERE media_id = :media",
    "DELETE FROM media WHERE id = :media",
)


def purge_media(
    conn: sqlite3.Connection, settings: Settings, *, media_id: str, actor: str
) -> MediaPurgeResult:
    """Delete one media file, everything derived from it, and its bytes (section 12).

    Opens its own transaction: the object and the crops are unlinked afterwards, and a file
    removed for a transaction that then rolled back would be evidence lost to a crash.

    Raises `MediaNotFoundError` before anything is written, so deleting a file that is
    already gone leaves no trace but the caller's 404.
    """
    row = conn.execute(
        "SELECT case_id, sha256, path FROM media WHERE id = ?", (media_id,)
    ).fetchone()
    if row is None:
        raise MediaNotFoundError("media not found")
    case_id = str(row["case_id"])
    sha256 = str(row["sha256"])
    stored_path = str(row["path"])

    crop_digests = {
        str(item["crop_sha256"])
        for item in conn.execute(
            "SELECT DISTINCT crop_sha256 FROM detections "
            "WHERE media_id = ? AND crop_sha256 IS NOT NULL",
            (media_id,),
        ).fetchall()
    }
    # Whose gallery membership this file was holding up. Read before the deletes, because
    # afterwards the templates that name them are gone.
    person_status = {
        str(item["id"]): str(item["status"])
        for item in conn.execute(
            "SELECT DISTINCT p.id, p.status FROM persons p "
            "JOIN templates t ON t.person_id = p.id "
            "JOIN detections d ON d.id = t.detection_id WHERE d.media_id = :media",
            {"media": media_id},
        ).fetchall()
    }

    counts = {name: _media_count(conn, sql, media_id) for name, sql in _MEDIA_COUNTS}

    with transaction(conn):
        for statement in _MEDIA_PURGE_STEPS:
            conn.execute(statement, {"media": media_id})

        # A person whose last template was enrolled from this file leaves the gallery
        # (spec 7, 12). Counted as a transition, not a state: someone parked at zero
        # templates before this delete did not lose anything to it.
        unenrolled = 0
        for person_id, before in person_status.items():
            sync_person_status(conn, person_id=person_id, actor=actor, case_id=case_id)
            after = conn.execute(
                "SELECT status FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
            if after is not None and str(after["status"]) != before:
                unenrolled += 1

        # Decided inside the transaction, where "no row points at these bytes" is a fact
        # rather than a race; the unlink itself happens once the delete has committed.
        orphan_object = _unreferenced(conn, "SELECT 1 FROM media WHERE sha256 = ?", sha256)
        orphan_crops = [
            digest
            for digest in sorted(crop_digests)
            if _unreferenced(conn, "SELECT 1 FROM detections WHERE crop_sha256 = ?", digest)
        ]

        audit.append(
            conn,
            actor=actor,
            case_id=case_id,
            action="media.purge",
            object_type="media",
            object_id=media_id,
            payload={
                "sha256": sha256,
                "persons_unenrolled": unenrolled,
                "object_orphaned": orphan_object,
                "crops_orphaned": len(orphan_crops),
                **counts,
            },
        )

    object_removed = orphan_object and _unlink(storage.resolve(settings.media_dir, stored_path))
    crops_removed = sum(
        _unlink(storage.path_for(settings.crops_dir, digest)) for digest in orphan_crops
    )
    return MediaPurgeResult(
        media_id=media_id,
        persons_unenrolled=unenrolled,
        object_removed=object_removed,
        crops_removed=crops_removed,
        **counts,
    )


def _media_count(conn: sqlite3.Connection, sql: str, media_id: str) -> int:
    row = conn.execute(sql, {"media": media_id}).fetchone()
    return 0 if row is None else int(row["count"])


def _unreferenced(conn: sqlite3.Connection, sql: str, digest: str) -> bool:
    return conn.execute(sql, (digest,)).fetchone() is None


def _unlink(path: Path) -> bool:
    """Remove a stored object. A file already gone is the wanted end state, not an error."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
