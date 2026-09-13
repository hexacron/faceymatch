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
from typing import Any

from app import audit


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
