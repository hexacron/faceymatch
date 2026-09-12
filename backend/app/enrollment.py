"""Template lifecycle: operator-only enrolment from stored detections, and revocation.

Revocation is one-way. There is deliberately no un-revoke endpoint and none should be
added: putting a face back into the gallery means enrolling it again from a stored
detection, which re-states the operator's decision and re-audits it under section 12. An
"unrevoke" would return special-category data to the matching gallery without anyone
claiming responsibility for that, and it would let a template outlive the reason it was
withdrawn for.

`persons.status` follows the active template count in one place (`sync_person_status`), so
the enrol and revoke paths cannot disagree about who is in the gallery (spec 7, 12).
"""

from __future__ import annotations

import json
import sqlite3

from app import audit
from app.ids import new_id


class EnrollmentError(ValueError):
    """A requested detection cannot become a template."""


class TemplateNotFoundError(LookupError):
    """No template with that id belongs to that person."""


class TemplateAlreadyRevokedError(ValueError):
    """The template is already revoked, and revocation is one-way."""


def create_template_from_detection(
    conn: sqlite3.Connection,
    *,
    person_id: str,
    detection_id: str,
    embedder_model_id: str,
    actor: str,
) -> str | None:
    """Create an active template inside the caller's transaction.

    Returns None when this exact person/detection/model template is already active. Callers
    are operator endpoints; automatic matching never imports this function.
    """
    person = conn.execute(
        "SELECT do_not_enroll FROM persons WHERE id = ?", (person_id,)
    ).fetchone()
    if person is None:
        raise EnrollmentError("person not found")
    if bool(person["do_not_enroll"]):
        raise EnrollmentError("person is marked do_not_enroll")

    row = conn.execute(
        "SELECT de.embedding, d.media_id, d.quality_json, m.case_id "
        "FROM detection_embeddings de "
        "JOIN detections d ON d.id = de.detection_id "
        "JOIN media m ON m.id = d.media_id "
        "WHERE de.detection_id = ? AND de.embedder_model_id = ?",
        (detection_id, embedder_model_id),
    ).fetchone()
    if row is None:
        raise EnrollmentError(
            "detection has no quality-passing embedding for the active embedder"
        )
    existing = conn.execute(
        "SELECT id FROM templates WHERE person_id = ? AND detection_id = ? "
        "AND embedder_model_id = ? AND status = 'active'",
        (person_id, detection_id, embedder_model_id),
    ).fetchone()
    if existing is not None:
        return None

    quality_data = json.loads(str(row["quality_json"]))
    quality_value = quality_data.get("det_score") if isinstance(quality_data, dict) else None
    template_id = new_id()
    now = audit.now_ts()
    conn.execute(
        "INSERT INTO templates (id, person_id, detection_id, source_case_id, embedding, "
        "embedder_model_id, quality, status, created_at, created_by) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
        (
            template_id,
            person_id,
            detection_id,
            str(row["case_id"]),
            row["embedding"],
            embedder_model_id,
            None if quality_value is None else float(quality_value),
            now,
            actor,
        ),
    )
    audit.append(
        conn,
        actor=actor,
        case_id=str(row["case_id"]),
        action="template.create",
        object_type="template",
        object_id=template_id,
        payload={
            "person_id": person_id,
            "detection_id": detection_id,
            "embedder_model_id": embedder_model_id,
        },
    )
    sync_person_status(conn, person_id=person_id, actor=actor, case_id=str(row["case_id"]))
    return template_id


def revoke_template(
    conn: sqlite3.Connection,
    *,
    person_id: str,
    template_id: str,
    reason: str | None,
    actor: str,
) -> None:
    """Revoke one template inside the caller's transaction. One-way (module docstring).

    The face leaves the matching gallery by construction: every gallery query already
    filters `templates.status = 'active'`, so there is no second exclusion list that could
    fall out of step with this one. The embedding row survives because the audit trail has
    to stay re-checkable against the bytes the claim was made from (spec 9, 12).
    """
    person = conn.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone()
    if person is None:
        raise TemplateNotFoundError("person not found")
    row = conn.execute(
        "SELECT status, detection_id, source_case_id, quality FROM templates "
        "WHERE id = ? AND person_id = ?",
        (template_id, person_id),
    ).fetchone()
    if row is None:
        raise TemplateNotFoundError("template not found for this person")
    if str(row["status"]) == "revoked":
        raise TemplateAlreadyRevokedError("template is already revoked")

    source_case_id = None if row["source_case_id"] is None else str(row["source_case_id"])
    conn.execute("UPDATE templates SET status = 'revoked' WHERE id = ?", (template_id,))
    audit.append(
        conn,
        actor=actor,
        case_id=source_case_id,
        action="template.revoke",
        object_type="template",
        object_id=template_id,
        payload={
            "person_id": person_id,
            "template_id": template_id,
            "detection_id": (
                None if row["detection_id"] is None else str(row["detection_id"])
            ),
            "source_case_id": source_case_id,
            "quality": None if row["quality"] is None else float(row["quality"]),
            "reason": reason,
        },
    )
    sync_person_status(conn, person_id=person_id, actor=actor, case_id=source_case_id)


def sync_person_status(
    conn: sqlite3.Connection,
    *,
    person_id: str,
    actor: str,
    case_id: str | None = None,
) -> None:
    """Make `persons.status` agree with the active template count (spec 7, 12).

    A person stranded at zero active templates is `unenrolled`: the row survives for audit
    and graph history, but they are out of the gallery and nothing may auto-accept to them.
    Enrolling again reverses that. Both directions live here so the two write paths state
    the rule once; a no-op transition writes nothing, including no audit entry.
    """
    row = conn.execute(
        "SELECT p.status, COUNT(t.id) AS active FROM persons p "
        "LEFT JOIN templates t ON t.person_id = p.id AND t.status = 'active' "
        "WHERE p.id = ? GROUP BY p.id",
        (person_id,),
    ).fetchone()
    if row is None:
        raise EnrollmentError("person not found")
    active = int(row["active"])
    wanted = "enrolled" if active > 0 else "unenrolled"
    if str(row["status"]) == wanted:
        return
    conn.execute("UPDATE persons SET status = ? WHERE id = ?", (wanted, person_id))
    audit.append(
        conn,
        actor=actor,
        case_id=case_id,
        action=f"person.{'enroll' if wanted == 'enrolled' else 'unenroll'}",
        object_type="person",
        object_id=person_id,
        payload={"status": wanted, "active_templates": active},
    )
