"""Operator-only template enrollment from stored detections."""

from __future__ import annotations

import json
import sqlite3

from app import audit
from app.ids import new_id


class EnrollmentError(ValueError):
    """A requested detection cannot become a template."""


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
    conn.execute("UPDATE persons SET status = 'enrolled' WHERE id = ?", (person_id,))
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
    return template_id
