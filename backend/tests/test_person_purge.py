"""Person purge (spec 12): the one operation that destroys rather than records.

What has to hold: it removes every claim naming the person, it leaves the evidence those
claims were made from, and the audit chain still verifies afterwards without reprinting the
name that was the point of deleting it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from fastapi.testclient import TestClient
from httpx2 import Response

from app import audit
from app.db.conn import transaction
from app.pipeline.matching import gallery_person_count
from tests.conftest import insert_match, seed_gallery

PERSON_NAME = "Person One"


def _delete(client: TestClient, person_id: str) -> Response:
    return client.delete(f"/api/persons/{person_id}")


def _count(conn: sqlite3.Connection, table: str, person_id: str) -> int:
    # Table names come from this module, never from input.
    query = f"SELECT COUNT(*) AS n FROM {table} WHERE person_id = ?"  # noqa: S608
    return int(conn.execute(query, (person_id,)).fetchone()["n"])


def test_delete_removes_every_claim_and_leaves_the_evidence(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    ids = seed_gallery(conn)
    insert_match(conn, ids)
    with transaction(conn):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES (?, ?, 'operator', 'match-1', ?, ?)",
            (ids["track"], ids["person"], ids["threshold_set"], audit.now_ts()),
        )
        conn.execute(
            "INSERT INTO identifications (id, track_id, person_id, decision, operator, "
            "created_at) VALUES ('ident-1', ?, ?, 'confirm', 'tester', ?)",
            (ids["track"], ids["person"], audit.now_ts()),
        )

    response = _delete(client, ids["person"])

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["templates"], body["identities"], body["identifications"]) == (1, 1, 1)
    assert body["matches"] == 1
    assert body["rematch_job_id"]

    assert conn.execute("SELECT 1 FROM persons WHERE id = ?", (ids["person"],)).fetchone() is None
    for table in ("templates", "identities", "identifications", "matches"):
        assert _count(conn, table, ids["person"]) == 0

    # The evidence those claims were made from is untouched: it is the record of what was
    # in the picture, not a claim about who it was.
    detection = conn.execute(
        "SELECT crop_sha256 FROM detections WHERE id = ?", (ids["detection"],)
    ).fetchone()
    assert detection is not None
    assert detection["crop_sha256"] == "e" * 64
    assert conn.execute("SELECT 1 FROM tracks WHERE id = ?", (ids["track"],)).fetchone() is not None
    assert conn.execute("SELECT 1 FROM media WHERE id = ?", (ids["media"],)).fetchone() is not None
    assert audit.verify(conn).ok


def test_a_deleted_person_leaves_the_matching_gallery(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """Nothing may score against a person who no longer exists."""
    ids = seed_gallery(conn)
    assert gallery_person_count(conn, embedder_model_id=ids["embedder"]) == 1

    assert _delete(client, ids["person"]).status_code == 200

    assert gallery_person_count(conn, embedder_model_id=ids["embedder"]) == 0


def test_deleting_the_same_person_twice_is_a_404_that_writes_nothing(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """Two windows, one row: the second attempt reports it is gone, it does not fail oddly."""
    ids = seed_gallery(conn)
    assert _delete(client, ids["person"]).status_code == 200
    head_after_delete = audit.head(conn)

    again = _delete(client, ids["person"])

    assert again.status_code == 404
    assert audit.head(conn) == head_after_delete


def test_the_audit_entry_hashes_the_name_it_deleted(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """Section 12: a purge entry keeps hashes only. A log that reprints the name has not
    deleted it."""
    ids = seed_gallery(conn)

    assert _delete(client, ids["person"]).status_code == 200

    row = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action = 'person.purge'"
    ).fetchone()
    assert row is not None
    payload = json.loads(str(row["payload_json"]))
    assert payload["display_name_sha256"] == hashlib.sha256(PERSON_NAME.encode()).hexdigest()
    assert PERSON_NAME not in str(row["payload_json"])
    assert payload["person_id"] == ids["person"]
