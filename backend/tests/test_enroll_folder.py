"""Folder enrolment (spec 6.6): a curated `Person Name/*.jpg` tree becomes a gallery.

What has to hold: one person per immediate subfolder with a template per image, everything it
cannot decide reported per file rather than guessed, a second run over the same folder writing
nothing, and the audit chain still verifying afterwards.

No models and no worker are involved: the path hashes each file to find the `media` row those
bytes were imported as and enrols from the *stored* detection (invariant 13), so a test writes
arbitrary bytes with a supported suffix and inserts the rows the import and the worker would
have left behind.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app import audit
from app.config import Settings
from app.db.conn import transaction
from tests.conftest import seed_gallery

CASE = "case-1"
REASON = "verifying folder enrolment"


def _seed_models(conn: sqlite3.Connection, settings: Settings) -> None:
    """The case, plus model rows under the ids the running settings name.

    `seed_gallery`'s models carry test ids; templates written by the endpoint reference
    `settings.embedder_model`, so that row has to exist under its real id.
    """
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES (?, 'Case', 'test warrant', ?, 'tester')",
            (CASE, now),
        )
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, "
            "dim) VALUES (?, 'yunet', '2023mar', 'detector', ?, 'MIT', 1, NULL)",
            (settings.detector_model, "a" * 64),
        )
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, "
            "dim) VALUES (?, 'sface', '2021dec', 'embedder', ?, 'Apache-2.0', 1, 128)",
            (settings.embedder_model, "b" * 64),
        )


def _write(root: Path, relative: str) -> Path:
    """One file with a supported suffix and bytes unique to its path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"pixels of {relative}".encode())
    return path


def _import(
    conn: sqlite3.Connection,
    settings: Settings,
    path: Path,
    *,
    media_id: str,
    status: str = "done",
    faces: int = 1,
) -> None:
    """The rows ingest and the worker would have written for `path`."""
    now = audit.now_ts()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with transaction(conn):
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES (?, ?, ?, 'image', ?, ?, ?)",
            (media_id, CASE, digest, f"media/{media_id}.jpg", now, status),
        )
        for index in range(faces):
            detection_id = f"{media_id}-d{index}"
            conn.execute(
                "INSERT INTO detections (id, media_id, t_ms, frame_idx, det_idx, x, y, w, h, "
                "landmarks_json, det_score, quality_json, crop_sha256, detector_model_id) "
                "VALUES (?, ?, 0, 0, ?, 1, 2, 30, 30, '[]', 0.9, '{\"det_score\": 0.9}', ?, ?)",
                (
                    detection_id,
                    media_id,
                    index,
                    hashlib.sha256(detection_id.encode()).hexdigest(),
                    settings.detector_model,
                ),
            )
            conn.execute(
                "INSERT INTO detection_embeddings (detection_id, embedding, "
                "embedder_model_id, created_at) VALUES (?, ?, ?, ?)",
                (detection_id, b"\x01" * 8, settings.embedder_model, now),
            )


def _enroll(client: TestClient, folder: Path) -> Any:
    response = client.post(
        "/api/persons/enroll_folder",
        json={"case_id": CASE, "folder_path": str(folder), "reason": REASON},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _two_people(conn: sqlite3.Connection, settings: Settings, root: Path) -> None:
    for index, name in enumerate(("Ada Lovelace", "Grace Hopper")):
        for number in (1, 2):
            path = _write(root, f"{name}/{number}.jpg")
            _import(conn, settings, path, media_id=f"media-{index}-{number}")


def test_each_person_folder_becomes_one_person_with_its_templates(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    _two_people(conn, settings, root)

    body = _enroll(client, root)

    assert body["templates_created"] == 4
    assert body["files_seen"] == 4
    assert body["skipped"] == []
    persons = sorted(body["persons"], key=lambda person: person["display_name"])
    assert [(p["display_name"], p["created"], p["templates_created"]) for p in persons] == [
        ("Ada Lovelace", True, 2),
        ("Grace Hopper", True, 2),
    ]

    listed = client.get("/api/persons").json()["items"]
    assert {(item["display_name"], item["status"], item["template_count"]) for item in listed} == {
        ("Ada Lovelace", "enrolled", 2),
        ("Grace Hopper", "enrolled", 2),
    }

    job = conn.execute(
        "SELECT kind, status FROM jobs WHERE id = ?", (body["rematch_job_id"],)
    ).fetchone()
    assert (job["kind"], job["status"]) == ("rematch", "queued")


def test_a_file_outside_a_person_folder_is_skipped_with_its_reason(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    _two_people(conn, settings, root)
    loose = _write(root, "loose.jpg")
    _import(conn, settings, loose, media_id="media-loose")

    body = _enroll(client, root)

    assert body["skipped"] == [
        {"file": "loose.jpg", "reason": "file is not inside a person folder"}
    ]
    assert body["templates_created"] == 4
    names = {person["display_name"] for person in body["persons"]}
    assert names == {"Ada Lovelace", "Grace Hopper"}
    assert conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"] == 2


def test_an_unimported_file_does_not_stop_the_folder(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    _two_people(conn, settings, root)
    _write(root, "Ada Lovelace/3.jpg")  # on disk, never imported

    body = _enroll(client, root)

    assert body["skipped"] == [
        {
            "file": "Ada Lovelace/3.jpg",
            "reason": "not imported: no media row for these bytes in this case",
        }
    ]
    assert body["templates_created"] == 4


def test_a_second_run_creates_nothing_new(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    _two_people(conn, settings, root)
    _enroll(client, root)

    body = _enroll(client, root)

    assert body["templates_created"] == 0
    assert body["rematch_job_id"] is None
    assert {skip["reason"] for skip in body["skipped"]} == {
        "already enrolled from this face"
    }
    assert len(body["skipped"]) == 4
    assert conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"] == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM templates").fetchone()["n"] == 4
    assert [person["created"] for person in body["persons"]] == [False, False]


def test_two_faces_in_one_image_are_left_for_the_operator(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    crowd = _write(root, "Ada Lovelace/crowd.jpg")
    _import(conn, settings, crowd, media_id="media-crowd", faces=2)

    body = _enroll(client, root)

    assert body["skipped"] == [
        {"file": "Ada Lovelace/crowd.jpg", "reason": "more than one face: enrol this one by hand"}
    ]
    assert body["templates_created"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM templates").fetchone()["n"] == 0


def test_a_do_not_enroll_person_is_skipped_without_aborting_the_batch(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    _two_people(conn, settings, root)
    with transaction(conn):
        conn.execute(
            "INSERT INTO persons (id, display_name, do_not_enroll, created_at, created_by) "
            "VALUES ('parked', 'Ada Lovelace', 1, ?, 'tester')",
            (audit.now_ts(),),
        )

    body = _enroll(client, root)

    assert body["templates_created"] == 2
    assert {skip["reason"] for skip in body["skipped"]} == {
        "person is marked do_not_enroll"
    }
    # The trigger would have aborted the statement; the check above it means it never fired.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM templates WHERE person_id = 'parked'"
    ).fetchone()["n"] == 0


def test_a_duplicate_person_name_refuses_the_request(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    _two_people(conn, settings, root)
    now = audit.now_ts()
    with transaction(conn):
        for person_id in ("twin-1", "twin-2"):
            conn.execute(
                "INSERT INTO persons (id, display_name, created_at, created_by) "
                "VALUES (?, 'Ada Lovelace', ?, 'tester')",
                (person_id, now),
            )

    response = client.post(
        "/api/persons/enroll_folder",
        json={"case_id": CASE, "folder_path": str(root), "reason": REASON},
    )

    assert response.status_code == 409, response.text
    assert "more than one person is named Ada Lovelace" in response.json()["detail"]
    assert conn.execute("SELECT COUNT(*) AS n FROM templates").fetchone()["n"] == 0


def test_the_audit_chain_still_verifies_after_a_folder_enrol(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)
    root = tmp_path / "tree"
    _two_people(conn, settings, root)
    _write(root, "loose.jpg")

    _enroll(client, root)

    assert audit.verify(conn).ok
    summary = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action = 'enrollment.folder'"
    ).fetchone()
    assert summary is not None
    assert REASON in str(summary["payload_json"])


def test_an_unknown_case_is_a_404(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    seed_gallery(conn)
    root = tmp_path / "tree"
    root.mkdir()

    response = client.post(
        "/api/persons/enroll_folder",
        json={"case_id": "nope", "folder_path": str(root), "reason": REASON},
    )

    assert response.status_code == 404, response.text


def test_a_path_that_is_not_a_folder_is_a_400(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    _seed_models(conn, settings)

    response = client.post(
        "/api/persons/enroll_folder",
        json={"case_id": CASE, "folder_path": str(tmp_path / "missing"), "reason": REASON},
    )

    assert response.status_code == 400, response.text
