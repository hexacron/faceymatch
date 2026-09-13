"""Media purge and previews (spec 12, 6.1): deleting a file from the library.

What has to hold: the file and every row derived from it goes, a template enrolled from one
of its faces goes with it and leaves the person `unenrolled` rather than deleted, a claim an
operator made survives losing the match it was scored against (invariant 5), the bytes are
only unlinked when no other case still points at them, and the chain still verifies.
"""

from __future__ import annotations

import io
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image

from app import audit
from app.config import Settings
from app.core import storage
from app.db.conn import transaction
from tests.conftest import insert_match, seed_gallery

SECOND_CASE = "case-2"
SECOND_MEDIA = "media-2"


def _store_pixels(settings: Settings, colour: tuple[int, int, int] = (90, 120, 160)) -> str:
    """Real bytes in the object store, so a purge has something to unlink."""
    encoded = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(encoded, format="PNG")
    return storage.store_bytes(settings.media_dir, encoded.getvalue())


def _back_media_with_bytes(
    conn: sqlite3.Connection, settings: Settings, media_id: str
) -> tuple[str, object]:
    """Point a seeded media row at a stored object. Returns (sha256, path)."""
    digest = _store_pixels(settings)
    with transaction(conn):
        conn.execute(
            "UPDATE media SET sha256 = ?, path = ? WHERE id = ?",
            (digest, storage.relative_path_for(digest), media_id),
        )
    return digest, storage.path_for(settings.media_dir, digest)


def _store_crop(conn: sqlite3.Connection, settings: Settings, detection_id: str) -> str:
    """Give a detection a crop on disk, the way `process` would have."""
    encoded = io.BytesIO()
    Image.new("RGB", (112, 112), (10, 20, 30)).save(encoded, format="PNG")
    digest = storage.store_bytes(settings.crops_dir, encoded.getvalue())
    with transaction(conn):
        conn.execute(
            "UPDATE detections SET crop_sha256 = ? WHERE id = ?", (digest, detection_id)
        )
    return digest


def test_delete_removes_the_file_and_everything_derived_from_it(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn, calibrated=True)
    insert_match(conn, ids)
    _back_media_with_bytes(conn, settings, ids["media"])
    with transaction(conn):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES (?, ?, 'auto', 'match-1', ?, ?)",
            (ids["track"], ids["person"], ids["threshold_set"], audit.now_ts()),
        )
        conn.execute(
            "INSERT INTO identifications (id, track_id, person_id, decision, operator, "
            "created_at) VALUES ('ident-1', ?, ?, 'confirm', 'tester', ?)",
            (ids["track"], ids["person"], audit.now_ts()),
        )

    response = client.delete(f"/api/media/{ids['media']}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["detections"], body["tracks"], body["templates"]) == (1, 1, 1)
    assert (body["identities"], body["identifications"], body["matches"]) == (1, 1, 1)
    assert body["object_removed"] is True
    for table in ("media", "detections", "tracks", "templates", "identities", "matches"):
        query = f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608 - table names are literals here
        assert conn.execute(query).fetchone()["n"] == 0, table
    assert conn.execute("SELECT COUNT(*) AS n FROM identifications").fetchone()["n"] == 0
    assert audit.verify(conn).ok


def test_a_person_left_with_no_template_leaves_the_gallery_but_not_the_database(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    _back_media_with_bytes(conn, settings, ids["media"])

    body = client.delete(f"/api/media/{ids['media']}").json()

    assert body["persons_unenrolled"] == 1
    person = conn.execute(
        "SELECT status FROM persons WHERE id = ?", (ids["person"],)
    ).fetchone()
    assert person is not None, "the person row is history, not evidence: it must survive"
    assert person["status"] == "unenrolled"
    # The gallery changed, so every stored auto score is stale.
    job = conn.execute(
        "SELECT kind, params_json FROM jobs WHERE id = ?", (body["rematch_job_id"],)
    ).fetchone()
    assert job["kind"] == "rematch"
    assert ids["media"] in str(job["params_json"])


def test_deleting_a_file_nobody_was_enrolled_from_queues_no_rematch(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    with transaction(conn):
        conn.execute("DELETE FROM templates WHERE id = ?", (ids["template"],))
    _back_media_with_bytes(conn, settings, ids["media"])

    body = client.delete(f"/api/media/{ids['media']}").json()

    assert body["templates"] == 0
    assert body["rematch_job_id"] is None
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


def test_an_operator_decision_elsewhere_survives_losing_its_match(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """Invariant 5: the operator named that face, and a deleted template does not unname it."""
    ids = seed_gallery(conn)
    insert_match(conn, ids)
    _back_media_with_bytes(conn, settings, ids["media"])
    # A second file, whose track the operator confirmed against the template about to go.
    with transaction(conn):
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES (?, ?, ?, 'image', 'media/two.jpg', ?, 'done')",
            (SECOND_MEDIA, ids["case"], "f" * 64, audit.now_ts()),
        )
        conn.execute(
            "INSERT INTO tracks (id, media_id, start_ms, end_ms, embedding_mean, "
            "embedder_model_id) VALUES ('track-2', ?, 0, 0, ?, ?)",
            (SECOND_MEDIA, b"\x00" * 8, ids["embedder"]),
        )
        conn.execute(
            "INSERT INTO matches (id, track_id, person_id, rank, score, band, "
            "best_template_id, threshold_set_id, embedder_model_id, created_at) "
            "VALUES ('match-2', 'track-2', ?, 1, 0.8, 'strong', ?, ?, ?, ?)",
            (
                ids["person"],
                ids["template"],
                ids["threshold_set"],
                ids["embedder"],
                audit.now_ts(),
            ),
        )
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES ('track-2', ?, 'operator', 'match-2', ?, ?)",
            (ids["person"], ids["threshold_set"], audit.now_ts()),
        )

    assert client.delete(f"/api/media/{ids['media']}").status_code == 200

    kept = conn.execute("SELECT * FROM identities WHERE track_id = 'track-2'").fetchone()
    assert kept is not None, "an operator decision was deleted with someone else's template"
    assert kept["person_id"] == ids["person"]
    assert kept["match_id"] is None
    assert conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"] == 0


def test_an_auto_identity_scored_against_a_deleted_template_goes(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn, calibrated=True)
    _back_media_with_bytes(conn, settings, ids["media"])
    with transaction(conn):
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES (?, ?, ?, 'image', 'media/two.jpg', ?, 'done')",
            (SECOND_MEDIA, ids["case"], "f" * 64, audit.now_ts()),
        )
        conn.execute(
            "INSERT INTO tracks (id, media_id, start_ms, end_ms, embedding_mean, "
            "embedder_model_id) VALUES ('track-2', ?, 0, 0, ?, ?)",
            (SECOND_MEDIA, b"\x00" * 8, ids["embedder"]),
        )
        conn.execute(
            "INSERT INTO matches (id, track_id, person_id, rank, score, band, "
            "best_template_id, threshold_set_id, embedder_model_id, created_at) "
            "VALUES ('match-2', 'track-2', ?, 1, 0.8, 'strong', ?, ?, ?, ?)",
            (
                ids["person"],
                ids["template"],
                ids["threshold_set"],
                ids["embedder"],
                audit.now_ts(),
            ),
        )
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES ('track-2', ?, 'auto', 'match-2', ?, ?)",
            (ids["person"], ids["threshold_set"], audit.now_ts()),
        )

    assert client.delete(f"/api/media/{ids['media']}").status_code == 200

    assert conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()["n"] == 0
    assert conn.execute("SELECT 1 FROM tracks WHERE id = 'track-2'").fetchone() is not None


def test_the_bytes_stay_while_another_case_still_holds_them(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """One blob, many cases (spec 6.1). Deleting one copy must not blind the other."""
    ids = seed_gallery(conn)
    digest, path = _back_media_with_bytes(conn, settings, ids["media"])
    with transaction(conn):
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES (?, 'Other', 'test warrant', ?, 'tester')",
            (SECOND_CASE, audit.now_ts()),
        )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES (?, ?, ?, 'image', ?, ?, 'done')",
            (
                SECOND_MEDIA,
                SECOND_CASE,
                digest,
                storage.relative_path_for(digest),
                audit.now_ts(),
            ),
        )

    body = client.delete(f"/api/media/{ids['media']}").json()

    assert body["object_removed"] is False
    assert path.is_file(), "the other case's copy of these bytes was deleted"

    assert client.delete(f"/api/media/{SECOND_MEDIA}").json()["object_removed"] is True
    assert not path.exists()


def test_a_stored_crop_goes_with_the_detection_that_produced_it(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    _back_media_with_bytes(conn, settings, ids["media"])
    crop = _store_crop(conn, settings, ids["detection"])

    body = client.delete(f"/api/media/{ids['media']}").json()

    assert body["crops_removed"] == 1
    assert not storage.path_for(settings.crops_dir, crop).exists()


def test_deleting_a_file_that_is_already_gone_is_a_404_that_writes_nothing(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    _back_media_with_bytes(conn, settings, ids["media"])
    assert client.delete(f"/api/media/{ids['media']}").status_code == 200
    head = audit.head(conn)

    again = client.delete(f"/api/media/{ids['media']}")

    assert again.status_code == 404
    assert audit.head(conn) == head


def _second_media(conn: sqlite3.Connection, settings: Settings, case_id: str) -> str:
    """A second file in the same case, with its own bytes and its own detection."""
    digest = _store_pixels(settings, (10, 200, 40))
    with transaction(conn):
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES (?, ?, ?, 'image', ?, ?, 'done')",
            (
                SECOND_MEDIA,
                case_id,
                digest,
                storage.relative_path_for(digest),
                audit.now_ts(),
            ),
        )
        conn.execute(
            "INSERT INTO detections (id, media_id, t_ms, frame_idx, det_idx, x, y, w, h, "
            "landmarks_json, det_score, quality_json, detector_model_id) "
            "VALUES ('detection-2', ?, 0, 0, 0, 1, 2, 30, 30, '[]', 0.9, '{}', 'det-model')",
            (SECOND_MEDIA,),
        )
    return SECOND_MEDIA


def test_a_bulk_delete_removes_every_selected_file_and_queues_one_rematch(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    _back_media_with_bytes(conn, settings, ids["media"])
    second = _second_media(conn, settings, ids["case"])

    response = client.post(
        "/api/media/bulk_delete", json={"media_ids": [ids["media"], second]}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted"] == [ids["media"], second]
    assert body["errors"] == []
    assert (body["detections"], body["templates"], body["objects_removed"]) == (2, 1, 2)
    assert body["persons_unenrolled"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 0
    # One gallery, one re-match, however many files were selected.
    jobs = conn.execute("SELECT id, kind FROM jobs").fetchall()
    assert [job["kind"] for job in jobs] == ["rematch"]
    assert body["rematch_job_id"] == jobs[0]["id"]
    assert audit.verify(conn).ok


def test_a_file_that_is_already_gone_does_not_stop_the_rest_of_the_selection(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    _back_media_with_bytes(conn, settings, ids["media"])

    body = client.post(
        "/api/media/bulk_delete", json={"media_ids": ["gone", ids["media"]]}
    ).json()

    assert body["deleted"] == [ids["media"]]
    assert body["errors"] == [{"media_id": "gone", "error": "media not found"}]
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 0


def test_a_bulk_delete_of_files_nobody_was_enrolled_from_queues_no_rematch(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    with transaction(conn):
        conn.execute("DELETE FROM templates WHERE id = ?", (ids["template"],))
    _back_media_with_bytes(conn, settings, ids["media"])
    second = _second_media(conn, settings, ids["case"])

    body = client.post(
        "/api/media/bulk_delete", json={"media_ids": [ids["media"], second, ids["media"]]}
    ).json()

    # The repeated id is the same delete asked for twice, not a failure to report.
    assert body["deleted"] == [ids["media"], second]
    assert body["errors"] == []
    assert body["rematch_job_id"] is None
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0


def test_an_empty_selection_is_refused_before_anything_is_deleted(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    seed_gallery(conn)
    head = audit.head(conn)

    response = client.post("/api/media/bulk_delete", json={"media_ids": []})

    assert response.status_code == 422
    assert audit.head(conn) == head


def test_a_preview_is_a_smaller_picture_of_the_same_file(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = seed_gallery(conn)
    _back_media_with_bytes(conn, settings, ids["media"])

    response = client.get(f"/api/media/{ids['media']}/thumbnail?size=64")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(response.content)) as preview:
        assert max(preview.size) == 64
    # The second request is the browser's: the preview is derived, so it must not be re-encoded.
    again = client.get(
        f"/api/media/{ids['media']}/thumbnail?size=64",
        headers={"If-None-Match": response.headers["etag"]},
    )
    assert again.status_code == 304
    assert again.content == b""


def test_bytes_that_are_not_a_video_report_a_decode_failure(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """A preview is derived, so a file that will not open is 415 and never a 500."""
    ids = seed_gallery(conn)
    digest = storage.store_bytes(settings.media_dir, b"not a container, just bytes")
    with transaction(conn):
        conn.execute(
            "UPDATE media SET kind = 'video', sha256 = ?, path = ? WHERE id = ?",
            (digest, storage.relative_path_for(digest), ids["media"]),
        )

    response = client.get(f"/api/media/{ids['media']}/thumbnail")

    assert response.status_code == 415
