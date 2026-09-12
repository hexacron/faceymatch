from __future__ import annotations

import io
import sqlite3

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import audit
from app.config import Settings
from app.core import storage
from app.core.registry import ActiveModels
from app.core.types import ARCFACE_TEMPLATE, Detection
from app.db.conn import transaction
from app.main import seed_default_threshold_set
from app.pipeline.process import process_image
from tests.conftest import seed_gallery


def _seed_review_queue(
    conn: sqlite3.Connection, ids: dict[str, str], rows: list[tuple[str, str, float]]
) -> None:
    """Add rank-1 candidate tracks (track_id, band, score) to the review queue."""
    now = audit.now_ts()
    with transaction(conn):
        for track_id, band, score in rows:
            conn.execute(
                "INSERT INTO tracks (id, media_id, start_ms, end_ms, embedding_mean, "
                "embedder_model_id) VALUES (?, ?, 0, 0, ?, ?)",
                (track_id, ids["media"], b"\x00" * 8, ids["embedder"]),
            )
            conn.execute(
                "INSERT INTO matches (id, track_id, person_id, rank, score, band, "
                "best_template_id, threshold_set_id, embedder_model_id, created_at) "
                "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
                (
                    f"match-{track_id}",
                    track_id,
                    ids["person"],
                    score,
                    band,
                    ids["template"],
                    ids["threshold_set"],
                    ids["embedder"],
                    now,
                ),
            )


class FakeDetector:
    model_id = "detector"

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(
                x=20.0,
                y=20.0,
                w=112.0,
                h=112.0,
                score=0.99,
                landmarks=ARCFACE_TEMPLATE + np.array([20.0, 20.0], dtype=np.float32),
            )
        ]


class FakeEmbedder:
    model_id = "embedder"
    dim = 2

    def embed(self, crops: np.ndarray) -> np.ndarray:
        count = 1 if crops.ndim == 3 else crops.shape[0]
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (count, 1))


def test_case_and_person_lists_use_wire_envelopes(client: TestClient) -> None:
    case = client.post(
        "/api/cases", json={"name": "Case", "authorization_basis": "consent"}
    ).json()
    person = client.post("/api/persons", json={"display_name": "Ada"}).json()

    assert client.get("/api/cases").json()["items"] == [case]
    assert client.get("/api/persons").json()["items"] == [person]


def _seed_faces(conn: sqlite3.Connection) -> None:
    """Two persons with templates, plus one with none, for the tile view's face reference."""
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, dim) "
            "VALUES ('detector', 'Detector', '1', 'detector', ?, 'MIT', 1, NULL)",
            ("b" * 64,),
        )
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, dim) "
            "VALUES ('embedder', 'Embedder', '1', 'embedder', ?, 'MIT', 1, 2)",
            ("c" * 64,),
        )
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES ('case', 'Case', 'consent', ?, 'tester')",
            (now,),
        )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES ('media', 'case', ?, 'image', 'media/one.jpg', ?, 'done')",
            ("d" * 64, now),
        )
        for index, crop in enumerate((("worse", "a" * 64), ("better", "b" * 64), ("old", None))):
            name, crop_sha256 = crop
            conn.execute(
                "INSERT INTO tracks (id, media_id, start_ms, end_ms) VALUES (?, 'media', 0, 0)",
                (f"track-{name}",),
            )
            conn.execute(
                "INSERT INTO detections (id, media_id, track_id, t_ms, frame_idx, det_idx, "
                "x, y, w, h, landmarks_json, det_score, quality_json, crop_sha256, "
                "detector_model_id) VALUES (?, 'media', ?, 0, 0, ?, 1, 2, 30, 30, '[]', 0.9, "
                "'{}', ?, 'detector')",
                (f"detection-{name}", f"track-{name}", index, crop_sha256),
            )
        for person_id, display_name in (
            ("person-enrolled", "Ada Lovelace"),
            ("person-empty", "Grace Hopper"),
        ):
            conn.execute(
                "INSERT INTO persons (id, display_name, status, created_at, created_by) "
                "VALUES (?, ?, 'enrolled', ?, 'tester')",
                (person_id, display_name, now),
            )
        # Two active templates of different quality, plus a revoked one that scores higher
        # than either and must not be chosen.
        for template_id, detection_id, quality, template_status in (
            ("template-worse", "detection-worse", 0.40, "active"),
            ("template-better", "detection-better", 0.90, "active"),
            ("template-revoked", "detection-better", 0.99, "revoked"),
        ):
            conn.execute(
                "INSERT INTO templates (id, person_id, detection_id, source_case_id, "
                "embedding, embedder_model_id, quality, status, created_at, created_by) "
                "VALUES (?, 'person-enrolled', ?, 'case', ?, 'embedder', ?, ?, ?, 'tester')",
                (template_id, detection_id, b"\x00" * 8, quality, template_status, now),
            )


def test_person_payloads_carry_the_best_active_templates_crop(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """The tile view needs one face per person without a request per person."""
    _seed_faces(conn)

    items = {item["id"]: item for item in client.get("/api/persons").json()["items"]}

    # Highest-quality ACTIVE template wins; the revoked 0.99 one does not.
    assert items["person-enrolled"]["crop_sha256"] == "b" * 64
    assert items["person-enrolled"]["template_count"] == 2
    # No template at all: the UI renders its placeholder.
    assert items["person-empty"]["crop_sha256"] is None

    # The detail payload must agree with the list, or the tile and the page disagree.
    detail = client.get("/api/persons/person-enrolled").json()
    assert detail["person"]["crop_sha256"] == "b" * 64
    assert client.get("/api/persons/person-empty").json()["person"]["crop_sha256"] is None


def test_a_template_without_a_stored_crop_is_not_a_representative_face(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """A template can exist with no crop on disk; it cannot be the tile's face."""
    _seed_faces(conn)
    now = audit.now_ts()
    with transaction(conn):
        conn.execute("UPDATE templates SET status = 'revoked' WHERE person_id = ?",
                     ("person-enrolled",))
        conn.execute(
            "INSERT INTO templates (id, person_id, detection_id, source_case_id, embedding, "
            "embedder_model_id, quality, created_at, created_by) "
            "VALUES ('template-cropless', 'person-enrolled', 'detection-old', 'case', ?, "
            "'embedder', 1.0, ?, 'tester')",
            (b"\x00" * 8, now),
        )

    item = next(
        item
        for item in client.get("/api/persons").json()["items"]
        if item["id"] == "person-enrolled"
    )

    assert item["template_count"] == 1  # the crop-less template is active
    assert item["crop_sha256"] is None


def test_default_threshold_is_audited_and_idempotent(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, dim) "
            "VALUES ('embedder', 'SFace', '1', 'embedder', ?, 'Apache-2.0', 1, 2)",
            ("a" * 64,),
        )

    first = seed_default_threshold_set(conn, model_id="embedder", actor="tester")
    second = seed_default_threshold_set(conn, model_id="embedder", actor="tester")

    assert first is not None
    assert second is None
    row = conn.execute("SELECT * FROM threshold_sets").fetchone()
    assert row is not None
    assert (row["t_strong"], row["t_possible"], row["margin"]) == (0.55, 0.35, 0.05)
    assert row["calibrated"] == 0
    assert row["active"] == 1
    actions = conn.execute(
        "SELECT action FROM audit_log WHERE action = 'threshold_set.seed'"
    ).fetchall()
    assert len(actions) == 1


def test_process_image_runs_through_matching_without_redecode(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    tuned = settings.model_copy(update={"min_embed_px": 40, "min_sharpness": 0.0})
    image = np.indices((180, 180)).sum(axis=0) % 2 * 255
    rgb = np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(rgb).save(encoded, format="PNG")
    digest = storage.store_bytes(tuned.media_dir, encoded.getvalue())
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, dim) "
            "VALUES ('detector', 'Detector', '1', 'detector', ?, 'MIT', 1, NULL)",
            ("b" * 64,),
        )
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, dim) "
            "VALUES ('embedder', 'Embedder', '1', 'embedder', ?, 'MIT', 1, 2)",
            ("c" * 64,),
        )
        conn.execute(
            "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
            "calibrated, active, created_at) "
            "VALUES ('threshold', 'embedder', .55, .35, .05, 0, 1, ?)",
            (now,),
        )
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES ('case', 'Case', 'consent', ?, 'tester')",
            (now,),
        )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES ('media', 'case', ?, 'image', ?, ?, 'new')",
            (digest, storage.relative_path_for(digest), now),
        )

    result = process_image(
        conn,
        tuned,
        ActiveModels(
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            detector_model_id="detector",
            embedder_model_id="embedder",
            execution_provider="CPUExecutionProvider",
        ),
        media_id="media",
        actor="tester",
    )

    assert result.detections == result.quality_passed == result.tracks == 1
    assert result.matching is not None
    assert result.matching["auto_accept_allowed"] is False
    track = conn.execute("SELECT * FROM tracks WHERE media_id = 'media'").fetchone()
    assert track is not None and bytes(track["embedding_mean"]) == np.array(
        [1.0, 0.0], dtype="<f4"
    ).tobytes()
    assert conn.execute("SELECT COUNT(*) FROM detection_embeddings").fetchone()[0] == 1
    actions = {
        str(row["action"])
        for row in conn.execute(
            "SELECT action FROM audit_log WHERE action IN ('media.processed', 'matching.rematch')"
        )
    }
    assert actions == {"media.processed", "matching.rematch"}


def test_review_band_filter_runs_before_the_limit(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """A band the operator asked for must not be crowded out by the other band's scores."""
    ids = seed_gallery(conn, calibrated=True)
    _seed_review_queue(
        conn,
        ids,
        [(f"possible-{n}", "possible", 0.90 - n / 100) for n in range(5)]
        + [("ambiguous-0", "ambiguous", 0.50), ("ambiguous-1", "ambiguous", 0.49)],
    )

    ambiguous = client.get("/api/review", params={"band": "ambiguous", "limit": 3})
    assert ambiguous.status_code == 200
    assert [item["track_id"] for item in ambiguous.json()["items"]] == [
        "ambiguous-0",
        "ambiguous-1",
    ]

    possible = client.get("/api/review", params={"band": "possible", "limit": 3})
    assert [item["track_id"] for item in possible.json()["items"]] == [
        "possible-0",
        "possible-1",
        "possible-2",
    ]

    # Unfiltered, the limit applies to the whole queue by score.
    both = client.get("/api/review", params={"limit": 3})
    assert [item["band"] for item in both.json()["items"]] == ["possible"] * 3


def test_review_bulk_can_create_a_new_person(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    ids = seed_gallery(conn, calibrated=True)
    _seed_review_queue(conn, ids, [("track-unknown", "possible", 0.5)])

    resp = client.post(
        "/api/review/bulk",
        json={
            "decisions": [
                {"track_id": "track-unknown", "decision": "new", "new_name": "Jane Doe"}
            ]
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"applied": 1, "errors": []}
    created = client.get("/api/persons", params={"q": "Jane Doe"}).json()["items"]
    assert [person["display_name"] for person in created] == ["Jane Doe"]
    track = client.get("/api/tracks/track-unknown").json()
    assert track["identity"]["name"] == "Jane Doe"
    assert track["identity"]["source"] == "operator"
