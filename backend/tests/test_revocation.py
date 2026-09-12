"""Template revocation and do_not_enroll (spec 6.4, 12).

Duplicate enrolments make auto-accept structurally impossible: two persons tie on the same
face, the 6.4 margin rule correctly refuses to name either, and a 0.99 match stays
`ambiguous` forever. Revocation is the way out, so what is tested here is the observable
consequence of withdrawing a face — who is left in the gallery, who is still enrolled, and
whose identity survives the re-match that follows.
"""

from __future__ import annotations

import io
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import audit
from app.config import Settings
from app.core.registry import ActiveModels
from app.core.types import ARCFACE_TEMPLATE, Detection
from app.core.vectors import to_blob
from app.db.conn import transaction
from app.pipeline import live
from app.pipeline.matching import rematch

_EMBEDDER = "sface-2021dec"  # settings.embedder_model: the enrolment path looks for it
_FACE = (1.0, 0.0)
_OTHER_FACE = (0.0, 1.0)


class _Detector:
    model_id = "detector"

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(
                x=0.0,
                y=0.0,
                w=112.0,
                h=112.0,
                score=0.99,
                landmarks=ARCFACE_TEMPLATE.copy(),
            )
        ]


class _Embedder:
    model_id = _EMBEDDER
    dim = 2

    def embed(self, crops: np.ndarray) -> np.ndarray:
        count = 1 if crops.ndim == 3 else crops.shape[0]
        return np.tile(np.array([_FACE], dtype=np.float32), (count, 1))


def _models() -> ActiveModels:
    return ActiveModels(
        detector=_Detector(),
        embedder=_Embedder(),
        detector_model_id="detector",
        embedder_model_id=_EMBEDDER,
        execution_provider="CPUExecutionProvider",
    )


def _frame() -> bytes:
    """A sharp checkerboard, so the quality gate passes on a synthetic frame."""
    pattern = np.indices((240, 240)).sum(axis=0) % 2 * 255
    rgb = np.repeat(pattern[:, :, None], 3, axis=2).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _tuned(settings: Settings) -> Settings:
    """112 px synthetic faces sit below the shipped quality floors."""
    return settings.model_copy(update={"min_embed_px": 40, "min_sharpness": 0.0})


def _seed(conn: sqlite3.Connection) -> None:
    """The duplicate-gallery shape: one person enrolled twice, one enrolled once.

    `dup` and `single` hold the same face vector, which is the tie that 6.4 refuses to
    resolve. One track carries that face too, so a re-match has something to score.
    """
    live.clear_gallery_cache()
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, "
            "commercial_use, dim) VALUES ('detector', 'YuNet', '1', 'detector', ?, "
            "'MIT', 1, NULL)",
            ("a" * 64,),
        )
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, "
            "commercial_use, dim) VALUES (?, 'SFace', '1', 'embedder', ?, "
            "'Apache-2.0', 1, 2)",
            (_EMBEDDER, "b" * 64),
        )
        conn.execute(
            "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
            "calibrated, calibrated_at, eval_report_sha256, gallery_size, "
            "execution_provider, active, created_at) VALUES ('ts', ?, 0.6, 0.4, 0.05, "
            "1, ?, ?, 10, 'CPUExecutionProvider', 1, ?)",
            (_EMBEDDER, now, "c" * 64, now),
        )
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES ('case', 'Case', 'consent', ?, 'tester')",
            (now,),
        )
        for person_id, name in (("dup", "Dup Face"), ("single", "Single Face")):
            conn.execute(
                "INSERT INTO persons (id, display_name, status, created_at, created_by) "
                "VALUES (?, ?, 'enrolled', ?, 'tester')",
                (person_id, name, now),
            )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES ('media', 'case', ?, 'image', 'media/one.png', ?, 'done')",
            ("d" * 64, now),
        )
        conn.execute(
            "INSERT INTO tracks (id, media_id, start_ms, end_ms, best_detection_id, "
            "embedding_mean, embedder_model_id) VALUES ('track', 'media', 0, 0, NULL, "
            "?, ?)",
            (to_blob(np.array(_FACE, dtype=np.float32)), _EMBEDDER),
        )
        for detection_id, face in (("det-a", _FACE), ("det-b", _OTHER_FACE)):
            conn.execute(
                "INSERT INTO detections (id, media_id, track_id, t_ms, frame_idx, "
                "det_idx, x, y, w, h, landmarks_json, det_score, quality_json, "
                "crop_sha256, detector_model_id) VALUES (?, 'media', 'track', 0, 0, ?, "
                "0, 0, 112, 112, '[]', 0.99, '{\"det_score\": 0.99}', ?, 'detector')",
                (detection_id, 0 if detection_id == "det-a" else 1, detection_id * 8),
            )
            conn.execute(
                "INSERT INTO detection_embeddings (detection_id, embedding, "
                "embedder_model_id, created_at) VALUES (?, ?, ?, ?)",
                (detection_id, to_blob(np.array(face, dtype=np.float32)), _EMBEDDER, now),
            )
        conn.execute(
            "UPDATE tracks SET best_detection_id = 'det-a' WHERE id = 'track'"
        )
        # dup holds the contested face twice; single holds it once.
        for template_id, person_id in (
            ("tpl-dup-1", "dup"),
            ("tpl-dup-2", "dup"),
            ("tpl-single", "single"),
        ):
            conn.execute(
                "INSERT INTO templates (id, person_id, detection_id, source_case_id, "
                "embedding, embedder_model_id, quality, status, created_at, created_by) "
                "VALUES (?, ?, 'det-a', 'case', ?, ?, 0.9, 'active', ?, 'tester')",
                (
                    template_id,
                    person_id,
                    to_blob(np.array(_FACE, dtype=np.float32)),
                    _EMBEDDER,
                    now,
                ),
            )


def _revoke(client: TestClient, person_id: str, template_id: str) -> object:
    return client.post(
        f"/api/persons/{person_id}/templates/{template_id}/revoke",
        json={"reason": "duplicate from an agent verification session"},
    )


def _person(client: TestClient, person_id: str) -> dict[str, object]:
    payload = client.get(f"/api/persons/{person_id}").json()["person"]
    assert isinstance(payload, dict)
    return payload


def _actions(conn: sqlite3.Connection, action: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE action = ?", (action,)
        ).fetchone()["n"]
    )


def test_revoking_the_last_active_template_unenrols_the_person(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """Spec 12: zero active templates strands the person outside the gallery."""
    _seed(conn)

    response = _revoke(client, "single", "tpl-single")

    assert response.status_code == 200
    assert response.json()["status"] == "revoked"
    person = _person(client, "single")
    assert person["status"] == "unenrolled"
    assert person["template_count"] == 0
    assert person["crop_sha256"] is None
    assert _actions(conn, "person.unenroll") == 1


def test_revoking_one_of_two_templates_keeps_the_person_enrolled(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    _seed(conn)

    assert _revoke(client, "dup", "tpl-dup-1").status_code == 200

    person = _person(client, "dup")
    assert person["status"] == "enrolled"
    assert person["template_count"] == 1
    assert _actions(conn, "person.unenroll") == 0


def test_a_revoked_template_is_no_longer_a_rematch_candidate(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """The tie that made the face ambiguous is gone, so the survivor can be named."""
    _seed(conn)
    before = rematch(
        conn,
        settings,
        embedder_model_id=_EMBEDDER,
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )
    assert before.gallery_persons == 2
    assert before.auto_accepted == 0  # two persons tie: 6.4 refuses to pick (ambiguous)

    assert _revoke(client, "single", "tpl-single").status_code == 200

    after = rematch(
        conn,
        settings,
        embedder_model_id=_EMBEDDER,
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )

    assert after.gallery_persons == 1
    named = {
        str(row["person_id"])
        for row in conn.execute("SELECT person_id FROM matches").fetchall()
    }
    assert named == {"dup"}
    scored_templates = {
        str(row["best_template_id"])
        for row in conn.execute("SELECT best_template_id FROM matches").fetchall()
    }
    assert "tpl-single" not in scored_templates
    assert after.auto_accepted == 1


def test_a_revoked_template_is_no_longer_a_live_match_candidate(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    _seed(conn)
    tuned = _tuned(settings)
    models = _models()

    before = live.match_frame(conn, tuned, models, frame=_frame())
    assert {candidate.person_id for candidate in before.faces[0].candidates} == {
        "dup",
        "single",
    }
    assert before.faces[0].candidates[0].band == "ambiguous"

    assert _revoke(client, "single", "tpl-single").status_code == 200

    after = live.match_frame(conn, tuned, models, frame=_frame())

    candidates = after.faces[0].candidates
    assert [candidate.person_id for candidate in candidates] == ["dup"]
    assert candidates[0].band == "strong"
    assert after.gallery_persons == 1


def test_an_operator_identity_survives_the_rematch_after_a_revocation(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """Invariant 5, enforced by the identities_operator_wins trigger, not by convention."""
    _seed(conn)
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, threshold_set_id, "
            "updated_at) VALUES ('track', 'single', 'operator', 'ts', ?)",
            (now,),
        )

    # The operator's answer names a person whose only template is about to disappear.
    assert _revoke(client, "single", "tpl-single").status_code == 200
    rematch(
        conn,
        settings,
        embedder_model_id=_EMBEDDER,
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )

    row = conn.execute(
        "SELECT person_id, source FROM identities WHERE track_id = 'track'"
    ).fetchone()
    assert (str(row["person_id"]), str(row["source"])) == ("single", "operator")

    # And the trigger is what guarantees it: the downgrade is refused at the DB edge,
    # with a calibrated threshold set supplied so this is the only rule in play.
    with pytest.raises(sqlite3.IntegrityError, match="operator identity"), transaction(conn):
        conn.execute(
            "UPDATE identities SET source = 'auto', person_id = 'dup', "
            "threshold_set_id = 'ts', updated_at = ? WHERE track_id = 'track'",
            (audit.now_ts(),),
        )


def test_a_second_revoke_is_a_conflict_and_appends_nothing(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """Revocation is one-way and recorded once, however many times it is asked for."""
    _seed(conn)
    assert _revoke(client, "dup", "tpl-dup-1").status_code == 200
    head = client.get("/api/audit").json()["head_seq"]

    repeat = _revoke(client, "dup", "tpl-dup-1")

    assert repeat.status_code == 409
    assert client.get("/api/audit").json()["head_seq"] == head
    assert _actions(conn, "template.revoke") == 1


def test_revoking_what_does_not_belong_to_the_person_is_a_404(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    _seed(conn)
    head = client.get("/api/audit").json()["head_seq"]

    assert _revoke(client, "nobody", "tpl-dup-1").status_code == 404
    assert _revoke(client, "dup", "tpl-missing").status_code == 404
    # The template exists, but not for this person: naming the wrong owner reveals nothing.
    assert _revoke(client, "dup", "tpl-single").status_code == 404

    assert client.get("/api/audit").json()["head_seq"] == head
    assert _person(client, "dup")["template_count"] == 2


def test_do_not_enroll_parks_a_person_without_touching_their_faces(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """Spec 12: parking a person and withdrawing their face are different acts."""
    _seed(conn)

    response = client.patch("/api/persons/dup", json={"do_not_enroll": True})

    assert response.status_code == 200
    assert response.json()["do_not_enroll"] is True
    # Their templates are untouched: still active, still theirs, still enrolled.
    assert response.json()["template_count"] == 2
    assert response.json()["status"] == "enrolled"
    assert _actions(conn, "template.revoke") == 0

    # But they are out of the gallery, so nothing can be matched to them.
    result = rematch(
        conn,
        settings,
        embedder_model_id=_EMBEDDER,
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )
    assert result.gallery_persons == 1
    named = {
        str(row["person_id"])
        for row in conn.execute("SELECT person_id FROM matches").fetchall()
    }
    assert named == {"single"}

    # And no new face may be added to them while parked.
    refused = client.post("/api/persons/dup/templates", json={"detection_id": "det-b"})
    assert refused.status_code == 409

    # Unparking is the same setter, and restores them to the gallery.
    assert client.patch("/api/persons/dup", json={"do_not_enroll": False}).json()[
        "do_not_enroll"
    ] is False
    restored = rematch(
        conn,
        settings,
        embedder_model_id=_EMBEDDER,
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )
    assert restored.gallery_persons == 2


def test_patching_an_unknown_person_is_a_404(client: TestClient) -> None:
    assert client.patch("/api/persons/nobody", json={"do_not_enroll": True}).status_code == 404


def test_a_revoke_queues_the_rematch_that_rescores_the_gallery(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """The gallery changed, so stored auto identities must be rescored, as on enrolment."""
    _seed(conn)

    assert _revoke(client, "dup", "tpl-dup-1").status_code == 200

    queued = conn.execute(
        "SELECT params_json FROM jobs WHERE kind = 'rematch' AND status = 'queued'"
    ).fetchall()
    assert len(queued) == 1
    assert "tpl-dup-1" in str(queued[0]["params_json"])
