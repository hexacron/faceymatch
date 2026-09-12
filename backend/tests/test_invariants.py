"""One test per invariant this build can enforce (AGENTS.md test rules).

Invariant 10 has no runtime hook of its own: no attribute model is listed in models.lock,
and `test_unlisted_weights_refuse_to_start` is what stops one being loaded.
"""

from __future__ import annotations

import hashlib
import io
import json
import socket
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app import audit, models_lock
from app.config import REPO_ROOT, Settings
from app.core import storage, vectors
from app.db.conn import transaction
from app.enrollment import create_template_from_detection
from app.pipeline.ingest import ingest_file
from app.pipeline.matching import rematch
from tests.conftest import insert_match, seed_gallery


# Invariant 1: no outbound network calls at runtime. The guard is the autouse
# forbid_outbound_network fixture in conftest, so it covers the whole suite; here it is
# proven to actually bite, and the full request surface is exercised under it.
def test_outbound_network_is_blocked_for_the_whole_suite() -> None:
    with pytest.raises(AssertionError, match="DNS resolution is forbidden"):
        socket.getaddrinfo("example.com", 443)
    with socket.socket() as probe, pytest.raises(AssertionError, match="outbound connection"):
        probe.connect(("93.184.216.34", 443))


def test_no_outbound_call_happens_during_startup_or_requests(settings: Settings) -> None:
    from app.main import create_app

    with TestClient(create_app(settings)) as client:
        assert client.get("/api/healthz").status_code == 200
        assert (
            client.post(
                "/api/cases", json={"name": "C", "authorization_basis": "warrant"}
            ).status_code
            == 201
        )
        assert client.get("/api/audit").status_code == 200
        assert client.post("/api/jobs/audit_verify").status_code == 202


# Invariant 2: never compare embeddings with different model_id values.
def test_match_across_embedder_models_is_rejected(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn)
    with pytest.raises(sqlite3.IntegrityError, match="embedder_model_id boundary"):
        insert_match(conn, ids, embedder_model_id=ids["other_embedder"])

    insert_match(conn, ids)  # same model is accepted
    assert conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"] == 1


# Invariant 4 / C5: auto-accept only under a calibrated threshold set.
def test_auto_identity_requires_a_calibrated_threshold_set(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn, calibrated=False)
    insert_match(conn, ids)

    with (
        pytest.raises(sqlite3.IntegrityError, match="calibrated threshold set"),
        transaction(conn),
    ):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES (?, ?, 'auto', ?, ?, ?)",
            (ids["track"], ids["person"], "match-1", ids["threshold_set"], audit.now_ts()),
        )

    # The operator may still decide on an uncalibrated system.
    with transaction(conn):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES (?, ?, 'operator', ?, ?, ?)",
            (ids["track"], ids["person"], "match-1", ids["threshold_set"], audit.now_ts()),
        )
    row = conn.execute("SELECT source FROM identities WHERE track_id = ?", (ids["track"],))
    assert row.fetchone()["source"] == "operator"


def test_auto_identity_is_allowed_once_calibrated(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn, calibrated=True)
    insert_match(conn, ids)
    with transaction(conn):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES (?, ?, 'auto', ?, ?, ?)",
            (ids["track"], ids["person"], "match-1", ids["threshold_set"], audit.now_ts()),
        )
    assert conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()["n"] == 1


# Invariant 5: an operator decision always wins.
def test_rematch_cannot_overwrite_an_operator_identity(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn, calibrated=True)
    insert_match(conn, ids)
    with transaction(conn):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES (?, ?, 'operator', ?, ?, ?)",
            (ids["track"], ids["person"], "match-1", ids["threshold_set"], audit.now_ts()),
        )

    with pytest.raises(sqlite3.IntegrityError, match="operator identity"), transaction(conn):
        conn.execute(
            "UPDATE identities SET source = 'auto', updated_at = ? WHERE track_id = ?",
            (audit.now_ts(), ids["track"]),
        )


def _rematchable(conn: sqlite3.Connection, *, execution_provider: str) -> dict[str, str]:
    """`seed_gallery` with real 2-d vectors, so `rematch` can actually score it."""
    ids = seed_gallery(conn, calibrated=True)
    blob = vectors.to_blob(np.array([1.0, 0.0], dtype=np.float32))
    with transaction(conn):
        conn.execute("UPDATE models SET dim = 2 WHERE kind = 'embedder'")
        conn.execute("UPDATE templates SET embedding = ? WHERE id = ?", (blob, ids["template"]))
        conn.execute("UPDATE tracks SET embedding_mean = ? WHERE id = ?", (blob, ids["track"]))
        conn.execute(
            "UPDATE threshold_sets SET execution_provider = ? WHERE id = ?",
            (execution_provider, ids["threshold_set"]),
        )
    return ids


# Spec 10: a threshold set is only reproducible on the execution provider it was
# calibrated on, so auto-accept requires the runtime provider to match it.
def test_auto_accept_requires_the_calibrated_execution_provider(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = _rematchable(conn, execution_provider="CoreMLExecutionProvider")

    blocked = rematch(
        conn,
        settings,
        embedder_model_id=ids["embedder"],
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )
    assert blocked.matches == 1
    assert blocked.auto_accepted == 0
    assert blocked.gate_allowed is False
    assert conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()["n"] == 0

    # Same scores, same gallery: only the calibrated provider changes.
    with transaction(conn):
        conn.execute(
            "UPDATE threshold_sets SET execution_provider = 'CPUExecutionProvider' WHERE id = ?",
            (ids["threshold_set"],),
        )
    allowed = rematch(
        conn,
        settings,
        embedder_model_id=ids["embedder"],
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )
    assert allowed.auto_accepted == 1
    row = conn.execute(
        "SELECT source FROM identities WHERE track_id = ?", (ids["track"],)
    ).fetchone()
    assert row["source"] == "auto"


# Invariant 3: only operator actions create templates. Auto-matches never do.
def test_rematch_never_creates_a_template(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = _rematchable(conn, execution_provider="CPUExecutionProvider")
    before = conn.execute("SELECT COUNT(*) AS n FROM templates").fetchone()["n"]

    result = rematch(
        conn,
        settings,
        embedder_model_id=ids["embedder"],
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )

    assert result.auto_accepted == 1  # the auto path ran, and still enrolled nothing
    assert conn.execute("SELECT COUNT(*) AS n FROM templates").fetchone()["n"] == before
    created_by = {
        str(row["created_by"])
        for row in conn.execute("SELECT created_by FROM templates").fetchall()
    }
    assert created_by == {"tester"}


def test_enrollment_from_a_detection_needs_an_embedding_for_the_active_model(
    conn: sqlite3.Connection,
) -> None:
    """An operator can only enrol what the active embedder actually produced (invariant 2)."""
    ids = seed_gallery(conn)
    with pytest.raises(ValueError, match="no quality-passing embedding"), transaction(conn):
        create_template_from_detection(
            conn,
            person_id=ids["person"],
            detection_id=ids["detection"],
            embedder_model_id=ids["other_embedder"],
            actor="tester",
        )


def _enrollable(conn: sqlite3.Connection, settings: Settings) -> dict[str, str]:
    """`seed_gallery` where the track's best detection has an embedding for the *active*
    embedder, which is what the enrollment path looks for."""
    ids = seed_gallery(conn)
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, "
            "dim) VALUES (?, 'SFace', '2021dec', 'embedder', ?, 'Apache-2.0', 1, 2)",
            (settings.embedder_model, "f" * 64),
        )
        conn.execute(
            "UPDATE tracks SET best_detection_id = ? WHERE id = ?",
            (ids["detection"], ids["track"]),
        )
        conn.execute(
            "INSERT INTO detection_embeddings (detection_id, embedding, embedder_model_id, "
            "created_at) VALUES (?, ?, ?, ?)",
            (
                ids["detection"],
                vectors.to_blob(np.array([1.0, 0.0], dtype=np.float32)),
                settings.embedder_model,
                audit.now_ts(),
            ),
        )
    return ids


def _templates(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM templates").fetchone()["n"])


def _last_payload(conn: sqlite3.Connection, action: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action = ? ORDER BY seq DESC LIMIT 1",
        (action,),
    ).fetchone()
    payload: dict[str, Any] = json.loads(str(row["payload_json"]))
    return payload


# D17 / spec 6.6: tagging and enrolling are separate operator actions.
def test_confirm_does_not_enrol_unless_asked(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = _enrollable(conn, settings)
    other = client.post("/api/persons", json={"display_name": "Someone Else"}).json()["id"]
    before = _templates(conn)

    resp = client.post(
        "/api/identifications",
        json={"track_id": ids["track"], "decision": "confirm", "person_id": other},
    )

    assert resp.status_code == 201
    assert _templates(conn) == before
    identity = client.get(f"/api/tracks/{ids['track']}").json()["identity"]
    assert identity["person_id"] == other
    assert identity["source"] == "operator"
    assert resp.json()["template_created"] is False
    assert _last_payload(conn, "identification.confirm")["template_created"] is False
    assert client.get(f"/api/persons/{other}").json()["person"]["status"] == "unenrolled"


def test_confirm_with_enroll_creates_the_template(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = _enrollable(conn, settings)
    other = client.post("/api/persons", json={"display_name": "Someone Else"}).json()["id"]
    before = _templates(conn)

    resp = client.post(
        "/api/identifications",
        json={
            "track_id": ids["track"],
            "decision": "confirm",
            "person_id": other,
            "enroll": True,
        },
    )

    assert resp.status_code == 201
    assert _templates(conn) == before + 1
    assert resp.json()["template_created"] is True
    assert _last_payload(conn, "identification.confirm")["template_created"] is True
    detail = client.get(f"/api/persons/{other}").json()
    assert detail["person"]["status"] == "enrolled"
    assert [t["created_by"] for t in detail["templates"]] == [settings.operator_name]


def test_new_person_is_always_bootstrapped_with_one_template(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """Without it the person is born unenrolled and could never match anything."""
    ids = _enrollable(conn, settings)
    before = _templates(conn)

    resp = client.post(
        "/api/identifications",
        json={"track_id": ids["track"], "decision": "new", "new_name": "Bootstrapped"},
    )

    assert resp.status_code == 201
    assert _templates(conn) == before + 1
    assert resp.json()["template_created"] is True
    assert _last_payload(conn, "identification.new")["template_created"] is True
    created = client.get("/api/persons", params={"q": "Bootstrapped"}).json()["items"]
    assert [(p["status"], p["template_count"]) for p in created] == [("enrolled", 1)]


def test_a_new_person_that_could_not_be_enrolled_says_so_on_the_response(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    """The silent gallery hole: `new` on a track with no quality-passing crop saves the
    decision, creates the person, and enrols nothing. The response has to admit that, or
    the operator is told "saved" about a person who can never be matched."""
    ids = seed_gallery(conn)  # no embedding for the active embedder: nothing to enrol from
    with transaction(conn):
        conn.execute(
            "UPDATE tracks SET best_detection_id = ? WHERE id = ?",
            (ids["detection"], ids["track"]),
        )
    before = _templates(conn)

    resp = client.post(
        "/api/identifications",
        json={"track_id": ids["track"], "decision": "new", "new_name": "Unenrollable"},
    )

    assert resp.status_code == 201
    assert resp.json()["template_created"] is False
    assert _templates(conn) == before
    created = client.get("/api/persons", params={"q": "Unenrollable"}).json()["items"]
    assert [(p["status"], p["template_count"], p["crop_sha256"]) for p in created] == [
        ("unenrolled", 0, None)
    ]
    # The decision itself still stands: tagging a track we cannot enrol from is valid.
    identity = client.get(f"/api/tracks/{ids['track']}").json()["identity"]
    assert identity["source"] == "operator"


def test_reject_refuses_an_enroll_request(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = _enrollable(conn, settings)
    resp = client.post(
        "/api/identifications",
        json={"track_id": ids["track"], "decision": "reject", "enroll": True},
    )
    assert resp.status_code == 422
    assert _templates(conn) == 1


def test_review_bulk_carries_the_enroll_opt_in(
    client: TestClient, conn: sqlite3.Connection, settings: Settings
) -> None:
    ids = _enrollable(conn, settings)
    other = client.post("/api/persons", json={"display_name": "Someone Else"}).json()["id"]
    before = _templates(conn)
    decision = {"track_id": ids["track"], "decision": "confirm", "person_id": other}

    plain = client.post("/api/review/bulk", json={"decisions": [decision]})
    assert plain.json() == {"applied": 1, "errors": []}
    assert _templates(conn) == before

    enrolling = client.post(
        "/api/review/bulk", json={"decisions": [{**decision, "enroll": True}]}
    )
    assert enrolling.json() == {"applied": 1, "errors": []}
    assert _templates(conn) == before + 1


# Invariant 7: hash every ingested file (SHA-256) before processing.
def test_ingest_hashes_and_stores_by_content(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    settings.ensure_dirs()
    encoded = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(encoded, format="PNG")
    payload = encoded.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    first_path = tmp_path / "first.png"
    first_path.write_bytes(payload)
    with transaction(conn):
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES ('case-1', 'Case', 'test warrant', ?, 'tester')",
            (audit.now_ts(),),
        )

    first = ingest_file(conn, settings, case_id="case-1", src=first_path, actor="tester")

    assert first.sha256 == digest
    assert first.reused is False
    stored = settings.media_dir / storage.relative_path_for(digest)
    assert stored.read_bytes() == payload
    row = conn.execute("SELECT sha256, path, status FROM media WHERE id = ?", (first.media_id,))
    media = row.fetchone()
    assert media["sha256"] == digest
    # Status is 'new' because the hash and the store happen before any processing.
    assert media["status"] == "new"

    # The same bytes under a different filename are the same evidence.
    second_path = tmp_path / "renamed.png"
    second_path.write_bytes(payload)
    second = ingest_file(conn, settings, case_id="case-1", src=second_path, actor="tester")

    assert second.reused is True
    assert second.media_id == first.media_id
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 1


# Invariant 6: the audit log is append-only.
def test_audit_log_rejects_update_and_delete(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        audit.append(conn, actor="tester", action="a", object_type="t")

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE audit_log SET actor = 'mallory' WHERE seq = 1")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM audit_log WHERE seq = 1")


# Invariant 8: verify model files against models.lock; refuse to start on mismatch.
def test_models_lock_mismatch_refuses(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    weights = models_dir / "sface.onnx"
    weights.write_bytes(b"real weights")
    (models_dir / "models.lock").write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "id": "sface-2021dec",
                        "name": "sface",
                        "version": "2021dec",
                        "kind": "embedder",
                        "file": "sface.onnx",
                        "sha256": "0" * 64,
                        "license": "Apache-2.0",
                        "commercial_use": True,
                        "dim": 128,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(models_lock.ModelsLockError, match=r"does not match models\.lock"):
        models_lock.verify(models_dir)

    # Correct digest verifies.
    digest = models_lock.sha256_file(weights)
    lock_path = models_dir / "models.lock"
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    data["models"][0]["sha256"] = digest
    lock_path.write_text(json.dumps(data), encoding="utf-8")
    assert models_lock.verify(models_dir).models[0].sha256 == digest


def test_unlisted_weights_refuse_to_start(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models.lock").write_text(
        json.dumps({"version": 1, "models": []}), encoding="utf-8"
    )
    (models_dir / "smuggled.onnx").write_bytes(b"unlisted")
    with pytest.raises(models_lock.ModelsLockError, match=r"absent from models\.lock"):
        models_lock.verify(models_dir)


def test_missing_models_lock_refuses_to_start(tmp_path: Path) -> None:
    with pytest.raises(models_lock.ModelsLockError, match="missing"):
        models_lock.load(tmp_path)


# Invariant 9: non-commercial models need ALLOW_NONCOMMERCIAL_MODELS.
def test_noncommercial_model_is_gated(settings: Settings) -> None:
    lock = models_lock.ModelsLock(
        version=1,
        models=[
            models_lock.ModelEntry(
                id="buffalo_l-w600k_r50",
                name="buffalo_l",
                version="w600k_r50",
                kind="embedder",
                file="w600k_r50.onnx",
                sha256="f" * 64,
                license="InsightFace non-commercial",
                commercial_use=False,
                dim=512,
            )
        ],
    )
    with pytest.raises(models_lock.ModelLicenseError, match="non-commercial"):
        models_lock.assert_loadable(lock, "buffalo_l-w600k_r50", settings)

    permitted = settings.model_copy(update={"allow_noncommercial_models": True})
    entry = models_lock.assert_loadable(lock, "buffalo_l-w600k_r50", permitted)
    assert entry.dim == 512


def test_shipped_default_embedder_is_permissively_licensed(settings: Settings) -> None:
    """A fresh install must not require the non-commercial flag to run."""
    assert settings.embedder_model == "sface-2021dec"
    assert settings.allow_noncommercial_models is False


def test_tracked_models_lock_keeps_a_fresh_clone_legal() -> None:
    """The committed lock must let the default models load with the flag off (C7).

    Reads the real models/models.lock, not a fixture: this is the file that decides
    whether a fresh clone starts in a permissively licensed state.
    """
    lock = models_lock.load(REPO_ROOT / "models")
    defaults = Settings()
    for model_id in (defaults.embedder_model, defaults.detector_model):
        entry = lock.by_id(model_id)
        assert entry is not None, f"{model_id} missing from models.lock"
        assert entry.commercial_use, f"{model_id} is the default but is not commercial-safe"
        # assert_loadable must not raise with allow_noncommercial_models = False.
        models_lock.assert_loadable(lock, model_id, defaults)

    for entry in lock.models:
        if not entry.commercial_use:
            with pytest.raises(models_lock.ModelLicenseError):
                models_lock.assert_loadable(lock, entry.id, defaults)


# Invariant 11: bind to 127.0.0.1 only.
def test_non_loopback_host_is_rejected() -> None:
    with pytest.raises(ValidationError, match="invariant 11"):
        Settings(host="0.0.0.0")  # noqa: S104 - asserting this is refused
    with pytest.raises(ValidationError, match="invariant 11"):
        Settings(host="100.64.0.1")
    assert Settings(host="127.0.0.1").host == "127.0.0.1"


# Invariant 12: every match stores best_template_id and threshold_set_id.
def test_match_without_provenance_is_rejected(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn)
    for column in ("best_template_id", "threshold_set_id"):
        columns = {
            "best_template_id": ids["template"],
            "threshold_set_id": ids["threshold_set"],
        }
        columns[column] = None  # type: ignore[assignment]  # proving NOT NULL bites
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"), transaction(conn):
            conn.execute(
                "INSERT INTO matches (id, track_id, person_id, rank, score, band, "
                "best_template_id, threshold_set_id, embedder_model_id, created_at) "
                "VALUES ('m', ?, ?, 1, 0.9, 'strong', ?, ?, ?, ?)",
                (
                    ids["track"],
                    ids["person"],
                    columns["best_template_id"],
                    columns["threshold_set_id"],
                    ids["embedder"],
                    audit.now_ts(),
                ),
            )


# Section 12: do_not_enroll blocks template creation.
def test_do_not_enroll_blocks_templates(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn)
    with transaction(conn):
        conn.execute(
            "UPDATE persons SET do_not_enroll = 1 WHERE id = ?", (ids["person"],)
        )
    with pytest.raises(sqlite3.IntegrityError, match="do_not_enroll"), transaction(conn):
        conn.execute(
            "INSERT INTO templates (id, person_id, embedding, embedder_model_id, "
            "created_at, created_by) VALUES ('t2', ?, ?, ?, ?, 'tester')",
            (ids["person"], b"\x00" * 8, ids["embedder"], audit.now_ts()),
        )


# Section 12: an unenrolled person is out of the gallery.
def test_unenrolled_person_cannot_be_auto_accepted(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn, calibrated=True)
    insert_match(conn, ids)
    with transaction(conn):
        conn.execute("UPDATE persons SET status = 'unenrolled' WHERE id = ?", (ids["person"],))

    with pytest.raises(sqlite3.IntegrityError, match="enrolled person"), transaction(conn):
        conn.execute(
            "INSERT INTO identities (track_id, person_id, source, match_id, "
            "threshold_set_id, updated_at) VALUES (?, ?, 'auto', ?, ?, ?)",
            (ids["track"], ids["person"], "match-1", ids["threshold_set"], audit.now_ts()),
        )


def test_only_one_threshold_set_can_be_active(conn: sqlite3.Connection) -> None:
    ids = seed_gallery(conn, calibrated=True)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"), transaction(conn):
        conn.execute(
            "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
            "calibrated, active, created_at) VALUES ('ts-2', ?, 0.6, 0.4, 0.05, 0, 1, ?)",
            (ids["embedder"], audit.now_ts()),
        )
