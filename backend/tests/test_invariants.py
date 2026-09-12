"""One test per invariant that M0 can actually enforce (AGENTS.md test rules).

Invariants 3, 7, 10 depend on the ingest and enrollment code paths and are tested with the
milestones that add them.
"""

from __future__ import annotations

import json
import socket
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import audit, models_lock
from app.config import REPO_ROOT, Settings
from app.db.conn import transaction
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
