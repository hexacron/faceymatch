"""GET/PATCH /api/config: what an operator may change, and what changing it costs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import audit, jobs, runtime_config
from app.config import Settings
from app.db.conn import connect, transaction
from app.main import create_app

DETECTOR = "yunet-2023mar"
SFACE = "sface-2021dec"
BUFFALO = "buffalo_l-w600k_r50"

# Stand-in weights. Nothing here loads an ONNX graph: these tests exercise the gate that
# decides whether a switch is allowed, and that gate reads models.lock and the digest of
# the bytes on disk, not the graph inside them.
_WEIGHTS: dict[str, tuple[str, str, bool, int | None]] = {
    # id: (filename, license, commercial_use, dim)
    DETECTOR: ("yunet.onnx", "MIT", True, None),
    SFACE: ("sface.onnx", "Apache-2.0", True, 128),
    BUFFALO: ("w600k_r50.onnx", "InsightFace: non-commercial research only", False, 512),
}


def _write_models(models_dir: Path) -> None:
    entries: list[dict[str, Any]] = []
    for model_id, (filename, license_name, commercial, dim) in _WEIGHTS.items():
        path = models_dir / filename
        path.write_bytes(f"weights for {model_id}".encode())
        entries.append(
            {
                "id": model_id,
                "name": model_id,
                "version": "1",
                "kind": "detector" if dim is None else "embedder",
                "file": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "license": license_name,
                "commercial_use": commercial,
                **({} if dim is None else {"dim": dim}),
            }
        )
    (models_dir / "models.lock").write_text(
        json.dumps({"version": 1, "models": entries}), encoding="utf-8"
    )


@pytest.fixture
def provisioned(tmp_path: Path) -> Settings:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_models(models_dir)
    return Settings(
        db_path=tmp_path / "data" / "facematch.db",
        media_dir=tmp_path / "media",
        crops_dir=tmp_path / "crops",
        models_dir=models_dir,
        frontend_dist=tmp_path / "dist",
        fixtures_dir=tmp_path / "fixtures",
        operator_name="tester",
        detector_model=DETECTOR,
        embedder_model=SFACE,
        allow_noncommercial_models=False,
    )


@pytest.fixture
def api(provisioned: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(provisioned)) as client:
        yield client


@pytest.fixture
def permissive(provisioned: Settings) -> Iterator[TestClient]:
    """The same install with the C7 licence flag set in the environment (invariant 9)."""
    settings = provisioned.model_copy(update={"allow_noncommercial_models": True})
    with TestClient(create_app(settings)) as client:
        yield client


def db(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path)


def audit_entries(client: TestClient, action: str) -> list[dict[str, Any]]:
    entries = client.get("/api/audit", params={"limit": 1000}).json()["entries"]
    return [entry for entry in entries if entry["action"] == action]


def test_config_reports_the_whole_operating_state(api: TestClient) -> None:
    body = api.get("/api/config").json()

    assert body["editable"]["embedder_model"] == SFACE
    assert body["editable"]["person_score_mode"] == "max"
    assert body["readonly"]["allow_noncommercial_models"] is False
    assert body["readonly"]["execution_provider"] == "CPUExecutionProvider"
    assert body["pending_job"] is None

    models = {item["id"]: item for item in body["models"]}
    assert models[SFACE]["active"] is True
    assert models[BUFFALO]["active"] is False
    assert models[BUFFALO]["commercial_use"] is False
    assert models[BUFFALO]["dim"] == 512
    assert all(item["present"] for item in models.values())


def test_a_change_is_audited_with_its_previous_and_new_value(
    api: TestClient, provisioned: Settings
) -> None:
    response = api.patch(
        "/api/config",
        json={"changes": {"min_embed_px": 120}, "reason": "small faces were noise"},
    )

    assert response.status_code == 200
    assert response.json()["editable"]["min_embed_px"] == 120
    entries = audit_entries(api, "config.change")
    assert len(entries) == 1
    assert entries[0]["payload"] == {
        "key": "min_embed_px",
        "previous": 80,
        "new": 120,
        "reason": "small faces were noise",
    }
    conn = db(provisioned)
    try:
        assert audit.verify(conn).ok
    finally:
        conn.close()


def test_a_change_takes_effect_without_a_restart(
    api: TestClient, provisioned: Settings
) -> None:
    """A configuration that only applies after a restart is a configuration nobody trusts."""
    api.patch("/api/config", json={"changes": {"sample_fps": 1.5}, "reason": None})

    # The API's very next request already resolves it...
    assert api.get("/api/config").json()["editable"]["sample_fps"] == 1.5
    # ...and so does the worker process, which shares only the database.
    conn = db(provisioned)
    try:
        assert runtime_config.effective(conn, provisioned).sample_fps == 1.5
    finally:
        conn.close()


def test_an_unchanged_value_is_not_audited_as_a_change(api: TestClient) -> None:
    api.patch("/api/config", json={"changes": {"min_embed_px": 80}, "reason": "no-op"})
    assert audit_entries(api, "config.change") == []


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"t_strong": 0.9}, id="thresholds are calibration output, not config"),
        pytest.param({"allow_noncommercial_models": True}, id="licence gate is env-only"),
        pytest.param({"execution_provider": "CoreMLExecutionProvider"}, id="ep is env-only"),
        pytest.param({"db_path": "elsewhere.db"}, id="the install owns where evidence lives"),
        pytest.param({"nonsense": 1}, id="unknown key"),
    ],
)
def test_an_unchangeable_key_is_refused_by_name(
    api: TestClient, changes: dict[str, Any]
) -> None:
    response = api.patch("/api/config", json={"changes": changes, "reason": None})

    assert response.status_code == 400
    assert "not editable" in response.json()["detail"]
    assert audit_entries(api, "config.change") == []


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"min_det_score": 2.0}, id="a detector score above 1 is not a score"),
        pytest.param({"top_k": 0}, id="ranking zero candidates ranks nothing"),
        pytest.param({"max_yaw": -5.0}, id="a negative angle bound rejects every face"),
        pytest.param({"person_score_mode": "median"}, id="not a scoring mode spec 6.4 has"),
    ],
)
def test_an_out_of_range_value_is_refused(api: TestClient, changes: dict[str, Any]) -> None:
    response = api.patch("/api/config", json={"changes": changes, "reason": None})

    assert response.status_code == 422
    assert audit_entries(api, "config.change") == []


def test_a_noncommercial_model_is_refused_unless_the_environment_allows_it(
    api: TestClient,
) -> None:
    """Invariant 9, C7: a licensing gate the UI can flip is not a gate."""
    response = api.patch(
        "/api/config", json={"changes": {"embedder_model": BUFFALO}, "reason": "eval"}
    )

    assert response.status_code == 422
    assert "non-commercial" in response.json()["detail"]
    assert api.get("/api/config").json()["editable"]["embedder_model"] == SFACE
    assert audit_entries(api, "config.change") == []


def test_the_same_switch_is_allowed_when_the_environment_permits_it(
    permissive: TestClient,
) -> None:
    response = permissive.patch(
        "/api/config", json={"changes": {"embedder_model": BUFFALO}, "reason": "eval"}
    )

    assert response.status_code == 200
    assert response.json()["editable"]["embedder_model"] == BUFFALO


def test_a_model_absent_from_models_lock_is_refused(api: TestClient) -> None:
    """Invariant 8: nothing loads that the lock does not name."""
    response = api.patch(
        "/api/config",
        json={"changes": {"embedder_model": "arcface-from-the-internet"}, "reason": None},
    )

    assert response.status_code == 422
    assert "models.lock" in response.json()["detail"]
    assert api.get("/api/config").json()["editable"]["embedder_model"] == SFACE


def test_a_model_whose_digest_no_longer_matches_is_refused(
    permissive: TestClient, provisioned: Settings
) -> None:
    """Invariant 8: the bytes on disk have to be the bytes models.lock vouches for."""
    (provisioned.models_dir / "w600k_r50.onnx").write_bytes(b"tampered")

    response = permissive.patch(
        "/api/config", json={"changes": {"embedder_model": BUFFALO}, "reason": None}
    )

    assert response.status_code == 422
    assert "does not match models.lock" in response.json()["detail"]
    assert permissive.get("/api/config").json()["editable"]["embedder_model"] == SFACE


def test_selecting_an_already_active_but_tampered_model_is_still_refused(
    api: TestClient, provisioned: Settings
) -> None:
    """"It was already set" is not evidence that the weights are still the right ones."""
    (provisioned.models_dir / "sface.onnx").write_bytes(b"tampered")

    response = api.patch(
        "/api/config", json={"changes": {"embedder_model": SFACE}, "reason": None}
    )

    assert response.status_code == 422
    assert "does not match models.lock" in response.json()["detail"]


def test_a_missing_weight_file_is_refused(
    permissive: TestClient, provisioned: Settings
) -> None:
    (provisioned.models_dir / "w600k_r50.onnx").unlink()

    response = permissive.patch(
        "/api/config", json={"changes": {"embedder_model": BUFFALO}, "reason": None}
    )

    assert response.status_code == 422
    assert "not present" in response.json()["detail"]


def test_switching_the_embedder_enqueues_the_reembed_then_the_rematch(
    permissive: TestClient, provisioned: Settings
) -> None:
    body = permissive.patch(
        "/api/config",
        json={"changes": {"embedder_model": BUFFALO}, "reason": "measuring the gap (D2)"},
    ).json()

    assert len(body["jobs_enqueued"]) == 2
    conn = db(provisioned)
    try:
        kinds = [
            jobs.get(conn, job_id).kind  # type: ignore[union-attr]
            for job_id in body["jobs_enqueued"]
        ]
        # A single worker runs them in creation order: vectors first, then scores.
        assert kinds == ["reembed", "rematch"]
        queued = jobs.get(conn, body["jobs_enqueued"][0])
        assert queued is not None
        assert queued.params["embedder_model_id"] == BUFFALO
        assert queued.params["previous_embedder_model_id"] == SFACE
        assert audit.verify(conn).ok
    finally:
        conn.close()
    assert body["pending_job"] == {
        "id": body["jobs_enqueued"][0],
        "kind": "reembed",
        "status": "queued",
    }


def test_switching_the_embedder_records_that_auto_accept_stopped_applying(
    permissive: TestClient, provisioned: Settings
) -> None:
    """C5, spec 10: the operator must be able to find out why identities stopped landing."""
    conn = db(provisioned)
    try:
        now = audit.now_ts()
        with transaction(conn):
            conn.execute("UPDATE threshold_sets SET active = 0 WHERE active = 1")
            conn.execute(
                "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
                "calibrated, calibrated_at, eval_report_sha256, gallery_size, "
                "execution_provider, active, created_at) "
                "VALUES ('ts-sface', ?, 0.6, 0.4, 0.05, 1, ?, ?, 10, "
                "'CPUExecutionProvider', 1, ?)",
                (SFACE, now, "a" * 64, now),
            )
    finally:
        conn.close()

    permissive.patch(
        "/api/config", json={"changes": {"embedder_model": BUFFALO}, "reason": "eval"}
    )

    suspended = audit_entries(permissive, "config.auto_accept_suspended")
    assert len(suspended) == 1
    payload = suspended[0]["payload"]
    assert payload["threshold_set_model_id"] == SFACE
    assert payload["embedder_model_id"] == BUFFALO
    assert payload["threshold_set_calibrated"] is True

    # And healthz now says so on the wire, not only in the chain.
    auto_accept = permissive.get("/api/healthz").json()["auto_accept"]
    assert auto_accept["allowed"] is False
    assert auto_accept["reason"] is not None


def test_a_second_switch_while_a_reembed_is_in_flight_is_refused(
    permissive: TestClient, provisioned: Settings
) -> None:
    """The running job must keep the parameters it started with, or it mixes two models."""
    first = permissive.patch(
        "/api/config", json={"changes": {"embedder_model": BUFFALO}, "reason": "eval"}
    )
    assert first.status_code == 200

    second = permissive.patch(
        "/api/config", json={"changes": {"embedder_model": SFACE}, "reason": "changed my mind"}
    )

    assert second.status_code == 409
    assert "reembed" in second.json()["detail"]
    assert permissive.get("/api/config").json()["editable"]["embedder_model"] == BUFFALO

    # Even an unrelated key waits: the in-flight job reads its settings from the same table.
    unrelated = permissive.patch(
        "/api/config", json={"changes": {"top_k": 5}, "reason": None}
    )
    assert unrelated.status_code == 409


def test_changing_the_detector_does_not_touch_stored_detections(
    api: TestClient, provisioned: Settings
) -> None:
    """Stored detections are the record of what was found then; a new detector is for later."""
    conn = db(provisioned)
    try:
        now = audit.now_ts()
        with transaction(conn):
            conn.execute(
                "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
                "VALUES ('case', 'C', 'consent', ?, 'tester')",
                (now,),
            )
            conn.execute(
                "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
                "VALUES ('media', 'case', ?, 'image', 'm.png', ?, 'done')",
                ("b" * 64, now),
            )
            conn.execute(
                "INSERT INTO tracks (id, media_id, start_ms, end_ms) "
                "VALUES ('track', 'media', 0, 0)"
            )
            conn.execute(
                "INSERT INTO detections (id, media_id, track_id, t_ms, frame_idx, det_idx, "
                "x, y, w, h, landmarks_json, det_score, quality_json, crop_sha256, "
                "detector_model_id) VALUES ('det', 'media', 'track', 0, 0, 0, 1, 1, 10, 10, "
                "'[]', 0.9, '{}', NULL, ?)",
                (DETECTOR,),
            )
    finally:
        conn.close()

    # Only one detector is locked, so switching to it is the only detector change available;
    # what matters is that the endpoint enqueues nothing for the detector slot.
    body = api.patch(
        "/api/config",
        json={"changes": {"detector_model": DETECTOR, "min_det_score": 0.8}, "reason": None},
    ).json()

    assert body["jobs_enqueued"] == []
    conn = db(provisioned)
    try:
        row = conn.execute("SELECT detector_model_id FROM detections").fetchone()
        assert str(row["detector_model_id"]) == DETECTOR
        assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0
    finally:
        conn.close()


def test_a_switch_survives_a_restart(provisioned: Settings) -> None:
    """The override is durable: a new process must not silently revert to the .env model."""
    settings = provisioned.model_copy(update={"allow_noncommercial_models": True})
    with TestClient(create_app(settings)) as first:
        first.patch(
            "/api/config", json={"changes": {"embedder_model": BUFFALO}, "reason": "eval"}
        )

    with TestClient(create_app(settings)) as second:
        body = second.get("/api/config").json()
        assert body["editable"]["embedder_model"] == BUFFALO
        assert {item["id"] for item in body["models"] if item["active"]} == {
            BUFFALO,
            DETECTOR,
        }
