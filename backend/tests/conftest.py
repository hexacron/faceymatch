from __future__ import annotations

import json
import socket
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import audit
from app.config import Settings
from app.db.conn import connect, transaction
from app.db.migrate import migrate
from app.main import create_app

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", ""})


@pytest.fixture(autouse=True)
def forbid_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 1: no outbound network calls at runtime.

    Sockets themselves are not blocked — asyncio's event loop needs a local self-pipe — so
    the guard sits on the two things that actually reach the network: name resolution and
    connecting to a non-loopback address.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else None
        if host not in _LOOPBACK:
            raise AssertionError(f"outbound connection to {address!r} (invariant 1)")
        real_connect(self, address)

    def forbidden_resolution(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("DNS resolution is forbidden at runtime (invariant 1)")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", forbidden_resolution)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden_resolution)


@pytest.fixture(autouse=True)
def isolate_settings_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test describes its own configuration; the machine's must not leak into it.

    `Settings` reads the repo-root `.env` and the process environment. Both are real on a
    developer's or the operator's machine, and a test that passes only because the local
    `.env` happens to agree with it is not testing anything. Both sources are removed here,
    so every fixture states the configuration it means.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models.lock").write_text(
        json.dumps({"version": 1, "models": []}), encoding="utf-8"
    )
    return Settings(
        db_path=tmp_path / "data" / "facematch.db",
        media_dir=tmp_path / "media",
        crops_dir=tmp_path / "crops",
        models_dir=models_dir,
        frontend_dist=tmp_path / "dist",
        fixtures_dir=tmp_path / "fixtures",
        operator_name="tester",
    )


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    settings.ensure_dirs()
    connection = connect(settings.db_path)
    migrate(connection)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def seed_gallery(conn: sqlite3.Connection, *, calibrated: bool = False) -> dict[str, str]:
    """Insert the minimum row graph needed to exercise the matching invariants.

    Returns the ids by role.
    """
    now = audit.now_ts()
    ids = {
        "detector": "det-model",
        "embedder": "emb-model",
        "other_embedder": "emb-model-2",
        "case": "case-1",
        "person": "person-1",
        "media": "media-1",
        "track": "track-1",
        "detection": "detection-1",
        "template": "template-1",
        "threshold_set": "ts-1",
    }
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, dim)"
            " VALUES (?, 'yunet', '2023mar', 'detector', ?, 'MIT', 1, NULL)",
            (ids["detector"], "a" * 64),
        )
        for model_id in (ids["embedder"], ids["other_embedder"]):
            conn.execute(
                "INSERT INTO models (id, name, version, kind, sha256, license, "
                "commercial_use, dim) VALUES (?, 'sface', '2021dec', 'embedder', ?, "
                "'Apache-2.0', 1, 128)",
                (model_id, model_id.ljust(64, "b")),
            )
        conn.execute(
            "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
            "calibrated, calibrated_at, eval_report_sha256, gallery_size, "
            "execution_provider, active, created_at) "
            "VALUES (?, ?, 0.6, 0.4, 0.05, ?, ?, ?, ?, ?, 1, ?)",
            (
                ids["threshold_set"],
                ids["embedder"],
                int(calibrated),
                now if calibrated else None,
                "c" * 64 if calibrated else None,
                10 if calibrated else None,
                "CPUExecutionProvider" if calibrated else None,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES (?, 'Case', 'test warrant', ?, 'tester')",
            (ids["case"], now),
        )
        conn.execute(
            "INSERT INTO persons (id, display_name, created_at, created_by) "
            "VALUES (?, 'Person One', ?, 'tester')",
            (ids["person"], now),
        )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES (?, ?, ?, 'image', 'media/one.jpg', ?, 'done')",
            (ids["media"], ids["case"], "d" * 64, now),
        )
        conn.execute(
            "INSERT INTO tracks (id, media_id, start_ms, end_ms, embedding_mean, "
            "embedder_model_id) VALUES (?, ?, 0, 0, ?, ?)",
            (ids["track"], ids["media"], b"\x00" * 8, ids["embedder"]),
        )
        conn.execute(
            "INSERT INTO detections (id, media_id, track_id, t_ms, frame_idx, det_idx, "
            "x, y, w, h, landmarks_json, det_score, quality_json, crop_sha256, "
            "detector_model_id) VALUES (?, ?, ?, 0, 0, 0, 1, 2, 30, 30, '[]', 0.9, '{}', "
            "?, ?)",
            (ids["detection"], ids["media"], ids["track"], "e" * 64, ids["detector"]),
        )
        conn.execute(
            "INSERT INTO templates (id, person_id, detection_id, source_case_id, embedding, "
            "embedder_model_id, quality, created_at, created_by) "
            "VALUES (?, ?, ?, ?, ?, ?, 0.8, ?, 'tester')",
            (
                ids["template"],
                ids["person"],
                ids["detection"],
                ids["case"],
                b"\x00" * 8,
                ids["embedder"],
                now,
            ),
        )
    return ids


def insert_match(
    conn: sqlite3.Connection,
    ids: dict[str, str],
    *,
    match_id: str = "match-1",
    embedder_model_id: str | None = None,
) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO matches (id, track_id, person_id, rank, score, band, "
            "best_template_id, threshold_set_id, embedder_model_id, created_at) "
            "VALUES (?, ?, ?, 1, 0.77, 'strong', ?, ?, ?, ?)",
            (
                match_id,
                ids["track"],
                ids["person"],
                ids["template"],
                ids["threshold_set"],
                embedder_model_id if embedder_model_id is not None else ids["embedder"],
                audit.now_ts(),
            ),
        )
