"""Faces-only folder import, and the library filter that sweeps what is already in (6.1).

What has to hold: a file the detector found no face in is gone once its job has looked at
it, the reason is in the chain so an automatic drop is distinguishable from an operator
pressing the bin, the chain still verifies across that write, and without the flag nothing
is destroyed. The filter that finds the same files afterwards answers only for processed
media: one still queued has not been asked yet.

The models are fakes, so this exercises the decision rather than the weights: a detector
that reports a face for a bright image and nothing for a dark one. The images are generated
here, because fixtures are gitignored and a test that skips on a fresh clone proves nothing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import audit
from app.config import Settings
from app.core.registry import ActiveModels
from app.core.types import ARCFACE_TEMPLATE, Detection
from app.db.conn import transaction
from app.pipeline.ingest import ingest_folder
from app.worker import PURGE_REASON_NO_FACES, run_once

IMAGE_W, IMAGE_H = 320, 240

# The two files a mixed folder holds, told apart by the only thing a detector sees.
FACE_LEVEL = 200
BLANK_LEVEL = 40


class BrightnessDetector:
    """One face in a bright image, nothing in a dark one."""

    model_id = "detector"

    def detect(self, image: np.ndarray) -> list[Detection]:
        if int(image[0, 0, 0]) < 128:
            return []
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


def _models() -> ActiveModels:
    return ActiveModels(
        detector=BrightnessDetector(),  # type: ignore[arg-type]  # a Detector by structure
        embedder=FakeEmbedder(),
        detector_model_id="detector",
        embedder_model_id="embedder",
        execution_provider="CPUExecutionProvider",
    )


def _tuned(settings: Settings) -> Settings:
    """These faces are synthetic, so the size gate is the only one that means anything."""
    return settings.model_copy(update={"min_embed_px": 40, "min_sharpness": 0.0})


def _flat_png(path: Path, level: int) -> None:
    array = np.full((IMAGE_H, IMAGE_W, 3), level, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _mixed_folder(root: Path) -> Path:
    folder = root / "mixed"
    folder.mkdir()
    _flat_png(folder / "face.png", FACE_LEVEL)
    _flat_png(folder / "landscape.png", BLANK_LEVEL)
    return folder


def _seed_models(conn: sqlite3.Connection, *, case_id: str = "case") -> None:
    """The rows the pipeline's foreign keys need, without loading any weights."""
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO models (id, name, version, kind, sha256, license, "
            "commercial_use, dim) VALUES ('detector', 'D', '1', 'detector', ?, 'MIT', 1, NULL)",
            ("b" * 64,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO models (id, name, version, kind, sha256, license, "
            "commercial_use, dim) VALUES ('embedder', 'E', '1', 'embedder', ?, 'MIT', 1, 2)",
            ("c" * 64,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO threshold_sets (id, model_id, t_strong, t_possible, "
            "margin, calibrated, active, created_at) "
            "VALUES ('threshold', 'embedder', .55, .35, .05, 0, 1, ?)",
            (now,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO cases (id, name, authorization_basis, created_at, "
            "created_by) VALUES (?, 'Case', 'consent', ?, 'tester')",
            (case_id, now),
        )


def _drain(conn: sqlite3.Connection, settings: Settings) -> None:
    while run_once(conn, settings) is not None:
        pass


def _use_fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.worker._verified_lock", lambda _settings: None)
    monkeypatch.setattr("app.worker.get_active_models", lambda _settings, _lock: _models())


def _purge_payloads(conn: sqlite3.Connection) -> list[dict[str, object]]:
    return [
        json.loads(str(row["payload_json"]))
        for row in conn.execute(
            "SELECT payload_json FROM audit_log WHERE action = 'media.purge' ORDER BY seq"
        ).fetchall()
    ]


def _import(
    conn: sqlite3.Connection, settings: Settings, folder: Path, *, faces_only: bool
) -> dict[str, str]:
    """Import the folder and return media id by filename stem."""
    results = ingest_folder(
        conn,
        settings,
        case_id="case",
        folder=folder,
        actor="tester",
        faces_only=faces_only,
    )
    stems = sorted(path.stem for path in folder.iterdir())
    assert len(results) == len(stems), "the folder walk did not register every file"
    return dict(zip(stems, (result.media_id for result in results), strict=True))


def test_a_file_with_no_face_is_purged_once_it_has_been_processed(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the flag: the library ends up holding only what was asked for."""
    tuned = _tuned(settings)
    _seed_models(conn)
    ids = _import(conn, tuned, _mixed_folder(tmp_path), faces_only=True)
    blank_path = str(
        conn.execute("SELECT path FROM media WHERE id = ?", (ids["landscape"],)).fetchone()["path"]
    )
    _use_fake_models(monkeypatch)

    _drain(conn, tuned)

    surviving = {
        str(row["id"])
        for row in conn.execute("SELECT id FROM media").fetchall()
    }
    assert surviving == {ids["face"]}, "the wrong set of files survived the import"
    assert not (tuned.media_dir / blank_path).exists(), "the purged file's bytes are still stored"

    payloads = _purge_payloads(conn)
    assert len(payloads) == 1
    assert payloads[0]["reason"] == PURGE_REASON_NO_FACES

    progress = json.loads(
        str(
            conn.execute(
                "SELECT progress FROM jobs WHERE json_extract(params_json, '$.media_id') = ?",
                (ids["landscape"],),
            ).fetchone()["progress"]
        )
    )
    assert progress["purged_no_faces"] is True
    assert progress["detections"] == 0


def test_without_the_flag_a_file_with_no_face_is_kept(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API default destroys nothing: an ordinary import is unchanged."""
    tuned = _tuned(settings)
    _seed_models(conn)
    ids = _import(conn, tuned, _mixed_folder(tmp_path), faces_only=False)
    _use_fake_models(monkeypatch)

    _drain(conn, tuned)

    surviving = {
        str(row["id"])
        for row in conn.execute("SELECT id FROM media").fetchall()
    }
    assert surviving == {ids["face"], ids["landscape"]}
    assert _purge_payloads(conn) == []


def test_the_audit_chain_still_verifies_after_an_automatic_purge(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant 6: a write the worker made on its own is still a hash-chained write."""
    tuned = _tuned(settings)
    _seed_models(conn)
    _import(conn, tuned, _mixed_folder(tmp_path), faces_only=True)
    _use_fake_models(monkeypatch)

    _drain(conn, tuned)

    assert _purge_payloads(conn) != [], "nothing was purged, so this proves nothing"
    assert audit.verify(conn).ok


def test_a_purge_by_an_operator_records_no_reason(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    """`reason` is what makes the automatic drops the distinguishable ones."""
    _seed_models(conn)
    ids = _import(conn, settings, _mixed_folder(tmp_path), faces_only=False)

    response = client.delete(f"/api/media/{ids['face']}")

    assert response.status_code == 200, response.text
    payloads = _purge_payloads(conn)
    assert len(payloads) == 1
    assert payloads[0]["reason"] is None


def test_the_library_can_be_filtered_on_whether_a_file_holds_a_face(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """A queued file appears in neither answer: it has not been asked yet."""
    _seed_models(conn)
    now = audit.now_ts()
    with transaction(conn):
        for media_id, status in (("with", "done"), ("without", "done"), ("queued", "new")):
            conn.execute(
                "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
                "VALUES (?, 'case', ?, 'image', ?, ?, ?)",
                (media_id, media_id.ljust(64, "f"), f"media/{media_id}.jpg", now, status),
            )
        conn.execute(
            "INSERT INTO detections (id, media_id, track_id, t_ms, frame_idx, det_idx, "
            "x, y, w, h, landmarks_json, det_score, quality_json, crop_sha256, "
            "detector_model_id) VALUES ('det-1', 'with', NULL, 0, 0, 0, 1, 2, 30, 30, "
            "'[]', 0.9, '{}', NULL, 'detector')"
        )

    def listed(query: str) -> list[str]:
        response = client.get(f"/api/media?{query}")
        assert response.status_code == 200, response.text
        return [item["id"] for item in response.json()["items"]]

    assert listed("has_faces=false") == ["without"]
    assert listed("has_faces=true") == ["with"]
    assert sorted(listed("case_id=case")) == ["queued", "with", "without"]
