"""Live match (tier 2): advisory, match-only, and above all non-persisting.

Fake detector/embedder, as in the stored-path tests: CI has no model weights. What is
exercised for real is everything that makes tier 2 safe — that it writes nothing, and that
its bands are the stored path's bands rather than a second opinion.
"""

from __future__ import annotations

import io
import sqlite3
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import audit
from app.api import live as live_api
from app.config import Settings
from app.core import storage
from app.core.registry import ActiveModels
from app.core.types import ARCFACE_TEMPLATE, Detection
from app.core.vectors import to_blob
from app.db.conn import transaction
from app.main import create_app
from app.pipeline import live
from app.pipeline.process import process_image

# Tables that the stored path writes and tier 2 must never touch.
_WRITTEN_TABLES = (
    "media",
    "detections",
    "detection_embeddings",
    "tracks",
    "matches",
    "identities",
    "identifications",
    "templates",
    "persons",
    "jobs",
    "audit_log",
)


class FakeDetector:
    """One face per configured offset, with landmarks the aligner accepts."""

    model_id = "detector"

    def __init__(self, offsets: list[tuple[float, float]]) -> None:
        self._offsets = offsets

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(
                x=x,
                y=y,
                w=112.0,
                h=112.0,
                score=0.99,
                landmarks=ARCFACE_TEMPLATE + np.array([x, y], dtype=np.float32),
            )
            for x, y in self._offsets
        ]


class FakeEmbedder:
    model_id = "embedder"
    dim = 2

    def __init__(self, vector: tuple[float, float] = (1.0, 0.0)) -> None:
        self._vector = np.array([vector], dtype=np.float32)

    def embed(self, crops: np.ndarray) -> np.ndarray:
        count = 1 if crops.ndim == 3 else crops.shape[0]
        return np.tile(self._vector, (count, 1))


@pytest.fixture
def tuned(settings: Settings) -> Settings:
    """The fake detector's 112 px boxes are below the shipped quality floors."""
    return settings.model_copy(update={"min_embed_px": 40, "min_sharpness": 0.0})


def _frame(size: tuple[int, int] = (240, 240)) -> bytes:
    """A high-frequency checkerboard: sharp enough to clear the quality gate."""
    pattern = np.indices(size).sum(axis=0) % 2 * 255
    rgb = np.repeat(pattern[:, :, None], 3, axis=2).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _models(detector: FakeDetector, embedder: FakeEmbedder) -> ActiveModels:
    return ActiveModels(
        detector=detector,
        embedder=embedder,
        detector_model_id="detector",
        embedder_model_id="embedder",
        execution_provider="CPUExecutionProvider",
    )


def _seed(conn: sqlite3.Connection, *, template: tuple[float, float] = (1.0, 0.0)) -> None:
    """Models, an active (uncalibrated) threshold set, a case, and one enrolled person."""
    live.clear_gallery_cache()
    now = "2026-01-01T00:00:00Z"
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
            "INSERT INTO persons (id, display_name, created_at, created_by) "
            "VALUES ('person', 'Ada Lovelace', ?, 'tester')",
            (now,),
        )
        conn.execute(
            "INSERT INTO templates (id, person_id, source_case_id, embedding, "
            "embedder_model_id, created_at, created_by) "
            "VALUES ('template', 'person', 'case', ?, 'embedder', ?, 'tester')",
            (to_blob(np.array(template, dtype=np.float32)), now),
        )


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _WRITTEN_TABLES:
        # Table names come from the module-level tuple, never from input.
        query = f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608
        counts[table] = int(conn.execute(query).fetchone()["n"])
    return counts


def test_live_match_persists_nothing(
    conn: sqlite3.Connection, tuned: Settings
) -> None:
    _seed(conn)
    before = _counts(conn)

    result = live.match_frame(
        conn,
        tuned,
        _models(FakeDetector([(20.0, 20.0)]), FakeEmbedder()),
        frame=_frame(),
    )

    assert len(result.faces) == 1
    assert result.faces[0].candidates  # it did real work
    assert _counts(conn) == before
    # No stored crop either: the aligned pixels never reach the crop store.
    assert [path for path in tuned.crops_dir.rglob("*") if path.is_file()] == []


def test_live_bands_are_the_stored_paths_bands(
    conn: sqlite3.Connection, tuned: Settings
) -> None:
    """Same vectors, same thresholds, same band: one band definition, not two."""
    _seed(conn)
    frame = _frame()
    models = _models(FakeDetector([(20.0, 20.0)]), FakeEmbedder())

    live_result = live.match_frame(conn, tuned, models, frame=frame)

    # Now the stored path over the identical pixels.
    digest = storage.store_bytes(tuned.media_dir, frame)
    with transaction(conn):
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES ('media', 'case', ?, 'image', ?, '2026-01-01T00:00:00Z', 'new')",
            (digest, storage.relative_path_for(digest)),
        )
    process_image(conn, tuned, models, media_id="media", actor="tester")

    stored = conn.execute("SELECT band, score, person_id FROM matches WHERE rank = 1").fetchone()
    assert stored is not None
    top = live_result.faces[0].candidates[0]
    assert top.rank == 1
    assert top.band == str(stored["band"])
    assert top.person_id == str(stored["person_id"])
    assert top.score == pytest.approx(float(stored["score"]))
    assert top.name == "Ada Lovelace"
    assert top.best_template_id == "template"


def test_an_unknown_face_gets_candidates_banded_unknown(
    conn: sqlite3.Connection, tuned: Settings
) -> None:
    _seed(conn)

    result = live.match_frame(
        conn,
        tuned,
        # Orthogonal to the only template: a face the gallery has never seen.
        _models(FakeDetector([(20.0, 20.0)]), FakeEmbedder((0.0, 1.0))),
        frame=_frame(),
    )

    face = result.faces[0]
    assert face.quality_passed is True
    assert [candidate.band for candidate in face.candidates] == ["unknown"]
    assert face.candidates[0].score == pytest.approx(0.0)


def test_live_match_never_claims_auto_acceptance(
    conn: sqlite3.Connection, tuned: Settings
) -> None:
    """The threshold set here is uncalibrated, so the gate is shut (invariant 4)."""
    _seed(conn)

    result = live.match_frame(
        conn, tuned, _models(FakeDetector([(20.0, 20.0)]), FakeEmbedder()), frame=_frame()
    )

    assert result.threshold_set_id == "threshold"
    assert result.auto_accept_allowed is False
    assert result.auto_accept_reason is not None


def test_quality_failures_are_reported_without_candidates(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """A face the gate rejects is still drawn, but it is never scored."""
    _seed(conn)

    result = live.match_frame(
        conn,
        settings,  # untuned: min_embed_px 80 with sharpness floor 40 rejects the flat crop
        _models(FakeDetector([(20.0, 20.0)]), FakeEmbedder()),
        frame=_png_flat(),
    )

    face = result.faces[0]
    assert face.quality_passed is False
    assert face.quality_reasons
    assert face.candidates == []


def _png_flat() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((240, 240, 3), 128, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_gallery_cache_follows_the_audit_chain_head(
    conn: sqlite3.Connection, tuned: Settings
) -> None:
    """A new template must be visible to the next frame, not after a restart."""
    _seed(conn)
    models = _models(FakeDetector([(20.0, 20.0)]), FakeEmbedder())

    first = live.match_frame(conn, tuned, models, frame=_frame())
    assert first.gallery_persons == 1

    with transaction(conn):
        conn.execute(
            "INSERT INTO persons (id, display_name, created_at, created_by) "
            "VALUES ('person-2', 'Grace Hopper', '2026-01-01T00:00:00Z', 'tester')"
        )
        conn.execute(
            "INSERT INTO templates (id, person_id, source_case_id, embedding, "
            "embedder_model_id, created_at, created_by) "
            "VALUES ('template-2', 'person-2', 'case', ?, 'embedder', "
            "'2026-01-01T00:00:00Z', 'tester')",
            (to_blob(np.array([0.0, 1.0], dtype=np.float32)),),
        )
        # Every state change appends to the chain; that is what invalidates the cache.
        audit.append(
            conn,
            actor="tester",
            action="person.template_add",
            object_type="template",
            object_id="template-2",
            case_id="case",
            payload={"person_id": "person-2"},
        )

    second = live.match_frame(conn, tuned, models, frame=_frame())
    assert second.gallery_persons == 2
    assert {candidate.person_id for candidate in second.faces[0].candidates} == {
        "person",
        "person-2",
    }


def _client_with_models(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Real app, fake weights: CI has no ONNX files to load."""

    def fake_active(*_args: Any, **_kwargs: Any) -> ActiveModels:
        return _models(FakeDetector([(20.0, 20.0)]), FakeEmbedder())

    monkeypatch.setattr(live_api, "get_active_models", fake_active)
    return TestClient(create_app(settings))


def test_live_match_endpoint_returns_boxes_and_candidates(
    settings: Settings, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuned = settings.model_copy(update={"min_embed_px": 40, "min_sharpness": 0.0})
    with _client_with_models(tuned, monkeypatch) as client:
        _seed(conn)
        before = _counts(conn)

        response = client.post(
            "/api/live/match",
            data={"case_id": "case"},
            files={"frame": ("frame.png", _frame(), "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert (body["width"], body["height"]) == (240, 240)
        assert body["threshold_set_id"] == "threshold"
        assert body["auto_accept_allowed"] is False
        assert body["elapsed_ms"] >= 0
        assert set(body["timings"]) == {
            "decode",
            "detect",
            "quality_align",
            "embed",
            "match",
        }
        face = body["faces"][0]
        assert (face["x"], face["y"], face["w"], face["h"]) == (20.0, 20.0, 112.0, 112.0)
        assert face["quality_passed"] is True
        assert face["candidates"][0]["name"] == "Ada Lovelace"
        # Still nothing written, through the HTTP edge as well.
        assert _counts(conn) == before


class RefusingEmbedder:
    """Proves the boxes-only path never reaches the embedder, rather than assuming it."""

    model_id = "embedder"
    dim = 2

    def embed(self, crops: np.ndarray) -> np.ndarray:
        raise AssertionError("identify=false must not embed anything")


def test_boxes_only_frames_skip_identification_entirely(
    settings: Settings, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick that only wants boxes must not pay for alignment, embedding or scoring."""
    tuned = settings.model_copy(update={"min_embed_px": 40, "min_sharpness": 0.0})

    def fake_active(*_args: Any, **_kwargs: Any) -> ActiveModels:
        return _models(FakeDetector([(20.0, 20.0), (60.0, 60.0)]), RefusingEmbedder())

    monkeypatch.setattr(live_api, "get_active_models", fake_active)
    with TestClient(create_app(tuned)) as client:
        _seed(conn)
        before = _counts(conn)

        response = client.post(
            "/api/live/match",
            data={"case_id": "case", "identify": "false"},
            files={"frame": ("frame.png", _frame(), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["identified"] is False
    assert len(body["faces"]) == 2
    assert all(face["candidates"] == [] for face in body["faces"])
    assert all(face["quality_passed"] is True for face in body["faces"])
    assert body["timings"]["embed"] == 0.0
    assert body["timings"]["match"] == 0.0
    # The reporting fields still mean what they mean; only identification was skipped.
    assert body["gallery_persons"] == 1
    assert body["threshold_set_id"] == "threshold"
    assert _counts(conn) == before


def test_an_identify_pass_says_so(
    settings: Settings, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`identified` is the client's only way to tell "nothing found" from "not asked"."""
    tuned = settings.model_copy(update={"min_embed_px": 40, "min_sharpness": 0.0})
    with _client_with_models(tuned, monkeypatch) as client:
        _seed(conn)
        response = client.post(
            "/api/live/match", files={"frame": ("frame.png", _frame(), "image/png")}
        )

    body = response.json()
    assert body["identified"] is True
    assert body["faces"][0]["candidates"]
    assert body["timings"]["embed"] > 0.0


def test_live_match_endpoint_rejects_an_unknown_case_and_undecodable_bytes(
    settings: Settings, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client_with_models(settings, monkeypatch) as client:
        _seed(conn)

        unknown = client.post(
            "/api/live/match",
            data={"case_id": "nope"},
            files={"frame": ("frame.png", _frame(), "image/png")},
        )
        garbage = client.post(
            "/api/live/match",
            files={"frame": ("frame.png", b"not an image", "image/png")},
        )

    assert unknown.status_code == 404
    assert garbage.status_code == 400
