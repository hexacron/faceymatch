"""Video processing (spec 6.2, milestone M2).

What has to hold: a sampled frame lands on a grid that is a function of its timestamp, one
face across many frames is one track with one mean, a job killed mid-file resumes from its
checkpoint without writing the same frame twice, and a flicker never becomes a track.

The models are fakes, so this exercises the pipeline rather than the weights: a detector
that reports a moving box and an embedder that returns a constant vector. The videos are
generated here, because fixtures are gitignored and a decode test that skips on a fresh
clone proves nothing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import audit
from app.config import Settings
from app.core import storage
from app.core.registry import ActiveModels
from app.core.types import ARCFACE_TEMPLATE, Detection
from app.db.conn import transaction
from app.jobs import enqueue
from app.pipeline import video
from app.pipeline.process import UnsupportedMediaKindError, process_video
from app.worker import run_once

FRAME_W, FRAME_H = 320, 240


class MovingFaceDetector:
    """One face drifting left to right, so association has something to get wrong."""

    model_id = "detector"

    def __init__(self, *, score: float = 0.99) -> None:
        self._score = score
        self._calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        x = 20.0 + 4.0 * self._calls
        self._calls += 1
        return [
            Detection(
                x=x,
                y=20.0,
                w=112.0,
                h=112.0,
                score=self._score,
                landmarks=ARCFACE_TEMPLATE + np.array([x, 20.0], dtype=np.float32),
            )
        ]


class FlickerDetector:
    """A face on the first frame and nothing afterwards: one box is not a track."""

    model_id = "detector"

    def __init__(self) -> None:
        self._calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self._calls += 1
        if self._calls > 1:
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


def _models(detector: object) -> ActiveModels:
    return ActiveModels(
        detector=detector,  # type: ignore[arg-type]  # a Detector by structure, not by import
        embedder=FakeEmbedder(),
        detector_model_id="detector",
        embedder_model_id="embedder",
        execution_provider="CPUExecutionProvider",
    )


def _tuned(settings: Settings, **overrides: object) -> Settings:
    """These faces are synthetic, so the size gate is the only one that means anything."""
    return settings.model_copy(
        update={"min_embed_px": 40, "min_sharpness": 0.0, **overrides}
    )


def write_clip(path: Path, *, seconds: float = 4.0, rate: int = 30) -> Path:
    """A real encoded clip: the decoder under test is the point, so nothing is stubbed.

    The frames carry a checkerboard rather than flat colour because the quality gate
    measures Laplacian variance, and a flat frame is indistinguishable from an out-of-focus
    one — which is the correct verdict on flat colour and the wrong fixture for a pipeline
    test.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checker = (np.indices((FRAME_H, FRAME_W)).sum(axis=0) % 2 * 255).astype(np.uint8)
    pixels = np.repeat(checker[:, :, None], 3, axis=2)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = FRAME_W, FRAME_H, "yuv420p"
        for _index in range(int(seconds * rate)):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _seed_video(
    conn: sqlite3.Connection, settings: Settings, path: Path, *, media_id: str = "media"
) -> str:
    """Store a clip the way ingest would and register it as `new`."""
    digest, _size = storage.store_file(settings.media_dir, path)
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
            "created_by) VALUES ('case', 'Case', 'consent', ?, 'tester')",
            (now,),
        )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES (?, 'case', ?, 'video', ?, ?, 'new')",
            (media_id, digest, storage.relative_path_for(digest), now),
        )
    return media_id


def test_sampling_lands_on_a_grid_read_off_the_timestamps(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "clip.mp4", seconds=5.0)

    samples = list(video.iter_samples(clip, sample_fps=3.0))

    assert [item.frame_idx for item in samples] == list(range(15))
    # Each sample is the first frame at or after its slot, never before it.
    assert all(item.t_ms >= item.frame_idx * 1000 / 3 - 1 for item in samples)
    assert samples[0].image.shape == (FRAME_H, FRAME_W, 3)


def test_a_resumed_decode_puts_the_same_frames_in_the_same_slots(tmp_path: Path) -> None:
    """The grid is what makes a resumed job idempotent; a decode ordinal would renumber."""
    clip = write_clip(tmp_path / "clip.mp4", seconds=5.0)
    whole = {item.frame_idx: item.t_ms for item in video.iter_samples(clip, sample_fps=3.0)}

    resumed = list(video.iter_samples(clip, sample_fps=3.0, start_ms=2000))

    assert resumed, "seeking into the file sampled nothing"
    assert all(whole[item.frame_idx] == item.t_ms for item in resumed)
    assert min(item.frame_idx for item in resumed) == 6


def test_a_face_across_a_clip_becomes_one_track_with_one_mean(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    tuned = _tuned(settings)
    media_id = _seed_video(conn, tuned, write_clip(tmp_path / "clip.mp4"))

    result = process_video(
        conn, tuned, _models(MovingFaceDetector()), media_id=media_id, actor="tester"
    )

    assert result.tracks == 1
    assert result.decoded_frames == 12  # 4 s at 3 fps
    assert result.detections == 12
    tracks = conn.execute("SELECT * FROM tracks WHERE media_id = ?", (media_id,)).fetchall()
    assert len(tracks) == 1
    track = tracks[0]
    assert track["embedder_model_id"] == "embedder"
    assert bytes(track["embedding_mean"]) == np.array([1.0, 0.0], dtype="<f4").tobytes()
    assert track["best_detection_id"] is not None
    assert int(track["end_ms"]) > int(track["start_ms"]), "a track spanning a clip has duration"
    # Every sampled frame contributed one box to that track, and each one is embedded.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM detections WHERE track_id = ?", (track["id"],)
    ).fetchone()["n"] == 12
    assert conn.execute("SELECT COUNT(*) AS n FROM detection_embeddings").fetchone()["n"] == 12
    media = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    assert media["status"] == "done"
    assert (media["width"], media["height"]) == (FRAME_W, FRAME_H)
    assert media["duration_ms"] is not None and media["fps"] is not None


def test_a_one_frame_flicker_leaves_its_detection_but_no_track(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    """The box is a record of what the detector saw; a track is a claim that it was a face."""
    tuned = _tuned(settings)
    media_id = _seed_video(conn, tuned, write_clip(tmp_path / "clip.mp4"))

    result = process_video(
        conn, tuned, _models(FlickerDetector()), media_id=media_id, actor="tester"
    )

    assert result.tracks == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"] == 0
    orphan = conn.execute(
        "SELECT COUNT(*) AS n FROM detections WHERE media_id = ? AND track_id IS NULL",
        (media_id,),
    ).fetchone()["n"]
    assert orphan == 1


def test_a_job_resumed_from_its_checkpoint_does_not_write_the_frame_twice(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    """The exit test in miniature: a killed job continues, it does not start again."""
    tuned = _tuned(settings, video_batch_frames=3)
    media_id = _seed_video(conn, tuned, write_clip(tmp_path / "clip.mp4", seconds=4.0))
    job = enqueue(conn, kind="process", actor="tester", params={"media_id": media_id})

    # First pass over the whole file, checkpointing as it goes.
    process_video(
        conn,
        tuned,
        _models(MovingFaceDetector()),
        media_id=media_id,
        actor="tester",
        job_id=job.id,
    )
    before = conn.execute(
        "SELECT COUNT(*) AS n FROM detections WHERE media_id = ?", (media_id,)
    ).fetchone()["n"]

    # Rewind the checkpoint to the middle and run again, which is what a kill looks like.
    with transaction(conn):
        conn.execute(
            "UPDATE jobs SET progress = ? WHERE id = ?",
            ('{"media_id": "media", "last_frame_idx": 5, "last_t_ms": 1667}', job.id),
        )
    process_video(
        conn,
        tuned,
        _models(MovingFaceDetector()),
        media_id=media_id,
        actor="tester",
        job_id=job.id,
    )

    after = conn.execute(
        "SELECT COUNT(*) AS n FROM detections WHERE media_id = ?", (media_id,)
    ).fetchone()["n"]
    assert after == before, "a resumed pass inserted frames that were already written"
    slots = conn.execute(
        "SELECT frame_idx, det_idx, COUNT(*) AS n FROM detections WHERE media_id = ? "
        "GROUP BY frame_idx, det_idx HAVING n > 1",
        (media_id,),
    ).fetchall()
    assert slots == [], "the same grid slot was written twice"
    assert audit.verify(conn).ok


def test_the_worker_routes_a_video_to_the_video_pipeline(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`handle_process` decides by `media.kind`, so ingest does not have to queue two kinds."""
    tuned = _tuned(settings)
    media_id = _seed_video(conn, tuned, write_clip(tmp_path / "clip.mp4", seconds=2.0))
    enqueue(conn, kind="process", actor="tester", params={"media_id": media_id})
    monkeypatch.setattr("app.worker._verified_lock", lambda _settings: None)
    monkeypatch.setattr(
        "app.worker.get_active_models", lambda _settings, _lock: _models(MovingFaceDetector())
    )

    job = run_once(conn, tuned)

    assert job is not None
    done = conn.execute("SELECT status, progress FROM jobs WHERE id = ?", (job.id,)).fetchone()
    assert done["status"] == "done"
    assert '"decoded_frames": 6' in str(done["progress"]).replace('":', '": ')
    assert conn.execute("SELECT status FROM media WHERE id = ?", (media_id,)).fetchone()[
        "status"
    ] == "done"


def test_a_still_image_is_refused_by_the_video_entry_point(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    tuned = _tuned(settings)
    media_id = _seed_video(conn, tuned, write_clip(tmp_path / "clip.mp4", seconds=1.0))
    with transaction(conn):
        conn.execute("UPDATE media SET kind = 'image' WHERE id = ?", (media_id,))

    with pytest.raises(UnsupportedMediaKindError):
        process_video(
            conn, tuned, _models(MovingFaceDetector()), media_id=media_id, actor="tester"
        )


def test_a_video_preview_is_its_first_frame(
    client: TestClient, conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    media_id = _seed_video(conn, settings, write_clip(tmp_path / "clip.mp4", seconds=1.0))

    response = client.get(f"/api/media/{media_id}/thumbnail?size=128")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    with Image.open(__import__("io").BytesIO(response.content)) as poster:
        assert max(poster.size) == 128
