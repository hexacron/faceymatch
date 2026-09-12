"""The reembed job: switching the embedder without touching source media (spec 6.2, 6.3)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pytest

from app import audit, jobs, worker
from app.config import Settings
from app.core import storage, vectors
from app.core.registry import ActiveModels
from app.core.types import Detection
from app.db.conn import transaction
from app.pipeline.matching import rematch
from app.pipeline.reembed import reembed

OLD_MODEL = "old-embedder"
NEW_MODEL = "new-embedder"
OLD_DIM = 2
NEW_DIM = 3


class UnusedDetector:
    """A re-embed reads stored crops; a detector here would mean media was re-decoded."""

    model_id = "detector"

    def detect(self, image: np.ndarray) -> list[Detection]:  # pragma: no cover
        raise AssertionError("reembed must never run detection (spec 6.2 step 5-6)")


@dataclass
class ShadeEmbedder:
    """Deterministic per-crop vectors: the crop's own grey level decides its direction."""

    model_id: str
    dim: int
    fail_after: int | None = None
    calls: int = 0

    def embed(self, crops: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("embedder died mid-job")
        stack = crops[None, ...] if crops.ndim == 3 else crops
        shades = stack[:, 0, 0, 0].astype(np.float32) / 255.0
        out = np.zeros((stack.shape[0], self.dim), dtype=np.float32)
        out[:, 0] = shades
        out[:, self.dim - 1] = 1.0
        return vectors.l2_normalize(out)


def models_for(embedder: ShadeEmbedder) -> ActiveModels:
    return ActiveModels(
        detector=UnusedDetector(),
        embedder=embedder,
        detector_model_id="detector",
        embedder_model_id=embedder.model_id,
        execution_provider="CPUExecutionProvider",
    )


def store_shade(settings: Settings, shade: int) -> str:
    """Write a solid 112x112 crop to the content-addressed store; return its digest."""
    crop = np.full((112, 112, 3), shade, dtype=np.uint8)
    return storage.store_crop(settings.crops_dir, crop)


@dataclass(frozen=True, slots=True)
class Seeded:
    person: str
    other_person: str
    tracks: list[str]
    detections: dict[str, tuple[str, float, str | None]]
    template_ok: str
    template_no_crop: str


def seed(
    conn: sqlite3.Connection, settings: Settings, *, calibrated_for: str | None = None
) -> Seeded:
    """A small case already embedded under OLD_MODEL, with one crop-less enrolled face.

    Track A has three crops of different detector scores, so best-K selection is visible.
    Track B has one. The crop-less detection is what a template can be stranded on.
    """
    now = audit.now_ts()
    detections: dict[str, tuple[str, float, str | None]] = {}
    with transaction(conn):
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, "
            "dim) VALUES ('detector', 'D', '1', 'detector', ?, 'MIT', 1, NULL)",
            ("a" * 64,),
        )
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, "
            "dim) VALUES (?, 'Old', '1', 'embedder', ?, 'MIT', 1, ?)",
            (OLD_MODEL, "b" * 64, OLD_DIM),
        )
        conn.execute(
            "INSERT INTO models (id, name, version, kind, sha256, license, commercial_use, "
            "dim) VALUES (?, 'New', '1', 'embedder', ?, 'MIT', 1, ?)",
            (NEW_MODEL, "c" * 64, NEW_DIM),
        )
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES ('case', 'Case', 'consent', ?, 'tester')",
            (now,),
        )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, ingested_at, status) "
            "VALUES ('media', 'case', ?, 'image', 'm/one.png', ?, 'done')",
            ("d" * 64, now),
        )
        for person_id, name in (("person-a", "Ada"), ("person-b", "Grace")):
            conn.execute(
                "INSERT INTO persons (id, display_name, created_at, created_by) "
                "VALUES (?, ?, ?, 'tester')",
                (person_id, name, now),
            )

        old = ShadeEmbedder(model_id=OLD_MODEL, dim=OLD_DIM)
        # (detection_id, track_id, det_score, shade or None for "no crop stored")
        layout = [
            ("det-a1", "track-a", 0.95, 40),
            ("det-a2", "track-a", 0.90, 120),
            ("det-a3", "track-a", 0.10, 200),
            ("det-b1", "track-b", 0.80, 240),
            ("det-c1", "track-c", 0.70, None),
        ]
        for track_id in ("track-a", "track-b", "track-c"):
            conn.execute(
                "INSERT INTO tracks (id, media_id, start_ms, end_ms) VALUES (?, 'media', 0, 0)",
                (track_id,),
            )
        for index, (det_id, track_id, det_score, shade) in enumerate(layout):
            crop_sha256 = None if shade is None else store_shade(settings, shade)
            conn.execute(
                "INSERT INTO detections (id, media_id, track_id, t_ms, frame_idx, det_idx, "
                "x, y, w, h, landmarks_json, det_score, quality_json, crop_sha256, "
                "detector_model_id) VALUES (?, 'media', ?, 0, 0, ?, 1, 1, 112, 112, '[]', "
                "?, '{}', ?, 'detector')",
                (det_id, track_id, index, det_score, crop_sha256),
            )
            detections[det_id] = (track_id, det_score, crop_sha256)
            if crop_sha256 is not None:
                crop = storage.load_crop(settings.crops_dir, crop_sha256)
                conn.execute(
                    "INSERT INTO detection_embeddings (detection_id, embedding, "
                    "embedder_model_id, created_at) VALUES (?, ?, ?, ?)",
                    (det_id, vectors.to_blob(old.embed(crop)[0]), OLD_MODEL, now),
                )
        for track_id, det_id in (("track-a", "det-a1"), ("track-b", "det-b1")):
            crop_sha256 = detections[det_id][2]
            assert crop_sha256 is not None
            mean = old.embed(storage.load_crop(settings.crops_dir, crop_sha256))[0]
            conn.execute(
                "UPDATE tracks SET embedding_mean = ?, embedder_model_id = ? WHERE id = ?",
                (vectors.to_blob(mean), OLD_MODEL, track_id),
            )

        # person-a is enrolled from a detection with a crop; person-b from one without.
        conn.execute(
            "INSERT INTO templates (id, person_id, detection_id, source_case_id, embedding, "
            "embedder_model_id, quality, created_at, created_by) "
            "VALUES ('tpl-ok', 'person-a', 'det-a1', 'case', ?, ?, 0.95, ?, 'tester')",
            (
                conn.execute(
                    "SELECT embedding FROM detection_embeddings WHERE detection_id = 'det-a1'"
                ).fetchone()["embedding"],
                OLD_MODEL,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO templates (id, person_id, detection_id, source_case_id, embedding, "
            "embedder_model_id, quality, created_at, created_by) "
            "VALUES ('tpl-nocrop', 'person-b', 'det-c1', 'case', ?, ?, 0.70, ?, 'tester')",
            (vectors.to_blob(np.array([0.6, 0.8], dtype=np.float32)), OLD_MODEL, now),
        )
        if calibrated_for is not None:
            conn.execute(
                "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
                "calibrated, calibrated_at, eval_report_sha256, gallery_size, "
                "execution_provider, active, created_at) "
                "VALUES ('ts-calibrated', ?, 0.5, 0.2, 0.01, 1, ?, ?, 10, "
                "'CPUExecutionProvider', 1, ?)",
                (calibrated_for, now, "e" * 64, now),
            )
    return Seeded(
        person="person-a",
        other_person="person-b",
        tracks=["track-a", "track-b", "track-c"],
        detections=detections,
        template_ok="tpl-ok",
        template_no_crop="tpl-nocrop",
    )


def enqueue_reembed(conn: sqlite3.Connection) -> str:
    job = jobs.enqueue(
        conn, kind="reembed", actor="tester", params={"embedder_model_id": NEW_MODEL}
    )
    return job.id


def test_reembed_adds_the_new_model_and_keeps_the_old_vectors(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """A switch must be reversible: the previous model's rows are additions away, not gone."""
    seeded = seed(conn, settings)
    before = {
        str(row["detection_id"]): bytes(row["embedding"])
        for row in conn.execute(
            "SELECT detection_id, embedding FROM detection_embeddings "
            "WHERE embedder_model_id = ?",
            (OLD_MODEL,),
        )
    }

    result = reembed(
        conn,
        settings,
        models_for(ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)),
        job_id=enqueue_reembed(conn),
        actor="tester",
    )

    assert result.crops_embedded == 4  # every detection with a stored crop
    new_rows = {
        str(row["detection_id"]): bytes(row["embedding"])
        for row in conn.execute(
            "SELECT detection_id, embedding FROM detection_embeddings "
            "WHERE embedder_model_id = ?",
            (NEW_MODEL,),
        )
    }
    assert set(new_rows) == set(before)
    assert all(len(blob) == NEW_DIM * 4 for blob in new_rows.values())

    still_there = {
        str(row["detection_id"]): bytes(row["embedding"])
        for row in conn.execute(
            "SELECT detection_id, embedding FROM detection_embeddings "
            "WHERE embedder_model_id = ?",
            (OLD_MODEL,),
        )
    }
    assert still_there == before
    assert (
        conn.execute(
            "SELECT status FROM templates WHERE id = ?", (seeded.template_ok,)
        ).fetchone()["status"]
        == "active"
    )


def test_track_means_are_the_best_k_crops_under_the_new_model(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """Spec 6.2 step 6: the mean of the best K crops, K = embed_k, by detector score."""
    tuned = settings.model_copy(update={"embed_k": 2})
    seed(conn, tuned)
    embedder = ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)

    reembed(conn, tuned, models_for(embedder), job_id=enqueue_reembed(conn), actor="tester")

    # track-a's crops score 0.95, 0.90, 0.10: the 0.10 crop must not reach the mean.
    best_two = np.stack(
        [
            embedder.embed(storage.load_crop(tuned.crops_dir, store_shade(tuned, shade)))[0]
            for shade in (40, 120)
        ]
    )
    expected = vectors.l2_normalize(best_two.mean(axis=0))
    row = conn.execute(
        "SELECT embedding_mean, embedder_model_id FROM tracks WHERE id = 'track-a'"
    ).fetchone()
    assert str(row["embedder_model_id"]) == NEW_MODEL
    assert vectors.from_blob(bytes(row["embedding_mean"]), NEW_DIM) == pytest.approx(
        expected, abs=1e-6
    )

    # A track whose only detection has no stored crop keeps the model it was embedded with:
    # it has nothing to compare under the new one, so it leaves that gallery (invariant 2).
    stranded = conn.execute(
        "SELECT embedding_mean, embedder_model_id FROM tracks WHERE id = 'track-c'"
    ).fetchone()
    assert stranded["embedding_mean"] is None
    assert stranded["embedder_model_id"] is None


def test_a_template_without_a_crop_is_reported_not_dropped(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """The person would leave the gallery in silence; the job has to name the face."""
    seeded = seed(conn, settings)

    result = reembed(
        conn,
        settings,
        models_for(ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)),
        job_id=enqueue_reembed(conn),
        actor="tester",
    )

    assert result.templates_reembedded == 1
    stranded = result.templates_without_crop
    assert [item.template_id for item in stranded] == [seeded.template_no_crop]
    assert stranded[0].person_id == seeded.other_person
    assert "no stored aligned crop" in stranded[0].reason

    # No vector was invented for it, so that person is genuinely out of the new gallery.
    assert (
        conn.execute(
            "SELECT COUNT(*) AS n FROM templates WHERE person_id = ? AND embedder_model_id = ?",
            (seeded.other_person, NEW_MODEL),
        ).fetchone()["n"]
        == 0
    )
    # And the operator can find it again from the chain, not only from the job result.
    payload = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action = 'reembed.complete'"
    ).fetchone()
    assert seeded.template_no_crop in str(payload["payload_json"])


def test_a_resumed_job_does_not_double_write(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """A killed job continues from its checkpoint; committed crops are not embedded twice."""
    seed(conn, settings)
    job_id = enqueue_reembed(conn)

    dying = ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM, fail_after=2)
    with pytest.raises(RuntimeError, match="embedder died"):
        reembed(conn, settings, models_for(dying), job_id=job_id, actor="tester", batch_size=1)

    partial = jobs.get(conn, job_id)
    assert partial is not None
    assert partial.progress["crops_embedded"] == 2
    checkpoint = partial.progress["last_detection_id"]

    healthy = ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)
    result = reembed(
        conn, settings, models_for(healthy), job_id=job_id, actor="tester", batch_size=1
    )

    # It resumed rather than restarting: two crops were already committed, so the second
    # run embedded the remaining two and re-embedded nothing.
    assert healthy.calls == 2
    assert checkpoint == "det-a2"
    assert result.crops_embedded == 4
    duplicates = conn.execute(
        "SELECT detection_id, COUNT(*) AS n FROM detection_embeddings "
        "WHERE embedder_model_id = ? GROUP BY detection_id HAVING n > 1",
        (NEW_MODEL,),
    ).fetchall()
    assert duplicates == []


def test_auto_accept_stays_off_until_a_set_is_calibrated_for_the_new_model(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """C5 and invariant 2: a threshold set calibrated on one model does not gate another."""
    seed(conn, settings, calibrated_for=OLD_MODEL)
    embedder = ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)

    result = reembed(
        conn, settings, models_for(embedder), job_id=enqueue_reembed(conn), actor="tester"
    )

    assert result.auto_accept_allowed is False
    assert result.auto_accept_reason is not None
    assert NEW_MODEL in result.auto_accept_reason
    suspended = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action = 'reembed.auto_accept_suspended'"
    ).fetchall()
    assert len(suspended) == 1

    after_switch = rematch(
        conn,
        settings,
        embedder_model_id=NEW_MODEL,
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )
    assert after_switch.tracks == 2  # the re-embedded tracks are matched...
    assert after_switch.auto_accepted == 0  # ...and nothing self-confirms
    assert conn.execute("SELECT COUNT(*) AS n FROM identities").fetchone()["n"] == 0

    # Calibrate for the new model and activate it: the same match now auto-accepts.
    now = audit.now_ts()
    with transaction(conn):
        conn.execute("UPDATE threshold_sets SET active = 0 WHERE active = 1")
        conn.execute(
            "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
            "calibrated, calibrated_at, eval_report_sha256, gallery_size, "
            "execution_provider, active, created_at) "
            "VALUES ('ts-new', ?, 0.5, 0.2, 0.01, 1, ?, ?, 10, 'CPUExecutionProvider', 1, ?)",
            (NEW_MODEL, now, "f" * 64, now),
        )

    after_calibration = rematch(
        conn,
        settings,
        embedder_model_id=NEW_MODEL,
        execution_provider="CPUExecutionProvider",
        actor="tester",
    )
    assert after_calibration.auto_accepted >= 1
    identity = conn.execute(
        "SELECT person_id, source FROM identities WHERE track_id = 'track-a'"
    ).fetchone()
    assert identity is not None
    assert (str(identity["person_id"]), str(identity["source"])) == ("person-a", "auto")


def test_reembed_never_decodes_original_media(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """Spec 6.2: the stored crops exist so a model switch does not touch source media."""
    seed(conn, settings)
    # The media file was never written to the store, so any decode attempt would fail; the
    # detector raises if it is called at all.
    result = reembed(
        conn,
        settings,
        models_for(ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)),
        job_id=enqueue_reembed(conn),
        actor="tester",
    )
    assert result.crops_embedded == 4


def test_the_moved_tracks_get_their_rescoring_queued(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """Stale scores must not outlive the vectors they were computed from."""
    seed(conn, settings)

    result = reembed(
        conn,
        settings,
        models_for(ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)),
        job_id=enqueue_reembed(conn),
        actor="tester",
    )

    assert result.rematch_job_id is not None
    queued = jobs.get(conn, result.rematch_job_id)
    assert queued is not None
    assert (queued.kind, queued.status) == ("rematch", "queued")


def test_a_rematch_already_waiting_is_not_queued_twice(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """`PATCH /api/config` queues the pair itself; the job must not pile a third on."""
    seed(conn, settings)
    already = jobs.enqueue(conn, kind="rematch", actor="tester")

    result = reembed(
        conn,
        settings,
        models_for(ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)),
        job_id=enqueue_reembed(conn),
        actor="tester",
    )

    assert result.rematch_job_id == already.id
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE kind = 'rematch'"
    ).fetchone()["n"] == 1


def test_a_recorded_crop_missing_from_the_store_is_counted_not_guessed(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """A digest with no object behind it produces no vector, and says so."""
    seeded = seed(conn, settings)
    lost = seeded.detections["det-b1"][2]
    assert lost is not None
    storage.path_for(settings.crops_dir, lost).unlink()

    result = reembed(
        conn,
        settings,
        models_for(ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)),
        job_id=enqueue_reembed(conn),
        actor="tester",
    )

    assert result.crops_missing == 1
    assert result.crops_missing_sample == ["det-b1"]
    assert result.crops_embedded == 3
    assert (
        conn.execute(
            "SELECT COUNT(*) AS n FROM detection_embeddings "
            "WHERE detection_id = 'det-b1' AND embedder_model_id = ?",
            (NEW_MODEL,),
        ).fetchone()["n"]
        == 0
    )


def test_a_reembed_job_runs_through_the_worker(
    conn: sqlite3.Connection, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kind is registered: with no handler the job would fail loudly instead."""
    seed(conn, settings)
    embedder = ShadeEmbedder(model_id=NEW_MODEL, dim=NEW_DIM)
    monkeypatch.setattr(worker.models_lock, "verify", lambda models_dir: None)
    monkeypatch.setattr(worker, "get_active_models", lambda *_: models_for(embedder))
    job_id = enqueue_reembed(conn)

    ran = worker.run_once(conn, settings)

    assert ran is not None and ran.id == job_id
    done = jobs.get(conn, job_id)
    assert done is not None
    assert done.status == "done", done.error
    assert done.progress["crops_embedded"] == 4
    assert audit.verify(conn).ok


def test_a_reembed_job_refuses_to_retarget_itself(
    conn: sqlite3.Connection, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job pinned to one model must not silently re-embed into another (invariant 2)."""
    seed(conn, settings)
    monkeypatch.setattr(worker.models_lock, "verify", lambda models_dir: None)
    monkeypatch.setattr(
        worker,
        "get_active_models",
        lambda *_: models_for(ShadeEmbedder(model_id=OLD_MODEL, dim=OLD_DIM)),
    )
    job_id = enqueue_reembed(conn)

    worker.run_once(conn, settings)

    failed = jobs.get(conn, job_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error is not None
    assert NEW_MODEL in failed.error
