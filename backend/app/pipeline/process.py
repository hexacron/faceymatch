"""Still-image and video processing pipeline (spec 6.2).

Both kinds end in the same place: one `tracks` row per face, its mean embedding, and one
`detections` row per box. A still is the degenerate video — one frame, one detection per
track, `start_ms = end_ms = 0` — which is why steps 4 to 10 are written once and called
from both, and why the matching, acceptance and review code never asks what kind of file a
track came from (spec 6.2, final paragraph).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app import audit
from app.config import Settings
from app.core import storage, vectors
from app.core.registry import ActiveModels
from app.db.conn import transaction
from app.ids import new_id
from app.jobs import write_progress
from app.pipeline import align, decode, quality, tracking, video
from app.pipeline.matching import rematch
from app.pipeline.tracking import Tracker

MEDIA_KIND_IMAGE = "image"
MEDIA_KIND_VIDEO = "video"


class MediaNotFoundError(LookupError):
    """The requested media row does not exist."""


class UnsupportedMediaKindError(ValueError):
    """A pipeline entry point was handed a media row of the kind it does not process."""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    media_id: str
    detections: int
    quality_passed: int
    tracks: int
    matching: dict[str, object] | None = None
    # The checkpoint fields (spec 6.2, "Job resume"). A still is one frame at t = 0; a
    # video reports how far the decode actually got, which is where a resume picks up.
    decoded_frames: int = 1
    last_frame_idx: int = 0
    last_t_ms: int = 0

    def as_progress(self) -> dict[str, object]:
        progress: dict[str, object] = {
            "media_id": self.media_id,
            "last_frame_idx": self.last_frame_idx,
            "last_t_ms": self.last_t_ms,
            "decoded_frames": self.decoded_frames,
            "detections": self.detections,
            "quality_passed": self.quality_passed,
            "tracks": self.tracks,
        }
        if self.matching is not None:
            progress["matching"] = self.matching
        return progress


@dataclass(frozen=True, slots=True)
class _PreparedDetection:
    track_id: str
    detection_id: str
    det_idx: int
    x: float
    y: float
    w: float
    h: float
    landmarks_json: str
    det_score: float
    quality_json: str
    crop_sha256: str | None
    embedding: np.ndarray | None


def process_image(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    *,
    media_id: str,
    actor: str,
) -> ProcessResult:
    """Decode through acceptance, without leaving a tracks-only intermediate result."""
    row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    if row is None:
        raise MediaNotFoundError(f"no media {media_id!r}")
    if str(row["kind"]) != "image":
        raise UnsupportedMediaKindError(
            f"media {media_id!r} is {row['kind']!r}; M1 processing supports images only"
        )

    existing = conn.execute(
        "SELECT COUNT(*) AS count FROM tracks WHERE media_id = ?", (media_id,)
    ).fetchone()
    if existing is not None and int(existing["count"]) > 0:
        count = int(existing["count"])
        passed = conn.execute(
            "SELECT COUNT(*) AS count FROM detections "
            "WHERE media_id = ? AND crop_sha256 IS NOT NULL",
            (media_id,),
        ).fetchone()
        return ProcessResult(media_id, count, 0 if passed is None else int(passed["count"]), count)

    with transaction(conn):
        conn.execute("UPDATE media SET status = 'processing' WHERE id = ?", (media_id,))
        audit.append(
            conn,
            actor=actor,
            case_id=str(row["case_id"]),
            action="media.processing",
            object_type="media",
            object_id=media_id,
            payload={},
        )

    image = decode.decode_image(_stored_path(settings.media_dir, str(row["path"])))
    detections = models.detector.detect(image)
    reports = [quality.evaluate(image, detection, settings) for detection in detections]
    # Store the crops first, then embed them in one call: ArcFace takes the whole stack as
    # a single Run and SFace overlaps its fixed-batch Runs, so one call per image beats one
    # call per face on both adapters. Every passing crop is still embedded — the live cap
    # has no business on the evidence path.
    crop_hashes: list[str] = []
    crops: list[np.ndarray] = []
    for detection, report in zip(detections, reports, strict=True):
        if report.passed:
            crop = align.align_crop(image, detection.landmarks)
            crop_hashes.append(storage.store_crop(settings.crops_dir, crop))
            crops.append(crop)
    embeddings = (
        models.embedder.embed(np.stack(crops))
        if crops
        else np.zeros((0, models.embedder.dim), dtype=np.float32)
    )

    prepared: list[_PreparedDetection] = []
    passed_idx = 0
    for det_idx, (detection, report) in enumerate(zip(detections, reports, strict=True)):
        crop_sha256: str | None = None
        embedding: np.ndarray | None = None
        if report.passed:
            crop_sha256 = crop_hashes[passed_idx]
            embedding = embeddings[passed_idx]
            passed_idx += 1
        prepared.append(
            _PreparedDetection(
                track_id=new_id(),
                detection_id=new_id(),
                det_idx=det_idx,
                x=detection.x,
                y=detection.y,
                w=detection.w,
                h=detection.h,
                landmarks_json=audit.canonical_json(detection.landmarks.tolist()).decode("utf-8"),
                det_score=detection.score,
                quality_json=audit.canonical_json(report.as_json()).decode("utf-8"),
                crop_sha256=crop_sha256,
                embedding=embedding,
            )
        )

    height, width = image.shape[:2]
    now = audit.now_ts()
    with transaction(conn):
        for item in prepared:
            # One crop per still-image track, but through the same definition of a track
            # mean the re-embed job uses, so the two writers cannot drift (spec 6.2 step 6).
            mean_blob = (
                None
                if item.embedding is None
                else vectors.to_blob(
                    vectors.track_mean(
                        item.embedding[None, :], [item.det_score], k=settings.embed_k
                    )
                )
            )
            conn.execute(
                "INSERT INTO tracks (id, media_id, start_ms, end_ms, best_detection_id, "
                "embedding_mean, embedder_model_id) VALUES (?, ?, 0, 0, NULL, ?, ?)",
                (
                    item.track_id,
                    media_id,
                    mean_blob,
                    models.embedder_model_id if mean_blob is not None else None,
                ),
            )
            conn.execute(
                "INSERT INTO detections (id, media_id, track_id, t_ms, frame_idx, det_idx, "
                "x, y, w, h, landmarks_json, det_score, quality_json, crop_sha256, "
                "detector_model_id) VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.detection_id,
                    media_id,
                    item.track_id,
                    item.det_idx,
                    item.x,
                    item.y,
                    item.w,
                    item.h,
                    item.landmarks_json,
                    item.det_score,
                    item.quality_json,
                    item.crop_sha256,
                    models.detector_model_id,
                ),
            )
            conn.execute(
                "UPDATE tracks SET best_detection_id = ? WHERE id = ?",
                (item.detection_id, item.track_id),
            )
            if item.embedding is not None:
                conn.execute(
                    "INSERT INTO detection_embeddings (detection_id, embedding, "
                    "embedder_model_id, created_at) VALUES (?, ?, ?, ?)",
                    (
                        item.detection_id,
                        vectors.to_blob(item.embedding),
                        models.embedder_model_id,
                        now,
                    ),
                )
        conn.execute(
            "UPDATE media SET width = ?, height = ?, duration_ms = NULL, fps = NULL, "
            "status = 'done' WHERE id = ?",
            (int(width), int(height), media_id),
        )
        audit.append(
            conn,
            actor=actor,
            case_id=str(row["case_id"]),
            action="media.processed",
            object_type="media",
            object_id=media_id,
            payload={
                "detections": len(prepared),
                "quality_passed": sum(item.embedding is not None for item in prepared),
                "detector_model_id": models.detector_model_id,
                "embedder_model_id": models.embedder_model_id,
            },
        )

    match_result = rematch(
        conn,
        settings,
        embedder_model_id=models.embedder_model_id,
        execution_provider=models.execution_provider,
        actor=actor,
        track_ids={item.track_id for item in prepared if item.embedding is not None},
    )
    return ProcessResult(
        media_id=media_id,
        detections=len(prepared),
        quality_passed=sum(item.embedding is not None for item in prepared),
        tracks=len(prepared),
        matching=match_result.as_progress(),
    )


@dataclass(slots=True)
class _PendingBox:
    """One detection of one frame, held until its batch commits."""

    frame_idx: int
    t_ms: int
    det_idx: int
    track_key: int
    x: float
    y: float
    w: float
    h: float
    landmarks_json: str
    det_score: float
    quality_json: str
    crop: np.ndarray | None
    crop_sha256: str | None = None
    embedding: np.ndarray | None = None


def process_video(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    *,
    media_id: str,
    actor: str,
    job_id: str | None = None,
) -> ProcessResult:
    """Sample, detect, track, embed and match one video (spec 6.2 steps 1 to 8).

    Work commits in batches of `video_batch_frames` sampled frames, each followed by a
    checkpoint in `jobs.progress`. A killed worker resumes from the last checkpoint rather
    than from the start: detections are unique on `(media_id, frame_idx, det_idx)` and
    `frame_idx` is a function of the timestamp (`video.iter_samples`), so the frames a dead
    batch had already written cannot be written twice.

    What a resume does not carry across is the tracker's own memory: a face on screen at
    the moment the worker died becomes two tracks, one either side of the checkpoint. That
    is a visible, correct-by-parts outcome — two tracks of the same person, each matched on
    its own merits — and the alternative is serialising filter state into the checkpoint so
    a crash can restore a Kalman covariance, which is a great deal of machinery for a case
    that only arises after a kill.
    """
    row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    if row is None:
        raise MediaNotFoundError(f"no media {media_id!r}")
    if str(row["kind"]) != MEDIA_KIND_VIDEO:
        raise UnsupportedMediaKindError(
            f"media {media_id!r} is {row['kind']!r}; process_video takes video only"
        )
    case_id = str(row["case_id"])
    path = _stored_path(settings.media_dir, str(row["path"]))
    info = video.probe(path)
    resume_ms, totals = _resume_from(conn, job_id)

    with transaction(conn):
        conn.execute(
            "UPDATE media SET status = 'processing', width = ?, height = ?, "
            "duration_ms = ?, fps = ? WHERE id = ?",
            (info.width, info.height, info.duration_ms, info.fps, media_id),
        )
        audit.append(
            conn,
            actor=actor,
            case_id=case_id,
            action="media.processing",
            object_type="media",
            object_id=media_id,
            payload={"resumed_from_t_ms": resume_ms} if resume_ms > 0 else {},
        )

    tracker = Tracker(
        min_iou=settings.track_min_iou,
        high_score=settings.track_high_score,
        min_hits=settings.track_min_hits,
        max_age_ms=settings.track_max_age_ms,
    )
    track_ids: dict[int, str] = {}
    written_tracks: set[str] = set()
    pending: list[_PendingBox] = []
    last_frame_idx = 0
    last_t_ms = resume_ms

    for sample in video.iter_samples(
        path, sample_fps=settings.sample_fps, start_ms=resume_ms
    ):
        detections = models.detector.detect(sample.image)
        step = tracker.update(
            sample.t_ms,
            [
                tracking.Observation(
                    x=det.x, y=det.y, w=det.w, h=det.h, score=det.score
                )
                for det in detections
            ],
        )
        for assignment in step.assignments:
            detection = detections[assignment.detection_index]
            report = quality.evaluate(sample.image, detection, settings)
            pending.append(
                _PendingBox(
                    frame_idx=sample.frame_idx,
                    t_ms=sample.t_ms,
                    det_idx=assignment.detection_index,
                    track_key=assignment.track_key,
                    x=detection.x,
                    y=detection.y,
                    w=detection.w,
                    h=detection.h,
                    landmarks_json=audit.canonical_json(
                        detection.landmarks.tolist()
                    ).decode("utf-8"),
                    det_score=detection.score,
                    quality_json=audit.canonical_json(report.as_json()).decode("utf-8"),
                    crop=align.align_crop(sample.image, detection.landmarks)
                    if report.passed
                    else None,
                )
            )
        totals["decoded"] += 1
        totals["detections"] += len(step.assignments)
        last_frame_idx = sample.frame_idx
        last_t_ms = sample.t_ms

        _unwind(conn, track_ids, written_tracks, step.discarded)
        if totals["decoded"] % settings.video_batch_frames == 0:
            totals["passed"] += _commit_batch(
                conn,
                settings,
                models,
                media_id=media_id,
                pending=pending,
                track_ids=track_ids,
                written_tracks=written_tracks,
            )
            pending = []
            _checkpoint(
                conn,
                job_id,
                media_id=media_id,
                totals=totals,
                last_frame_idx=last_frame_idx,
                last_t_ms=last_t_ms,
                tracks=_track_count(conn, media_id),
            )

    totals["passed"] += _commit_batch(
        conn,
        settings,
        models,
        media_id=media_id,
        pending=pending,
        track_ids=track_ids,
        written_tracks=written_tracks,
    )
    # Whatever is still tentative when the file ends never became a face. Swept from the
    # rows rather than from the tracker's memory, so it also catches what a previous run
    # left behind when it was killed: a confirmed track holds at least `track_min_hits`
    # detections by construction, so a shorter one is a flicker whoever wrote it.
    _sweep_short_tracks(conn, media_id=media_id, min_hits=settings.track_min_hits)

    # Every track of this file, not just this run's: a resumed job has to finish the work
    # its predecessor started, and a mean is only written once the file is through.
    surviving = [
        str(item["id"])
        for item in conn.execute(
            "SELECT id FROM tracks WHERE media_id = ? ORDER BY id", (media_id,)
        ).fetchall()
    ]
    _write_track_means(conn, settings, models, track_ids=surviving)
    with transaction(conn):
        conn.execute("UPDATE media SET status = 'done' WHERE id = ?", (media_id,))
        audit.append(
            conn,
            actor=actor,
            case_id=case_id,
            action="media.processed",
            object_type="media",
            object_id=media_id,
            payload={
                "kind": MEDIA_KIND_VIDEO,
                "decoded_frames": totals["decoded"],
                "detections": totals["detections"],
                "quality_passed": totals["passed"],
                "tracks": len(surviving),
                "detector_model_id": models.detector_model_id,
                "embedder_model_id": models.embedder_model_id,
            },
        )

    match_result = rematch(
        conn,
        settings,
        embedder_model_id=models.embedder_model_id,
        execution_provider=models.execution_provider,
        actor=actor,
        track_ids=set(surviving),
    )
    return ProcessResult(
        media_id=media_id,
        detections=totals["detections"],
        quality_passed=totals["passed"],
        tracks=len(surviving),
        matching=match_result.as_progress(),
        decoded_frames=totals["decoded"],
        last_frame_idx=last_frame_idx,
        last_t_ms=last_t_ms,
    )


def _resume_from(
    conn: sqlite3.Connection, job_id: str | None
) -> tuple[int, dict[str, int]]:
    """The checkpoint: where to restart, and the counts to carry on from.

    Reading the checkpoint rather than scanning for the last written row is what spec 6.2
    asks for, and carrying the counts means a resumed job's progress continues rather than
    restarting at zero — a progress bar that goes backwards after a crash reports the crash
    as lost work, which it is not.
    """
    empty = {"decoded": 0, "detections": 0, "passed": 0}
    if job_id is None:
        return 0, empty
    row = conn.execute("SELECT progress FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return 0, empty
    stored = json.loads(str(row["progress"]))
    if not isinstance(stored, dict):
        return 0, empty
    last_t_ms = _as_count(stored.get("last_t_ms"))
    totals = {
        "decoded": _as_count(stored.get("decoded_frames")),
        "detections": _as_count(stored.get("detections")),
        "passed": _as_count(stored.get("quality_passed")),
    }
    # Strictly after the last committed frame: that one is already written, and a repeat
    # would be ignored by the unique index rather than corrected.
    return (last_t_ms + 1 if last_t_ms else 0), totals


def _as_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _track_count(conn: sqlite3.Connection, media_id: str) -> int:
    """From the rows, so a resumed job's checkpoint counts its predecessor's tracks too."""
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM tracks WHERE media_id = ?", (media_id,)
    ).fetchone()
    return 0 if row is None else int(row["count"])


def _sweep_short_tracks(conn: sqlite3.Connection, *, media_id: str, min_hits: int) -> None:
    """Unwind tracks with fewer detections than a confirmation takes."""
    rows = conn.execute(
        "SELECT t.id FROM tracks t LEFT JOIN detections d ON d.track_id = t.id "
        "WHERE t.media_id = ? GROUP BY t.id HAVING COUNT(d.id) < ?",
        (media_id, min_hits),
    ).fetchall()
    if not rows:
        return
    with transaction(conn):
        for item in rows:
            conn.execute(
                "UPDATE detections SET track_id = NULL WHERE track_id = ?", (item["id"],)
            )
            conn.execute("DELETE FROM tracks WHERE id = ?", (item["id"],))


def _commit_batch(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    *,
    media_id: str,
    pending: list[_PendingBox],
    track_ids: dict[int, str],
    written_tracks: set[str],
) -> int:
    """Store crops, embed them in one call, and write the batch. Returns crops embedded."""
    if not pending:
        return 0
    passing = [(box, box.crop) for box in pending if box.crop is not None]
    for box, crop in passing:
        box.crop_sha256 = storage.store_crop(settings.crops_dir, crop)
    if passing:
        # One embed call per batch, not per frame: both adapters are faster on a stack, and
        # a 3 fps video hands them a few dozen faces at a time.
        embeddings = models.embedder.embed(np.stack([crop for _box, crop in passing]))
        for (box, _crop), embedding in zip(passing, embeddings, strict=True):
            box.embedding = embedding
            # The pixels are in the object store and the vector is in hand; holding a batch
            # of 112x112 crops through the write adds nothing but resident memory.
            box.crop = None

    now = audit.now_ts()
    with transaction(conn):
        for box in pending:
            track_id = track_ids.get(box.track_key)
            if track_id is None:
                track_id = new_id()
                track_ids[box.track_key] = track_id
            if track_id not in written_tracks:
                conn.execute(
                    "INSERT INTO tracks (id, media_id, start_ms, end_ms) VALUES (?, ?, ?, ?)",
                    (track_id, media_id, box.t_ms, box.t_ms),
                )
                written_tracks.add(track_id)
            else:
                conn.execute(
                    "UPDATE tracks SET end_ms = ? WHERE id = ? AND end_ms < ?",
                    (box.t_ms, track_id, box.t_ms),
                )
            detection_id = new_id()
            # `OR IGNORE` is the resume guard: a frame this job already committed before it
            # died is unique on (media_id, frame_idx, det_idx) and is left exactly as it
            # was written, rather than inserted a second time under a new track.
            written = conn.execute(
                "INSERT OR IGNORE INTO detections (id, media_id, track_id, t_ms, frame_idx, "
                "det_idx, x, y, w, h, landmarks_json, det_score, quality_json, crop_sha256, "
                "detector_model_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    detection_id,
                    media_id,
                    track_id,
                    box.t_ms,
                    box.frame_idx,
                    box.det_idx,
                    box.x,
                    box.y,
                    box.w,
                    box.h,
                    box.landmarks_json,
                    box.det_score,
                    box.quality_json,
                    box.crop_sha256,
                    models.detector_model_id,
                ),
            ).rowcount
            if written and box.embedding is not None:
                conn.execute(
                    "INSERT INTO detection_embeddings (detection_id, embedding, "
                    "embedder_model_id, created_at) VALUES (?, ?, ?, ?)",
                    (
                        detection_id,
                        vectors.to_blob(box.embedding),
                        models.embedder_model_id,
                        now,
                    ),
                )
    return len(passing)


def _unwind(
    conn: sqlite3.Connection,
    track_ids: dict[int, str],
    written_tracks: set[str],
    keys: list[int],
) -> None:
    """Drop tracks that never became a face, keeping their detections as evidence.

    The boxes stay — something was in the picture and the audit trail says so — but with no
    track they are never embedded into a mean, never matched and never named. Deleting them
    instead would be deleting the record of a detector decision.
    """
    for key in keys:
        track_id = track_ids.pop(key, None)
        if track_id is None:
            continue
        written_tracks.discard(track_id)
        with transaction(conn):
            conn.execute(
                "UPDATE detections SET track_id = NULL WHERE track_id = ?", (track_id,)
            )
            conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))


def _write_track_means(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    *,
    track_ids: list[str],
) -> None:
    """`tracks.embedding_mean` and `best_detection_id`, read back from what was stored.

    Read back rather than accumulated in memory, because a resumed job never saw the crops
    the run before it embedded, and a mean over half a track is not the track's mean. Same
    `vectors.track_mean` the still path and the re-embed job use (spec 6.2 step 6).
    """
    for track_id in track_ids:
        rows = conn.execute(
            "SELECT d.id, d.det_score, e.embedding FROM detections d "
            "JOIN detection_embeddings e ON e.detection_id = d.id "
            "AND e.embedder_model_id = ? WHERE d.track_id = ? ORDER BY d.det_score DESC, d.id",
            (models.embedder_model_id, track_id),
        ).fetchall()
        with transaction(conn):
            if not rows:
                # Every box in this track failed the quality gate: a real face the camera
                # never showed well enough to embed. The track stays, unmatched.
                continue
            mean = vectors.track_mean(
                vectors.stack_blobs([bytes(item["embedding"]) for item in rows],
                                    models.embedder.dim),
                [float(item["det_score"]) for item in rows],
                k=settings.embed_k,
            )
            conn.execute(
                "UPDATE tracks SET embedding_mean = ?, embedder_model_id = ?, "
                "best_detection_id = ? WHERE id = ?",
                (
                    vectors.to_blob(mean),
                    models.embedder_model_id,
                    str(rows[0]["id"]),
                    track_id,
                ),
            )


def _checkpoint(
    conn: sqlite3.Connection,
    job_id: str | None,
    *,
    media_id: str,
    totals: dict[str, int],
    last_frame_idx: int,
    last_t_ms: int,
    tracks: int,
) -> None:
    if job_id is None:
        return
    with transaction(conn):
        write_progress(
            conn,
            job_id,
            {
                "media_id": media_id,
                "last_frame_idx": last_frame_idx,
                "last_t_ms": last_t_ms,
                "decoded_frames": totals["decoded"],
                "detections": totals["detections"],
                "quality_passed": totals["passed"],
                "tracks": tracks,
            },
        )


def mark_failed(conn: sqlite3.Connection, *, media_id: str, actor: str, error: str) -> None:
    """Mark a failed pipeline outcome and audit the same transaction."""
    row = conn.execute("SELECT case_id FROM media WHERE id = ?", (media_id,)).fetchone()
    if row is None:
        return
    with transaction(conn):
        conn.execute("UPDATE media SET status = 'failed' WHERE id = ?", (media_id,))
        audit.append(
            conn,
            actor=actor,
            case_id=str(row["case_id"]),
            action="media.process_failed",
            object_type="media",
            object_id=media_id,
            payload={"error": error},
        )


def _stored_path(root: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else root / path
