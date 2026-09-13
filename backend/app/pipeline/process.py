"""Still-image processing pipeline (spec 6.2)."""

from __future__ import annotations

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
from app.pipeline import align, decode, quality
from app.pipeline.matching import rematch


class MediaNotFoundError(LookupError):
    """The requested media row does not exist."""


class UnsupportedMediaKindError(ValueError):
    """The M1 execution pipeline only processes still images."""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    media_id: str
    detections: int
    quality_passed: int
    tracks: int
    matching: dict[str, object] | None = None

    def as_progress(self) -> dict[str, object]:
        progress: dict[str, object] = {
            "media_id": self.media_id,
            "last_frame_idx": 0,
            "last_t_ms": 0,
            "decoded_frames": 1,
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
