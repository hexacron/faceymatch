"""Re-embed stored aligned crops under a new embedder (spec 6.2 steps 5-6, 6.3, 7, 9).

The original media is never decoded here. Step 5 of the pipeline writes the aligned 112x112
crop of every quality-passing detection to the content-addressed store precisely so that a
model switch is a re-read of those crops (spec 9): the bytes an embedding was computed from
stay re-checkable, and a switch costs one pass over `detections.crop_sha256` instead of a
re-decode of every file in the case.

Nothing is overwritten. `detection_embeddings` is keyed `(detection_id, embedder_model_id)`
and templates are re-created rather than mutated, so the previous model's vectors survive
the switch intact. That is what makes a switch reversible: pointing `embedder_model` back
at the old id restores the old gallery without re-embedding anything.

Nothing is mixed either (invariant 2). Every row written carries the new model id, the
gallery and track queries filter on it, and the `matches_same_model` trigger refuses a
score that crosses the boundary.

Three phases, each checkpointed into `jobs.progress` after its committed batches so a
killed job resumes where it stopped (spec 6.2, "Job resume"):

1. crops     -> one `detection_embeddings` row per stored crop, for the new model;
2. tracks    -> `tracks.embedding_mean` and `tracks.embedder_model_id` recomputed as the
                L2-normalized mean of the best `embed_k` crops (`vectors.track_mean`, the
                same function the still-image pipeline writes its means with);
3. templates -> a new template row per active template, carrying the same person, source
                case and quality, under the new model.

The checkpoint is an optimisation, not the correctness argument: every write is guarded on
the row it would create, so a resumed job cannot double-write even with the checkpoint lost.

A template whose detection has no stored crop cannot be re-embedded, and no vector is
invented for it. It is reported by id in the result and in the audit entry, because that
template's person otherwise leaves the gallery in silence: the operator has to know which
faces stopped being searchable so they can re-enrol them from media that still exists.

Re-embedding a template is not a new enrolment (invariant 3, C6). It re-states an enrolment
an operator already made, under a different model, and the audit entry carries the id of the
template it came from so the chain runs back to that operator's `template.create`. No
template is created for a person, detection or crop nobody enrolled.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from app import audit
from app.config import Settings
from app.core import acceptance, storage, vectors
from app.core.registry import ActiveModels
from app.core.storage import ObjectNotFoundError
from app.db.conn import transaction
from app.ids import new_id
from app.jobs import insert as jobs_insert
from app.jobs import write_progress
from app.pipeline.matching import active_threshold_set, gallery_person_count

# Crops per embed call and per commit. Large enough that ONNX batching pays off, small
# enough that a kill loses about a second of work.
DEFAULT_BATCH_SIZE = 256

# Missing crops are counted exactly and sampled by id: on a large case the count is the
# actionable number, and an unbounded id list would bloat every progress read.
MAX_SAMPLED_IDS = 50


@dataclass(frozen=True, slots=True)
class MissingTemplateCrop:
    """An active template that could not follow the switch, and why."""

    template_id: str
    person_id: str
    detection_id: str | None
    reason: str

    def as_json(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "person_id": self.person_id,
            "detection_id": self.detection_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ReembedResult:
    embedder_model_id: str
    crops_embedded: int
    crops_missing: int
    crops_missing_sample: list[str]
    tracks_updated: int
    tracks_left_on_previous_model: int
    templates_reembedded: int
    templates_without_crop: list[MissingTemplateCrop]
    templates_skipped_do_not_enroll: int
    gallery_persons: int
    auto_accept_allowed: bool
    auto_accept_reason: str | None
    # The rematch that will rescore the moved tracks. Never None once the job has run.
    rematch_job_id: str | None = None

    def as_progress(self) -> dict[str, Any]:
        return {
            "phase": "done",
            "embedder_model_id": self.embedder_model_id,
            "crops_embedded": self.crops_embedded,
            "crops_missing": self.crops_missing,
            "crops_missing_sample": self.crops_missing_sample,
            "tracks_updated": self.tracks_updated,
            "tracks_left_on_previous_model": self.tracks_left_on_previous_model,
            "templates_reembedded": self.templates_reembedded,
            "templates_without_crop": [item.as_json() for item in self.templates_without_crop],
            "templates_skipped_do_not_enroll": self.templates_skipped_do_not_enroll,
            "gallery_persons": self.gallery_persons,
            "auto_accept_allowed": self.auto_accept_allowed,
            "auto_accept_reason": self.auto_accept_reason,
            "rematch_job_id": self.rematch_job_id,
        }


class _Checkpoint:
    """The job's resume state, carried across phases and written whole after each batch.

    Written whole rather than per key so that `jobs.progress` always describes one coherent
    point in the run: a reader (or a resuming worker) never sees phase 3's counters beside
    phase 1's cursor.
    """

    def __init__(self, conn: sqlite3.Connection, job_id: str, *, model_id: str) -> None:
        self.job_id = job_id
        row = conn.execute("SELECT progress FROM jobs WHERE id = ?", (job_id,)).fetchone()
        stored = {} if row is None else json.loads(str(row["progress"]))
        # A checkpoint left by a run against a different model says nothing about this one.
        resumable = (
            isinstance(stored, dict) and stored.get("embedder_model_id") == model_id
        )
        self.state: dict[str, Any] = dict(stored) if resumable else {}
        self.state["embedder_model_id"] = model_id

    def text(self, key: str) -> str:
        value = self.state.get(key)
        return value if isinstance(value, str) else ""

    def count(self, key: str) -> int:
        value = self.state.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def ids(self, key: str) -> list[str]:
        value = self.state.get(key)
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)][:MAX_SAMPLED_IDS]

    def save(self, conn: sqlite3.Connection, **fields: Any) -> None:
        """Update and persist, inside the caller's transaction."""
        self.state.update(fields)
        write_progress(conn, self.job_id, self.state)


def reembed(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    *,
    job_id: str,
    actor: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ReembedResult:
    """Re-embed every stored crop, track and template under `models.embedder_model_id`."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    model_id = models.embedder_model_id
    checkpoint = _Checkpoint(conn, job_id, model_id=model_id)

    _embed_crops(conn, settings, models, checkpoint=checkpoint, batch_size=batch_size)
    _recompute_track_means(
        conn, settings, model_id=model_id, checkpoint=checkpoint, batch_size=batch_size
    )
    missing_templates = _reembed_templates(
        conn, model_id=model_id, actor=actor, checkpoint=checkpoint
    )

    gallery_persons = gallery_person_count(conn, embedder_model_id=model_id)
    threshold_set = active_threshold_set(conn)
    gate = (
        None
        if threshold_set is None
        else acceptance.build_gate(
            threshold_set,
            embedder_model_id=model_id,
            execution_provider=models.execution_provider,
            live_gallery_size=gallery_persons,
        )
    )
    allowed = False if gate is None else gate.allowed
    reason = "no active threshold set" if gate is None else gate.reason

    result = ReembedResult(
        embedder_model_id=model_id,
        crops_embedded=checkpoint.count("crops_embedded"),
        crops_missing=checkpoint.count("crops_missing"),
        crops_missing_sample=checkpoint.ids("crops_missing_sample"),
        tracks_updated=checkpoint.count("tracks_updated"),
        tracks_left_on_previous_model=_count_tracks_on_other_models(conn, model_id=model_id),
        templates_reembedded=checkpoint.count("templates_reembedded"),
        templates_without_crop=missing_templates,
        templates_skipped_do_not_enroll=checkpoint.count("templates_skipped_do_not_enroll"),
        gallery_persons=gallery_persons,
        auto_accept_allowed=allowed,
        auto_accept_reason=reason,
        rematch_job_id=_pending_rematch(conn),
    )

    with transaction(conn):
        if result.rematch_job_id is None:
            # Track means now belong to a different model than the `matches` rows scored
            # from them. Nothing may read a stale score, so the job that moved the vectors
            # is the job that queues their rescoring. `PATCH /api/config` already queues
            # one, which is why this is conditional rather than unconditional.
            result = replace(
                result,
                rematch_job_id=jobs_insert(
                    conn,
                    kind="rematch",
                    actor=actor,
                    params={"reason": "reembed_followup", "embedder_model_id": model_id},
                ),
            )
        audit.append(
            conn,
            actor=actor,
            action="reembed.complete",
            object_type="model",
            object_id=model_id,
            payload=result.as_progress(),
        )
        if not allowed:
            # Spec 10 and C5: the switch puts the active threshold set out of scope, so
            # auto-accept is off until a set calibrated for this model is activated. Stated
            # as its own entry rather than left implicit in a counter, because "why did the
            # system stop confirming identities" is a question asked months later.
            audit.append(
                conn,
                actor=actor,
                action="reembed.auto_accept_suspended",
                object_type="model",
                object_id=model_id,
                payload={
                    "embedder_model_id": model_id,
                    "threshold_set_id": None if threshold_set is None else threshold_set.id,
                    "threshold_set_model_id": (
                        None if threshold_set is None else threshold_set.model_id
                    ),
                    "reason": reason,
                },
            )
    return result


_PENDING_CROPS = (
    "SELECT d.id, d.crop_sha256 FROM detections d "
    "WHERE d.crop_sha256 IS NOT NULL AND d.id > ? "
    "AND NOT EXISTS (SELECT 1 FROM detection_embeddings de "
    "                WHERE de.detection_id = d.id AND de.embedder_model_id = ?) "
    "ORDER BY d.id LIMIT ?"
)


def _embed_crops(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    *,
    checkpoint: _Checkpoint,
    batch_size: int,
) -> None:
    """Phase 1: one `detection_embeddings` row per stored crop, for the new model."""
    model_id = models.embedder_model_id
    cursor = checkpoint.text("last_detection_id")
    embedded = checkpoint.count("crops_embedded")
    missing = checkpoint.count("crops_missing")
    sample = checkpoint.ids("crops_missing_sample")

    while True:
        rows = conn.execute(_PENDING_CROPS, (cursor, model_id, batch_size)).fetchall()
        if not rows:
            break
        cursor = str(rows[-1]["id"])
        crops: list[np.ndarray] = []
        detection_ids: list[str] = []
        for row in rows:
            detection_id = str(row["id"])
            try:
                crops.append(storage.load_crop(settings.crops_dir, str(row["crop_sha256"])))
            except ObjectNotFoundError:
                # The digest is recorded but the object is gone. Never fabricate a vector
                # for it: the detection simply has no embedding under the new model, and
                # the count says how much of the case that is.
                missing += 1
                if len(sample) < MAX_SAMPLED_IDS:
                    sample.append(detection_id)
                continue
            detection_ids.append(detection_id)

        vectorised = (
            models.embedder.embed(np.stack(crops))
            if crops
            else np.zeros((0, models.embedder.dim), dtype=np.float32)
        )
        now = audit.now_ts()
        with transaction(conn):
            for detection_id, embedding in zip(detection_ids, vectorised, strict=True):
                conn.execute(
                    "INSERT INTO detection_embeddings (detection_id, embedding, "
                    "embedder_model_id, created_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (detection_id, embedder_model_id) DO NOTHING",
                    (detection_id, vectors.to_blob(embedding), model_id, now),
                )
            embedded += len(detection_ids)
            checkpoint.save(
                conn,
                phase="crops",
                last_detection_id=cursor,
                crops_embedded=embedded,
                crops_missing=missing,
                crops_missing_sample=sample,
            )


_TRACK_PAGE = (
    "SELECT DISTINCT d.track_id AS track_id FROM detections d "
    "JOIN detection_embeddings de ON de.detection_id = d.id "
    "                            AND de.embedder_model_id = ? "
    "WHERE d.track_id IS NOT NULL AND d.track_id > ? "
    "ORDER BY d.track_id LIMIT ?"
)


def _recompute_track_means(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    model_id: str,
    checkpoint: _Checkpoint,
    batch_size: int,
) -> None:
    """Phase 2: `tracks.embedding_mean` from the new crop embeddings.

    A track whose crops all failed to re-embed keeps pointing at the previous model. Its old
    mean is still a true statement about that model, and every matching query filters on
    `embedder_model_id`, so the track drops out of the new gallery rather than being scored
    against vectors it cannot be compared with (invariant 2).

    Paged by track id rather than streamed, so no read cursor is left open across a commit.
    """
    dim = _model_dim(conn, model_id)
    cursor = checkpoint.text("last_track_id")
    updated = checkpoint.count("tracks_updated")

    while True:
        page = conn.execute(_TRACK_PAGE, (model_id, cursor, batch_size)).fetchall()
        if not page:
            break
        track_ids = [str(row["track_id"]) for row in page]
        cursor = track_ids[-1]
        crops = _crops_by_track(conn, model_id=model_id, track_ids=track_ids)
        with transaction(conn):
            for track_id in track_ids:
                rows = crops.get(track_id)
                if not rows:  # pragma: no cover - the page query selected it by join
                    continue
                mean = vectors.track_mean(
                    vectors.stack_blobs([blob for blob, _ in rows], dim),
                    [score for _, score in rows],
                    k=settings.embed_k,
                )
                updated += conn.execute(
                    "UPDATE tracks SET embedding_mean = ?, embedder_model_id = ? "
                    "WHERE id = ? AND (embedder_model_id IS NOT ? OR embedding_mean IS NOT ?)",
                    (
                        vectors.to_blob(mean),
                        model_id,
                        track_id,
                        model_id,
                        vectors.to_blob(mean),
                    ),
                ).rowcount
            checkpoint.save(
                conn, phase="tracks", last_track_id=cursor, tracks_updated=updated
            )


def _crops_by_track(
    conn: sqlite3.Connection, *, model_id: str, track_ids: list[str]
) -> dict[str, list[tuple[bytes, float]]]:
    """Every stored crop embedding of one page of tracks, in detection order."""
    placeholders = ",".join("?" * len(track_ids))
    rows = conn.execute(
        "SELECT d.track_id AS track_id, d.det_score AS det_score, "  # noqa: S608
        "       de.embedding AS embedding FROM detections d "
        "JOIN detection_embeddings de ON de.detection_id = d.id "
        "                            AND de.embedder_model_id = ? "
        f"WHERE d.track_id IN ({placeholders}) ORDER BY d.track_id, d.id",
        (model_id, *track_ids),
    ).fetchall()
    grouped: dict[str, list[tuple[bytes, float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["track_id"]), []).append(
            (bytes(row["embedding"]), float(row["det_score"]))
        )
    return grouped


_ORPHAN_TEMPLATES = (
    "SELECT t.id, t.person_id, t.detection_id, t.source_case_id, t.quality, "
    "       p.do_not_enroll AS do_not_enroll, d.crop_sha256 AS crop_sha256, "
    "       de.embedding AS embedding "
    "FROM templates t "
    "JOIN persons p ON p.id = t.person_id "
    "LEFT JOIN detections d ON d.id = t.detection_id "
    "LEFT JOIN detection_embeddings de ON de.detection_id = t.detection_id "
    "                                 AND de.embedder_model_id = ? "
    "WHERE t.status = 'active' AND t.embedder_model_id <> ? "
    "AND NOT EXISTS (SELECT 1 FROM templates t2 WHERE t2.person_id = t.person_id "
    "                AND t2.detection_id IS t.detection_id "
    "                AND t2.embedder_model_id = ? AND t2.status = 'active') "
    "ORDER BY t.id"
)


def _reembed_templates(
    conn: sqlite3.Connection, *, model_id: str, actor: str, checkpoint: _Checkpoint
) -> list[MissingTemplateCrop]:
    """Phase 3: carry the gallery across the switch, and name what could not come along.

    One transaction: a gallery that is half-migrated is worse than one that is not migrated,
    and templates are counted in thousands, not millions.
    """
    rows = conn.execute(_ORPHAN_TEMPLATES, (model_id, model_id, model_id)).fetchall()
    created = 0
    blocked = 0
    missing: list[MissingTemplateCrop] = []
    seen: set[tuple[str, str | None]] = set()
    now = audit.now_ts()

    with transaction(conn):
        for row in rows:
            template_id = str(row["id"])
            person_id = str(row["person_id"])
            detection_id = None if row["detection_id"] is None else str(row["detection_id"])
            if bool(row["do_not_enroll"]):
                # Section 12: the person is out of the gallery anyway, and the
                # templates_respect_do_not_enroll trigger would abort the batch.
                blocked += 1
                continue
            key = (person_id, detection_id)
            if key in seen:
                continue
            seen.add(key)
            if row["embedding"] is None:
                missing.append(
                    MissingTemplateCrop(
                        template_id=template_id,
                        person_id=person_id,
                        detection_id=detection_id,
                        reason=_missing_reason(detection_id, row["crop_sha256"]),
                    )
                )
                continue
            new_template_id = new_id()
            source_case_id = (
                None if row["source_case_id"] is None else str(row["source_case_id"])
            )
            conn.execute(
                "INSERT INTO templates (id, person_id, detection_id, source_case_id, "
                "embedding, embedder_model_id, quality, status, created_at, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (
                    new_template_id,
                    person_id,
                    detection_id,
                    source_case_id,
                    row["embedding"],
                    model_id,
                    None if row["quality"] is None else float(row["quality"]),
                    now,
                    actor,
                ),
            )
            audit.append(
                conn,
                actor=actor,
                case_id=source_case_id,
                action="template.reembed",
                object_type="template",
                object_id=new_template_id,
                payload={
                    "person_id": person_id,
                    "detection_id": detection_id,
                    "embedder_model_id": model_id,
                    "source_template_id": template_id,
                },
            )
            created += 1
        checkpoint.save(
            conn,
            phase="templates",
            templates_reembedded=created,
            templates_without_crop=[item.as_json() for item in missing],
            templates_skipped_do_not_enroll=blocked,
        )
    return missing


def _missing_reason(detection_id: str | None, crop_sha256: Any) -> str:
    if detection_id is None:
        return "template has no source detection to re-embed from"
    if crop_sha256 is None:
        return "source detection has no stored aligned crop (it never passed the quality gate)"
    return f"stored crop {str(crop_sha256)[:12]} is missing from the crop store"


def _model_dim(conn: sqlite3.Connection, model_id: str) -> int:
    row = conn.execute(
        "SELECT dim FROM models WHERE id = ? AND kind = 'embedder'", (model_id,)
    ).fetchone()
    if row is None or row["dim"] is None:
        raise ValueError(f"embedder {model_id!r} has no registered dimension")
    return int(row["dim"])


def _count_tracks_on_other_models(conn: sqlite3.Connection, *, model_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM tracks "
        "WHERE embedder_model_id IS NOT NULL AND embedder_model_id <> ?",
        (model_id,),
    ).fetchone()
    return 0 if row is None else int(row["count"])


def _pending_rematch(conn: sqlite3.Connection) -> str | None:
    """A rematch already waiting to run, so the follow-up is queued once, not twice."""
    row = conn.execute(
        "SELECT id FROM jobs WHERE kind = 'rematch' AND status IN ('queued', 'running') "
        "ORDER BY created_at, id LIMIT 1"
    ).fetchone()
    return None if row is None else str(row["id"])

