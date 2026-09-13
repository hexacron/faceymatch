"""Live match (tier 2): detect, gate, align, embed, score — and persist nothing.

This is a read-only query against the gallery, for the operator browsing media in another
application. It shares every stage with the stored pipeline (`process.py`): the same
`Detector`, the same quality gate, the same alignment, the same `Embedder`, and the same
`scoring`/`acceptance` band logic. Only the writes are missing.

What it deliberately does *not* do, and why:

- No `media`, `detections`, `detection_embeddings`, `tracks`, `matches`, `identities` or
  `templates` rows, no stored crop, and no per-frame audit entry. Nothing here is a state
  change, so there is nothing to append (invariant 6 stays intact rather than being
  flooded with un-actionable frame entries).
- Because nothing is stored, nothing here can be tagged or enrolled. To enrol what the
  operator sees, the frame must first go through tier 1 (`POST /api/capture`), which makes
  it hashed, content-addressed evidence; the operator then acts on the resulting
  detection. A biometric claim never rests on pixels we did not keep (invariant 3, spec 12).
- The result is advisory. `AutoAcceptGate` is reported so the UI can explain why nothing
  self-confirms, but no live face is ever marked accepted: auto-accept is a property of
  stored matches (invariant 4, D16).

Gallery caching: the template matrix is loaded once per `embedder_model_id` and reused
across frames, keyed by the audit chain head. Invariant 6 makes that head a total version
counter for the database — every state change appends to it — so a cached matrix can never
outlive a template, person-status or embedding change, and the check costs one indexed
`MAX(seq)` read instead of a full gallery reload per frame.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app import audit
from app.config import Settings
from app.core import acceptance, scoring, vectors
from app.core.registry import ActiveModels
from app.core.types import Band, PersonCandidate
from app.pipeline import align, decode, matching, quality


@dataclass(frozen=True, slots=True)
class LiveCandidate:
    person_id: str
    name: str
    rank: int
    score: float
    band: Band
    best_template_id: str


@dataclass(frozen=True, slots=True)
class LiveFace:
    x: float
    y: float
    w: float
    h: float
    det_score: float
    quality_passed: bool
    quality_reasons: list[str]
    candidates: list[LiveCandidate]


@dataclass(frozen=True, slots=True)
class LiveTimings:
    """Per-stage wall time in ms. Advisory: the UI sizes its sampling interval from it."""

    decode: float
    detect: float
    quality_align: float
    embed: float
    match: float

    def as_dict(self) -> dict[str, float]:
        return {
            "decode": self.decode,
            "detect": self.detect,
            "quality_align": self.quality_align,
            "embed": self.embed,
            "match": self.match,
        }


@dataclass(frozen=True, slots=True)
class LiveResult:
    width: int
    height: int
    faces: list[LiveFace]
    # Whether identification was asked for. A client has to be able to tell "no candidates
    # were found" from "we did not look" — they render as very different things.
    identified: bool
    threshold_set_id: str | None
    auto_accept_allowed: bool
    auto_accept_reason: str | None
    gallery_persons: int
    elapsed_ms: int
    timings: LiveTimings


@dataclass(frozen=True, slots=True)
class Gallery:
    """One `(templates, dim)` matrix plus the row provenance needed to rank persons."""

    matrix: np.ndarray
    person_ids: list[str]
    template_ids: list[str]
    names: dict[str, str]
    # Built with the matrix, because every frame reduces over the same rows and deriving
    # the grouping from the id strings per face is the reduction's whole cost.
    person_index: scoring.PersonIndex
    # The audit chain head hash the matrix was loaded at. A hash rather than the sequence
    # number: it identifies the chain as well as its length, so a cached matrix can never
    # be served to a different database that happens to be the same number of entries in.
    version: str

    @property
    def persons(self) -> int:
        return len(self.person_index.persons)


@dataclass
class _GalleryCache:
    lock: threading.Lock = field(default_factory=threading.Lock)
    entries: dict[str, Gallery] = field(default_factory=dict)


_cache = _GalleryCache()


def clear_gallery_cache() -> None:
    """Drop cached gallery matrices. For tests and for a model switch in one process."""
    with _cache.lock:
        _cache.entries.clear()


def gallery_for(conn: sqlite3.Connection, *, embedder_model_id: str) -> Gallery:
    """The active gallery for one embedder, cached until the audit chain head moves."""
    _, version = audit.head(conn)
    with _cache.lock:
        cached = _cache.entries.get(embedder_model_id)
        if cached is not None and cached.version == version:
            return cached

    loaded = _load_gallery(conn, embedder_model_id=embedder_model_id, version=version)
    with _cache.lock:
        _cache.entries[embedder_model_id] = loaded
    return loaded


def _load_gallery(
    conn: sqlite3.Connection, *, embedder_model_id: str, version: str
) -> Gallery:
    dim_row = conn.execute(
        "SELECT dim FROM models WHERE id = ? AND kind = 'embedder'", (embedder_model_id,)
    ).fetchone()
    if dim_row is None or dim_row["dim"] is None:
        raise ValueError(f"active embedder {embedder_model_id!r} has no registered dimension")
    dim = int(dim_row["dim"])

    # Same filter as the stored path (matching.rematch): active templates of enrolled,
    # enrollable persons, for this embedder only (invariant 2).
    rows = conn.execute(
        "SELECT t.id, t.person_id, t.embedding, p.display_name FROM templates t "
        "JOIN persons p ON p.id = t.person_id "
        "WHERE t.status = 'active' AND p.status = 'enrolled' "
        "AND p.do_not_enroll = 0 AND t.embedder_model_id = ? "
        "ORDER BY t.person_id, t.id",
        (embedder_model_id,),
    ).fetchall()
    person_ids = [str(row["person_id"]) for row in rows]
    return Gallery(
        matrix=vectors.stack_blobs([bytes(row["embedding"]) for row in rows], dim),
        person_ids=person_ids,
        template_ids=[str(row["id"]) for row in rows],
        names={str(row["person_id"]): str(row["display_name"]) for row in rows},
        person_index=scoring.PersonIndex.build(person_ids),
        version=version,
    )


def match_frame(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    *,
    frame: bytes | np.ndarray,
    identify: bool = True,
) -> LiveResult:
    """Score one frame against the gallery without writing anything.

    `frame` is either encoded image bytes (what the endpoint receives) or an already
    decoded RGB array. Raises `decode.ImageDecodeError` for undecodable bytes.

    `identify=False` answers "where are the faces" and nothing else: the quality gate still
    runs, so a rejected face is still reported as rejected, but no crop is warped, nothing
    is embedded and no gallery row is scored. It exists because a box tracking a moving
    face is worth more to the operator than a name arriving a third of a second late, and
    the embedder is most of that third of a second. `embed` and `match` are reported as
    0.0 because no embedding and no scoring happened; the gallery and gate are still read
    (both cached) so `gallery_persons` and `auto_accept_allowed` mean what they always mean.
    """
    started = time.perf_counter()

    decode_start = started
    image = decode.decode_bytes(frame) if isinstance(frame, bytes) else frame
    decode_ms = _since(decode_start)

    detect_start = time.perf_counter()
    detections = models.detector.detect(image)
    detect_ms = _since(detect_start)

    prepare_start = time.perf_counter()
    reports = [quality.evaluate(image, detection, settings) for detection in detections]
    crops = (
        [
            align.align_crop(image, detection.landmarks)
            for detection, report in zip(detections, reports, strict=True)
            if report.passed
        ]
        if identify
        else []
    )
    quality_align_ms = _since(prepare_start)

    embed_start = time.perf_counter()
    # One call with the whole stack: the adapter owns the per-graph batching, exactly as
    # the stored path leaves it (spec 6.3).
    embeddings = (
        models.embedder.embed(np.stack(crops))
        if crops
        else np.zeros((0, models.embedder.dim), dtype=np.float32)
    )
    embed_ms = _since(embed_start) if identify else 0.0

    match_start = time.perf_counter()
    gallery = gallery_for(conn, embedder_model_id=models.embedder_model_id)
    threshold_set = matching.active_threshold_set(conn)
    gate = (
        None
        if threshold_set is None
        else acceptance.build_gate(
            threshold_set,
            embedder_model_id=models.embedder_model_id,
            execution_provider=models.execution_provider,
            live_gallery_size=gallery.persons,
        )
    )

    ranked: list[tuple[list[PersonCandidate], Band]] = []
    if identify and len(crops) > 0 and gate is not None and gallery.matrix.shape[0] > 0:
        scores = scoring.cosine_scores(embeddings, gallery.matrix)
        for row in scores:
            scored = scoring.rank_persons(
                row,
                gallery.person_ids,
                gallery.template_ids,
                mode=settings.person_score_mode,
                top_k=settings.top_k,
                index=gallery.person_index,
            )
            band, _, _ = scoring.band_for_candidates(scored, gate.thresholds)
            ranked.append((scored, band))
    match_ms = _since(match_start) if identify else 0.0

    faces: list[LiveFace] = []
    passed_idx = 0
    for detection, report in zip(detections, reports, strict=True):
        live_candidates: list[LiveCandidate] = []
        if report.passed:
            if passed_idx < len(ranked):
                scored, band = ranked[passed_idx]
                live_candidates = [
                    LiveCandidate(
                        person_id=candidate.person_id,
                        name=gallery.names.get(candidate.person_id, ""),
                        rank=candidate.rank,
                        score=candidate.score,
                        band=band,
                        best_template_id=candidate.best_template_id,
                    )
                    for candidate in scored
                ]
            passed_idx += 1
        faces.append(
            LiveFace(
                x=detection.x,
                y=detection.y,
                w=detection.w,
                h=detection.h,
                det_score=detection.score,
                quality_passed=report.passed,
                quality_reasons=list(report.reasons),
                candidates=live_candidates,
            )
        )

    height, width = int(image.shape[0]), int(image.shape[1])
    return LiveResult(
        width=width,
        height=height,
        faces=faces,
        identified=identify,
        threshold_set_id=None if gate is None else gate.threshold_set_id,
        auto_accept_allowed=False if gate is None else gate.allowed,
        auto_accept_reason=(
            "no active threshold set" if gate is None else gate.reason
        ),
        gallery_persons=gallery.persons,
        elapsed_ms=round(_since(started)),
        timings=LiveTimings(
            decode=decode_ms,
            detect=detect_ms,
            quality_align=quality_align_ms,
            embed=embed_ms,
            match=match_ms,
        ),
    )


def _since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


