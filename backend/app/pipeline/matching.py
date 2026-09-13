"""Gallery matching and re-matching over stored track means (spec 6.4-6.5)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app import audit
from app.config import Settings
from app.core import acceptance, scoring, vectors
from app.core.types import Band, Thresholds
from app.db.conn import transaction
from app.ids import new_id


@dataclass(frozen=True, slots=True)
class RematchResult:
    tracks: int
    gallery_templates: int
    gallery_persons: int
    matches: int
    auto_accepted: int
    gate_allowed: bool
    gate_reason: str | None
    gate_warning: str | None

    def as_progress(self) -> dict[str, object]:
        return {
            "tracks": self.tracks,
            "gallery_templates": self.gallery_templates,
            "gallery_persons": self.gallery_persons,
            "matches": self.matches,
            "auto_accepted": self.auto_accepted,
            "auto_accept_allowed": self.gate_allowed,
            "auto_accept_reason": self.gate_reason,
            "warning": self.gate_warning,
        }


@dataclass(frozen=True, slots=True)
class _MatchWrite:
    match_id: str
    track_id: str
    person_id: str
    rank: int
    score: float
    band: Band
    best_template_id: str


# One statement for the whole track scan. A rematch scoped to a handful of tracks (one per
# enrolment, one per revoke) used to load every track in the database and filter in Python.
_TRACK_SELECT = (
    "SELECT t.id, t.embedding_mean, m.case_id FROM tracks t "
    "JOIN media m ON m.id = t.media_id "
    "WHERE t.embedding_mean IS NOT NULL AND t.embedder_model_id = ?"
)
# Bound parameters per statement. Well under SQLite's limit, and it keeps the plan simple.
_ID_CHUNK = 500


def _track_rows(
    conn: sqlite3.Connection, *, embedder_model_id: str, track_ids: set[str] | None
) -> list[sqlite3.Row]:
    """Stored track means for this embedder, optionally narrowed to `track_ids`.

    Chunked because the id set has no upper bound. Each chunk is a contiguous run of the
    sorted ids ordered by id, so the concatenation is still globally ordered by id and a
    re-match writes its rows in the same sequence every time.
    """
    if track_ids is None:
        return conn.execute(f"{_TRACK_SELECT} ORDER BY t.id", (embedder_model_id,)).fetchall()
    ordered = sorted(track_ids)
    rows: list[sqlite3.Row] = []
    for start in range(0, len(ordered), _ID_CHUNK):
        chunk = ordered[start : start + _ID_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows.extend(
            conn.execute(
                f"{_TRACK_SELECT} AND t.id IN ({placeholders}) ORDER BY t.id",
                (embedder_model_id, *chunk),
            ).fetchall()
        )
    return rows


def rematch(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    embedder_model_id: str,
    execution_provider: str,
    actor: str,
    track_ids: set[str] | None = None,
) -> RematchResult:
    """Score stored means against one in-memory gallery matrix, in probe blocks.

    Original media and crops are never decoded here. Model boundaries are enforced by both
    SQL filters and the schema trigger on `matches`.
    """
    threshold_set = active_threshold_set(conn)
    if threshold_set is None:
        return RematchResult(0, 0, 0, 0, 0, False, "no active threshold set", None)

    dim_row = conn.execute(
        "SELECT dim FROM models WHERE id = ? AND kind = 'embedder'", (embedder_model_id,)
    ).fetchone()
    if dim_row is None or dim_row["dim"] is None:
        raise ValueError(f"active embedder {embedder_model_id!r} has no registered dimension")
    dim = int(dim_row["dim"])

    gallery_rows = conn.execute(
        "SELECT t.id, t.person_id, t.embedding FROM templates t "
        "JOIN persons p ON p.id = t.person_id "
        "WHERE t.status = 'active' AND p.status = 'enrolled' "
        "AND p.do_not_enroll = 0 AND t.embedder_model_id = ? "
        "ORDER BY t.person_id, t.id",
        (embedder_model_id,),
    ).fetchall()
    person_ids = [str(row["person_id"]) for row in gallery_rows]
    template_ids = [str(row["id"]) for row in gallery_rows]
    gallery = vectors.stack_blobs([bytes(row["embedding"]) for row in gallery_rows], dim)
    person_index = scoring.PersonIndex.build(person_ids)
    gallery_persons = len(person_index.persons)

    track_rows = _track_rows(
        conn, embedder_model_id=embedder_model_id, track_ids=track_ids
    )

    gate = acceptance.build_gate(
        threshold_set,
        embedder_model_id=embedder_model_id,
        execution_provider=execution_provider,
        live_gallery_size=gallery_persons,
    )
    writes: list[_MatchWrite] = []
    top_match_by_track: dict[str, _MatchWrite] = {}
    if gallery.shape[0] > 0:
        for start in range(0, len(track_rows), settings.rematch_block_size):
            block_rows = track_rows[start : start + settings.rematch_block_size]
            probes = vectors.stack_blobs(
                [bytes(row["embedding_mean"]) for row in block_rows], dim
            )
            block_scores = scoring.cosine_scores(probes, gallery)
            for row, scores_for_track in zip(block_rows, block_scores, strict=True):
                candidates = scoring.rank_persons(
                    scores_for_track,
                    person_ids,
                    template_ids,
                    mode=settings.person_score_mode,
                    top_k=settings.top_k,
                    index=person_index,
                )
                band, _, _ = scoring.band_for_candidates(candidates, gate.thresholds)
                track_id = str(row["id"])
                for candidate in candidates:
                    write = _MatchWrite(
                        match_id=new_id(),
                        track_id=track_id,
                        person_id=candidate.person_id,
                        rank=candidate.rank,
                        score=candidate.score,
                        band=band,
                        best_template_id=candidate.best_template_id,
                    )
                    writes.append(write)
                    if candidate.rank == 1:
                        top_match_by_track[track_id] = write

    auto_accepted = 0
    with transaction(conn):
        for row in track_rows:
            track_id = str(row["id"])
            current = conn.execute(
                "SELECT source FROM identities WHERE track_id = ?", (track_id,)
            ).fetchone()
            if current is not None and str(current["source"]) == "operator":
                conn.execute(
                    "UPDATE identities SET match_id = NULL, threshold_set_id = NULL "
                    "WHERE track_id = ?",
                    (track_id,),
                )
            else:
                conn.execute("DELETE FROM identities WHERE track_id = ?", (track_id,))
            conn.execute("DELETE FROM matches WHERE track_id = ?", (track_id,))
        for item in writes:
            conn.execute(
                "INSERT INTO matches (id, track_id, person_id, rank, score, band, "
                "best_template_id, threshold_set_id, embedder_model_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.match_id,
                    item.track_id,
                    item.person_id,
                    item.rank,
                    item.score,
                    item.band,
                    item.best_template_id,
                    threshold_set.id,
                    embedder_model_id,
                    audit.now_ts(),
                ),
            )

        for row in track_rows:
            track_id = str(row["id"])
            operator = conn.execute(
                "SELECT source FROM identities WHERE track_id = ?", (track_id,)
            ).fetchone()
            if operator is not None and str(operator["source"]) == "operator":
                continue
            latest = conn.execute(
                "SELECT decision FROM identifications WHERE track_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (track_id,),
            ).fetchone()
            top = top_match_by_track.get(track_id)
            rejected = latest is not None and str(latest["decision"]) == "reject"
            if top is not None and gate.accepts(top.band) and not rejected:
                now = audit.now_ts()
                conn.execute(
                    "INSERT INTO identities (track_id, person_id, source, match_id, "
                    "threshold_set_id, updated_at) VALUES (?, ?, 'auto', ?, ?, ?) "
                    "ON CONFLICT(track_id) DO UPDATE SET person_id = excluded.person_id, "
                    "source = excluded.source, match_id = excluded.match_id, "
                    "threshold_set_id = excluded.threshold_set_id, "
                    "updated_at = excluded.updated_at",
                    (
                        track_id,
                        top.person_id,
                        top.match_id,
                        threshold_set.id,
                        now,
                    ),
                )
                auto_accepted += 1
                audit.append(
                    conn,
                    actor=actor,
                    case_id=str(row["case_id"]),
                    action="identity.auto_accept",
                    object_type="track",
                    object_id=track_id,
                    payload={
                        "person_id": top.person_id,
                        "match_id": top.match_id,
                        "threshold_set_id": threshold_set.id,
                        "embedder_model_id": embedder_model_id,
                        "score": top.score,
                    },
                )
            elif operator is None or str(operator["source"]) == "auto":
                conn.execute("DELETE FROM identities WHERE track_id = ?", (track_id,))

        audit.append(
            conn,
            actor=actor,
            action="matching.rematch",
            object_type="model",
            object_id=embedder_model_id,
            payload={
                "tracks": len(track_rows),
                "gallery_templates": len(gallery_rows),
                "gallery_persons": gallery_persons,
                "matches": len(writes),
                "auto_accepted": auto_accepted,
                "threshold_set_id": threshold_set.id,
                "gate_allowed": gate.allowed,
                "gate_reason": gate.reason,
                "gate_warning": gate.warning,
            },
        )

    return RematchResult(
        tracks=len(track_rows),
        gallery_templates=len(gallery_rows),
        gallery_persons=gallery_persons,
        matches=len(writes),
        auto_accepted=auto_accepted,
        gate_allowed=gate.allowed,
        gate_reason=gate.reason,
        gate_warning=gate.warning,
    )


def active_threshold_set(conn: sqlite3.Connection) -> acceptance.ThresholdSet | None:
    """The active threshold set as the gate needs it, or None when none is active.

    Public because the live match path must read bands from the very same row the stored
    path does; two loaders would be two band definitions waiting to drift.
    """
    row = conn.execute(
        "SELECT id, model_id, t_strong, t_possible, margin, calibrated, gallery_size, "
        "execution_provider FROM threshold_sets WHERE active = 1"
    ).fetchone()
    if row is None:
        return None
    return acceptance.ThresholdSet(
        id=str(row["id"]),
        model_id=str(row["model_id"]),
        thresholds=Thresholds(
            t_strong=float(row["t_strong"]),
            t_possible=float(row["t_possible"]),
            margin=float(row["margin"]),
        ),
        calibrated=bool(row["calibrated"]),
        gallery_size=None if row["gallery_size"] is None else int(row["gallery_size"]),
        execution_provider=(
            None if row["execution_provider"] is None else str(row["execution_provider"])
        ),
    )


def gallery_person_count(conn: sqlite3.Connection, *, embedder_model_id: str) -> int:
    """How many persons are actually in the gallery for one embedder.

    The same filter `rematch` and `live.gallery_for` load their matrices with: an active
    template, an enrolled person, not `do_not_enroll`, and this model only (invariant 2).
    Public because spec 10's 2x warn and 5x block rule compares it against the calibrated
    gallery size, and a second definition of "in the gallery" would move that line.
    """
    row = conn.execute(
        "SELECT COUNT(DISTINCT t.person_id) AS count FROM templates t "
        "JOIN persons p ON p.id = t.person_id "
        "WHERE t.status = 'active' AND p.status = 'enrolled' AND p.do_not_enroll = 0 "
        "AND t.embedder_model_id = ?",
        (embedder_model_id,),
    ).fetchone()
    return 0 if row is None else int(row["count"])
