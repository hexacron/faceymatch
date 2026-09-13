"""Similarity, person scoring, and band assignment (spec 6.4).

No thresholds are hard-coded here; they arrive as a `Thresholds` value taken from the active
threshold set. Band assignment is a pure function of (top1, top2, thresholds) so the eval
harness and the pipeline cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.types import Band, PersonCandidate, Thresholds

PersonScoreMode = Literal["max", "mean_top3"]
MEAN_TOP3_MIN_TEMPLATES = 5


@dataclass(frozen=True, slots=True)
class PersonIndex:
    """Which gallery row belongs to which person, built once per gallery.

    The reduction from template scores to person scores runs once per probe, and every
    probe reduces over the same rows. Deriving the grouping from the id strings each time
    made a 5000-template gallery cost a dictionary build and a 5000-element sort per face;
    here it is one integer array the numpy reduction indexes directly.
    """

    #: Unique person ids, in first-appearance order. The reduction's output is aligned to it.
    persons: tuple[str, ...]
    #: (templates,) int64 — the position in `persons` of each gallery row's owner.
    row_person: np.ndarray
    #: (persons,) int64 — each person's rank in lexicographic id order, the documented
    #: tie-break for equal scores.
    sort_rank: np.ndarray

    @classmethod
    def build(cls, person_ids: Sequence[str]) -> PersonIndex:
        persons: list[str] = []
        position: dict[str, int] = {}
        row_person = np.empty(len(person_ids), dtype=np.int64)
        for row, person_id in enumerate(person_ids):
            index = position.get(person_id)
            if index is None:
                index = len(persons)
                position[person_id] = index
                persons.append(person_id)
            row_person[row] = index
        ranks = {person_id: rank for rank, person_id in enumerate(sorted(persons))}
        return cls(
            persons=tuple(persons),
            row_person=row_person,
            sort_rank=np.array([ranks[person_id] for person_id in persons], dtype=np.int64),
        )


def cosine_scores(probes: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """(P, dim) x (G, dim) -> (P, G) cosine similarity.

    Both sides are assumed L2-normalized, so cosine is a single matmul. This is the blocked
    path spec 6.2 requires for re-match; there is no per-row vector-store query.

    float32 is required rather than coerced. Every producer of an embedding in this system
    already emits float32 (`vectors.stack_blobs`, both embedder adapters), so a coercion
    here only bought three full-size copies of the largest arrays in the process — a 4096
    probe block against a 5000-template gallery copies 80 MB per call. Anything else
    arriving here is a bug worth hearing about, not worth silently casting.
    """
    if probes.ndim != 2 or gallery.ndim != 2:
        raise ValueError(f"expected 2-D inputs, got {probes.shape} and {gallery.shape}")
    if probes.shape[1] != gallery.shape[1]:
        raise ValueError(
            f"dimension mismatch: probes {probes.shape[1]} vs gallery {gallery.shape[1]}. "
            "Never compare embeddings across model_id values (invariant 2)."
        )
    if probes.dtype != np.float32 or gallery.dtype != np.float32:
        raise ValueError(
            f"expected float32 embeddings, got probes {probes.dtype} and "
            f"gallery {gallery.dtype}"
        )
    if gallery.shape[0] == 0:
        return np.zeros((probes.shape[0], 0), dtype=np.float32)
    scores: np.ndarray = probes @ gallery.T
    return scores


def rank_persons(
    template_scores: np.ndarray,
    person_ids: list[str],
    template_ids: list[str],
    *,
    mode: PersonScoreMode = "max",
    top_k: int | None = None,
    index: PersonIndex | None = None,
) -> list[PersonCandidate]:
    """Reduce per-template scores to ranked per-person candidates.

    `template_scores` is one probe's scores over the gallery, aligned with `person_ids` and
    `template_ids`. `best_template_id` is always the template that produced the person's
    single highest score, even in `mean_top3` mode, because that is the row spec 6.4 wants
    recorded as the provenance of the match.

    `index` is the same gallery's `PersonIndex`, built once by a caller that scores many
    probes against one gallery. It is derived from `person_ids` and changes nothing about
    the result; omitting it just rebuilds it here.
    """
    if not (len(template_scores) == len(person_ids) == len(template_ids)):
        raise ValueError("template_scores, person_ids and template_ids must align")
    if index is not None and len(index.row_person) != len(person_ids):
        raise ValueError(
            f"index covers {len(index.row_person)} rows but the gallery has {len(person_ids)}"
        )

    if mode == "max":
        resolved = PersonIndex.build(person_ids) if index is None else index
        return _rank_by_max(template_scores, template_ids, resolved, top_k)
    return _rank_by_mean_top3(template_scores, person_ids, template_ids, top_k)


def _rank_by_max(
    template_scores: np.ndarray,
    template_ids: list[str],
    index: PersonIndex,
    top_k: int | None,
) -> list[PersonCandidate]:
    """Segment maximum over the gallery rows, entirely in numpy."""
    scores = np.asarray(template_scores, dtype=np.float32)
    count = len(index.persons)
    best = np.full(count, -np.inf, dtype=np.float32)
    np.maximum.at(best, index.row_person, scores)

    # The template that produced each person's best score, lowest row index on a tie —
    # which is the row the grouped path's stable sort picks, so provenance is unchanged.
    hits = np.flatnonzero(scores == best[index.row_person])
    best_row = np.full(count, scores.size, dtype=np.int64)
    np.minimum.at(best_row, index.row_person[hits], hits)

    # Score desc, then person_id, so equal scores never reorder between runs (a reviewer
    # re-running a match must get the same top-1). lexsort's last key is the primary one.
    order = np.lexsort((index.sort_rank, -best))
    limited = order if top_k is None else order[:top_k]
    return [
        PersonCandidate(
            person_id=index.persons[person],
            score=float(best[person]),
            best_template_id=template_ids[int(best_row[person])],
            rank=rank + 1,
        )
        for rank, person in enumerate(limited.tolist())
    ]


def _rank_by_mean_top3(
    template_scores: np.ndarray,
    person_ids: list[str],
    template_ids: list[str],
    top_k: int | None,
) -> list[PersonCandidate]:
    """Spec 6.4's mean-of-top-3, off the hot path: a gallery-wide rewrite buys nothing here."""
    grouped: dict[str, list[tuple[float, str]]] = {}
    for score, person_id, template_id in zip(
        template_scores.tolist(), person_ids, template_ids, strict=True
    ):
        grouped.setdefault(person_id, []).append((float(score), template_id))

    scored: list[tuple[float, str, str]] = []
    for person_id, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_template = pairs[0]
        if len(pairs) >= MEAN_TOP3_MIN_TEMPLATES:
            top3 = [score for score, _ in pairs[:3]]
            person_score = float(sum(top3) / len(top3))
        else:
            person_score = best_score
        scored.append((person_score, person_id, best_template))

    scored.sort(key=lambda item: (-item[0], item[1]))
    limited = scored if top_k is None else scored[:top_k]
    return [
        PersonCandidate(person_id=person_id, score=score, best_template_id=template_id, rank=i + 1)
        for i, (score, person_id, template_id) in enumerate(limited)
    ]


def assign_band(top1: float, top2: float, thresholds: Thresholds) -> Band:
    """Spec 6.4 band rules, evaluated in the fixed order; first rule that holds wins.

    `top2` is the second best *person* score, or 0.0 with fewer than two persons in the
    gallery. Note the ordering: a score above `t_strong` with an insufficient margin is
    `ambiguous`, not `strong` and not `possible`.
    """
    margin = top1 - top2
    if top1 >= thresholds.t_strong and margin >= thresholds.margin:
        return "strong"
    if top1 >= thresholds.t_possible and margin < thresholds.margin:
        return "ambiguous"
    if top1 >= thresholds.t_possible:
        return "possible"
    return "unknown"


def band_for_candidates(
    candidates: list[PersonCandidate], thresholds: Thresholds
) -> tuple[Band, float, float]:
    """Convenience wrapper: returns (band, top1, top2) for a ranked candidate list."""
    if not candidates:
        return "unknown", 0.0, 0.0
    top1 = candidates[0].score
    top2 = candidates[1].score if len(candidates) > 1 else 0.0
    return assign_band(top1, top2, thresholds), top1, top2
