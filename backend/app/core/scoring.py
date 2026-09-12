"""Similarity, person scoring, and band assignment (spec 6.4).

No thresholds are hard-coded here; they arrive as a `Thresholds` value taken from the active
threshold set. Band assignment is a pure function of (top1, top2, thresholds) so the eval
harness and the pipeline cannot drift apart.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from app.core.types import Band, PersonCandidate, Thresholds

PersonScoreMode = Literal["max", "mean_top3"]
MEAN_TOP3_MIN_TEMPLATES = 5


def cosine_scores(probes: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """(P, dim) x (G, dim) -> (P, G) cosine similarity.

    Both sides are assumed L2-normalized, so cosine is a single matmul. This is the blocked
    path spec 6.2 requires for re-match; there is no per-row vector-store query.
    """
    if probes.ndim != 2 or gallery.ndim != 2:
        raise ValueError(f"expected 2-D inputs, got {probes.shape} and {gallery.shape}")
    if probes.shape[1] != gallery.shape[1]:
        raise ValueError(
            f"dimension mismatch: probes {probes.shape[1]} vs gallery {gallery.shape[1]}. "
            "Never compare embeddings across model_id values (invariant 2)."
        )
    if gallery.shape[0] == 0:
        return np.zeros((probes.shape[0], 0), dtype=np.float32)
    return (probes.astype(np.float32) @ gallery.astype(np.float32).T).astype(np.float32)


def rank_persons(
    template_scores: np.ndarray,
    person_ids: list[str],
    template_ids: list[str],
    *,
    mode: PersonScoreMode = "max",
    top_k: int | None = None,
) -> list[PersonCandidate]:
    """Reduce per-template scores to ranked per-person candidates.

    `template_scores` is one probe's scores over the gallery, aligned with `person_ids` and
    `template_ids`. `best_template_id` is always the template that produced the person's
    single highest score, even in `mean_top3` mode, because that is the row spec 6.4 wants
    recorded as the provenance of the match.
    """
    if not (len(template_scores) == len(person_ids) == len(template_ids)):
        raise ValueError("template_scores, person_ids and template_ids must align")

    grouped: dict[str, list[tuple[float, str]]] = {}
    for score, person_id, template_id in zip(
        template_scores.tolist(), person_ids, template_ids, strict=True
    ):
        grouped.setdefault(person_id, []).append((float(score), template_id))

    scored: list[tuple[float, str, str]] = []
    for person_id, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_template = pairs[0]
        if mode == "mean_top3" and len(pairs) >= MEAN_TOP3_MIN_TEMPLATES:
            top3 = [score for score, _ in pairs[:3]]
            person_score = float(sum(top3) / len(top3))
        else:
            person_score = best_score
        scored.append((person_score, person_id, best_template))

    # Deterministic order: score desc, then person_id so equal scores never reorder between
    # runs (a reviewer re-running a match must get the same top-1).
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
