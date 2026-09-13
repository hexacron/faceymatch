"""The vectorised person reduction must produce exactly what the grouped one produced.

A faster reduction that moves a score by one ULP silently invalidates the active calibrated
threshold set: the bands were measured against the old numbers, and nothing in the system
would notice the difference. So the contract is equality, not closeness.
"""

from __future__ import annotations

import numpy as np

from app.core.scoring import PersonIndex, cosine_scores, rank_persons
from app.core.types import PersonCandidate
from app.core.vectors import l2_normalize


def _reference(
    template_scores: np.ndarray,
    person_ids: list[str],
    template_ids: list[str],
    *,
    top_k: int | None,
) -> list[PersonCandidate]:
    """The grouped Python reduction this replaced, kept here as the thing to match."""
    grouped: dict[str, list[tuple[float, str]]] = {}
    for score, person_id, template_id in zip(
        template_scores.tolist(), person_ids, template_ids, strict=True
    ):
        grouped.setdefault(person_id, []).append((float(score), template_id))

    scored: list[tuple[float, str, str]] = []
    for person_id, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_template = pairs[0]
        scored.append((best_score, person_id, best_template))

    scored.sort(key=lambda item: (-item[0], item[1]))
    limited = scored if top_k is None else scored[:top_k]
    return [
        PersonCandidate(person_id=person, score=score, best_template_id=template, rank=i + 1)
        for i, (score, person, template) in enumerate(limited)
    ]


def _gallery(persons: int, per_person: int, dim: int) -> tuple[list[str], list[str], np.ndarray]:
    """Rows ordered as the SQL orders them: by person, then template id."""
    rng = np.random.default_rng(4242)
    person_ids: list[str] = []
    template_ids: list[str] = []
    for person in range(persons):
        for template in range(per_person):
            person_ids.append(f"person-{person:03d}")
            template_ids.append(f"tmpl-{person:03d}-{template:02d}")
    matrix = l2_normalize(rng.standard_normal((len(person_ids), dim)).astype(np.float32), axis=1)
    return person_ids, template_ids, matrix


def test_the_vectorised_reduction_matches_the_grouped_one() -> None:
    person_ids, template_ids, gallery = _gallery(persons=8, per_person=5, dim=16)
    probes = l2_normalize(
        np.random.default_rng(9).standard_normal((6, 16)).astype(np.float32), axis=1
    )
    index = PersonIndex.build(person_ids)

    for row in cosine_scores(probes, gallery):
        for top_k in (None, 3, 1):
            assert rank_persons(
                row, person_ids, template_ids, top_k=top_k, index=index
            ) == _reference(row, person_ids, template_ids, top_k=top_k)


def test_ties_pick_the_same_person_and_the_same_template() -> None:
    """Equal scores are the case where two orderings can silently disagree."""
    person_ids = ["b", "b", "a", "a", "c"]
    template_ids = ["b-1", "b-2", "a-1", "a-2", "c-1"]
    scores = np.array([0.5, 0.5, 0.5, 0.4, 0.1], dtype=np.float32)

    ranked = rank_persons(scores, person_ids, template_ids, top_k=None)

    assert ranked == _reference(scores, person_ids, template_ids, top_k=None)
    # Lexicographic person id breaks the score tie, and the earliest row breaks the
    # template tie inside a person.
    assert [candidate.person_id for candidate in ranked] == ["a", "b", "c"]
    assert ranked[0].best_template_id == "a-1"
    assert ranked[1].best_template_id == "b-1"


def test_an_unordered_gallery_reduces_the_same_way() -> None:
    """The reduction indexes rows by person; it must not assume they arrive grouped."""
    person_ids = ["a", "b", "a", "c", "b"]
    template_ids = ["a-1", "b-1", "a-2", "c-1", "b-2"]
    scores = np.array([0.2, 0.9, 0.7, 0.4, 0.3], dtype=np.float32)

    assert rank_persons(scores, person_ids, template_ids, top_k=None) == _reference(
        scores, person_ids, template_ids, top_k=None
    )


def test_cosine_scores_are_bit_identical_to_the_coercing_form() -> None:
    """Dropping the three astype copies must not move a single score."""
    _, _, gallery = _gallery(persons=40, per_person=3, dim=128)
    probes = l2_normalize(
        np.random.default_rng(5).standard_normal((17, 128)).astype(np.float32), axis=1
    )

    scores = cosine_scores(probes, gallery)
    coerced = (probes.astype(np.float32) @ gallery.astype(np.float32).T).astype(np.float32)

    assert scores.dtype == np.float32
    assert np.array_equal(scores, coerced)
