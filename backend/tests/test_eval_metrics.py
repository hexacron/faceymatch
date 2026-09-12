"""Calibration metric tests (spec 10).

Every case here uses synthetic score distributions with answers that can be derived
by hand, never real embeddings: these functions decide when the system is allowed to
claim an identity by itself, so their definitions have to be pinned to arithmetic a
reviewer can check, not to whatever a model happened to produce.

`eval/` is a script directory, not a package (spec 13 repo layout), so it is put on
sys.path here rather than imported as `eval.metrics`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import metrics  # noqa: E402


def _normal(rng: np.random.Generator, mean: float, sd: float, n: int) -> np.ndarray:
    return np.asarray(rng.normal(mean, sd, n), dtype=np.float64)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260912)


# --- threshold_at_rate: the primitive every operating point is built on -------


def test_threshold_at_rate_admits_exactly_the_allowed_fraction() -> None:
    # 1000 scores 0.000..0.999. A 1% target allows 10 of them above threshold.
    scores = np.arange(1000, dtype=np.float64) / 1000.0
    threshold = metrics.threshold_at_rate(scores, 0.01)

    assert metrics.rate_at_or_above(scores, threshold) == pytest.approx(0.01)
    # The 11th highest score (0.989) must be excluded, the 10th (0.990) included.
    assert threshold > 0.989
    assert threshold <= 0.990


def test_threshold_at_rate_excludes_ties_at_the_boundary() -> None:
    # Half the population sits on one value. A 10% target cannot accept any of it.
    scores = np.concatenate([np.full(50, 0.5), np.linspace(0.6, 0.9, 50)])
    threshold = metrics.threshold_at_rate(scores, 0.10)

    assert metrics.rate_at_or_above(scores, threshold) <= 0.10


def test_threshold_at_rate_returns_floor_when_everything_is_allowed() -> None:
    scores = np.linspace(0.0, 1.0, 10)
    assert metrics.threshold_at_rate(scores, 1.0) == metrics.SCORE_FLOOR
    assert metrics.rate_at_or_above(scores, metrics.SCORE_FLOOR) == 1.0


def test_threshold_at_rate_rejects_impossible_targets() -> None:
    with pytest.raises(ValueError, match="rate_target"):
        metrics.threshold_at_rate(np.array([0.1, 0.2]), 1.5)
    with pytest.raises(ValueError, match="empty"):
        metrics.threshold_at_rate(np.array([]), 0.1)


# --- verification: FNMR at a target FMR --------------------------------------


def test_fnmr_is_near_zero_for_well_separated_distributions(
    rng: np.random.Generator,
) -> None:
    # N(0.7, 0.05) vs N(0.1, 0.05): 12 sigma apart, so an FMR=1e-3 threshold sits
    # far below the genuine mass and almost nothing is falsely rejected.
    genuine = _normal(rng, 0.7, 0.05, 5000)
    impostor = _normal(rng, 0.1, 0.05, 5000)

    threshold, fnmr = metrics.fnmr_at_fmr(genuine, impostor, 1e-3)

    assert 0.2 < threshold < 0.5
    assert fnmr < 0.001
    assert metrics.fmr_at_threshold(impostor, threshold) <= 1e-3


def test_overlapping_distributions_give_a_predictable_nonzero_fnmr() -> None:
    # Deterministic, hand-countable populations.
    # impostor: 0.00..0.99 (100 values). FMR=0.01 allows 1 -> threshold just above
    # the 2nd highest (0.98), i.e. in (0.98, 0.99].
    impostor = np.arange(100, dtype=np.float64) / 100.0
    # genuine: 0.50..1.49 (100 values). Exactly 49 of them are < 0.99.
    genuine = impostor + 0.5

    threshold, fnmr = metrics.fnmr_at_fmr(genuine, impostor, 0.01)

    assert 0.98 < threshold <= 0.99
    assert fnmr == pytest.approx(0.49)
    assert metrics.fmr_at_threshold(impostor, threshold) == pytest.approx(0.01)


def test_stricter_fmr_target_costs_fnmr(rng: np.random.Generator) -> None:
    genuine = _normal(rng, 0.45, 0.12, 4000)
    impostor = _normal(rng, 0.15, 0.10, 4000)

    threshold_3, fnmr_3 = metrics.fnmr_at_fmr(genuine, impostor, 1e-3)
    threshold_2, fnmr_2 = metrics.fnmr_at_fmr(genuine, impostor, 1e-2)

    assert threshold_3 > threshold_2
    assert fnmr_3 > fnmr_2 > 0.0


def test_fnmr_is_monotone_in_the_fmr_target(rng: np.random.Generator) -> None:
    # Raising the FMR target can never raise the FNMR: the threshold can only fall,
    # and FNMR is non-decreasing in the threshold.
    genuine = _normal(rng, 0.5, 0.15, 3000)
    impostor = _normal(rng, 0.2, 0.15, 3000)

    previous_threshold = float("inf")
    previous_fnmr = 1.0
    for target in (1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5):
        threshold, fnmr = metrics.fnmr_at_fmr(genuine, impostor, target)
        assert threshold <= previous_threshold
        assert fnmr <= previous_fnmr + 1e-12
        previous_threshold, previous_fnmr = threshold, fnmr


def test_fnmr_at_threshold_counts_rejected_genuine_pairs() -> None:
    genuine = np.array([0.1, 0.4, 0.6, 0.9])
    assert metrics.fnmr_at_threshold(genuine, 0.5) == pytest.approx(0.5)
    assert metrics.fnmr_at_threshold(genuine, 0.6) == pytest.approx(0.5)
    assert metrics.fnmr_at_threshold(genuine, 0.0) == 0.0


# --- all-pairs split ---------------------------------------------------------


def test_genuine_impostor_split_counts_unordered_pairs() -> None:
    # 3 identities x 2 samples: 3 genuine pairs, 12 impostor pairs, 15 total.
    basis = np.eye(3, dtype=np.float64)
    embeddings = np.repeat(basis, 2, axis=0)
    labels = ["a", "a", "b", "b", "c", "c"]

    genuine, impostor = metrics.genuine_impostor_scores(embeddings, labels)

    assert genuine.size == 3
    assert impostor.size == 12
    # Orthonormal basis vectors: identical within an identity, orthogonal across.
    assert np.allclose(genuine, 1.0)
    assert np.allclose(impostor, 0.0)


def test_genuine_impostor_split_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="labels"):
        metrics.genuine_impostor_scores(np.eye(3), ["a", "b"])


# --- open-set identification -------------------------------------------------


def test_fnir_counts_wrong_rank1_as_a_miss_at_every_threshold() -> None:
    mated = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([True, True, True, False])

    # Even with the floor threshold the wrong-person probe is still a miss.
    assert metrics.fnir_at_threshold(mated, correct, metrics.SCORE_FLOOR) == 0.25
    assert metrics.dir_at_threshold(mated, correct, metrics.SCORE_FLOOR) == 0.75
    # Above every score, nothing is detected at all.
    assert metrics.fnir_at_threshold(mated, correct, 0.95) == 1.0


def test_fpir_fnir_curve_is_monotone_and_spans_the_operating_range() -> None:
    mated = np.linspace(0.4, 0.9, 50)
    correct = np.ones(50, dtype=bool)
    nonmated = np.linspace(0.0, 0.5, 50)

    curve = metrics.fpir_fnir_curve(
        mated, correct, nonmated, gallery_size=12
    )

    assert curve.mated_count == 50
    assert curve.nonmated_count == 50
    assert curve.gallery_size == 12
    assert np.all(np.diff(curve.thresholds) >= 0.0)
    # FPIR falls and FNIR rises as the threshold rises.
    assert np.all(np.diff(curve.fpir) <= 1e-12)
    assert np.all(np.diff(curve.fnir) >= -1e-12)
    assert curve.fpir[0] == 1.0
    assert curve.fnir[0] == 0.0
    assert curve.fpir[-1] == 0.0
    assert curve.fnir[-1] == 1.0


def test_fpir_fnir_curve_requires_nonmated_probes() -> None:
    with pytest.raises(ValueError, match="nonmated_top1"):
        metrics.fpir_fnir_curve(
            np.array([0.8]), np.array([True]), np.array([]), gallery_size=4
        )


def test_fpir_fnir_curve_honours_an_explicit_threshold_grid() -> None:
    curve = metrics.fpir_fnir_curve(
        np.array([0.8, 0.6]),
        np.array([True, True]),
        np.array([0.3, 0.7]),
        gallery_size=2,
        thresholds=[0.75, 0.5],
    )

    assert curve.thresholds.tolist() == [0.5, 0.75]
    assert curve.fpir.tolist() == [0.5, 0.0]
    assert curve.fnir.tolist() == [0.0, 0.5]
    assert curve.as_json()["gallery_size"] == 2


# --- review recall -----------------------------------------------------------


def test_review_recall_threshold_is_the_highest_that_holds_the_target() -> None:
    # 10 mated probes, all correct at rank 1, scores 0.10..1.00.
    mated = np.arange(1, 11, dtype=np.float64) / 10.0
    correct = np.ones(10, dtype=bool)

    threshold, recall = metrics.threshold_for_review_recall(mated, correct, 0.8)

    # 8 of 10 needed -> the 8th highest score is 0.3; anything higher drops to 0.7.
    assert threshold == pytest.approx(0.3)
    assert recall == pytest.approx(0.8)
    assert metrics.dir_at_threshold(mated, correct, np.nextafter(0.3, 1.0)) < 0.8


def test_review_recall_reports_shortfall_instead_of_faking_the_target() -> None:
    mated = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, False, False, False])

    threshold, recall = metrics.threshold_for_review_recall(mated, correct, 0.9)

    assert threshold == metrics.SCORE_FLOOR
    assert recall == pytest.approx(0.25)


# --- margin ------------------------------------------------------------------


def test_margin_separates_genuine_and_impostor_top1_top2_gaps() -> None:
    genuine_gaps = np.linspace(0.20, 0.40, 100)
    impostor_gaps = np.linspace(0.00, 0.10, 100)

    margin, notes = metrics.choose_margin(
        genuine_gaps, impostor_gaps, fpir_target=0.01
    )

    # Above nearly every impostor gap, below nearly every genuine gap.
    assert 0.09 < margin <= 0.20
    assert metrics.rate_at_or_above(impostor_gaps, margin) <= 0.01
    assert not notes


def test_margin_is_capped_by_genuine_recall_when_impostors_are_wide() -> None:
    genuine_gaps = np.linspace(0.05, 0.15, 100)
    impostor_gaps = np.linspace(0.0, 0.6, 100)

    margin, notes = metrics.choose_margin(
        genuine_gaps, impostor_gaps, fpir_target=0.01, genuine_quantile=0.05
    )

    # The 5% genuine-gap quantile binds instead of the impostor target.
    assert margin == pytest.approx(float(np.quantile(genuine_gaps, 0.05)))
    assert any("capped" in note for note in notes)


def test_margin_is_zero_without_runner_up_data() -> None:
    margin, notes = metrics.choose_margin([], [], fpir_target=1e-3)
    assert margin == 0.0
    assert any("no non-mated" in note for note in notes)


# --- the policy: choose_thresholds -------------------------------------------


def test_choose_thresholds_meets_the_fpir_target_and_band_ordering() -> None:
    rng = np.random.default_rng(7)
    mated = np.asarray(rng.normal(0.72, 0.05, 400), dtype=np.float64)
    correct = np.ones(400, dtype=bool)
    nonmated = np.asarray(rng.normal(0.20, 0.05, 2000), dtype=np.float64)

    choice = metrics.choose_thresholds(
        mated_top1=mated,
        mated_rank1_correct=correct,
        nonmated_top1=nonmated,
        gallery_size=12,
        fpir_target=1e-2,
        review_recall_target=0.95,
        genuine_gaps=np.linspace(0.25, 0.45, 400),
        impostor_gaps=np.linspace(0.0, 0.08, 2000),
    )

    # Spec 6.4 and the threshold_sets CHECK both require this ordering.
    assert choice.t_strong >= choice.t_possible
    assert choice.margin >= 0.0
    # The FPIR target is the binding promise behind auto-accept.
    assert choice.fpir_at_t_strong <= 1e-2
    assert metrics.fpir_at_threshold(nonmated, choice.t_strong) <= 1e-2
    assert choice.review_recall_at_t_possible >= 0.95
    assert choice.gallery_size == 12
    assert choice.mated_probes == 400
    assert choice.nonmated_probes == 2000


def test_choose_thresholds_keeps_recall_threshold_below_strong() -> None:
    # A high FPIR threshold and low review threshold are compatible: the possible band
    # intentionally spans the interval between them, so no clamp should destroy recall.
    mated = np.linspace(0.30, 0.95, 100)
    correct = np.ones(100, dtype=bool)
    # Non-mated probes score high: the FPIR target forces t_strong up to ~0.9.
    nonmated = np.linspace(0.50, 0.90, 100)

    choice = metrics.choose_thresholds(
        mated_top1=mated,
        mated_rank1_correct=correct,
        nonmated_top1=nonmated,
        gallery_size=8,
        fpir_target=0.01,
        review_recall_target=0.95,
    )

    assert choice.t_strong > choice.t_possible
    assert not any("clamped" in note for note in choice.notes)
    assert choice.review_recall_at_t_possible >= 0.95


def test_choose_thresholds_flags_an_unresolvable_fpir_target() -> None:
    choice = metrics.choose_thresholds(
        mated_top1=np.linspace(0.6, 0.9, 20),
        mated_rank1_correct=np.ones(20, dtype=bool),
        nonmated_top1=np.linspace(0.0, 0.3, 20),
        gallery_size=5,
        fpir_target=1e-3,
        review_recall_target=0.9,
    )

    # 20 non-mated probes cannot measure a 1e-3 rate; the report must say that
    # rather than imply the number was measured.
    assert any("cannot resolve" in note for note in choice.notes)
    assert choice.fpir_at_t_strong == 0.0


def test_choose_thresholds_is_monotone_in_the_fpir_target() -> None:
    rng = np.random.default_rng(11)
    mated = np.asarray(rng.normal(0.6, 0.1, 500), dtype=np.float64)
    correct = np.ones(500, dtype=bool)
    nonmated = np.asarray(rng.normal(0.3, 0.1, 500), dtype=np.float64)

    previous = float("inf")
    for target in (1e-3, 1e-2, 0.05, 0.1):
        choice = metrics.choose_thresholds(
            mated_top1=mated,
            mated_rank1_correct=correct,
            nonmated_top1=nonmated,
            gallery_size=10,
            fpir_target=target,
        )
        # A looser FPIR target can only lower t_strong, never raise it.
        assert choice.t_strong <= previous
        assert choice.fnir_at_t_strong <= 1.0
        previous = choice.t_strong


def test_choose_thresholds_refuses_a_degenerate_gallery() -> None:
    with pytest.raises(ValueError, match="gallery_size"):
        metrics.choose_thresholds(
            mated_top1=np.array([0.8]),
            mated_rank1_correct=np.array([True]),
            nonmated_top1=np.array([0.2]),
            gallery_size=0,
            fpir_target=1e-3,
        )


def test_choose_thresholds_json_round_trips_every_justifying_rate() -> None:
    choice = metrics.choose_thresholds(
        mated_top1=np.array([0.8, 0.7]),
        mated_rank1_correct=np.array([True, True]),
        nonmated_top1=np.array([0.2, 0.1]),
        gallery_size=2,
        fpir_target=0.5,
    )
    payload = choice.as_json()

    assert payload["t_strong"] == choice.t_strong
    assert payload["fpir_at_t_strong"] == choice.fpir_at_t_strong
    assert payload["review_recall_target"] == metrics.DEFAULT_REVIEW_RECALL_TARGET
    assert isinstance(payload["notes"], list)


# --- report helpers ----------------------------------------------------------


def test_summarize_scores_reports_the_distribution_not_just_a_mean() -> None:
    scores = np.linspace(-1.0, 1.0, 201)
    summary = metrics.summarize_scores(scores, bins=4)

    assert summary["count"] == 201
    assert summary["mean"] == pytest.approx(0.0)
    assert summary["min"] == -1.0
    assert summary["max"] == 1.0
    percentiles = summary["percentiles"]
    assert isinstance(percentiles, dict)
    assert percentiles["p50"] == pytest.approx(0.0)
    histogram = summary["histogram"]
    assert isinstance(histogram, dict)
    assert histogram["bin_edges"] == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert sum(histogram["counts"]) == 201


def test_summarize_scores_handles_an_empty_population() -> None:
    assert metrics.summarize_scores(np.array([]))["count"] == 0


def test_as_scores_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        metrics.as_scores([0.1, float("nan")])
