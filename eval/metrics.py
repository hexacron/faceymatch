"""Calibration metrics for spec section 10.

Pure functions over score arrays. No I/O, no DB, no model loading: everything here
takes numbers and returns numbers, so the numbers that authorise auto-accept can be
tested against synthetic distributions with known answers.

Vocabulary (ISO/IEC 19795-1 terms, biometric convention):

Verification (1:1), over *pairs* of samples:
  - genuine pair   two samples of the same identity.
  - impostor pair  two samples of different identities.
  - FMR(t)  false match rate  = fraction of impostor pairs scoring >= t.
  - FNMR(t) false non-match rate = fraction of genuine pairs scoring < t.

Open-set identification (1:N), over *probes* against a gallery of enrolled persons:
  - mated probe     its identity IS enrolled in the gallery.
  - non-mated probe its identity is NOT enrolled (the open-set impostor).
  - FPIR(t) false positive identification rate = fraction of non-mated probes whose
            top-1 person score is >= t, i.e. the system names someone who was never
            enrolled. This is the rate the spec's `FPIR_TARGET` bounds.
  - FNIR(t) false negative identification rate = fraction of mated probes that are
            missed, either because the top-1 score is < t (rejected) or because the
            rank-1 person is the wrong person (misidentified). FNIR = 1 - DIR.
  - DIR(t)  detection and identification rate at rank 1 = 1 - FNIR(t). Also the
            "review recall" the spec asks `t_possible` to hold.

Threshold convention throughout: a score is *accepted* at threshold t when
`score >= t`. Every threshold-selection function returns a value that satisfies its
target on the evaluated data, choosing the least restrictive such value, so a
reported rate is an achieved rate on the eval set and never an interpolation.

Scores are cosine similarities of L2-normalized embeddings, so they live in
[-1, 1]; `SCORE_FLOOR` is the "accept everything" threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Cosine similarity of L2-normalized vectors cannot go below -1, so this threshold
# accepts every score. Returned when the requested error rate permits everything.
SCORE_FLOOR = -1.0

DEFAULT_REVIEW_RECALL_TARGET = 0.95
# A margin is never allowed to cost more than this fraction of genuine top matches.
DEFAULT_MARGIN_GENUINE_QUANTILE = 0.05
SUMMARY_PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
DEFAULT_HISTOGRAM_BINS = 20


def as_scores(values: ArrayLike, *, name: str = "scores") -> FloatArray:
    """Coerce to a finite 1-D float64 score array, or raise."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _require_nonempty(array: FloatArray, name: str) -> FloatArray:
    if array.size == 0:
        raise ValueError(f"{name} is empty; there is nothing to calibrate on")
    return array


def genuine_impostor_scores(
    embeddings: ArrayLike, labels: list[str]
) -> tuple[FloatArray, FloatArray]:
    """All-pairs verification scores, split into genuine and impostor.

    `embeddings` is (N, dim) and assumed L2-normalized (cosine == dot product);
    `labels[i]` is the identity of row i. Returns (genuine, impostor) over the
    N*(N-1)/2 unordered pairs, so no pair is counted twice and no sample is paired
    with itself.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {matrix.shape}")
    if matrix.shape[0] != len(labels):
        raise ValueError(
            f"got {matrix.shape[0]} embeddings but {len(labels)} labels"
        )
    if matrix.shape[0] < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    scores = matrix @ matrix.T
    index = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    codes = np.fromiter(
        (index[label] for label in labels), dtype=np.int64, count=len(labels)
    )
    same = codes[:, None] == codes[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    genuine = scores[upper & same]
    impostor = scores[upper & ~same]
    return genuine.astype(np.float64), impostor.astype(np.float64)


def rate_at_or_above(scores: ArrayLike, threshold: float) -> float:
    """Fraction of `scores` that are >= `threshold`. 0.0 for an empty array."""
    array = as_scores(scores)
    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(array >= threshold) / array.size)


def threshold_at_rate(scores: ArrayLike, rate_target: float) -> float:
    """Lowest threshold whose accept rate over `scores` is <= `rate_target`.

    This is the one primitive behind every error-rate operating point here: give it
    the error population (impostor pair scores for FMR, non-mated top-1 scores for
    FPIR, non-mated top1-top2 gaps for the margin) and it returns the least
    restrictive threshold that keeps that population's accept rate within target.

    Method, with `n` scores sorted descending as `s[0] >= s[1] >= ...`: at most
    `m = floor(rate_target * n)` of them may be accepted, so the threshold must sit
    just above `s[m]`. `np.nextafter` is used rather than `s[m]` itself so ties at
    that score are excluded and the achieved rate is guaranteed <= target rather
    than one tie over it. With `m >= n` every score may be accepted and the floor is
    returned.

    Resolution limit: a target below `1 / n` collapses to `m = 0`, i.e. a threshold
    just above the single worst error case. That is a valid conservative bound but
    it is not a measurement of that rate; `n >= 1 / rate_target` samples are needed
    for the operating point to be resolvable, which is why the eval report records
    the population sizes next to the rates.
    """
    if not 0.0 <= rate_target <= 1.0:
        raise ValueError(f"rate_target must be in [0, 1], got {rate_target}")
    array = _require_nonempty(as_scores(scores), "scores")
    allowed = int(np.floor(rate_target * array.size))
    if allowed >= array.size:
        return SCORE_FLOOR
    descending = np.sort(array)[::-1]
    return float(np.nextafter(descending[allowed], np.inf))


def fmr_at_threshold(impostor: ArrayLike, threshold: float) -> float:
    """False match rate: fraction of impostor pairs accepted at `threshold`."""
    return rate_at_or_above(impostor, threshold)


def fnmr_at_threshold(genuine: ArrayLike, threshold: float) -> float:
    """False non-match rate: fraction of genuine pairs rejected at `threshold`."""
    array = _require_nonempty(as_scores(genuine), "genuine")
    return float(np.count_nonzero(array < threshold) / array.size)


def fnmr_at_fmr(
    genuine: ArrayLike, impostor: ArrayLike, fmr_target: float
) -> tuple[float, float]:
    """Verification operating point: (threshold, FNMR) at FMR <= `fmr_target`.

    The threshold is the lowest one whose false match rate over the impostor pairs
    is at or below `fmr_target` (see `threshold_at_rate`), and the FNMR is the
    genuine-pair rejection rate achieved *at that same threshold*. This is the
    "FNMR at FMR = 1e-3 / 1e-4" pair the spec asks for.

    Monotone by construction: raising `fmr_target` can only lower (never raise) the
    threshold, and FNMR is non-decreasing in the threshold, so a looser FMR target
    can never produce a worse FNMR.
    """
    genuine_scores = _require_nonempty(as_scores(genuine), "genuine")
    impostor_scores = _require_nonempty(as_scores(impostor), "impostor")
    threshold = threshold_at_rate(impostor_scores, fmr_target)
    return threshold, fnmr_at_threshold(genuine_scores, threshold)


def fpir_at_threshold(nonmated_top1: ArrayLike, threshold: float) -> float:
    """FPIR: fraction of non-mated probes whose top-1 score is accepted."""
    return rate_at_or_above(nonmated_top1, threshold)


def fnir_at_threshold(
    mated_top1: ArrayLike, mated_rank1_correct: ArrayLike, threshold: float
) -> float:
    """FNIR: fraction of mated probes missed at `threshold`.

    A mated probe is missed when its top-1 score is below the threshold *or* its
    rank-1 person is the wrong person. Wrong-person hits therefore count as misses
    at every threshold, which is what separates open-set identification from
    verification.
    """
    scores = _require_nonempty(as_scores(mated_top1), "mated_top1")
    correct = _as_bool(mated_rank1_correct, scores.size)
    hit = correct & (scores >= threshold)
    return float(1.0 - np.count_nonzero(hit) / scores.size)


def dir_at_threshold(
    mated_top1: ArrayLike, mated_rank1_correct: ArrayLike, threshold: float
) -> float:
    """Rank-1 detection and identification rate = 1 - FNIR. The review recall."""
    return 1.0 - fnir_at_threshold(mated_top1, mated_rank1_correct, threshold)


def _as_bool(values: ArrayLike, size: int) -> BoolArray:
    array = np.asarray(values, dtype=bool).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected {size} rank-1 flags, got {array.size}")
    return array


@dataclass(frozen=True, slots=True)
class OpenSetCurve:
    """FPIR vs FNIR sampled at a shared set of thresholds (spec 10 metrics).

    `thresholds[i]`, `fpir[i]` and `fnir[i]` describe one operating point. Points
    are ordered by ascending threshold, so FPIR is non-increasing and FNIR is
    non-decreasing along the arrays.
    """

    thresholds: FloatArray
    fpir: FloatArray
    fnir: FloatArray
    mated_count: int
    nonmated_count: int
    gallery_size: int

    def as_json(self) -> dict[str, object]:
        return {
            "thresholds": [float(t) for t in self.thresholds],
            "fpir": [float(v) for v in self.fpir],
            "fnir": [float(v) for v in self.fnir],
            "mated_probes": self.mated_count,
            "nonmated_probes": self.nonmated_count,
            "gallery_size": self.gallery_size,
        }


def fpir_fnir_curve(
    mated_top1: ArrayLike,
    mated_rank1_correct: ArrayLike,
    nonmated_top1: ArrayLike,
    *,
    gallery_size: int,
    thresholds: ArrayLike | None = None,
) -> OpenSetCurve:
    """Open-set identification curve over a probe set that mixes both populations.

    The probe set must contain mated probes (identity enrolled) and non-mated
    probes (identity never enrolled); the non-mated probes are the only source of
    FPIR, so a curve computed without them would be meaningless. `gallery_size` is
    carried through because open-set false positives grow with gallery size, and a
    threshold set is only valid near the size it was measured at (spec 10).

    With `thresholds=None` the curve is sampled at every distinct observed score
    plus a floor point, which is the finest sampling the data supports.
    """
    mated = _require_nonempty(as_scores(mated_top1), "mated_top1")
    correct = _as_bool(mated_rank1_correct, mated.size)
    nonmated = _require_nonempty(as_scores(nonmated_top1), "nonmated_top1")

    if thresholds is None:
        observed = np.unique(np.concatenate([mated, nonmated]))
        ceiling = np.nextafter(observed[-1], np.inf)
        grid = np.concatenate([[SCORE_FLOOR], observed, [ceiling]])
    else:
        grid = np.sort(as_scores(thresholds, name="thresholds"))

    fpir = np.array([fpir_at_threshold(nonmated, t) for t in grid], dtype=np.float64)
    fnir = np.array(
        [fnir_at_threshold(mated, correct, t) for t in grid], dtype=np.float64
    )
    return OpenSetCurve(
        thresholds=grid.astype(np.float64),
        fpir=fpir,
        fnir=fnir,
        mated_count=int(mated.size),
        nonmated_count=int(nonmated.size),
        gallery_size=gallery_size,
    )


def threshold_for_review_recall(
    mated_top1: ArrayLike, mated_rank1_correct: ArrayLike, recall_target: float
) -> tuple[float, float]:
    """Highest threshold whose rank-1 recall over mated probes is >= target.

    Returns (threshold, achieved recall). `t_possible` wants the *highest* such
    threshold, not the lowest: any lower value would only add non-mated noise to
    the review queue without finding another enrolled person.

    Method: only mated probes whose rank-1 person is correct can ever be recalled,
    so with `m` mated probes we need `k = ceil(recall_target * m)` of their scores
    at or above the threshold. Sorting those correct scores descending, the k-th of
    them is the highest threshold that keeps k hits. When fewer than k probes are
    correct at rank 1 the target is unreachable at any threshold; the floor is
    returned with the achieved recall so the caller can report the shortfall
    instead of pretending the target was met.
    """
    if not 0.0 <= recall_target <= 1.0:
        raise ValueError(f"recall_target must be in [0, 1], got {recall_target}")
    scores = _require_nonempty(as_scores(mated_top1), "mated_top1")
    correct = _as_bool(mated_rank1_correct, scores.size)
    needed = int(np.ceil(recall_target * scores.size))
    correct_scores = np.sort(scores[correct])[::-1]
    if needed == 0:
        # No recall demanded: the most restrictive threshold still satisfies it.
        top = float(correct_scores[0]) if correct_scores.size else SCORE_FLOOR
        threshold = float(np.nextafter(top, np.inf))
    elif correct_scores.size < needed:
        threshold = SCORE_FLOOR
    else:
        threshold = float(correct_scores[needed - 1])
    return threshold, dir_at_threshold(scores, correct, threshold)


def choose_margin(
    genuine_gaps: ArrayLike,
    impostor_gaps: ArrayLike,
    *,
    fpir_target: float,
    genuine_quantile: float = DEFAULT_MARGIN_GENUINE_QUANTILE,
) -> tuple[float, tuple[str, ...]]:
    """Pick the spec 6.4 `margin` from observed top1-top2 separation.

    The margin is the second half of the `strong` rule: a top-1 score above
    `t_strong` is only `strong` when it also beats the runner-up *person* by
    `margin`, otherwise it is `ambiguous`. So the margin must be large enough to
    catch the near-ties that open-set false positives produce, and small enough
    that genuine top matches keep their separation.

    `genuine_gaps` are the top1-top2 gaps of correctly identified mated probes;
    `impostor_gaps` are the top1-top2 gaps of non-mated probes. The chosen value is

        margin = min(threshold_at_rate(impostor_gaps, fpir_target),
                     quantile(genuine_gaps, genuine_quantile))

    clamped at 0.0 (the schema requires margin >= 0). The first term is the
    smallest gap that keeps at most `fpir_target` of non-mated probes looking
    unambiguous; the second is a ceiling that stops the margin rule from demoting
    more than `genuine_quantile` of genuine top matches out of `strong`. Returns
    (margin, notes) where notes record any binding constraint or missing data.
    """
    if not 0.0 <= genuine_quantile <= 1.0:
        raise ValueError(
            f"genuine_quantile must be in [0, 1], got {genuine_quantile}"
        )
    genuine = as_scores(genuine_gaps, name="genuine_gaps")
    impostor = as_scores(impostor_gaps, name="impostor_gaps")
    notes: list[str] = []

    if impostor.size == 0:
        notes.append(
            "no non-mated top1-top2 gaps observed; margin left at 0.0, so the "
            "strong band rests on t_strong alone"
        )
        return 0.0, tuple(notes)

    from_impostors = threshold_at_rate(impostor, fpir_target)
    margin = from_impostors
    if genuine.size == 0:
        notes.append(
            "no genuine top1-top2 gaps observed; margin not capped for recall"
        )
    else:
        ceiling = float(np.quantile(genuine, genuine_quantile))
        if ceiling < margin:
            notes.append(
                f"margin capped at the {genuine_quantile:.0%} genuine-gap quantile "
                f"{ceiling:.4f} (impostor gaps wanted {from_impostors:.4f})"
            )
            margin = ceiling
    if margin < 0.0:
        margin = 0.0
    return float(margin), tuple(notes)


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """The chosen threshold set plus the rates that justify it."""

    t_strong: float
    t_possible: float
    margin: float
    fpir_at_t_strong: float
    fnir_at_t_strong: float
    review_recall_at_t_possible: float
    fpir_target: float
    review_recall_target: float
    gallery_size: int
    mated_probes: int
    nonmated_probes: int
    notes: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "t_strong": self.t_strong,
            "t_possible": self.t_possible,
            "margin": self.margin,
            "fpir_at_t_strong": self.fpir_at_t_strong,
            "fnir_at_t_strong": self.fnir_at_t_strong,
            "review_recall_at_t_possible": self.review_recall_at_t_possible,
            "fpir_target": self.fpir_target,
            "review_recall_target": self.review_recall_target,
            "gallery_size": self.gallery_size,
            "mated_probes": self.mated_probes,
            "nonmated_probes": self.nonmated_probes,
            "notes": list(self.notes),
        }


def choose_thresholds(
    *,
    mated_top1: ArrayLike,
    mated_rank1_correct: ArrayLike,
    nonmated_top1: ArrayLike,
    gallery_size: int,
    fpir_target: float,
    review_recall_target: float = DEFAULT_REVIEW_RECALL_TARGET,
    genuine_gaps: ArrayLike | None = None,
    impostor_gaps: ArrayLike | None = None,
    margin_genuine_quantile: float = DEFAULT_MARGIN_GENUINE_QUANTILE,
) -> ThresholdChoice:
    """The spec section 10 threshold policy, as one auditable decision.

    - `t_strong` is the lowest score whose false positive *identification* rate over
      the non-mated probes is at or below `fpir_target` at the evaluated
      `gallery_size`. It is an identification rate, not a verification FMR: the
      population is "probes of people who were never enrolled", which is the
      population auto-accept actually faces.
    - `t_possible` is the highest score that still recalls `review_recall_target` of
      the mated probes at rank 1, so the review queue keeps that recall while
      staying as small as possible.
    - `margin` comes from the observed top1-top2 separation (`choose_margin`).
    - Finally `t_possible` is clamped to at most `t_strong`. The band rules and the
      `threshold_sets` CHECK both require `t_strong >= t_possible`; when the FPIR
      target forces `t_strong` below the recall point, recall is what gives way,
      and the clamp is recorded in `notes`.

    Every returned rate is measured at the returned threshold on the supplied data.
    """
    mated = _require_nonempty(as_scores(mated_top1), "mated_top1")
    correct = _as_bool(mated_rank1_correct, mated.size)
    nonmated = _require_nonempty(as_scores(nonmated_top1), "nonmated_top1")
    if gallery_size <= 0:
        raise ValueError(f"gallery_size must be positive, got {gallery_size}")

    notes: list[str] = []
    if fpir_target > 0.0 and nonmated.size < 1.0 / fpir_target:
        notes.append(
            f"{nonmated.size} non-mated probes cannot resolve FPIR={fpir_target:g}; "
            f"t_strong is a conservative bound above the worst observed non-mated "
            f"score, not a measurement at that rate"
        )

    t_strong = threshold_at_rate(nonmated, fpir_target)
    t_possible, recall = threshold_for_review_recall(
        mated, correct, review_recall_target
    )
    if recall < review_recall_target:
        notes.append(
            f"review recall target {review_recall_target:.4f} unreachable: only "
            f"{recall:.4f} of mated probes are correct at rank 1 at any threshold"
        )
    if t_possible > t_strong:
        notes.append(
            f"t_possible clamped from {t_possible:.4f} to t_strong {t_strong:.4f}: "
            f"the FPIR target binds before the review recall target"
        )
        t_possible = t_strong

    margin, margin_notes = choose_margin(
        genuine_gaps if genuine_gaps is not None else np.empty(0),
        impostor_gaps if impostor_gaps is not None else np.empty(0),
        fpir_target=fpir_target,
        genuine_quantile=margin_genuine_quantile,
    )
    notes.extend(margin_notes)

    return ThresholdChoice(
        t_strong=t_strong,
        t_possible=t_possible,
        margin=margin,
        fpir_at_t_strong=fpir_at_threshold(nonmated, t_strong),
        fnir_at_t_strong=fnir_at_threshold(mated, correct, t_strong),
        review_recall_at_t_possible=dir_at_threshold(mated, correct, t_possible),
        fpir_target=fpir_target,
        review_recall_target=review_recall_target,
        gallery_size=gallery_size,
        mated_probes=int(mated.size),
        nonmated_probes=int(nonmated.size),
        notes=tuple(notes),
    )



def summarize_scores(
    scores: ArrayLike, *, bins: int = DEFAULT_HISTOGRAM_BINS
) -> dict[str, object]:
    """Summarize a score population for the report: moments, percentiles, histogram.

    The report stores distributions rather than only the chosen thresholds, so a
    reviewer can re-derive any other operating point from the same run without
    re-running the models.
    """
    array = as_scores(scores)
    if array.size == 0:
        return {"count": 0, "percentiles": {}, "histogram": {}}
    counts, edges = np.histogram(array, bins=bins, range=(-1.0, 1.0))
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "percentiles": {
            f"p{percentile:g}": float(np.percentile(array, percentile))
            for percentile in SUMMARY_PERCENTILES
        },
        "histogram": {
            "bin_edges": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
        },
    }
