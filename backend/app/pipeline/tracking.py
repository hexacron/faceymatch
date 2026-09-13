"""Face tracking across sampled video frames (spec 6.2 step 3).

Kalman filter, IoU association, a high-score pass then a low-score pass, and the four states
the spec names: tentative until `min_hits`, confirmed, lost while unseen, deleted past
`max_age_ms`.

Pure numpy and no I/O, so the association rules are testable without a model, a database or
a video file — which matters because this is the one stage whose bugs look like someone
else's: a track that swaps identity mid-shot produces a perfectly valid embedding of the
wrong face.

Three choices worth stating:

**Two passes, high score then low.** The second pass is ByteTrack's: a face that blurs or
half-occludes drops below the detector's confidence bar for a frame or two without leaving
the frame. Associating those weak boxes to *existing* tracks keeps the track alive; they are
never allowed to start one, because a weak box with nothing to attach to is what a false
positive looks like.

**Greedy matching by descending IoU, not Hungarian.** A frame holds a handful of faces, and
at that size the optimal assignment and the greedy one differ only when two boxes overlap
each other more than they overlap themselves — which the IoU floor rejects anyway. The
alternative costs a scipy dependency for a 3x3 matrix.

**The filter steps once per sampled frame, not once per millisecond.** Sampling is a fixed
grid (`video.iter_samples`), so one step is one grid interval and velocity is measured in
pixels per sample. Ageing, by contrast, is in milliseconds: "lost for a second" has to mean
a second whatever the sample rate is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

TrackState = Literal["tentative", "confirmed", "lost", "deleted"]


@dataclass(frozen=True, slots=True)
class Observation:
    """One detector box in a frame, in image pixels."""

    x: float
    y: float
    w: float
    h: float
    score: float


@dataclass(frozen=True, slots=True)
class Assignment:
    """Which track a frame's detection belongs to, and whether that track is real yet."""

    detection_index: int
    track_key: int
    state: TrackState


@dataclass(frozen=True, slots=True)
class StepResult:
    assignments: list[Assignment]
    #: Tracks deleted this step that never reached `confirmed`. Their detections were a
    #: flicker, not a face crossing the frame, and the caller unwinds them.
    discarded: list[int]


def iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """IoU of every (track, detection) pair. Both are (N, 4) arrays of x, y, w, h."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    t = tracks[:, None, :]
    d = detections[None, :, :]
    left = np.maximum(t[..., 0], d[..., 0])
    top = np.maximum(t[..., 1], d[..., 1])
    right = np.minimum(t[..., 0] + t[..., 2], d[..., 0] + d[..., 2])
    bottom = np.minimum(t[..., 1] + t[..., 3], d[..., 1] + d[..., 3])
    overlap = np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)
    union = t[..., 2] * t[..., 3] + d[..., 2] * d[..., 3] - overlap
    return np.where(union > 0.0, overlap / union, 0.0).astype(np.float32)


def greedy_match(
    scores: np.ndarray, *, min_score: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair rows to columns, best score first. Returns (pairs, free rows, free columns)."""
    rows, cols = scores.shape
    taken_rows: set[int] = set()
    taken_cols: set[int] = set()
    pairs: list[tuple[int, int]] = []
    if rows and cols:
        # argsort on the flattened matrix: one sort instead of a scan per pair.
        for flat in np.argsort(scores, axis=None)[::-1]:
            row, col = divmod(int(flat), cols)
            if scores[row, col] < min_score:
                break
            if row in taken_rows or col in taken_cols:
                continue
            taken_rows.add(row)
            taken_cols.add(col)
            pairs.append((row, col))
    return (
        pairs,
        [row for row in range(rows) if row not in taken_rows],
        [col for col in range(cols) if col not in taken_cols],
    )


@dataclass
class _Track:
    key: int
    filter: _KalmanBox
    state: TrackState
    hits: int
    last_update_ms: int
    ever_confirmed: bool = False

    def box(self) -> np.ndarray:
        return self.filter.box()


class Tracker:
    """One tracker per media file. `update` is called once per sampled frame, in time order."""

    def __init__(
        self,
        *,
        min_iou: float,
        high_score: float,
        min_hits: int,
        max_age_ms: int,
    ) -> None:
        self._min_iou = min_iou
        self._high_score = high_score
        self._min_hits = min_hits
        self._max_age_ms = max_age_ms
        self._tracks: list[_Track] = []
        self._next_key = 0
        # Keys of tracks that were confirmed and have since been deleted. The row is gone,
        # but "was this ever a real face" is asked once, at the end of the file.
        self._retired: set[int] = set()

    def update(self, t_ms: int, observations: Sequence[Observation]) -> StepResult:
        """Advance every track to `t_ms` and attach this frame's boxes to them."""
        for track in self._tracks:
            track.filter.predict()

        boxes = np.array(
            [[o.x, o.y, o.w, o.h] for o in observations], dtype=np.float32
        ).reshape(-1, 4)
        scores = np.array([o.score for o in observations], dtype=np.float32)
        strong = [index for index in range(len(observations)) if scores[index] >= self._high_score]
        weak = [index for index in range(len(observations)) if scores[index] < self._high_score]

        assignments: list[Assignment] = []
        # Pass one: the boxes the detector is sure about, against every live track.
        live = [track for track in self._tracks if track.state != "deleted"]
        matched, free_tracks, free_strong = self._associate(live, boxes, strong)
        assignments.extend(self._absorb(matched, boxes, t_ms))

        # Pass two: what is left of the tracks, against the boxes the detector doubted.
        # Same IoU floor: a weak box may only rescue a track it already lines up with.
        remaining = [live[index] for index in free_tracks]
        matched_weak, _still_free, _free_weak = self._associate(remaining, boxes, weak)
        assignments.extend(self._absorb(matched_weak, boxes, t_ms))

        # A strong box that matched nothing starts a track. A weak one never does.
        for index in free_strong:
            track = _Track(
                key=self._next_key,
                filter=_KalmanBox(boxes[index]),
                state="tentative",
                hits=1,
                last_update_ms=t_ms,
            )
            self._next_key += 1
            self._tracks.append(track)
            if self._min_hits <= 1:
                track.state = "confirmed"
                track.ever_confirmed = True
            assignments.append(
                Assignment(detection_index=index, track_key=track.key, state=track.state)
            )

        discarded = self._age(t_ms)
        assignments.sort(key=lambda item: item.detection_index)
        return StepResult(assignments=assignments, discarded=discarded)

    def confirmed_keys(self) -> set[int]:
        """Every track that reached `confirmed`, including ones since deleted."""
        return {track.key for track in self._tracks if track.ever_confirmed} | self._retired

    def _associate(
        self, tracks: list[_Track], boxes: np.ndarray, indices: list[int]
    ) -> tuple[list[tuple[_Track, int]], list[int], list[int]]:
        if not tracks or not indices:
            return [], list(range(len(tracks))), list(indices)
        predicted = np.stack([track.box() for track in tracks])
        candidates = boxes[indices]
        pairs, free_rows, free_cols = greedy_match(
            iou_matrix(predicted, candidates), min_score=self._min_iou
        )
        return (
            [(tracks[row], indices[col]) for row, col in pairs],
            free_rows,
            [indices[col] for col in free_cols],
        )

    def _absorb(
        self, matched: list[tuple[_Track, int]], boxes: np.ndarray, t_ms: int
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        for track, index in matched:
            track.filter.correct(boxes[index])
            track.hits += 1
            track.last_update_ms = t_ms
            if track.state != "confirmed" and track.hits >= self._min_hits:
                track.state = "confirmed"
                track.ever_confirmed = True
            elif track.state == "lost":
                # Seen again: a lost track resumes as what it was, never as a new face.
                track.state = "confirmed" if track.ever_confirmed else "tentative"
            assignments.append(
                Assignment(detection_index=index, track_key=track.key, state=track.state)
            )
        return assignments

    def _age(self, t_ms: int) -> list[int]:
        """Mark unseen tracks lost, then delete them past `max_age_ms`."""
        discarded: list[int] = []
        for track in self._tracks:
            if track.state == "deleted" or track.last_update_ms == t_ms:
                continue
            if t_ms - track.last_update_ms > self._max_age_ms:
                track.state = "deleted"
                if track.ever_confirmed:
                    # Remembered after the row is dropped: the caller asks at the end of
                    # the file which tracks were ever real, long after this one left.
                    self._retired.add(track.key)
                else:
                    discarded.append(track.key)
            else:
                track.state = "lost"
        self._tracks = [track for track in self._tracks if track.state != "deleted"]
        return discarded


class _KalmanBox:
    """SORT's constant-velocity filter over (centre x, centre y, area, aspect).

    Area and aspect rather than width and height: a face that turns changes its aspect
    slowly and its area smoothly, while width alone jumps whenever the detector clips an
    edge. The written-out matrices are SORT's, whose tuning is the one part of this that is
    inherited rather than reasoned from first principles.
    """

    _DIM = 7

    def __init__(self, box: np.ndarray) -> None:
        self._x = np.zeros(self._DIM, dtype=np.float64)
        self._x[:4] = _to_measurement(box)
        self._p = np.eye(self._DIM, dtype=np.float64)
        # Velocities are unobserved at birth, so they start with a wide covariance; the
        # ratios are SORT's.
        self._p[4:, 4:] *= 1000.0
        self._p *= 10.0
        self._f = np.eye(self._DIM, dtype=np.float64)
        self._f[0, 4] = self._f[1, 5] = self._f[2, 6] = 1.0
        self._h = np.zeros((4, self._DIM), dtype=np.float64)
        self._h[:4, :4] = np.eye(4, dtype=np.float64)
        self._q = np.eye(self._DIM, dtype=np.float64)
        self._q[4:, 4:] *= 0.01
        self._q[-1, -1] *= 0.01
        self._r = np.eye(4, dtype=np.float64)
        self._r[2:, 2:] *= 10.0

    def predict(self) -> None:
        if self._x[2] + self._x[6] <= 0.0:
            # A shrinking box would pass through zero area; freeze the rate instead of
            # letting the state go negative, which `_to_box` cannot represent.
            self._x[6] = 0.0
        self._x = self._f @ self._x
        self._p = self._f @ self._p @ self._f.T + self._q

    def correct(self, box: np.ndarray) -> None:
        measurement = _to_measurement(box)
        residual = measurement - self._h @ self._x
        s = self._h @ self._p @ self._h.T + self._r
        gain = self._p @ self._h.T @ np.linalg.inv(s)
        self._x = self._x + gain @ residual
        self._p = (np.eye(self._DIM) - gain @ self._h) @ self._p

    def box(self) -> np.ndarray:
        return _to_box(self._x[:4])


def _to_measurement(box: np.ndarray) -> np.ndarray:
    x, y, w, h = (float(value) for value in box)
    w = max(w, 1e-6)
    h = max(h, 1e-6)
    return np.array([x + w / 2.0, y + h / 2.0, w * h, w / h], dtype=np.float64)


def _to_box(state: np.ndarray) -> np.ndarray:
    cx, cy, area, ratio = (float(value) for value in state)
    area = max(area, 1e-6)
    ratio = max(ratio, 1e-6)
    w = float(np.sqrt(area * ratio))
    h = area / w if w > 0.0 else 0.0
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)

