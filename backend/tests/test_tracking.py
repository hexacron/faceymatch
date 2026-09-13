"""Tracking across sampled frames (spec 6.2 step 3).

What has to hold: a face crossing the frame stays one track, two faces never swap, a track
only becomes real after `min_hits`, a flicker is reported as discarded rather than left as a
one-frame identity, and a face that disappears for longer than `max_age_ms` does not get
resurrected by whoever walks past next.

No model, no database, no video file: this is arithmetic, and its bugs look like someone
else's — a swapped track produces a valid embedding of the wrong face.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.tracking import Observation, Tracker, greedy_match, iou_matrix

STEP_MS = 333  # the 3 fps sample grid the pipeline decodes on


def _tracker(*, min_hits: int = 3, max_age_ms: int = 1000) -> Tracker:
    return Tracker(min_iou=0.3, high_score=0.6, min_hits=min_hits, max_age_ms=max_age_ms)


def _walk(tracker: Tracker, boxes_per_frame: list[list[Observation]]) -> list[list[int]]:
    """Run frames on the sample grid; return the track key behind each frame's boxes."""
    keys: list[list[int]] = []
    for index, boxes in enumerate(boxes_per_frame):
        result = tracker.update(index * STEP_MS, boxes)
        keys.append([item.track_key for item in result.assignments])
    return keys


def test_a_face_crossing_the_frame_stays_one_track() -> None:
    frames = [
        [Observation(x=10.0 + 12 * step, y=40.0, w=60.0, h=60.0, score=0.9)] for step in range(8)
    ]

    keys = _walk(_tracker(), frames)

    assert {key for frame in keys for key in frame} == {0}


def test_two_faces_keep_their_own_tracks() -> None:
    """The failure this rules out is silent: swapped tracks still embed and still match."""
    frames = [
        [
            Observation(x=10.0 + 8 * step, y=40.0, w=60.0, h=60.0, score=0.9),
            Observation(x=300.0 - 8 * step, y=40.0, w=60.0, h=60.0, score=0.9),
        ]
        for step in range(6)
    ]

    keys = _walk(_tracker(), frames)

    assert all(frame == keys[0] for frame in keys)
    assert len(set(keys[0])) == 2


def test_a_track_is_tentative_until_min_hits() -> None:
    tracker = _tracker(min_hits=3)
    frames = [[Observation(x=10.0, y=10.0, w=50.0, h=50.0, score=0.9)] for _ in range(3)]

    states = [
        tracker.update(index * STEP_MS, boxes).assignments[0].state
        for index, boxes in enumerate(frames)
    ]

    assert states == ["tentative", "tentative", "confirmed"]
    assert tracker.confirmed_keys() == {0}


def test_a_one_frame_flicker_is_discarded_rather_than_left_as_a_track() -> None:
    tracker = _tracker(min_hits=3, max_age_ms=1000)
    tracker.update(0, [Observation(x=10.0, y=10.0, w=50.0, h=50.0, score=0.9)])

    discarded = [tracker.update(step * STEP_MS, []).discarded for step in range(1, 6)]

    assert [key for step in discarded for key in step] == [0]
    assert tracker.confirmed_keys() == set()


def test_a_confirmed_track_survives_a_gap_shorter_than_max_age() -> None:
    tracker = _tracker(min_hits=2, max_age_ms=1000)
    seen = Observation(x=10.0, y=10.0, w=50.0, h=50.0, score=0.9)
    tracker.update(0, [seen])
    tracker.update(STEP_MS, [seen])
    tracker.update(2 * STEP_MS, [])  # occluded for one sample
    tracker.update(3 * STEP_MS, [])

    result = tracker.update(4 * STEP_MS, [seen])

    assert [item.track_key for item in result.assignments] == [0]
    assert result.assignments[0].state == "confirmed"
    assert result.discarded == []


def test_a_face_gone_longer_than_max_age_does_not_come_back_as_itself() -> None:
    """Otherwise the next person to walk through that spot inherits the first one's track."""
    tracker = _tracker(min_hits=2, max_age_ms=1000)
    seen = Observation(x=10.0, y=10.0, w=50.0, h=50.0, score=0.9)
    tracker.update(0, [seen])
    tracker.update(STEP_MS, [seen])
    for step in range(2, 8):
        tracker.update(step * STEP_MS, [])

    result = tracker.update(8 * STEP_MS, [seen])

    assert result.assignments[0].track_key != 0
    assert result.assignments[0].state == "tentative"


def test_a_weak_box_keeps_a_track_alive_but_never_starts_one() -> None:
    """ByteTrack's second pass: a face that blurs for a frame has not left the room."""
    tracker = _tracker(min_hits=2)
    strong = Observation(x=10.0, y=10.0, w=50.0, h=50.0, score=0.9)
    tracker.update(0, [strong])
    tracker.update(STEP_MS, [strong])

    rescued = tracker.update(2 * STEP_MS, [Observation(x=12.0, y=10.0, w=50.0, h=50.0, score=0.2)])
    elsewhere = tracker.update(
        3 * STEP_MS, [Observation(x=400.0, y=300.0, w=50.0, h=50.0, score=0.2)]
    )

    assert [item.track_key for item in rescued.assignments] == [0]
    assert elsewhere.assignments == [], "a weak box with nothing to attach to started a track"


def test_boxes_that_do_not_overlap_enough_are_not_associated() -> None:
    tracker = _tracker(min_hits=1)
    tracker.update(0, [Observation(x=10.0, y=10.0, w=50.0, h=50.0, score=0.9)])

    result = tracker.update(STEP_MS, [Observation(x=200.0, y=10.0, w=50.0, h=50.0, score=0.9)])

    assert result.assignments[0].track_key == 1


def test_iou_is_zero_for_disjoint_boxes_and_one_for_identical_ones() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 10.0, 10.0]], dtype=np.float32)

    scores = iou_matrix(boxes, boxes)

    assert scores[0, 0] == 1.0
    assert scores[0, 1] == 0.0
    assert iou_matrix(np.zeros((0, 4), dtype=np.float32), boxes).shape == (0, 2)


def test_greedy_matching_takes_the_best_pair_first_and_leaves_the_rest_free() -> None:
    scores = np.array([[0.9, 0.4], [0.8, 0.35]], dtype=np.float32)

    pairs, free_rows, free_cols = greedy_match(scores, min_score=0.5)

    assert pairs == [(0, 0)]
    assert (free_rows, free_cols) == ([1], [1])
