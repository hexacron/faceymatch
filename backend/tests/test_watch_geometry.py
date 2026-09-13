"""The watch helper's arithmetic (spec 6.11).

Importing `watch.geometry` and nothing else is itself an assertion: the module that decides
where a box is drawn must not need Qt, `mss` or Quartz, so it can be reasoned about — and
tested — without a display or a Screen Recording grant.
"""

from __future__ import annotations

import sys

from watch.geometry import (
    Rect,
    best_overlap,
    capture_size,
    chip_rect,
    frame_to_overlay,
    identifies,
    normalized,
)


def test_geometry_pulls_in_no_display_stack() -> None:
    assert not {name for name in sys.modules if name.startswith(("PySide6", "mss", "Quartz"))}


def test_a_retina_capture_is_capped_by_area_and_still_maps_back() -> None:
    """A 2x display would hand the encoder 1.92 MP for an 800x600 window; the cap shrinks it."""
    rect = Rect(100.0, 50.0, 800.0, 600.0)

    width, height = capture_size(rect, scale=2.0, max_pixels=1_000_000)

    assert width < 1600
    assert height < 1200
    assert width * height <= 1_000_000
    # Aspect ratio survives the shrink, so nothing is stretched.
    assert abs(width / height - 800 / 600) < 0.01

    centre = Rect(width / 2 - 10, height / 2 - 10, 20.0, 20.0)
    mapped = frame_to_overlay(centre, width, height, rect)
    assert abs(mapped.x + mapped.w / 2 - rect.w / 2) < 1.0
    assert abs(mapped.y + mapped.h / 2 - rect.h / 2) < 1.0


def test_a_small_capture_is_not_enlarged_to_the_budget() -> None:
    assert capture_size(Rect(0.0, 0.0, 320.0, 240.0), scale=1.0, max_pixels=1_000_000) == (320, 240)


def test_the_frames_far_corner_lands_on_the_targets_far_corner() -> None:
    """The overlay is placed at the target's origin, so its coordinates are target-local."""
    rect = Rect(1000.0, 700.0, 640.0, 480.0)

    corner = frame_to_overlay(Rect(1260.0, 940.0, 20.0, 20.0), 1280, 960, rect)

    assert abs(corner.x + corner.w - rect.w) < 1.0
    assert abs(corner.y + corner.h - rect.h) < 1.0


def test_a_box_normalises_to_fractions_of_its_own_frame() -> None:
    box = normalized(Rect(480.0, 270.0, 960.0, 540.0), 1920, 1080)

    assert (box.x, box.y, box.w, box.h) == (0.25, 0.25, 0.5, 0.5)


def test_a_label_carries_onto_the_box_it_overlaps() -> None:
    """IoU 0.5 is well clear of the 0.3 floor: same face, two cadences, one name."""
    rects = [Rect(0.0, 0.0, 10.0, 10.0), Rect(100.0, 100.0, 10.0, 10.0)]

    assert best_overlap(rects, Rect(100.0, 100.0, 10.0, 15.0), 0.3) == 1


def test_touching_boxes_do_not_share_a_label() -> None:
    """A name that snapped to the neighbouring face would be worse than no name at all."""
    rects = [Rect(0.0, 0.0, 10.0, 10.0)]

    assert best_overlap(rects, Rect(10.0, 0.0, 10.0, 10.0), 0.3) is None
    assert best_overlap([], Rect(0.0, 0.0, 10.0, 10.0), 0.3) is None


def test_every_third_tick_asks_for_names() -> None:
    assert [identifies(tick) for tick in range(7)] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_a_label_sits_above_its_box_when_there_is_room() -> None:
    chip = chip_rect(Rect(100.0, 200.0, 80.0, 80.0), (120.0, 18.0), (1000.0, 800.0), 4.0)

    assert (chip.x, chip.y, chip.w, chip.h) == (100.0, 178.0, 120.0, 18.0)


def test_a_face_at_the_top_edge_gets_its_label_below_instead() -> None:
    """Otherwise the label for the face nearest the title bar is the one that is never read."""
    box = Rect(100.0, 2.0, 80.0, 80.0)

    chip = chip_rect(box, (120.0, 18.0), (1000.0, 800.0), 4.0)

    assert chip.y == box.y + box.h + 4.0


def test_a_face_at_the_right_edge_keeps_its_whole_label_on_the_overlay() -> None:
    chip = chip_rect(Rect(960.0, 200.0, 80.0, 80.0), (120.0, 18.0), (1000.0, 800.0), 4.0)

    assert chip.x + chip.w <= 1000.0
