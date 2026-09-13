"""The helper's arithmetic: pixel budgets, coordinate conversion, box matching.

Pure functions, no Qt and no platform imports, so this is unit-testable without a display.
It is the one place the coordinate conversion lives, because getting it wrong is the classic
Retina bug: capture happens in physical pixels, the overlay is placed in logical points, and
the pixel cap adds a third scale between them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from watch.config import IDENTIFY_EVERY


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def capture_size(rect: Rect, scale: float, max_pixels: int) -> tuple[int, int]:
    """Physical pixel size to grab for `rect`, shrunk to fit `max_pixels` by area.

    `scale` is the display's device pixel ratio: a 800x600 window on a 2x display is
    1600x1200 real pixels, and that is what the encoder would otherwise be handed.
    """
    if rect.w <= 0 or rect.h <= 0 or scale <= 0:
        raise ValueError("capture_size needs a positive rectangle and scale")
    width = rect.w * scale
    height = rect.h * scale
    area = width * height
    if area > max_pixels:
        # Floored, not rounded: `max_pixels` is a budget, and rounding both axes up can
        # spend a few hundred pixels more than the caller allowed.
        shrink = math.sqrt(max_pixels / area)
        return max(1, math.floor(width * shrink)), max(1, math.floor(height * shrink))
    return max(1, round(width)), max(1, round(height))


def frame_to_overlay(box: Rect, frame_w: int, frame_h: int, rect: Rect) -> Rect:
    """A box in captured-frame pixels, as overlay-local logical points.

    One ratio per axis absorbs the device pixel ratio and the pixel cap together: whatever
    chain of scales produced a frame of `frame_w` x `frame_h` for a target of `rect`, the
    ratio between the two is the only thing the overlay needs to know.

    The result is overlay-local — the overlay is positioned at the target's origin, so
    `rect.x` and `rect.y` do not enter.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame_to_overlay needs a non-empty frame")
    sx = rect.w / frame_w
    sy = rect.h / frame_h
    return Rect(box.x * sx, box.y * sy, box.w * sx, box.h * sy)


def chip_rect(
    box: Rect, size: tuple[float, float], bounds: tuple[float, float], gap: float
) -> Rect:
    """Where a label chip of `size` (w, h) sits for `box`, kept inside `bounds` (w, h).

    Above the box by `gap` when there is room, below it otherwise, and clamped on both axes
    so a face at the window edge does not push its own label off the overlay.
    """
    x = min(max(box.x, 0.0), max(bounds[0] - size[0], 0.0))
    y = box.y - gap - size[1]
    if y < 0.0:
        y = box.y + box.h + gap
    y = min(max(y, 0.0), max(bounds[1] - size[1], 0.0))
    return Rect(x, y, size[0], size[1])


def normalized(box: Rect, frame_w: int, frame_h: int) -> Rect:
    """A box as fractions of its frame — the form the `#/media/{id}/box/...` hash takes."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("normalized needs a non-empty frame")
    return Rect(box.x / frame_w, box.y / frame_h, box.w / frame_w, box.h / frame_h)


def best_overlap(rects: Sequence[Rect], target: Rect, min_iou: float) -> int | None:
    """Index of the rect best overlapping `target` by IoU, or None below `min_iou`.

    The same rule and the same floor `frontend/src/lib/geometry.ts::bestOverlapRect` uses to
    carry labels from the last identify pass onto boxes-only ticks. IoU rather than centre
    distance: a detector run twice over the same frame shifts a box by a few pixels, but a
    *different* face never reaches a meaningful overlap, so a miss stays a miss instead of
    snapping to a neighbour.

    Both sides must be in the same coordinate space; the helper normalises to fractions of
    the frame, because the two cadences produce frames of different sizes.
    """
    best: int | None = None
    best_iou = min_iou
    target_area = target.w * target.h
    for index, rect in enumerate(rects):
        overlap_w = min(rect.x + rect.w, target.x + target.w) - max(rect.x, target.x)
        overlap_h = min(rect.y + rect.h, target.y + target.h) - max(rect.y, target.y)
        if overlap_w <= 0 or overlap_h <= 0:
            continue
        intersection = overlap_w * overlap_h
        union = rect.w * rect.h + target_area - intersection
        if union <= 0:
            continue
        iou = intersection / union
        if iou > best_iou:
            best_iou = iou
            best = index
    return best


def identifies(tick: int) -> bool:
    """True on every `IDENTIFY_EVERY`-th tick, starting with tick 0."""
    return tick % IDENTIFY_EVERY == 0
