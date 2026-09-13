"""Detector geometry shared by every fixed-input ONNX detector adapter (spec 6.1, 6.3).

Two operations, and every such adapter needs both: an aspect-preserving resize with a
centred pad onto the square canvas the graph demands, together with the inverse map that
takes decoded network-pixel coordinates back to original-image pixels; and greedy IoU
non-maximum suppression over the concatenated per-stride decode. They live here rather
than in one adapter so a second detector does not import from the first.

No cv2: the only image op needed is a resize, which Pillow does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Letterbox fill. Black is neutral for a detector trained without a border prior.
PAD_VALUE = 0


@dataclass(frozen=True, slots=True)
class Letterbox:
    """The aspect-preserving resize + centred pad applied before inference."""

    scale: float
    pad_x: float
    pad_y: float
    scaled_width: int
    scaled_height: int

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Map (..., 2) letterboxed pixel coordinates back to original-image pixels."""
        offset = np.array([self.pad_x, self.pad_y], dtype=np.float32)
        return ((np.asarray(points, dtype=np.float32) - offset) / self.scale).astype(
            np.float32, copy=False
        )


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize `image` to fit `size` x `size` preserving aspect ratio, then centre-pad."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"image must be non-empty, got shape {image.shape!r}")

    scale = min(size / width, size / height)
    scaled_width = max(1, min(size, round(width * scale)))
    scaled_height = max(1, min(size, round(height * scale)))
    pad_x = (size - scaled_width) / 2.0
    pad_y = (size - scaled_height) / 2.0

    if (scaled_width, scaled_height) == (width, height):
        resized = image
    else:
        resized = np.asarray(
            Image.fromarray(image).resize(
                (scaled_width, scaled_height), resample=Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )

    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    top = int(pad_y)
    left = int(pad_x)
    canvas[top : top + scaled_height, left : left + scaled_width] = resized
    return canvas, Letterbox(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        scaled_width=scaled_width,
        scaled_height=scaled_height,
    )


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy IoU non-maximum suppression. `boxes` is (N, 4) as (x1, y1, x2, y2)."""
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0].astype(np.float64)
    y1 = boxes[:, 1].astype(np.float64)
    x2 = boxes[:, 2].astype(np.float64)
    y2 = boxes[:, 3].astype(np.float64)
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores.astype(np.float64))[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        inter_w = np.maximum(
            0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest])
        )
        inter_h = np.maximum(
            0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])
        )
        intersection = inter_w * inter_h
        union = areas[best] + areas[rest] - intersection
        iou = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0.0
        )
        order = rest[iou <= iou_threshold]
    return keep
