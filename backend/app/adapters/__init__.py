"""Model adapters. Raw ONNX via onnxruntime only; never the insightface package."""

from __future__ import annotations

import numpy as np

from app.core.types import CROP_SIZE


def validate_crops(crops: np.ndarray) -> np.ndarray:
    """Shared aligned-crop contract for every embedder (spec 6.3).

    Accepts (112, 112, 3) or (N, 112, 112, 3) uint8 RGB and always returns the batched
    form. Shape and dtype are checked here rather than left to onnxruntime, because a
    float or BGR batch produces a valid-looking embedding with a silently wrong score.
    """
    batch = np.asarray(crops)
    if batch.ndim == 3:
        batch = batch[None]
    if batch.ndim != 4 or batch.shape[1:] != (CROP_SIZE, CROP_SIZE, 3):
        raise ValueError(
            f"expected (N, {CROP_SIZE}, {CROP_SIZE}, 3) crops, got shape {batch.shape!r}"
        )
    if batch.dtype != np.uint8:
        raise ValueError(f"expected uint8 crops, got dtype {batch.dtype!r}")
    return batch
