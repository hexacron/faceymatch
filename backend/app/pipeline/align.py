"""Similarity alignment of detected landmarks onto the ArcFace template (spec 6.2 step 5).

Both embedders were trained on crops produced by the same 5-point similarity warp, so the
alignment is part of the model contract, not a cosmetic resize: a wrong scale or a
transposed matrix costs more accuracy than a model swap gains.

The transform is the Umeyama least-squares similarity (uniform scale + rotation +
translation, no shear) from the detected landmarks to `ARCFACE_TEMPLATE`. Five point pairs
over-determine four parameters, so the fit is a minimisation, not an interpolation.

Pillow's `Image.transform(..., AFFINE, data)` takes the **inverse** map: for each *output*
pixel `(x, y)` it samples the input at `(a*x + b*y + c, d*x + e*y + f)`. Feeding it the
forward image->crop matrix (or its transpose) produces a plausible-looking but wrong crop,
which is why `align_crop` inverts explicitly and `test_align.py` pins the pixel mapping.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.types import ARCFACE_TEMPLATE, CROP_SIZE, LANDMARK_COUNT


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama 1991) mapping `src` onto `dst`.

    `src`, `dst`: (N, 2) point sets. Returns the 2x3 forward matrix `[sR | t]` such that
    `dst ~= src @ (sR).T + t`. Raises when the source points are degenerate (all identical),
    because no scale is recoverable from a single point.
    """
    source = np.asarray(src, dtype=np.float64)
    target = np.asarray(dst, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(
            f"expected matching (N, 2) point sets, got {source.shape!r} and {target.shape!r}"
        )
    count = source.shape[0]
    if count < 2:
        raise ValueError(f"need at least 2 point pairs, got {count}")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centred = source - src_mean
    dst_centred = target - dst_mean

    src_var = float((src_centred**2).sum() / count)
    if src_var <= 0.0:
        raise ValueError("source points are degenerate; no similarity transform exists")

    covariance = (dst_centred.T @ src_centred) / count
    u_mat, singular, vt_mat = np.linalg.svd(covariance)

    # Reflections are not similarities: force det(R) = +1. When the covariance is rank
    # deficient the sign of the smallest singular value is unconstrained, so it is chosen
    # from the factors instead of from det(covariance), which is zero there.
    correction = np.eye(2)
    reflects = np.linalg.det(covariance) < 0
    rank_deficient_reflection = (
        np.isclose(singular[1], 0.0)
        and np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0
    )
    if reflects or rank_deficient_reflection:
        correction[1, 1] = -1.0

    rotation = u_mat @ correction @ vt_mat
    scale = float((singular * np.diag(correction)).sum() / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)

    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (..., 2) points through a 2x3 forward affine matrix."""
    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (2, 3):
        raise ValueError(f"expected a 2x3 affine matrix, got {affine.shape!r}")
    coords = np.asarray(points, dtype=np.float64)
    return (coords @ affine[:, :2].T + affine[:, 2]).astype(np.float32)


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine. Raises when the linear part is singular."""
    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (2, 3):
        raise ValueError(f"expected a 2x3 affine matrix, got {affine.shape!r}")
    linear_inv = np.linalg.inv(affine[:, :2])
    inverse = np.empty((2, 3), dtype=np.float64)
    inverse[:, :2] = linear_inv
    inverse[:, 2] = -linear_inv @ affine[:, 2]
    return inverse


def align_crop(image: np.ndarray, landmarks: np.ndarray, *, size: int = CROP_SIZE) -> np.ndarray:
    """Warp `image` so `landmarks` land on the ArcFace template. Returns (size, size, 3) RGB."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image.shape!r}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got dtype {image.dtype!r}")
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (LANDMARK_COUNT, 2):
        raise ValueError(f"expected ({LANDMARK_COUNT}, 2) landmarks, got {points.shape!r}")

    template = ARCFACE_TEMPLATE.astype(np.float64)
    if size != CROP_SIZE:
        template = template * (size / CROP_SIZE)

    forward = similarity_transform(points, template)
    # Pillow samples the input at inverse(output_pixel), so the crop needs the inverse map.
    inverse = invert_transform(forward)
    data = (
        inverse[0, 0],
        inverse[0, 1],
        inverse[0, 2],
        inverse[1, 0],
        inverse[1, 1],
        inverse[1, 2],
    )
    warped = Image.fromarray(image).transform(
        (size, size),
        Image.Transform.AFFINE,
        data,
        resample=Image.Resampling.BICUBIC,
    )
    return np.asarray(warped, dtype=np.uint8)
