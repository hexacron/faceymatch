"""Quality gate before embedding (spec 6.2 step 4).

Four independent criteria, every threshold from `Settings` (AGENTS.md code rules):

    width    >= settings.min_embed_px     too few pixels to embed
    |yaw|     < settings.max_yaw          off-angle faces embed poorly
    sharpness > settings.min_sharpness    Laplacian variance of the *normalised* crop
    score     > settings.min_det_score    weak detections are often not faces

A failing detection is still recorded (spec 6.2 step 5 only stores crops for passing ones),
so `evaluate` reports *every* failed criterion rather than short-circuiting: the audit trail
should say a face was small AND blurred, not just small.

Sharpness is measured on the detection box resampled to the 112x112 embed size, never on
native pixels. Laplacian variance is a function of sampling density as well as focus, so on
native pixels the same face scores wildly differently depending on how large it happens to
be rendered. That is not academic: a screen capture of a photo on a 2.5x-DPR display is an
interpolated upscale, and measured over 58 fixture faces the native metric passed 100% of
them at 1x but scored the *same faces* upscaled 2.5x at 4% of their native value, dropping
them below any usable floor while a genuinely out-of-focus face at 1x scored higher. Judging
the pixels at the size the embedder will see them makes the gate a focus test again: the
same 58 faces pass at 100% (1x) and 98% (2.5x), while gaussian blur of radius 1.5 and above
is rejected outright at both scales.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from app.config import Settings
from app.core.types import CROP_SIZE, LANDMARK_COUNT, Detection, QualityReport

# Anthropometric constant for the yaw estimate below: nose-tip protrusion ahead of the
# pupil plane divided by interpupillary distance. Mean adult IPD is about 63 mm and mean
# nasal projection about 25 mm, so 0.40. It is a face-geometry constant, not a tunable
# threshold, which is why it lives here and not in Settings.
NOSE_DEPTH_RATIO = 0.40

# 4-neighbour discrete Laplacian. Sharp edges give large responses; a blurred copy of the
# same image gives small ones, so the variance of the response ranks focus.
LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

# ITU-R BT.601 luma weights, for RGB input (spec convention: images are RGB).
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

REASON_WIDTH = "width_below_min_embed_px"
REASON_YAW = "yaw_above_max"
REASON_SHARPNESS = "sharpness_below_min"
REASON_DET_SCORE = "det_score_below_min"


def to_gray(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB (or (H, W) already-gray) -> float64 luma."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64)
    if array.ndim == 3 and array.shape[2] == 3:
        return array.astype(np.float64) @ _LUMA
    raise ValueError(f"expected (H, W) or (H, W, 3), got shape {array.shape!r}")


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 3x3 Laplacian response over the valid (interior) region.

    Returns 0.0 for regions smaller than the kernel: a 2-pixel-wide crop has no focus
    information, and it fails the width criterion anyway.
    """
    array = np.asarray(gray, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {array.shape!r}")
    height, width = array.shape
    if height < 3 or width < 3:
        return 0.0

    # Explicit 3x3 correlation over the interior. The kernel is symmetric, so correlation
    # and convolution agree and no flip is needed.
    windows = np.lib.stride_tricks.sliding_window_view(array, (3, 3))
    response = np.tensordot(windows, LAPLACIAN_KERNEL, axes=((2, 3), (0, 1)))
    return float(response.var())


def yaw_from_landmarks(landmarks: np.ndarray) -> float:
    """Estimate yaw in degrees from horizontal landmark symmetry.

    Model: treat the two eyes and the nose tip as rigid points. Let `u` be the unit vector
    along the eye axis (right eye -> left eye), `span` the interocular distance and
    `offset = (nose - eye_midpoint) . u` the nose displacement along that axis. For a face
    rotated by yaw `theta`, the eye axis foreshortens as `span_0 * cos(theta)` while the
    protruding nose tip shifts by `depth * sin(theta)`, so

        offset / span = (depth / span_0) * tan(theta)
        theta         = atan2(offset, NOSE_DEPTH_RATIO * span)

    Sign: positive when the nose sits towards the subject's left eye (image right of the
    eye midpoint). Callers gate on the magnitude.

    Limits. This is a coarse monocular estimate, adequate for a gate and not for pose
    reporting: it assumes an average nose depth (NOSE_DEPTH_RATIO), so it is biased on
    unusually flat or prominent noses; it cannot separate yaw from a horizontally
    asymmetric face; it is blind to pitch (the projection onto the eye axis cancels
    vertical displacement) and unreliable past roughly 60 degrees, where the far eye is
    occluded and its landmark is a guess. It is invariant to in-plane roll and to scale
    because everything is measured along, and normalised by, the eye axis.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (LANDMARK_COUNT, 2):
        raise ValueError(f"expected ({LANDMARK_COUNT}, 2) landmarks, got {points.shape!r}")

    right_eye, left_eye, nose = points[0], points[1], points[2]
    axis = left_eye - right_eye
    span = float(np.hypot(axis[0], axis[1]))
    if span <= 0.0:
        # Both eyes on one pixel: degenerate landmarks, treat as maximally off-angle so
        # the gate rejects rather than silently passing.
        return 90.0
    unit = axis / span
    offset = float(np.dot(nose - (right_eye + left_eye) / 2.0, unit))
    return math.degrees(math.atan2(offset, NOSE_DEPTH_RATIO * span))


def crop_region(image: np.ndarray, detection: Detection) -> np.ndarray:
    """The detection's box as a view into the image, clipped to the image bounds."""
    height, width = image.shape[:2]
    box = detection.clipped(width, height)
    x0 = math.floor(box.x)
    y0 = math.floor(box.y)
    x1 = math.ceil(box.x + box.w)
    y1 = math.ceil(box.y + box.h)
    return image[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]


def normalized_sharpness(image: np.ndarray, detection: Detection) -> float:
    """Laplacian variance of the detection box resampled to the 112x112 embed size.

    Resampling first is what makes the number comparable between a face rendered at 100 px
    and the same face rendered at 250 px on a high-DPI display. LANCZOS is used rather than
    a plain affine resample because it prefilters: without that, downscaling aliases and the
    measurement picks up the aliasing instead of the focus. Regions too small to resample
    score 0.0, which also fails the width criterion.
    """
    region = crop_region(image, detection)
    if region.shape[0] < 2 or region.shape[1] < 2:
        return 0.0
    resized = Image.fromarray(region).resize(
        (CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS
    )
    return laplacian_variance(to_gray(np.asarray(resized, dtype=np.uint8)))


def evaluate(image: np.ndarray, detection: Detection, settings: Settings) -> QualityReport:
    """Run all four criteria. `reasons` lists every failure, in a fixed order."""
    height, width = image.shape[:2]
    box = detection.clipped(width, height)
    sharpness = normalized_sharpness(image, detection)
    yaw = yaw_from_landmarks(detection.landmarks)

    reasons: list[str] = []
    if box.w < settings.min_embed_px:
        reasons.append(REASON_WIDTH)
    if abs(yaw) >= settings.max_yaw:
        reasons.append(REASON_YAW)
    if sharpness <= settings.min_sharpness:
        reasons.append(REASON_SHARPNESS)
    if detection.score <= settings.min_det_score:
        reasons.append(REASON_DET_SCORE)

    return QualityReport(
        passed=not reasons,
        width_px=box.w,
        yaw_deg=yaw,
        sharpness=sharpness,
        det_score=detection.score,
        reasons=tuple(reasons),
    )
