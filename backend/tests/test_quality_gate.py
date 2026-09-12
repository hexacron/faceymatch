"""The quality gate must measure focus, not resolution.

A screen capture of a photo on a high-DPI display is an interpolated upscale: the same face
arrives two or three times larger, with no extra detail. A gate that measures Laplacian
variance on native pixels rejects exactly those frames while passing genuinely out-of-focus
ones, which makes enrolment from a screen capture a coin toss. These tests pin the property
that fixes it: the verdict follows the face, not the rendering scale.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.config import Settings
from app.core.types import ARCFACE_TEMPLATE, Detection
from app.pipeline import quality

SCALE = 2.5


def _textured(size: int = 200, period: int = 8) -> np.ndarray:
    """A hard-edged pattern well below Nyquist, so resampling preserves its energy."""
    grid = np.indices((size, size))
    pattern = ((grid[0] // period + grid[1] // period) % 2 * 255).astype(np.uint8)
    return np.dstack([pattern] * 3)


def _upscaled(image: np.ndarray, scale: float = SCALE) -> np.ndarray:
    """What a capture of this content on a high-DPI display looks like: interpolated."""
    source = Image.fromarray(image)
    grown = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.BICUBIC,
    )
    return np.asarray(grown.convert("RGB"), dtype=np.uint8)


def _blurred(image: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)).convert("RGB"),
        dtype=np.uint8,
    )


def _detection(x: float, y: float, side: float) -> Detection:
    """A box with landmarks placed by scaling the ArcFace template into it."""
    landmarks = ARCFACE_TEMPLATE * (side / 112.0) + np.array([x, y], dtype=np.float32)
    return Detection(x=x, y=y, w=side, h=side, score=0.95, landmarks=landmarks)


def test_the_gate_gives_the_same_verdict_at_both_rendering_scales(
    settings: Settings,
) -> None:
    native = _textured()
    upscaled = _upscaled(native)

    at_1x = quality.evaluate(native, _detection(20.0, 20.0, 120.0), settings)
    at_2_5x = quality.evaluate(
        upscaled, _detection(20.0 * SCALE, 20.0 * SCALE, 120.0 * SCALE), settings
    )

    assert at_1x.passed is True
    assert at_2_5x.passed is True
    # Same content, so the score must be of the same order. The native-pixel measurement
    # this replaced scored the 2.5x copy at ~4% of the 1x copy, which is the bug.
    ratio = at_2_5x.sharpness / at_1x.sharpness
    assert 0.4 < ratio < 2.5, f"sharpness moved with scale: ratio {ratio:.3f}"


def test_an_out_of_focus_face_is_still_rejected_at_both_scales(settings: Settings) -> None:
    blurred = _blurred(_textured(), radius=6.0)
    upscaled = _upscaled(blurred)

    at_1x = quality.evaluate(blurred, _detection(20.0, 20.0, 120.0), settings)
    at_2_5x = quality.evaluate(
        upscaled, _detection(20.0 * SCALE, 20.0 * SCALE, 120.0 * SCALE), settings
    )

    assert at_1x.passed is False
    assert at_2_5x.passed is False
    assert quality.REASON_SHARPNESS in at_1x.reasons
    assert quality.REASON_SHARPNESS in at_2_5x.reasons


def test_sharp_and_blurred_are_separated_by_more_than_the_scale_effect(
    settings: Settings,
) -> None:
    """Focus must dominate the measurement; rendering scale must not."""
    sharp = _textured()
    focus_gap = quality.normalized_sharpness(
        sharp, _detection(20.0, 20.0, 120.0)
    ) / max(
        quality.normalized_sharpness(_blurred(sharp, 6.0), _detection(20.0, 20.0, 120.0)),
        1e-9,
    )
    scale_gap = quality.normalized_sharpness(
        sharp, _detection(20.0, 20.0, 120.0)
    ) / max(
        quality.normalized_sharpness(
            _upscaled(sharp), _detection(20.0 * SCALE, 20.0 * SCALE, 120.0 * SCALE)
        ),
        1e-9,
    )

    assert focus_gap > 100.0
    assert scale_gap < 2.5


def test_a_box_too_small_to_resample_scores_zero_rather_than_raising(
    settings: Settings,
) -> None:
    report = quality.evaluate(_textured(), _detection(10.0, 10.0, 1.0), settings)

    assert report.sharpness == pytest.approx(0.0)
    assert quality.REASON_WIDTH in report.reasons
    assert quality.REASON_SHARPNESS in report.reasons


def test_every_failed_criterion_is_reported(settings: Settings) -> None:
    """The audit trail has to say a face was small AND blurred, not just small."""
    tiny_and_blurred = _blurred(_textured(), radius=6.0)
    detection = Detection(
        x=20.0,
        y=20.0,
        w=30.0,
        h=30.0,
        score=0.1,
        landmarks=ARCFACE_TEMPLATE * 0.1 + np.array([40.0, 40.0], dtype=np.float32),
    )

    report = quality.evaluate(tiny_and_blurred, detection, settings)

    assert report.passed is False
    assert set(report.reasons) >= {
        quality.REASON_WIDTH,
        quality.REASON_SHARPNESS,
        quality.REASON_DET_SCORE,
    }
