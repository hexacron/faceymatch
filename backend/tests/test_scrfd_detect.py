"""SCRFD's decode and its landmark order, pinned against the detector already trusted.

Both are things that can be wrong while looking right. A transposed anchor grid, a wrong
stride multiplier or an `(x1,y1,x2,y2)` / `(x,y,w,h)` confusion all produce plausible boxes
in the wrong places, and a permuted landmark order produces a plausibly-warped crop; the
quality gate and the gallery catch neither. So the boxes are compared with YuNet's by IoU
and the landmark order is compared through the aligned crop's embedding.

Needs the real weights, which are gitignored; skips cleanly where they are absent (CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.adapters.scrfd import ScrfdDetector
from app.adapters.sface import SFaceEmbedder
from app.adapters.yunet import YuNetDetector
from app.config import Settings
from app.core.registry import CPU_PROVIDER, build_session
from app.core.types import Detection
from app.pipeline.align import align_crop
from app.pipeline.decode import decode_image

SCRFD_FILE = "det_10g.onnx"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

SCORE_THRESHOLD = 0.6
NMS_IOU = 0.3


def _model(name: str) -> Path | None:
    path = Settings().models_dir / name
    return path if path.is_file() else None


def _fixture_image() -> np.ndarray | None:
    """First fixture face, recursively. Fixtures are gitignored, so this may be empty."""
    fixtures = Settings().fixtures_dir
    if not fixtures.is_dir():
        return None
    for path in sorted(fixtures.rglob("*.jpg")):
        return decode_image(path)
    return None


def _detectors() -> tuple[ScrfdDetector, YuNetDetector] | None:
    scrfd_path = _model(SCRFD_FILE)
    yunet_path = _model(YUNET_FILE)
    if scrfd_path is None or yunet_path is None:
        return None
    chain = [CPU_PROVIDER]
    scrfd = ScrfdDetector(
        build_session(scrfd_path, chain),
        model_id="buffalo_l-det_10g",
        score_threshold=SCORE_THRESHOLD,
        nms_iou=NMS_IOU,
    )
    yunet = YuNetDetector(
        build_session(yunet_path, chain),
        model_id="yunet-2023mar",
        score_threshold=SCORE_THRESHOLD,
        nms_iou=NMS_IOU,
    )
    return scrfd, yunet


def _best(detections: list[Detection]) -> Detection:
    return max(detections, key=lambda detection: detection.score)


def _iou(left: Detection, right: Detection) -> float:
    ax0, ay0, ax1, ay1 = left.x, left.y, left.x + left.w, left.y + left.h
    bx0, by0, bx1, by1 = right.x, right.y, right.x + right.w, right.y + right.h
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    union = left.w * left.h + right.w * right.h - intersection
    return intersection / union if union > 0.0 else 0.0


def test_scrfd_and_yunet_find_the_same_face() -> None:
    pair = _detectors()
    image = _fixture_image()
    if pair is None:
        pytest.skip(f"models/{SCRFD_FILE} or models/{YUNET_FILE} is not provisioned")
    if image is None:
        pytest.skip("no *.jpg under FIXTURES_DIR")
    scrfd, yunet = pair

    scrfd_faces = scrfd.detect(image)
    yunet_faces = yunet.detect(image)

    assert scrfd_faces, "SCRFD found no face in a fixture YuNet detects"
    assert yunet_faces
    assert _iou(_best(scrfd_faces), _best(yunet_faces)) >= 0.5


def test_scrfd_landmarks_align_like_yunet_landmarks() -> None:
    """The landmark-order test: any permutation warps the crop differently."""
    pair = _detectors()
    sface_path = _model(SFACE_FILE)
    image = _fixture_image()
    if pair is None or sface_path is None:
        pytest.skip("detector or embedder weights are not provisioned")
    if image is None:
        pytest.skip("no *.jpg under FIXTURES_DIR")
    scrfd, yunet = pair

    crops = np.stack(
        [
            align_crop(image, _best(scrfd.detect(image)).landmarks),
            align_crop(image, _best(yunet.detect(image)).landmarks),
        ]
    )
    embedder = SFaceEmbedder(build_session(sface_path, [CPU_PROVIDER]), model_id="sface-2021dec")
    vectors = embedder.embed(crops)

    assert float(vectors[0] @ vectors[1]) >= 0.9


def test_a_frame_with_no_face_returns_no_detections() -> None:
    """The empty-decode path, before NMS is ever reached."""
    scrfd_path = _model(SCRFD_FILE)
    if scrfd_path is None:
        pytest.skip(f"models/{SCRFD_FILE} is not provisioned")
    detector = ScrfdDetector(
        build_session(scrfd_path, [CPU_PROVIDER]),
        model_id="buffalo_l-det_10g",
        score_threshold=SCORE_THRESHOLD,
        nms_iou=NMS_IOU,
    )

    blank = np.full((480, 640, 3), 128, dtype=np.uint8)

    assert detector.detect(blank) == []
