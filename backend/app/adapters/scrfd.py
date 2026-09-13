"""SCRFD-10GF face detector adapter (spec 6.1, 6.3, C7).

Non-commercial weights: loading is gated on the `allow_noncommercial_models` setting
(invariant 9), an audited operator decision that `app.core.registry` enforces through
`app.models_lock.assert_loadable`. Raw ONNX through onnxruntime; the `insightface` package
is never imported, since its loader fetches weights over the network and would break C1
(spec 6.3). The letterbox and the NMS come from `app.adapters.boxes`, shared with YuNet.

Verified graph IO for `models/det_10g.onnx` (onnxruntime 1.30):

    input  'input.1'     [1, 3, ?, ?] float32   (dynamic spatial axes)
    output '448','471','494'  [N, 1]    scores,          strides 8, 16, 32
           '451','474','497'  [N, 4]    box distances,   strides 8, 16, 32
           '454','477','500'  [N, 10]   keypoint offsets (5 landmarks), same order

with N = (640 / stride) ** 2 * 2, i.e. 12800 / 3200 / 800: two anchors per feature-map
location, laid out row-major and tiled consecutively per location. The outputs are named
numerically and so are read by index, never by name; the index layout is all three score
heads, then all three box heads, then all three keypoint heads.

Preprocessing is the SCRFD one, `cv2.dnn.blobFromImage(img, 1.0/128, size, (127.5,)*3,
swapRB=True)`: RGB channel order, `(pixel - 127.5) / 128.0`, NCHW. Note the std is 128 and
not the 127.5 the ArcFace embedders use; getting it wrong shifts every score slightly
instead of failing loudly. The insightface export takes a dynamic input, so an arbitrary
image is letterboxed to `DEFAULT_INPUT_SIZE` (aspect-preserving resize, centred pad) and
every decoded coordinate is mapped back through that transform.

Decoding follows the SCRFD post-process:

* The score head is the confidence as it stands. There is no `sqrt(cls * obj)` geometric
  mean as in YuNet, and no sigmoid: applying either would move every score.
* Boxes are `distance2bbox`: four distances in stride units from the anchor point to the
  left, top, right and bottom edges, so `(x1, y1) = centre - d[:2]` and
  `(x2, y2) = centre + d[2:]`.
* Keypoints are 5 `(x, y)` offsets in stride units from the same anchor point.
* Each feature-map location carries `ANCHORS_PER_CELL` anchors, tiled consecutively per
  point (not as a tile of the whole grid), over a row-major `(x, y)` grid.

Landmark order is the ArcFace one -- `[right eye, left eye, nose, right mouth, left mouth]`,
exactly as `app.core.types.ARCFACE_TEMPLATE` is laid out -- so **no permutation is applied**.
insightface feeds these keypoints straight into its `estimate_norm` against the identical
reference array. Both `app.pipeline.align` and `app.pipeline.quality.yaw_from_landmarks`
depend on that order and neither would fail loudly if it were wrong, so
`backend/tests/test_scrfd_detect.py` pins it against YuNet through the aligned crop.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from app.adapters.boxes import letterbox, nms_indices
from app.core.types import LANDMARK_COUNT, Detection

STRIDES: tuple[int, ...] = (8, 16, 32)
# SCRFD tiles two anchors per feature-map location (the 6- and 9-output exports).
ANCHORS_PER_CELL = 2
# The insightface export takes a dynamic input; 640 is the size SCRFD-10GF is evaluated at.
DEFAULT_INPUT_SIZE = 640
# SCRFD normalisation: input_mean 127.5, input_std 128 (not 127.5, unlike ArcFace).
INPUT_MEAN = 127.5
INPUT_STD = 128.0
# One score, four box distances and five (x, y) keypoint offsets per anchor.
SCORE_WIDTH = 1
BBOX_WIDTH = 4
KPS_WIDTH = LANDMARK_COUNT * 2
# Three strides x (scores, boxes, keypoints).
EXPECTED_OUTPUTS = len(STRIDES) * 3


def _empty_decode() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0, LANDMARK_COUNT, 2), dtype=np.float32),
    )


def _check_head(head: np.ndarray, *, width: int, rows: int, stride: int, name: str) -> None:
    """A mis-decoded anchor grid yields plausible boxes in the wrong places, so assert."""
    if head.shape[-1] != width:
        raise ValueError(
            f"stride {stride}: expected {width} values per anchor in the {name} head, "
            f"got {head.shape[-1]}"
        )
    if head.size != rows * width:
        raise ValueError(
            f"stride {stride}: expected {rows} anchors in the {name} head, "
            f"got {head.size // width}"
        )


def decode_stride(
    scores: np.ndarray,
    bbox_distances: np.ndarray,
    kps_distances: np.ndarray,
    *,
    stride: int,
    size: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one SCRFD stride head into (boxes_xyxy, scores, landmarks) in network pixels."""
    side = size // stride
    rows = side * side * ANCHORS_PER_CELL
    _check_head(scores, width=SCORE_WIDTH, rows=rows, stride=stride, name="score")
    _check_head(bbox_distances, width=BBOX_WIDTH, rows=rows, stride=stride, name="bbox")
    _check_head(kps_distances, width=KPS_WIDTH, rows=rows, stride=stride, name="keypoint")

    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    kept = flat_scores >= score_threshold
    if not kept.any():
        return _empty_decode()

    # Row-major (x, y) grid in network pixels, each point repeated per anchor. The repeat
    # is per point and not a tile of the whole grid: insightface builds the same ordering
    # with `np.stack([centres] * num_anchors, axis=1).reshape(-1, 2)`.
    grid_rows, grid_cols = np.mgrid[:side, :side]
    centres = np.stack([grid_cols, grid_rows], axis=-1).astype(np.float32).reshape(-1, 2)
    centres = np.repeat(centres * stride, ANCHORS_PER_CELL, axis=0)[kept]

    # distance2bbox: distances in stride units to the four edges.
    distances = np.asarray(bbox_distances, dtype=np.float32).reshape(-1, BBOX_WIDTH)[kept]
    distances = distances * stride
    boxes = np.stack(
        [
            centres[:, 0] - distances[:, 0],
            centres[:, 1] - distances[:, 1],
            centres[:, 0] + distances[:, 2],
            centres[:, 1] + distances[:, 3],
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    offsets = np.asarray(kps_distances, dtype=np.float32).reshape(
        -1, LANDMARK_COUNT, 2
    )[kept]
    landmarks = (offsets * stride + centres[:, None, :]).astype(np.float32, copy=False)
    return boxes, flat_scores[kept], landmarks


class ScrfdDetector:
    """`app.core.types.Detector` over the buffalo_l det_10g (SCRFD-10GF) ONNX export."""

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        model_id: str,
        score_threshold: float,
        nms_iou: float,
    ) -> None:
        self.model_id = model_id
        self._session = session
        self._score_threshold = float(score_threshold)
        self._nms_iou = float(nms_iou)

        graph_input = session.get_inputs()[0]
        self._input_name = graph_input.name
        shape = graph_input.shape
        height = shape[2] if len(shape) == 4 else None
        width = shape[3] if len(shape) == 4 else None
        # A fixed square export is honoured; the insightface one has dynamic axes instead.
        if isinstance(height, int) and isinstance(width, int) and height == width:
            self._input_size = height
        else:
            self._input_size = DEFAULT_INPUT_SIZE

        outputs = session.get_outputs()
        if len(outputs) != EXPECTED_OUTPUTS:
            raise ValueError(
                f"{model_id}: expected {EXPECTED_OUTPUTS} SCRFD outputs (scores, bbox, kps "
                f"per stride), got {len(outputs)}"
            )
        # Read by index, not by name: this export names its outputs numerically.
        self._output_names = [output.name for output in outputs]

    def detect(self, image: np.ndarray) -> list[Detection]:
        """image: (H, W, 3) uint8 RGB. Returns detections in original-image pixels."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) image, got shape {image.shape!r}")
        if image.dtype != np.uint8:
            raise ValueError(f"expected a uint8 image, got dtype {image.dtype!r}")

        padded, geometry = letterbox(image, self._input_size)
        # RGB already (spec 6.3 hands RGB uint8) and SCRFD wants RGB, so unlike the YuNet
        # adapter next door there is no channel swap here. Then (v - 127.5) / 128, NCHW.
        blob = np.ascontiguousarray(
            ((padded.astype(np.float32) - INPUT_MEAN) / INPUT_STD).transpose(2, 0, 1)[None]
        )
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        fmc = len(STRIDES)

        boxes_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        landmark_parts: list[np.ndarray] = []
        for index, stride in enumerate(STRIDES):
            boxes, scores, landmarks = decode_stride(
                np.asarray(outputs[index]),
                np.asarray(outputs[index + fmc]),
                np.asarray(outputs[index + fmc * 2]),
                stride=stride,
                size=self._input_size,
                score_threshold=self._score_threshold,
            )
            boxes_parts.append(boxes)
            score_parts.append(scores)
            landmark_parts.append(landmarks)

        boxes = np.concatenate(boxes_parts, axis=0)
        scores = np.concatenate(score_parts, axis=0)
        landmarks = np.concatenate(landmark_parts, axis=0)
        if boxes.shape[0] == 0:
            return []

        height, width = image.shape[:2]
        detections: list[Detection] = []
        for index in nms_indices(boxes, scores, self._nms_iou):
            corners = geometry.to_original(boxes[index].reshape(2, 2))
            x0, y0 = float(corners[0, 0]), float(corners[0, 1])
            x1, y1 = float(corners[1, 0]), float(corners[1, 1])
            detection = Detection(
                x=x0,
                y=y0,
                w=x1 - x0,
                h=y1 - y0,
                score=float(scores[index]),
                landmarks=geometry.to_original(landmarks[index]),
            )
            detections.append(detection.clipped(width, height))
        return detections
