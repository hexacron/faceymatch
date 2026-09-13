"""YuNet face detector adapter (spec 6.1, 6.3).

Raw ONNX through onnxruntime; the `insightface` package is never imported (C1, spec 6.3).
No cv2 either: the letterbox and the NMS live in `app.adapters.boxes`, shared with the
other detector adapters, and the only image op either needs is a resize, which Pillow does.

Verified graph IO for `models/face_detection_yunet_2023mar.onnx` (onnxruntime 1.30):

    input  'input'          [1, 3, 640, 640] float32
    output 'cls_{8,16,32}'  [1, N, 1]
           'obj_{8,16,32}'  [1, N, 1]
           'bbox_{8,16,32}' [1, N, 4]
           'kps_{8,16,32}'  [1, N, 10]   (5 landmarks)

with N = (640 / stride) ** 2, i.e. exactly one anchor per feature-map location, laid out
row-major (`index = row * cols + col`).

Preprocessing is the OpenCV `FaceDetectorYN` one: BGR channel order, raw 0..255 values as
float32, NCHW, no mean subtraction and no scaling. Because the export has a fixed
640x640 input, an arbitrary image is letterboxed (aspect-preserving resize, centred pad)
and every decoded coordinate is mapped back through that transform.

Decoding follows the OpenCV YuNet post-process: the per-anchor score is the geometric mean
`sqrt(cls * obj)` of the two heads, the box is `(cx, cy, w, h)` where the centre is an
offset in stride units from the anchor point and the size is `exp(v) * stride`, and the
keypoints are 5 `(x, y)` offsets in stride units from the same anchor point.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from app.adapters.boxes import letterbox, nms_indices
from app.core.types import LANDMARK_COUNT, Detection

STRIDES: tuple[int, ...] = (8, 16, 32)
DEFAULT_INPUT_SIZE = 640


def _anchor_grid(cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major anchor column/row indices, one anchor per feature-map location."""
    columns = np.tile(np.arange(cols, dtype=np.float32), rows)
    row_indices = np.repeat(np.arange(rows, dtype=np.float32), cols)
    return columns, row_indices


def decode_head(
    cls_head: np.ndarray,
    obj_head: np.ndarray,
    bbox_head: np.ndarray,
    kps_head: np.ndarray,
    *,
    stride: int,
    cols: int,
    rows: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one stride head into (boxes_xyxy, scores, landmarks) in network pixels."""
    cls_scores = np.clip(np.asarray(cls_head, dtype=np.float32).reshape(-1), 0.0, 1.0)
    obj_scores = np.clip(np.asarray(obj_head, dtype=np.float32).reshape(-1), 0.0, 1.0)
    scores = np.sqrt(cls_scores * obj_scores)

    kept = np.flatnonzero(scores >= score_threshold)
    if kept.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, LANDMARK_COUNT, 2), dtype=np.float32),
        )

    columns, row_indices = _anchor_grid(cols, rows)
    columns = columns[kept]
    row_indices = row_indices[kept]
    bbox = np.asarray(bbox_head, dtype=np.float32).reshape(-1, 4)[kept]
    kps = np.asarray(kps_head, dtype=np.float32).reshape(-1, LANDMARK_COUNT * 2)[kept]

    cx = (columns + bbox[:, 0]) * stride
    cy = (row_indices + bbox[:, 1]) * stride
    width = np.exp(bbox[:, 2]) * stride
    height = np.exp(bbox[:, 3]) * stride
    boxes = np.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0], axis=1
    ).astype(np.float32, copy=False)

    landmarks = np.empty((kept.size, LANDMARK_COUNT, 2), dtype=np.float32)
    landmarks[:, :, 0] = (kps[:, 0::2] + columns[:, None]) * stride
    landmarks[:, :, 1] = (kps[:, 1::2] + row_indices[:, None]) * stride
    return boxes, scores[kept].astype(np.float32, copy=False), landmarks


class YuNetDetector:
    """`app.core.types.Detector` over the YuNet 2023mar ONNX export."""

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
        if not isinstance(height, int) or not isinstance(width, int) or height != width:
            # The pinned export is a fixed square 640; anything else is a different file.
            raise ValueError(
                f"{model_id}: expected a fixed square [1,3,S,S] input, got {shape!r}"
            )
        self._input_size = height
        self._output_names = [output.name for output in session.get_outputs()]

    def detect(self, image: np.ndarray) -> list[Detection]:
        """image: (H, W, 3) uint8 RGB. Returns detections in original-image pixels."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) image, got shape {image.shape!r}")
        if image.dtype != np.uint8:
            raise ValueError(f"expected a uint8 image, got dtype {image.dtype!r}")

        padded, geometry = letterbox(image, self._input_size)
        # BGR, raw 0..255 float32, NCHW -- OpenCV FaceDetectorYN preprocessing.
        blob = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        )
        raw = self._session.run(self._output_names, {self._input_name: blob})
        heads = dict(zip(self._output_names, raw, strict=True))

        boxes_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        landmark_parts: list[np.ndarray] = []
        for stride in STRIDES:
            side = self._input_size // stride
            boxes, scores, landmarks = decode_head(
                heads[f"cls_{stride}"],
                heads[f"obj_{stride}"],
                heads[f"bbox_{stride}"],
                heads[f"kps_{stride}"],
                stride=stride,
                cols=side,
                rows=side,
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
