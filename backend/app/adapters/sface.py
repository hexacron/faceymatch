"""SFace embedder adapter (spec 6.3). Default embedder: Apache-2.0, commercial-safe.

Raw ONNX through onnxruntime; the `insightface` package is never imported (C1, spec 6.3).

Verified graph IO for `models/face_recognition_sface_2021dec.onnx` (onnxruntime 1.30):

    input  'data' [1, 3, 112, 112] float32
    output 'fc1'  [1, 128]         float32

The batch dimension is fixed at 1, so `embed` accepts an (N, 112, 112, 3) batch and loops.

Preprocessing, verified empirically on LFW-funneled pairs rather than taken on trust
(the same-identity vs impostor cosine gap collapses if the channel order or the value
range is wrong). Measured mean cosine over 25 identities, aligned 112x112 crops:

    BGR, raw 0..255            same 0.573  impostor 0.041   gap 0.532   <- used
    RGB, raw 0..255            same 0.436  impostor 0.104   gap 0.332
    BGR, (v - 127.5) / 127.5   same 0.271  impostor 0.140   gap 0.131
    RGB, (v - 127.5) / 127.5   same 0.238  impostor 0.146   gap 0.092

So: BGR channel order, uint8-valued float32 with no mean subtraction and no scaling,
NCHW -- i.e. OpenCV's `blobFromImage(aligned, 1.0, (112,112), Scalar(), false, false)`.
Output is L2-normalized here so cosine is a plain dot product downstream.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from app.core.types import CROP_SIZE
from app.core.vectors import l2_normalize

SFACE_DIM = 128


class SFaceEmbedder:
    """`app.core.types.Embedder` over the SFace 2021dec ONNX export."""

    def __init__(
        self, session: ort.InferenceSession, *, model_id: str, dim: int = SFACE_DIM
    ) -> None:
        self.model_id = model_id
        self.dim = dim
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [output.name for output in session.get_outputs()]

    def embed(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, 112, 112, 3) uint8 RGB aligned. Returns (N, dim) float32, L2-normed."""
        batch = _validate_crops(crops)
        out = np.empty((batch.shape[0], self.dim), dtype=np.float32)
        for index in range(batch.shape[0]):
            # RGB -> BGR, raw values, NCHW, batch of 1 (the graph's batch is fixed).
            blob = np.ascontiguousarray(
                batch[index][:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
            )
            raw = self._session.run(self._output_names, {self._input_name: blob})[0]
            vector = np.asarray(raw, dtype=np.float32).reshape(-1)
            if vector.size != self.dim:
                raise ValueError(
                    f"{self.model_id}: model returned {vector.size} values, expected {self.dim}"
                )
            out[index] = vector
        return l2_normalize(out, axis=1)


def _validate_crops(crops: np.ndarray) -> np.ndarray:
    """Shared shape/dtype contract for aligned crop batches (spec 6.3)."""
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
