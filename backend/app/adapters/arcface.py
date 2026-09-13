"""ArcFace embedder adapter: buffalo_l w600k_r50 and antelopev2 glintr100 (spec 6.3, C7).

Non-commercial weights: loading is gated on the `allow_noncommercial_models` setting
(invariant 9), an audited operator decision that `app.core.registry` enforces through
`app.models_lock.assert_loadable`. Raw ONNX through
onnxruntime; the `insightface` package is never imported, since its loader fetches weights
over the network and would break C1 (spec 6.3).

Verified graph IO (onnxruntime 1.30). Both exports, one contract:

    models/w600k_r50.onnx    input 'input.1' [N, 3, 112, 112] -> output '683'  [N, 512]
    models/glintr100.onnx    input 'input.1' [N, 3, 112, 112] -> output '1333' [N, 512]

Both take a dynamic batch, so `MAX_BATCH` chunking applies to either.

Preprocessing, verified empirically on LFW-funneled pairs rather than taken on trust
(a wrong channel order or a missing normalisation collapses the same-identity vs impostor
cosine gap). Measured mean cosine over 25 identities, aligned 112x112 crops:

    PLACEHOLDER

So: RGB channel order, `(pixel - 127.5) / 127.5`, NCHW -- the ArcFace/insightface recipe.
Output is L2-normalized here so cosine is a plain dot product downstream.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from app.core.types import CROP_SIZE
from app.core.vectors import l2_normalize

ARCFACE_DIM = 512
# insightface ArcFace normalisation: input_mean = input_std = 127.5.
INPUT_MEAN = 127.5
INPUT_STD = 127.5
# Cap on one forward pass so a large re-embed job cannot allocate an unbounded blob.
MAX_BATCH = 32


class ArcFaceEmbedder:
    """`app.core.types.Embedder` over an ArcFace ONNX export (R50 or R100).

    Both exports take the same aligned 112x112 crop, the same normalisation and the same
    512-d L2-normalized output, so one class serves both ids; only the graph differs.
    """

    def __init__(
        self, session: ort.InferenceSession, *, model_id: str, dim: int = ARCFACE_DIM
    ) -> None:
        self.model_id = model_id
        self.dim = dim
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [output.name for output in session.get_outputs()]

    def embed(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, 112, 112, 3) uint8 RGB aligned. Returns (N, dim) float32, L2-normed."""
        batch = _validate_crops(crops)
        count = batch.shape[0]
        out = np.empty((count, self.dim), dtype=np.float32)
        for start in range(0, count, MAX_BATCH):
            chunk = batch[start : start + MAX_BATCH]
            # RGB, (v - 127.5) / 127.5, NCHW. The graph takes a dynamic batch.
            blob = np.ascontiguousarray(
                ((chunk.astype(np.float32) - INPUT_MEAN) / INPUT_STD).transpose(0, 3, 1, 2)
            )
            raw = self._session.run(self._output_names, {self._input_name: blob})[0]
            vectors = np.asarray(raw, dtype=np.float32).reshape(chunk.shape[0], -1)
            if vectors.shape[1] != self.dim:
                raise ValueError(
                    f"{self.model_id}: model returned {vectors.shape[1]} values per crop, "
                    f"expected {self.dim}"
                )
            out[start : start + chunk.shape[0]] = vectors
        return l2_normalize(out, axis=1)


def _validate_crops(crops: np.ndarray) -> np.ndarray:
    """Shape/dtype contract for aligned crop batches (spec 6.3)."""
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
