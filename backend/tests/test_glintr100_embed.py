"""glintr100 on the shared ArcFace adapter: the output contract and the batch chunking.

The adapter was written for w600k_r50 and is reused here, so the two things that could
differ between the exports are checked against this one: the 512-d L2-normalized row
contract every downstream cosine assumes, and that `MAX_BATCH` chunking is numerically
invisible -- a whole-gallery re-embed runs in chunks and must produce the same vectors a
one-crop call would.

Needs the real weights, which are gitignored; skips cleanly where they are absent (CI).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from app.adapters.arcface import ARCFACE_DIM, MAX_BATCH, ArcFaceEmbedder
from app.config import Settings
from app.core.registry import CPU_PROVIDER, build_session

GLINTR100_FILE = "glintr100.onnx"
MODEL_ID = "antelopev2-glintr100"


def _embedder() -> ArcFaceEmbedder | None:
    path = Settings().models_dir / GLINTR100_FILE
    if not path.is_file():
        return None
    session: ort.InferenceSession = build_session(path, [CPU_PROVIDER])
    return ArcFaceEmbedder(session, model_id=MODEL_ID, dim=ARCFACE_DIM)


def _crops() -> np.ndarray:
    return np.random.default_rng(13).integers(0, 256, size=(5, 112, 112, 3), dtype=np.uint8)


def test_glintr100_returns_512_l2_normed_rows() -> None:
    embedder = _embedder()
    if embedder is None:
        pytest.skip(f"models/{GLINTR100_FILE} is not provisioned")

    vectors = embedder.embed(_crops())

    assert vectors.shape == (5, ARCFACE_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_glintr100_batches_match_single_crops() -> None:
    """A re-embed larger than MAX_BATCH is split; the split must not move a vector."""
    embedder = _embedder()
    if embedder is None:
        pytest.skip(f"models/{GLINTR100_FILE} is not provisioned")
    count = MAX_BATCH + 3
    crops = np.random.default_rng(17).integers(
        0, 256, size=(count, 112, 112, 3), dtype=np.uint8
    )

    batched = embedder.embed(crops)

    assert batched.shape == (count, ARCFACE_DIM)
    # The chunk boundary and both ends of it: where a wrong slice would show up.
    for index in (0, MAX_BATCH - 1, MAX_BATCH, count - 1):
        single = embedder.embed(crops[index])
        assert np.allclose(batched[index], single[0], atol=1e-6)
