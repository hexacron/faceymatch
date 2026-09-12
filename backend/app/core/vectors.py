"""Embedding serialization and normalization.

On-disk form for every embedding BLOB (`templates.embedding`, `tracks.embedding_mean`,
`detection_embeddings.embedding`): raw little-endian float32, no header. The dimension comes
from the model row, so the bytes carry no schema of their own; `from_blob` validates length
against the expected dim rather than guessing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DTYPE = np.dtype("<f4")


def l2_normalize(vectors: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """L2-normalize along `axis`. Zero vectors are returned unchanged, not divided by zero."""
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return (array / safe).astype(np.float32, copy=False)


def to_blob(vector: np.ndarray) -> bytes:
    """Serialize one embedding. The caller is responsible for having normalized it."""
    array = np.ascontiguousarray(np.asarray(vector, dtype=DTYPE).reshape(-1))
    return array.tobytes()


def from_blob(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize one embedding and check it against the model's dimension."""
    expected = dim * DTYPE.itemsize
    if len(blob) != expected:
        raise ValueError(
            f"embedding blob is {len(blob)} bytes; expected {expected} for dim {dim}"
        )
    return np.frombuffer(blob, dtype=DTYPE).astype(np.float32, copy=True)


def stack_blobs(blobs: list[bytes], dim: int) -> np.ndarray:
    """Deserialize many embeddings into one (N, dim) float32 matrix for blocked matmul."""
    if not blobs:
        return np.zeros((0, dim), dtype=np.float32)
    matrix = np.empty((len(blobs), dim), dtype=np.float32)
    for i, blob in enumerate(blobs):
        matrix[i] = from_blob(blob, dim)
    return matrix


def track_mean(embeddings: np.ndarray, scores: Sequence[float], *, k: int) -> np.ndarray:
    """L2-normalized mean of the best `k` crop embeddings of one track (spec 6.2 step 6).

    "Best" is by detector score, descending: it is the per-detection quality signal the
    pipeline already records (`detections.det_score`, and `templates.quality` derived from
    it), so the crops that define a track are the ones the detector was most sure of.

    One function for both writers. The still-image pipeline gives it a single crop and the
    re-embed job gives it every stored crop of a track; a second implementation would be a
    second definition of what a track's vector means, and the two models' means would stop
    being comparable quantities.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be (N, dim), got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("a track mean needs at least one crop embedding")
    if len(scores) != matrix.shape[0]:
        raise ValueError(
            f"got {len(scores)} scores for {matrix.shape[0]} embeddings"
        )
    if matrix.shape[0] > k:
        # Stable: equal scores keep insertion order, so the mean is reproducible.
        order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable")[:k]
        matrix = matrix[order]
    return l2_normalize(matrix.mean(axis=0))
