"""Embedding serialization and normalization.

On-disk form for every embedding BLOB (`templates.embedding`, `tracks.embedding_mean`,
`detection_embeddings.embedding`): raw little-endian float32, no header. The dimension comes
from the model row, so the bytes carry no schema of their own; `from_blob` validates length
against the expected dim rather than guessing.
"""

from __future__ import annotations

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
