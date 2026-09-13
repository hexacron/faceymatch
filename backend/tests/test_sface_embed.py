"""Overlapping SFace's fixed-batch Runs must not move a single bit of a score.

The graph declares `data [1, 3, 112, 112]`, so one Run per crop is forced. Running them
concurrently is a latency change and nothing else: if it were also a numeric change, every
threshold set calibrated against the serial path would silently stop meaning what it says
(invariant 2 territory, and the calibrated auto-accept gate of invariant 4 rests on it).

Needs the real weights, which are gitignored; skips cleanly where they are absent (CI).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from app.adapters.sface import SFACE_DIM, SFaceEmbedder
from app.config import Settings
from app.core.registry import CPU_PROVIDER, build_session
from app.core.vectors import l2_normalize

SFACE_FILE = "face_recognition_sface_2021dec.onnx"


def _providers() -> list[str]:
    available = ort.get_available_providers()
    return [CPU_PROVIDER] + [p for p in ("CoreMLExecutionProvider",) if p in available]


def _session(provider: str) -> ort.InferenceSession | None:
    path = Settings().models_dir / SFACE_FILE
    if not path.is_file():
        return None
    chain = [provider] if provider == CPU_PROVIDER else [provider, CPU_PROVIDER]
    return build_session(path, chain)


def _serial_embed(session: ort.InferenceSession, crops: np.ndarray) -> np.ndarray:
    """The loop this adapter used before the pool: one Run at a time, same tensors."""
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    out = np.empty((crops.shape[0], SFACE_DIM), dtype=np.float32)
    for index in range(crops.shape[0]):
        blob = np.ascontiguousarray(
            crops[index][:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        )
        raw = session.run(output_names, {input_name: blob})[0]
        out[index] = np.asarray(raw, dtype=np.float32).reshape(-1)
    return l2_normalize(out, axis=1)


@pytest.mark.parametrize("provider", _providers())
def test_pooled_runs_are_byte_identical_to_serial_runs(provider: str) -> None:
    session = _session(provider)
    if session is None:
        pytest.skip(f"models/{SFACE_FILE} is not provisioned")
    # Five crops: more than the pool's four workers, so the queued run is covered too.
    crops = np.random.default_rng(7).integers(0, 256, size=(5, 112, 112, 3), dtype=np.uint8)

    pooled = SFaceEmbedder(session, model_id="sface-2021dec").embed(crops)
    serial = _serial_embed(session, crops)

    assert pooled.shape == (5, SFACE_DIM)
    assert np.array_equal(pooled, serial)


def test_one_crop_still_returns_one_row() -> None:
    """The single-crop shortcut skips the pool; it must not skip the contract."""
    session = _session(CPU_PROVIDER)
    if session is None:
        pytest.skip(f"models/{SFACE_FILE} is not provisioned")
    crop = np.random.default_rng(11).integers(0, 256, size=(112, 112, 3), dtype=np.uint8)

    embedder = SFaceEmbedder(session, model_id="sface-2021dec")
    single = embedder.embed(crop)
    batched = embedder.embed(crop[None])

    assert single.shape == (1, SFACE_DIM)
    assert np.array_equal(single, batched)
