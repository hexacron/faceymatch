"""Active model registry (spec 6.3).

One place builds onnxruntime sessions, so every other module gets its detector and embedder
through the `Detector` / `Embedder` interfaces (AGENTS.md code rules) and nothing else opens
a `.onnx` file. Three things this module owns:

- the license gate: every load goes through `models_lock.assert_loadable`, so a
  non-commercial model is refused unless the effective settings have
  `allow_noncommercial_models` set (invariant 9, C7). The flag is an audited runtime
  setting, and it is part of the cache key, so turning it off does not keep serving a
  session built while it was on;
- the execution provider that is *actually* in use. A CoreML request on a build without
  CoreML falls back to CPU with a logged warning, and `ActiveModels.execution_provider`
  reports the real provider, never the requested one -- `acceptance.build_gate` compares it
  against the threshold set's calibrated provider, and an echo would open the gate on scores
  that are not reproducible (spec 10);
- session caching. Building an ONNX session costs hundreds of milliseconds, so the job
  worker must not rebuild one per job. The cache key is hashable on purpose: `Settings` and
  `ModelsLock` are pydantic models and unhashable, so callers can load a fresh lock per job
  and still reuse the sessions.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort

from app.adapters.arcface_r50 import ArcFaceR50Embedder
from app.adapters.sface import SFaceEmbedder
from app.adapters.yunet import YuNetDetector
from app.config import Settings
from app.core.types import Detector, Embedder
from app.models_lock import ModelEntry, ModelsLock, ModelsLockError, assert_loadable

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"

# Embedder id -> adapter class (spec 6.3). A detector or embedder id that is not here is a
# configuration error, not a silent fallback: scores from an unexpected model would be
# compared against templates from another one (invariant 2).
_EMBEDDER_CLASSES: dict[str, type[SFaceEmbedder] | type[ArcFaceR50Embedder]] = {
    "sface-2021dec": SFaceEmbedder,
    "buffalo_l-w600k_r50": ArcFaceR50Embedder,
}
_DETECTOR_IDS = frozenset({"yunet-2023mar"})

_CacheKey = tuple[str, str, str, str, bool, tuple[tuple[str, str, str], ...]]
_cache: dict[_CacheKey, ActiveModels] = {}
_cache_lock = threading.Lock()


def has_adapter(model_id: str, *, kind: str) -> bool:
    """Whether this build can actually run that model id.

    A model can be listed in models.lock, present on disk and correctly licensed and still
    be unrunnable here, because no adapter implements it. Exposed so a configuration change
    is refused by the request that made it rather than by the next job.
    """
    return model_id in (_DETECTOR_IDS if kind == "detector" else _EMBEDDER_CLASSES)


@dataclass(frozen=True, slots=True)
class ActiveModels:
    """The detector and embedder this process is running, plus their provenance."""

    detector: Detector
    embedder: Embedder
    detector_model_id: str
    embedder_model_id: str
    execution_provider: str


def _resolve_path(entry: ModelEntry, models_dir: Path) -> Path:
    path = models_dir / entry.file
    if not path.is_file():
        raise ModelsLockError(
            f"model {entry.id!r} needs {path}, which is not present; provision weights with "
            "tools/fetch_models.py (they are never fetched at runtime, C1)"
        )
    return path


def _provider_chain(requested: str) -> list[str]:
    """Requested provider first, CPU as the fallback, warning when it is unavailable."""
    if requested == CPU_PROVIDER:
        return [CPU_PROVIDER]
    available = ort.get_available_providers()
    if requested not in available:
        logger.warning(
            "execution provider %s is not available in this onnxruntime build (%s); "
            "falling back to %s",
            requested,
            ", ".join(available),
            CPU_PROVIDER,
        )
        return [CPU_PROVIDER]
    return [requested, CPU_PROVIDER]


def build_session(path: Path, providers: list[str]) -> ort.InferenceSession:
    """Open one ONNX graph. Local file only; onnxruntime never reaches the network (C1)."""
    options = ort.SessionOptions()
    # The pinned exports declare weights as graph inputs, which floods stderr with
    # "Initializer ... appears in graph inputs" warnings on every session build.
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def active_provider(session: ort.InferenceSession) -> str:
    """The provider a built session actually runs on, highest priority first."""
    providers = session.get_providers()
    return providers[0] if providers else CPU_PROVIDER


def load_active_models(settings: Settings, lock: ModelsLock) -> ActiveModels:
    """Build the configured detector and embedder. Always builds; see `get_active_models`."""
    detector_entry = assert_loadable(lock, settings.detector_model, settings)
    embedder_entry = assert_loadable(lock, settings.embedder_model, settings)

    if detector_entry.id not in _DETECTOR_IDS:
        raise ModelsLockError(
            f"no detector adapter for {detector_entry.id!r}; known: "
            f"{', '.join(sorted(_DETECTOR_IDS))}"
        )
    embedder_class = _EMBEDDER_CLASSES.get(embedder_entry.id)
    if embedder_class is None:
        raise ModelsLockError(
            f"no embedder adapter for {embedder_entry.id!r}; known: "
            f"{', '.join(sorted(_EMBEDDER_CLASSES))}"
        )
    if embedder_entry.dim is None:
        raise ModelsLockError(
            f"models.lock entry {embedder_entry.id!r} has no dim; embeddings cannot be "
            "stored without one"
        )

    detector_path = _resolve_path(detector_entry, settings.models_dir)
    embedder_path = _resolve_path(embedder_entry, settings.models_dir)
    providers = _provider_chain(settings.execution_provider)

    detector_session = build_session(detector_path, providers)
    embedder_session = build_session(embedder_path, providers)

    detector = YuNetDetector(
        detector_session,
        model_id=detector_entry.id,
        score_threshold=settings.min_det_score,
        nms_iou=settings.nms_iou,
    )
    embedder = embedder_class(
        embedder_session, model_id=embedder_entry.id, dim=embedder_entry.dim
    )

    return ActiveModels(
        detector=detector,
        embedder=embedder,
        detector_model_id=detector_entry.id,
        embedder_model_id=embedder_entry.id,
        # The real provider, not settings.execution_provider (spec 10).
        execution_provider=active_provider(embedder_session),
    )


def _cache_key(settings: Settings, lock: ModelsLock) -> _CacheKey:
    """Hashable key over everything that changes which weights get loaded, and how."""
    relevant = tuple(
        (entry.id, entry.file, entry.sha256)
        for entry in sorted(lock.models, key=lambda item: item.id)
        if entry.id in {settings.detector_model, settings.embedder_model}
    )
    return (
        str(settings.models_dir),
        settings.detector_model,
        settings.embedder_model,
        settings.execution_provider,
        settings.allow_noncommercial_models,
        relevant,
    )


def get_active_models(settings: Settings, lock: ModelsLock) -> ActiveModels:
    """Cached `load_active_models`, so ONNX sessions are built once per process."""
    key = _cache_key(settings, lock)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        models = load_active_models(settings, lock)
        _cache[key] = models
        return models


def clear_cache() -> None:
    """Drop cached sessions. For tests and for a model switch inside one process."""
    with _cache_lock:
        _cache.clear()
