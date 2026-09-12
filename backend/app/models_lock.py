"""models.lock verification (invariants 7, 8, 9; spec 6.3 and C7).

Rules enforced here:
- Every model file present in models/ must be listed in models.lock and match its digest.
- An ONNX file in models/ that is not listed is a refusal, not a warning: unlisted weights
  are exactly what invariant 8 exists to catch.
- A model whose license is not cleared for commercial use loads only when
  ALLOW_NONCOMMERCIAL_MODELS=true.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings

LOCK_FILENAME = "models.lock"
_WEIGHT_SUFFIXES = frozenset({".onnx"})
_CHUNK = 1024 * 1024

ModelKind = Literal["detector", "embedder"]


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    version: str
    kind: ModelKind
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str
    commercial_use: bool
    dim: int | None = None


class ModelsLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    note: str | None = None
    models: list[ModelEntry] = Field(default_factory=list)

    def by_id(self, model_id: str) -> ModelEntry | None:
        for entry in self.models:
            if entry.id == model_id:
                return entry
        return None


class ModelsLockError(RuntimeError):
    """models.lock is missing, malformed, or contradicted by the files on disk."""


class ModelLicenseError(RuntimeError):
    """A non-commercial model was requested without ALLOW_NONCOMMERCIAL_MODELS."""


@dataclass(frozen=True, slots=True)
class ModelStatus:
    model_id: str
    present: bool
    license: str | None
    dim: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load(models_dir: Path) -> ModelsLock:
    lock_path = models_dir / LOCK_FILENAME
    if not lock_path.is_file():
        raise ModelsLockError(f"{lock_path} is missing; refusing to start (invariant 8)")
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelsLockError(f"{lock_path} is not valid JSON: {exc}") from exc
    return ModelsLock.model_validate(raw)


def verify(models_dir: Path, lock: ModelsLock | None = None) -> ModelsLock:
    """Verify every weight file in models_dir against models.lock. Raises on mismatch."""
    resolved = lock if lock is not None else load(models_dir)

    listed = {entry.file: entry for entry in resolved.models}
    for entry in resolved.models:
        path = models_dir / entry.file
        if not path.is_file():
            # Not an error: weights are gitignored and provisioned per install. The active
            # model's presence is checked when it is loaded.
            continue
        actual = sha256_file(path)
        if actual != entry.sha256:
            raise ModelsLockError(
                f"{entry.file} does not match models.lock "
                f"(expected {entry.sha256[:12]}, found {actual[:12]}); refusing to start"
            )

    for path in sorted(models_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES and path.name not in listed:
            raise ModelsLockError(
                f"{path.name} is present in models/ but absent from models.lock; "
                "refusing to start (invariant 8)"
            )
    return resolved


def assert_loadable(lock: ModelsLock, model_id: str, settings: Settings) -> ModelEntry:
    """Gate a model load on its license (invariant 9, C7)."""
    entry = lock.by_id(model_id)
    if entry is None:
        raise ModelsLockError(f"model {model_id!r} is not listed in models.lock")
    if not entry.commercial_use and not settings.allow_noncommercial_models:
        raise ModelLicenseError(
            f"model {model_id!r} is licensed {entry.license!r} (non-commercial); "
            "set ALLOW_NONCOMMERCIAL_MODELS=true to load it"
        )
    return entry


def status(lock: ModelsLock, model_id: str, models_dir: Path) -> ModelStatus:
    """Reporting helper for /api/healthz and the C7 license banner."""
    entry = lock.by_id(model_id)
    if entry is None:
        return ModelStatus(model_id=model_id, present=False, license=None, dim=None)
    return ModelStatus(
        model_id=model_id,
        present=(models_dir / entry.file).is_file(),
        license=entry.license,
        dim=entry.dim,
    )
