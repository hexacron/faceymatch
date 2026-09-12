"""Provisioned model inventory and active-model status."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import LockDep, SettingsDep
from app.models_lock import ModelKind

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelInfoOut(BaseModel):
    id: str
    name: str
    version: str
    kind: ModelKind
    sha256: str
    # Nullable to match /api/healthz: both endpoints describe the same models.lock field.
    license: str | None
    dim: int | None
    active: bool
    present: bool


class ModelsInfoOut(BaseModel):
    items: list[ModelInfoOut]
    execution_provider: str
    allow_noncommercial_models: bool


@router.get("", response_model=ModelsInfoOut)
def list_models(settings: SettingsDep, lock: LockDep) -> ModelsInfoOut:
    return ModelsInfoOut(
        items=[
            ModelInfoOut(
                id=entry.id,
                name=entry.name,
                version=entry.version,
                kind=entry.kind,
                sha256=entry.sha256,
                license=entry.license,
                dim=entry.dim,
                active=(
                    entry.id == settings.detector_model
                    if entry.kind == "detector"
                    else entry.id == settings.embedder_model
                ),
                present=(settings.models_dir / entry.file).is_file(),
            )
            for entry in lock.models
        ],
        execution_provider=settings.execution_provider,
        allow_noncommercial_models=settings.allow_noncommercial_models,
    )
