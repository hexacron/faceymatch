"""Health and model-license surface.

C7 requires the active model license to be visible in the UI, so it is reported here even
when the weights are not provisioned yet (license: null).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import audit
from app.api.deps import ConnDep, LockDep, SettingsDep
from app.db.migrate import current_version
from app.models_lock import status as model_status

router = APIRouter(prefix="/api", tags=["health"])

VERSION = "0.1.0"


class ModelOut(BaseModel):
    model_id: str
    license: str | None
    present: bool
    dim: int | None = None


class ThresholdSetOut(BaseModel):
    id: str
    calibrated: bool
    gallery_size: int | None
    execution_provider: str | None


class HealthOut(BaseModel):
    status: str
    version: str
    db_path: str
    migration_version: int
    embedder: ModelOut
    detector: ModelOut
    execution_provider: str
    allow_noncommercial_models: bool
    threshold_set: ThresholdSetOut | None
    audit_head_seq: int
    audit_head_hash: str | None


@router.get("/healthz", response_model=HealthOut)
def healthz(conn: ConnDep, settings: SettingsDep, lock: LockDep) -> HealthOut:
    embedder = model_status(lock, settings.embedder_model, settings.models_dir)
    detector = model_status(lock, settings.detector_model, settings.models_dir)
    head_seq, head_hash = audit.head(conn)

    row = conn.execute(
        "SELECT id, calibrated, gallery_size, execution_provider "
        "FROM threshold_sets WHERE active = 1"
    ).fetchone()
    threshold_set = (
        None
        if row is None
        else ThresholdSetOut(
            id=str(row["id"]),
            calibrated=bool(row["calibrated"]),
            gallery_size=row["gallery_size"],
            execution_provider=row["execution_provider"],
        )
    )

    return HealthOut(
        status="ok",
        version=VERSION,
        db_path=str(settings.db_path),
        migration_version=current_version(conn),
        embedder=ModelOut(
            model_id=embedder.model_id,
            license=embedder.license,
            present=embedder.present,
            dim=embedder.dim,
        ),
        detector=ModelOut(
            model_id=detector.model_id,
            license=detector.license,
            present=detector.present,
        ),
        execution_provider=settings.execution_provider,
        allow_noncommercial_models=settings.allow_noncommercial_models,
        threshold_set=threshold_set,
        audit_head_seq=head_seq,
        audit_head_hash=None if head_seq == 0 else head_hash,
    )
