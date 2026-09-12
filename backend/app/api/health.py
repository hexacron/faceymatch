"""Health, model licence, and whether anything may confirm an identity by itself.

C7 requires the active model licence to be visible in the UI, so it is reported here even
when the weights are not provisioned yet (license: null).

`auto_accept` is reported for the same reason. Auto-accept being off is the normal state of
an uncalibrated or freshly-switched install (C5, invariant 4, spec 10), and an operator who
cannot see that reads an empty identity column as "no matches" rather than as "nothing is
allowed to self-confirm yet". The gate is built with the same `acceptance.build_gate` and
the same active threshold set the job worker uses, so this endpoint cannot disagree with
what actually happens on the next match.
"""

from __future__ import annotations

import sqlite3
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import audit
from app.api.deps import ConnDep, LockDep, SettingsDep
from app.config import Settings
from app.core import registry
from app.core.acceptance import build_gate
from app.db.migrate import current_version
from app.models_lock import ModelsLock
from app.models_lock import status as model_status
from app.pipeline.capture import capability as capture_capability
from app.pipeline.matching import active_threshold_set, gallery_person_count

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


class CaptureCapabilityOut(BaseModel):
    """Whether `POST /api/capture` can work here, so the UI can disable it with a reason.

    `available` is the only field the button needs; `reason` is non-null exactly when
    `available` is false.
    """

    available: bool
    platform_supported: bool
    binary_present: bool
    reason: str | None


class AutoAcceptOut(BaseModel):
    """Why the system will or will not confirm an identity without an operator (spec 6.5).

    `reason` is non-null exactly when `allowed` is false. `warning` is independent: the gate
    is open, but the live gallery has grown past twice the size the active set was
    calibrated at, so the false-positive rate is above the calibrated target (spec 10).
    """

    allowed: bool
    reason: str | None
    warning: str | None
    embedder_model_id: str
    execution_provider: str


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    db_path: str
    migration_version: int
    embedder: ModelOut
    detector: ModelOut
    execution_provider: str
    allow_noncommercial_models: bool
    threshold_set: ThresholdSetOut | None
    auto_accept: AutoAcceptOut
    capture: CaptureCapabilityOut
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
    capture = capture_capability()
    auto_accept = _auto_accept(conn, settings, lock)

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
        auto_accept=auto_accept,
        capture=CaptureCapabilityOut(
            available=capture.available,
            platform_supported=capture.platform_supported,
            binary_present=capture.binary_present,
            reason=capture.reason,
        ),
        audit_head_seq=head_seq,
        audit_head_hash=None if head_seq == 0 else head_hash,
    )


def _auto_accept(
    conn: sqlite3.Connection, settings: Settings, lock: ModelsLock
) -> AutoAcceptOut:
    """Build the real gate, from the real execution provider where that is knowable.

    The provider a session runs on is not always the one that was requested: a CoreML
    request on a build without CoreML falls back to CPU, and a threshold set calibrated on
    CoreML does not apply to CPU scores (spec 10). `registry.get_active_models` is cached,
    so this costs one dictionary lookup after the first call.

    Any failure to build the models — absent, unlicensed, corrupt or unparseable weights —
    is reported as a closed gate rather than raised. Health is what an operator reads when
    something is already wrong, so it has to answer while the install is broken; and a model
    that will not load is a model that will auto-accept nothing.
    """
    threshold_set = active_threshold_set(conn)
    try:
        models = registry.get_active_models(settings, lock)
    except Exception as exc:  # see the docstring: health must still answer when models fail
        return AutoAcceptOut(
            allowed=False,
            reason=str(exc),
            warning=None,
            embedder_model_id=settings.embedder_model,
            execution_provider=settings.execution_provider,
        )
    if threshold_set is None:
        return AutoAcceptOut(
            allowed=False,
            reason="no active threshold set",
            warning=None,
            embedder_model_id=models.embedder_model_id,
            execution_provider=models.execution_provider,
        )
    gate = build_gate(
        threshold_set,
        embedder_model_id=models.embedder_model_id,
        execution_provider=models.execution_provider,
        live_gallery_size=gallery_person_count(conn, embedder_model_id=models.embedder_model_id),
    )
    return AutoAcceptOut(
        allowed=gate.allowed,
        reason=gate.reason,
        warning=gate.warning,
        embedder_model_id=models.embedder_model_id,
        execution_provider=models.execution_provider,
    )
