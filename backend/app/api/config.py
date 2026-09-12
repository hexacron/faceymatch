"""Operator configuration: read the whole operating state, change the editable part.

What is editable and what is not is a safety decision, not a UI convenience:

- `allow_noncommercial_models` and `execution_provider` are read-only and environment-only.
  A licensing gate the UI can flip is not a gate (invariant 9, C7), and a threshold set is
  only reproducible on the execution provider it was calibrated on (spec 10), so moving the
  provider from a web request would silently invalidate every calibrated band.
- Threshold values are not here at all. They come from a calibration run and are activated
  by their own audited action (spec 10, `POST /api/threshold_sets/{id}/activate`). A
  threshold an operator can type is a threshold nobody measured.
- `db_path`, `models_dir` and `operator_name` describe where the evidence lives and who is
  claiming it. Those belong to the install, not to a session.

Every accepted change is written to `runtime_config` and audited as `config.change` with
the previous value, the new value and the operator's reason, in one transaction with any
job the change implies. The change is durable and takes effect immediately: the API reads
effective settings per request and the worker per job, so no restart is involved.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app import audit, jobs, runtime_config
from app.api.deps import ConnDep, LockDep, SettingsDep
from app.config import Settings
from app.core import registry
from app.db.conn import transaction
from app.models_lock import ModelKind, ModelsLock
from app.pipeline.live import clear_gallery_cache
from app.thresholds import seed_default_threshold_set

router = APIRouter(prefix="/api/config", tags=["config"])

# Kinds that make the configuration unsafe to move while they run.
_BLOCKING_JOB_KINDS = ("reembed",)
# Kinds worth surfacing as "something is still catching up with your last change".
_PENDING_JOB_KINDS = ("reembed", "rematch")


class EditableConfigOut(BaseModel):
    detector_model: str
    embedder_model: str
    min_embed_px: int
    max_yaw: float
    min_sharpness: float
    min_det_score: float
    sample_fps: float
    top_k: int
    person_score_mode: Literal["max", "mean_top3"]


class ReadonlyConfigOut(BaseModel):
    execution_provider: str
    allow_noncommercial_models: bool
    operator_name: str
    db_path: str
    models_dir: str
    max_upload_bytes: int
    fpir_target: float
    embed_k: int
    rematch_block_size: int


class ConfigModelOut(BaseModel):
    id: str
    name: str
    version: str
    kind: ModelKind
    license: str
    commercial_use: bool
    dim: int | None
    present: bool
    active: bool


class PendingJobOut(BaseModel):
    id: str
    kind: str
    status: str


class ConfigOut(BaseModel):
    editable: EditableConfigOut
    readonly: ReadonlyConfigOut
    models: list[ConfigModelOut]
    pending_job: PendingJobOut | None


class ConfigPatchOut(ConfigOut):
    jobs_enqueued: list[str]


class ConfigPatch(BaseModel):
    changes: Annotated[dict[str, Any], Field(default_factory=dict)]
    reason: str | None = None


@router.get("", response_model=ConfigOut)
def get_config(conn: ConnDep, settings: SettingsDep, lock: LockDep) -> ConfigOut:
    """The whole operating configuration: what can move, what cannot, and what is in flight."""
    return _config_out(conn, settings, lock)


@router.patch("", response_model=ConfigPatchOut)
def patch_config(
    body: ConfigPatch, conn: ConnDep, settings: SettingsDep, lock: LockDep
) -> ConfigPatchOut:
    """Change any subset of the editable settings, audited, with the jobs the change implies.

    Changing `embedder_model` is the heavy path. Stored crops are re-embedded under the new
    model and the gallery is carried across (`reembed`), then every stored track is scored
    again (`rematch`). Old vectors are kept, so the switch is reversible. Auto-accept stops:
    the active threshold set was calibrated for the previous model, and a set calibrated for
    the new one has to be activated before anything self-confirms again (C5, spec 10).

    Changing `detector_model` does **not** re-run detection on stored media. Existing
    detections, their boxes and their crops are the record of what was found at the time and
    are left exactly as they are; the new detector applies to media processed from now on.
    Re-detecting stored media would rewrite evidence, which is a different operation with a
    different justification, and it is not this endpoint.
    """
    blocking = _pending_job(conn, _BLOCKING_JOB_KINDS)
    if blocking is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"a {blocking.kind} job ({blocking.id}) is {blocking.status}; it must finish "
                "before the configuration moves again, or it would commit half its work "
                "under one model and half under another"
            ),
        )

    try:
        candidate = runtime_config.validate(settings, lock, body.changes)
    except runtime_config.UnknownKeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except runtime_config.InvalidValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    applied = {
        key: value
        for key, value in body.changes.items()
        if value != getattr(settings, key)
    }
    switching_embedder = "embedder_model" in applied

    enqueued: list[str] = []
    with transaction(conn):
        runtime_config.write(
            conn,
            changes=applied,
            previous=settings,
            reason=body.reason,
            actor=settings.operator_name,
        )
        if switching_embedder:
            _audit_auto_accept_suspension(
                conn,
                previous_model_id=settings.embedder_model,
                new_model_id=candidate.embedder_model,
                actor=settings.operator_name,
                reason=body.reason,
            )
            seed_default_threshold_set(
                conn,
                model_id=candidate.embedder_model,
                actor=settings.operator_name,
            )
            # Order matters and a single worker preserves it: re-embed the stored crops
            # into the new model, then score every track against the gallery it produced.
            enqueued.append(
                jobs.insert(
                    conn,
                    kind="reembed",
                    actor=settings.operator_name,
                    params={
                        "embedder_model_id": candidate.embedder_model,
                        "previous_embedder_model_id": settings.embedder_model,
                        "reason": body.reason,
                    },
                )
            )
            enqueued.append(
                jobs.insert(
                    conn,
                    kind="rematch",
                    actor=settings.operator_name,
                    params={"reason": "embedder_switch"},
                )
            )

    _invalidate_caches()
    return ConfigPatchOut(
        **_config_out(conn, candidate, lock).model_dump(),
        jobs_enqueued=enqueued,
    )


def _audit_auto_accept_suspension(
    conn: sqlite3.Connection,
    *,
    previous_model_id: str,
    new_model_id: str,
    actor: str,
    reason: str | None,
) -> None:
    """Record that the active threshold set stopped applying, in the switch's transaction.

    The gate would refuse auto-accept anyway — `acceptance.build_gate` compares the set's
    `model_id` against the active embedder — but "it happened not to fire" is not a record.
    A reviewer asking why identities stopped being confirmed on a given date needs the
    answer in the chain, not inferred from a model id two tables away (C5, spec 10).
    """
    row = conn.execute(
        "SELECT id, model_id, calibrated FROM threshold_sets WHERE active = 1"
    ).fetchone()
    audit.append(
        conn,
        actor=actor,
        action="config.auto_accept_suspended",
        object_type="threshold_set",
        object_id=None if row is None else str(row["id"]),
        payload={
            "previous_embedder_model_id": previous_model_id,
            "embedder_model_id": new_model_id,
            "threshold_set_model_id": None if row is None else str(row["model_id"]),
            "threshold_set_calibrated": None if row is None else bool(row["calibrated"]),
            "reason": reason,
            "effect": (
                "the active threshold set is not calibrated for the new embedder, so every "
                "match stays a candidate until a set calibrated for it is activated"
            ),
        },
    )


def _invalidate_caches() -> None:
    """Drop everything in this process that was built from the previous configuration.

    ONNX sessions are keyed by model id and would not be *served* wrongly, but they hold
    hundreds of megabytes of weights that the switch just made dead. The gallery matrix is
    keyed by embedder id and audit head, so the config entry alone already moves it; it is
    cleared here anyway rather than left resting on that coupling.

    The worker is a separate process and needs no signal: it resolves effective settings and
    looks up its models per job.
    """
    registry.clear_cache()
    clear_gallery_cache()


def _pending_job(conn: sqlite3.Connection, kinds: tuple[str, ...]) -> PendingJobOut | None:
    placeholders = ",".join("?" * len(kinds))
    row = conn.execute(
        "SELECT id, kind, status FROM jobs "  # noqa: S608 - placeholders are generated
        f"WHERE kind IN ({placeholders}) AND status IN ('queued', 'running') "
        "ORDER BY created_at, id LIMIT 1",
        kinds,
    ).fetchone()
    if row is None:
        return None
    return PendingJobOut(
        id=str(row["id"]), kind=str(row["kind"]), status=str(row["status"])
    )


def _config_out(
    conn: sqlite3.Connection, settings: Settings, lock: ModelsLock
) -> ConfigOut:
    return ConfigOut(
        editable=EditableConfigOut(
            detector_model=settings.detector_model,
            embedder_model=settings.embedder_model,
            min_embed_px=settings.min_embed_px,
            max_yaw=settings.max_yaw,
            min_sharpness=settings.min_sharpness,
            min_det_score=settings.min_det_score,
            sample_fps=settings.sample_fps,
            top_k=settings.top_k,
            person_score_mode=settings.person_score_mode,
        ),
        readonly=ReadonlyConfigOut(
            execution_provider=settings.execution_provider,
            allow_noncommercial_models=settings.allow_noncommercial_models,
            operator_name=settings.operator_name,
            db_path=str(settings.db_path),
            models_dir=str(settings.models_dir),
            max_upload_bytes=settings.max_upload_bytes,
            fpir_target=settings.fpir_target,
            embed_k=settings.embed_k,
            rematch_block_size=settings.rematch_block_size,
        ),
        models=[
            ConfigModelOut(
                id=entry.id,
                name=entry.name,
                version=entry.version,
                kind=entry.kind,
                license=entry.license,
                commercial_use=entry.commercial_use,
                dim=entry.dim,
                present=(settings.models_dir / entry.file).is_file(),
                active=(
                    entry.id == settings.detector_model
                    if entry.kind == "detector"
                    else entry.id == settings.embedder_model
                ),
            )
            for entry in lock.models
        ],
        pending_job=_pending_job(conn, _PENDING_JOB_KINDS),
    )
