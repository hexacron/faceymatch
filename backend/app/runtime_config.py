"""Durable runtime overrides for the editable subset of `Settings` (spec 6.3, 8, 9).

The environment (and `.env`) is the base configuration. This module is a sparse overlay on
top of it, stored in the `runtime_config` table, so that an operator change is a database
write with an audit entry beside it rather than an untraceable edit to a file.

Three rules hold the shape together:

- `EDITABLE_FIELDS` is the whole editable surface. Anything else is refused by name, and a
  row written for it by hand is ignored on read. `execution_provider` is deliberately not
  in it: a threshold set is only reproducible on the provider it was calibrated on
  (spec 10), so moving the provider from a web request would silently invalidate every
  calibrated band.
- Validation happens by constructing a `Settings`, so the bounds live in one place
  (`app/config.py`) and the wire cannot accept a value the pipeline would reject.
- A model id is refused unless `models.lock` lists it, its weight file is present, its
  SHA-256 still matches, its kind fits the slot, and its licence is cleared on this
  install (invariants 8 and 9).

`allow_noncommercial_models` is editable here (invariant 9 as amended). It used to be
environment-only on the argument that a gate the UI can flip is not a gate. That argument
lost: the operator who owns the installation is the person entitled to decide what its
licences permit, and forcing them to edit a file and restart the service did not make the
decision more considered, only less visible. So the friction is gone and the record is not:
the change is a `config.change` entry carrying the previous value, the new value, the actor
and a reason that is *required* when the flag is turned on, because "who enabled a
non-commercial model, when, and why" is a question this system exists to be able to answer.
The licence text itself stays visible everywhere it already was — `models.lock`, the C7
banner, `GET /api/models`, `GET /api/config` — informational rather than blocking.

What did not move: `models.lock` verification (invariant 8). Unknown or SHA-mismatched
weights are still refused outright. That is integrity, not licensing, and no operator
setting touches it.

Thresholds are deliberately absent: they come from calibration only (spec 10).

(Migration 0003's header still describes the flag as environment-only. Migration files are
frozen once applied — `db/migrate.py` refuses to start on a changed digest — so the rule as
it stands today is stated here, which is where it is enforced.)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from app import audit
from app.config import Settings
from app.core import registry
from app.models_lock import (
    ModelEntry,
    ModelLicenseError,
    ModelsLock,
    ModelsLockError,
    assert_loadable,
    sha256_file,
)

# The editable surface, in the order the API reports it.
EDITABLE_FIELDS: tuple[str, ...] = (
    "detector_model",
    "embedder_model",
    "allow_noncommercial_models",
    "min_embed_px",
    "max_yaw",
    "min_sharpness",
    "min_det_score",
    "sample_fps",
    "top_k",
    "person_score_mode",
)

# Reported so the UI can show the whole operating configuration, never written here.
READONLY_FIELDS: tuple[str, ...] = (
    "execution_provider",
    "operator_name",
    "db_path",
    "models_dir",
    "max_upload_bytes",
    "fpir_target",
    "embed_k",
    "rematch_block_size",
)

# Turning the licence gate on is the one change whose whole justification is the reason.
REASON_REQUIRED_TO_ENABLE: tuple[str, ...] = ("allow_noncommercial_models",)

_EDITABLE = frozenset(EDITABLE_FIELDS)

# Which editable key selects which kind of model, so a detector cannot be set as an embedder.
_MODEL_FIELD_KINDS: dict[str, str] = {
    "detector_model": "detector",
    "embedder_model": "embedder",
}

# The keys that can change which models load, and so the only ones worth re-verifying
# weights for.
_EDITABLE_LICENCE_RELEVANT = frozenset(
    {*_MODEL_FIELD_KINDS, "allow_noncommercial_models"}
)


class UnknownKeyError(ValueError):
    """A requested key is not part of the editable configuration surface."""


class InvalidValueError(ValueError):
    """A requested value is out of range, or names a model this install may not load."""


class StateConflictError(ValueError):
    """The request is well formed but conflicts with what is currently running."""


class ReasonRequiredError(ValueError):
    """This change is only meaningful with the operator's reason recorded beside it."""


def load(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stored overrides, filtered to the editable surface. Unknown rows are ignored."""
    rows = conn.execute("SELECT key, value_json FROM runtime_config").fetchall()
    overrides: dict[str, Any] = {}
    for row in rows:
        key = str(row["key"])
        if key in _EDITABLE:
            overrides[key] = json.loads(str(row["value_json"]))
    return overrides


def effective(conn: sqlite3.Connection, base: Settings) -> Settings:
    """`base` with the stored overrides applied.

    Read fresh every time rather than cached: the API process and the worker process each
    need to see a change the other made, and a stale settings cache is exactly the "takes
    effect after a restart" bug this table exists to remove. The cost is one indexed read
    of a table with at most `len(EDITABLE_FIELDS)` rows.
    """
    overrides = load(conn)
    if not overrides:
        return base
    return _with(base, overrides)


def _with(base: Settings, changes: Mapping[str, Any]) -> Settings:
    return Settings(**{**base.model_dump(), **changes})


def validate(
    base: Settings,
    lock: ModelsLock,
    changes: Mapping[str, Any],
    *,
    reason: str | None = None,
) -> Settings:
    """Return the settings `changes` would produce, or raise.

    Validation runs against the *candidate* — the stored configuration overlaid with this
    request — not against what is stored now. One PATCH that both enables
    `allow_noncommercial_models` and selects a non-commercial embedder is a coherent final
    state and is accepted; splitting it over two requests would be an artificial gate.

    Raises `UnknownKeyError` for a key outside the editable surface, `ReasonRequiredError`
    when a change's whole justification is its reason, `InvalidValueError` for a value the
    pipeline could not run on, and `StateConflictError` when the value is fine but the
    resulting state is not.
    """
    unknown = sorted(key for key in changes if key not in _EDITABLE)
    if unknown:
        raise UnknownKeyError(
            f"not editable: {', '.join(unknown)}; editable keys are "
            f"{', '.join(EDITABLE_FIELDS)}"
        )
    try:
        candidate = _with(base, changes)
    except ValidationError as exc:
        raise InvalidValueError(_first_error(exc)) from exc

    for key in REASON_REQUIRED_TO_ENABLE:
        turning_on = key in changes and bool(changes[key]) and not getattr(base, key)
        if turning_on and (reason is None or not reason.strip()):
            raise ReasonRequiredError(
                f"turning on {key} needs a reason; it is recorded in the audit log with "
                "the previous and new value, and it is the only record of why this "
                "installation was permitted to run a non-commercially licensed model"
            )

    # Only a model or licence change can alter which models load, and each check costs a
    # SHA-256 over the weights, so the coherence pass is skipped for everything else.
    if _EDITABLE_LICENCE_RELEVANT & changes.keys():
        for field, kind in _MODEL_FIELD_KINDS.items():
            model_id = str(getattr(candidate, field))
            if field in changes:
                # Newly selected: the ordinary refusal, naming the remedy for this model.
                assert_model_usable(lock, model_id, kind=kind, settings=candidate)
            else:
                _assert_active_model_survives(lock, model_id, kind=kind, candidate=candidate)
    return candidate


def _assert_active_model_survives(
    lock: ModelsLock, model_id: str, *, kind: str, candidate: Settings
) -> None:
    """Refuse a change that would strand the model that is already running.

    Turning the licence gate off while a non-commercial model is active is refused rather
    than fixed for the operator. Silently forcing the embedder back would re-embed the whole
    gallery off a checkbox, which is exactly what the Config page's confirm step exists to
    prevent. Sending both keys in one request still works, and then the switch goes through
    the ordinary embedder-change consequences instead of happening as a side effect.
    """
    entry = lock.by_id(model_id)
    if entry is None or entry.commercial_use or candidate.allow_noncommercial_models:
        return
    raise StateConflictError(
        f"allow_noncommercial_models cannot be turned off while {model_id!r} is the active "
        f"{kind}: it is licensed {entry.license!r} (non-commercial) and would no longer "
        f"load. Send {kind}_model set to a commercially licensed model in the same request "
        "to do both at once, or leave the flag on."
    )


def assert_model_usable(
    lock: ModelsLock, model_id: str, *, kind: str, settings: Settings
) -> ModelEntry:
    """Refuse a model this install must not run (invariants 8, 9).

    Checked here rather than at load time because a rejected switch must leave the previous
    model in place. Discovering the mismatch when the worker next builds a session would
    mean the configuration already says one thing and the evidence says another.

    The licence is checked *last*, deliberately. Every other refusal is something no setting
    can fix — not locked, wrong kind, no adapter, weights absent, digest mismatch — so
    reporting one of those first means a licence refusal states the only remaining obstacle.
    A caller drafting a licence change can then trust that turning the flag on is enough.
    """
    entry = lock.by_id(model_id)
    if entry is None:
        raise InvalidValueError(f"model {model_id!r} is not listed in models.lock")
    if entry.kind != kind:
        raise InvalidValueError(f"model {model_id!r} is a {entry.kind}, not a {kind}")
    if entry.kind == "embedder" and entry.dim is None:
        raise InvalidValueError(
            f"models.lock entry {model_id!r} has no dim; embeddings cannot be stored"
        )
    if not registry.has_adapter(model_id, kind=kind):
        raise InvalidValueError(
            f"this build has no {kind} adapter for {model_id!r}; it is listed in "
            "models.lock but nothing here can run it"
        )

    path = settings.models_dir / entry.file
    if not path.is_file():
        raise InvalidValueError(
            f"model {model_id!r} needs {entry.file}, which is not present in "
            f"{settings.models_dir}; provision weights with tools/fetch_models.py "
            "(they are never fetched at runtime, C1)"
        )
    actual = sha256_file(path)
    if actual != entry.sha256:
        raise InvalidValueError(
            f"{entry.file} does not match models.lock (expected {entry.sha256[:12]}, "
            f"found {actual[:12]}); refusing to switch (invariant 8)"
        )

    try:
        # The licence rule itself lives in models_lock, so the config gate and the load
        # gate cannot drift apart. Everything it could raise besides the licence has
        # already been ruled out above.
        return assert_loadable(lock, model_id, settings)
    except (ModelsLockError, ModelLicenseError) as exc:
        raise InvalidValueError(str(exc)) from exc


def blocked_reason(
    lock: ModelsLock, model_id: str, *, kind: str, settings: Settings
) -> str | None:
    """Why `PATCH /api/config` would refuse this model right now, or None if it would not.

    The same `assert_model_usable` the endpoint enforces, so the picker and the endpoint
    cannot disagree about what is selectable. The model that is currently active gets no
    exemption: if the licence gate is off and a non-commercial model is somehow running,
    that has to be visible here rather than quietly fine.
    """
    try:
        assert_model_usable(lock, model_id, kind=kind, settings=settings)
    except InvalidValueError as exc:
        return str(exc)
    return None


def write(
    conn: sqlite3.Connection,
    *,
    changes: Mapping[str, Any],
    previous: Settings,
    reason: str | None,
    actor: str,
) -> None:
    """Persist overrides and audit each one, inside the caller's transaction.

    One `config.change` entry per key, carrying the previous and the new value, because the
    question a reviewer asks later is about a single setting, not about a batch.
    """
    now = audit.now_ts()
    for key in EDITABLE_FIELDS:
        if key not in changes:
            continue
        new_value = changes[key]
        old_value = getattr(previous, key)
        if new_value == old_value:
            continue
        conn.execute(
            "INSERT INTO runtime_config (key, value_json, updated_at, updated_by) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
            "value_json = excluded.value_json, updated_at = excluded.updated_at, "
            "updated_by = excluded.updated_by",
            (key, audit.canonical_json(new_value).decode("utf-8"), now, actor),
        )
        audit.append(
            conn,
            actor=actor,
            action="config.change",
            object_type="config",
            object_id=key,
            payload={
                "key": key,
                "previous": _jsonable(old_value),
                "new": _jsonable(new_value),
                "reason": reason,
            },
        )


def _jsonable(value: Any) -> Any:
    """Audit payloads are canonical JSON; a Path is not, and no editable value needs to be."""
    return value if isinstance(value, str | int | float | bool | None) else str(value)


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}"
