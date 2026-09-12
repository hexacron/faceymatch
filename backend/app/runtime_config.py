"""Durable runtime overrides for the editable subset of `Settings` (spec 6.3, 8, 9).

The environment (and `.env`) is the base configuration. This module is a sparse overlay on
top of it, stored in the `runtime_config` table, so that an operator change is a database
write with an audit entry beside it rather than an untraceable edit to a file.

Three rules hold the shape together:

- `EDITABLE_FIELDS` is the whole editable surface. Anything else is refused by name, and a
  row written for it by hand is ignored on read. That is what keeps
  `allow_noncommercial_models` env-only: a licensing gate the UI can flip is not a gate
  (invariant 9, C7). The execution provider is likewise env-only, because a threshold set
  is only reproducible on the provider it was calibrated on (spec 10).
- Validation happens by constructing a `Settings`, so the bounds live in one place
  (`app/config.py`) and the wire cannot accept a value the pipeline would reject.
- A model id is refused unless `models.lock` lists it, its weight file is present, its
  SHA-256 still matches, its kind fits the slot, and its licence is cleared for this
  install (invariants 8 and 9).

Thresholds are deliberately absent: they come from calibration only (spec 10).
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
    "allow_noncommercial_models",
    "operator_name",
    "db_path",
    "models_dir",
    "max_upload_bytes",
    "fpir_target",
    "embed_k",
    "rematch_block_size",
)

_EDITABLE = frozenset(EDITABLE_FIELDS)

# Which editable key selects which kind of model, so a detector cannot be set as an embedder.
_MODEL_FIELD_KINDS: dict[str, str] = {
    "detector_model": "detector",
    "embedder_model": "embedder",
}


class UnknownKeyError(ValueError):
    """A requested key is not part of the editable configuration surface."""


class InvalidValueError(ValueError):
    """A requested value is out of range, or names a model this install may not load."""


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


def validate(base: Settings, lock: ModelsLock, changes: Mapping[str, Any]) -> Settings:
    """Return the settings `changes` would produce, or raise.

    `UnknownKeyError` for a key outside the editable surface, `InvalidValueError` for a
    value the pipeline could not run on.
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

    for field, kind in _MODEL_FIELD_KINDS.items():
        if field in changes:
            assert_model_usable(lock, str(changes[field]), kind=kind, settings=candidate)
    return candidate


def assert_model_usable(
    lock: ModelsLock, model_id: str, *, kind: str, settings: Settings
) -> ModelEntry:
    """Refuse a model this install must not run (invariants 8, 9).

    Checked here rather than at load time because a rejected switch must leave the previous
    model in place. Discovering the mismatch when the worker next builds a session would
    mean the configuration already says one thing and the evidence says another.
    """
    try:
        entry = assert_loadable(lock, model_id, settings)
    except ModelsLockError as exc:
        raise InvalidValueError(str(exc)) from exc
    except ModelLicenseError as exc:
        raise InvalidValueError(str(exc)) from exc

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
    return entry


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
