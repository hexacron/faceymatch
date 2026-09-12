"""Request-scoped dependencies.

One SQLite connection per request: cheap for a local single-operator install, and it keeps
write transactions short so the worker is never starved of the write lock.

`SettingsDep` resolves the *effective* configuration — the environment overlaid with the
durable overrides in `runtime_config` — on every request. `app.state.settings` stays the
base, and nothing outside this module reads it, so a change made through
`PATCH /api/config` is visible to the next request instead of the next restart.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from app import runtime_config
from app.config import Settings
from app.db.conn import connect
from app.models_lock import ModelsLock


def db_conn(request: Request) -> Iterator[sqlite3.Connection]:
    settings: Settings = request.app.state.settings
    conn = connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


ConnDep = Annotated[sqlite3.Connection, Depends(db_conn)]


def settings_of(request: Request, conn: ConnDep) -> Settings:
    base: Settings = request.app.state.settings
    return runtime_config.effective(conn, base)


def models_lock_of(request: Request) -> ModelsLock:
    lock: ModelsLock = request.app.state.models_lock
    return lock


SettingsDep = Annotated[Settings, Depends(settings_of)]
LockDep = Annotated[ModelsLock, Depends(models_lock_of)]
