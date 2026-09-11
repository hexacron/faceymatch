"""Request-scoped dependencies.

One SQLite connection per request: cheap for a local single-operator install, and it keeps
write transactions short so the worker is never starved of the write lock.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.db.conn import connect
from app.models_lock import ModelsLock


def settings_of(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def models_lock_of(request: Request) -> ModelsLock:
    lock: ModelsLock = request.app.state.models_lock
    return lock


def db_conn(request: Request) -> Iterator[sqlite3.Connection]:
    settings: Settings = request.app.state.settings
    conn = connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


SettingsDep = Annotated[Settings, Depends(settings_of)]
LockDep = Annotated[ModelsLock, Depends(models_lock_of)]
ConnDep = Annotated[sqlite3.Connection, Depends(db_conn)]
