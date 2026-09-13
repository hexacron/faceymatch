"""Request-scoped dependencies.

One SQLite connection per (thread, database), held for the life of the process. Opening a
connection re-runs every PRAGMA in `db.conn.connect` and throws away the 64 MiB page cache
it just declared, which is a poor trade for a local single-operator install serving a live
loop at several requests a second. FastAPI runs sync endpoints in a threadpool and a
`sqlite3` connection is not safe to use from two threads at once, so the pool is
thread-local rather than global.

Writes still serialise through `transaction()`'s BEGIN IMMEDIATE, so the single-writer
property the audit chain depends on is exactly as it was.

`SettingsDep` resolves the *effective* configuration — the environment overlaid with the
durable overrides in `runtime_config` — on every request. `app.state.settings` stays the
base, and nothing outside this module reads it, so a change made through
`PATCH /api/config` is visible to the next request instead of the next restart.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request

from app import runtime_config
from app.config import Settings
from app.db.conn import connect
from app.models_lock import ModelsLock

_local = threading.local()
# Every per-thread pool dict ever handed out, so `reset_pool` can reach the ones this
# thread does not own. The dicts are the same objects the thread-locals hold, so clearing
# one here clears that thread's view of it too.
_pools_lock = threading.Lock()
_pools: list[dict[str, sqlite3.Connection]] = []


def _pooled(db_path: Path) -> sqlite3.Connection:
    """This thread's connection to `db_path`, opening it the first time."""
    pool: dict[str, sqlite3.Connection] | None = getattr(_local, "pool", None)
    if pool is None:
        pool = {}
        _local.pool = pool
        with _pools_lock:
            _pools.append(pool)
    key = str(db_path.resolve())
    conn = pool.get(key)
    if conn is None:
        conn = connect(db_path)
        pool[key] = conn
    return conn


def reset_pool() -> None:
    """Close and forget every pooled connection.

    For tests: each one gets its own temporary database, and a pooled connection outliving
    the file it was opened on would serve the next test another test's rows. Only safe with
    no request in flight, which is exactly when a test fixture runs.
    """
    with _pools_lock:
        pools = list(_pools)
        _pools.clear()
    for pool in pools:
        for conn in pool.values():
            conn.close()
        pool.clear()
    _local.pool = None


def db_conn(request: Request) -> Iterator[sqlite3.Connection]:
    settings: Settings = request.app.state.settings
    conn = _pooled(settings.db_path)
    try:
        yield conn
    finally:
        # A handler that failed outside `transaction()` would otherwise hand the next
        # request a connection holding the write lock, and the worker would block on it
        # until the process ended.
        if conn.in_transaction:
            conn.execute("ROLLBACK")


ConnDep = Annotated[sqlite3.Connection, Depends(db_conn)]


def settings_of(request: Request, conn: ConnDep) -> Settings:
    base: Settings = request.app.state.settings
    return runtime_config.effective(conn, base)


def models_lock_of(request: Request) -> ModelsLock:
    lock: ModelsLock = request.app.state.models_lock
    return lock


SettingsDep = Annotated[Settings, Depends(settings_of)]
LockDep = Annotated[ModelsLock, Depends(models_lock_of)]
