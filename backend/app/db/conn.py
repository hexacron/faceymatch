"""SQLite connection policy.

Every write in this system goes through `transaction()`, which opens BEGIN IMMEDIATE. That
gives the audit chain a single writer at a time across the API process and the job worker:
the chain head is read and the new entry appended inside one exclusive write transaction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

BUSY_TIMEOUT_MS = 10_000


class ExtensionLoadingUnavailableError(RuntimeError):
    """The running interpreter's sqlite3 was built without loadable extension support."""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        isolation_level=None,  # explicit transactions only
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Exclusive write transaction. Nesting is a bug, so it is rejected."""
    if conn.in_transaction:
        raise RuntimeError("transaction() is already open on this connection")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def assert_extension_loading_available() -> None:
    """Fail fast when sqlite-vec could never be loaded.

    macOS system Python ships a sqlite3 without `enable_load_extension`. uv-managed and
    Homebrew CPython builds have it. Checked at startup so the failure is one clear line
    instead of an import error deep in the vector store.
    """
    probe = sqlite3.connect(":memory:")
    try:
        if not hasattr(probe, "enable_load_extension"):
            raise ExtensionLoadingUnavailableError(
                "this Python's sqlite3 has no enable_load_extension; sqlite-vec cannot load. "
                "Use the uv-managed interpreter (`uv sync`), not the macOS system Python."
            )
    finally:
        probe.close()


def load_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into a connection."""
    import sqlite_vec

    assert_extension_loading_available()
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
