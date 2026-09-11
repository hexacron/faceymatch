"""Migration runner. Migrations are the single source of truth for the schema.

Applied migrations are recorded with the SHA-256 of the file that was applied. If a file
changes after the fact, startup refuses: the database no longer matches the code that is
supposed to describe it.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_NAME_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class MigrationDriftError(RuntimeError):
    """An already-applied migration file has changed on disk."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    path: Path
    sql: str
    sha256: str


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _NAME_RE.match(path.name)
        if match is None:
            raise ValueError(f"migration filename must be NNNN_name.sql, got {path.name!r}")
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=int(match.group(1)),
                name=match.group(2),
                path=path,
                sql=sql,
                sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    versions = [m.version for m in migrations]
    if len(set(versions)) != len(versions):
        raise ValueError(f"duplicate migration versions: {versions}")
    return migrations


def _ensure_bookkeeping(conn: sqlite3.Connection) -> None:
    """The ledger belongs to the runner, not to any migration.

    If migration 0001 created it, a fresh runner against a different migration set would
    have nowhere to record what it applied.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY,"
        " name TEXT NOT NULL,"
        " sha256 TEXT NOT NULL,"
        " applied_at TEXT NOT NULL)"
    )


def _applied(conn: sqlite3.Connection) -> dict[int, str]:
    _ensure_bookkeeping(conn)
    rows = conn.execute("SELECT version, sha256 FROM schema_migrations").fetchall()
    return {int(row["version"]): str(row["sha256"]) for row in rows}


def current_version(conn: sqlite3.Connection) -> int:
    applied = _applied(conn)
    return max(applied) if applied else 0


def migrate(conn: sqlite3.Connection, directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Apply pending migrations. Returns the migrations applied by this call."""
    migrations = discover(directory)
    applied = _applied(conn)

    for migration in migrations:
        known = applied.get(migration.version)
        if known is not None and known != migration.sha256:
            raise MigrationDriftError(
                f"migration {migration.version:04d}_{migration.name} changed after it was "
                f"applied (recorded {known[:12]}, file {migration.sha256[:12]})"
            )

    pending = [m for m in migrations if m.version not in applied]
    if not pending:
        return []

    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for migration in pending:
        # executescript() implicitly commits any pending transaction, so transaction control
        # must live inside the script itself for the DDL and its bookkeeping row to be atomic.
        # Scripts take no bound parameters; every interpolated value below is validated
        # (version int, name ^[a-z0-9_]+$, sha256 hex, timestamp generated here).
        if not _HEX64_RE.match(migration.sha256):
            raise ValueError(f"non-hex migration digest: {migration.sha256!r}")
        bookkeeping = (

            # executescript() accepts no bound parameters.
            "INSERT INTO schema_migrations (version, name, sha256, applied_at) VALUES "  # noqa: S608
            f"({migration.version:d}, '{migration.name}', '{migration.sha256}', '{now}');"
        )
        try:
            conn.executescript(f"BEGIN IMMEDIATE;\n{migration.sql}\n{bookkeeping}\nCOMMIT;")
        except BaseException:
            # A statement-level error aborts the script mid-way with the transaction still
            # open: roll back so no partial DDL survives and the write lock is released.
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return pending
