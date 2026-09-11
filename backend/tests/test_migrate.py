"""Migration runner behaviour."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.config import Settings
from app.db.conn import connect
from app.db.migrate import MigrationDriftError, current_version, discover, migrate


def test_migrate_is_idempotent(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    try:
        first = migrate(conn)
        assert [m.version for m in first] == [m.version for m in discover()]
        assert migrate(conn) == []
        assert current_version(conn) == max(m.version for m in discover())
    finally:
        conn.close()


def test_applied_migration_that_changes_on_disk_refuses(
    settings: Settings, tmp_path: Path
) -> None:
    directory = tmp_path / "migrations"
    directory.mkdir()
    migration = directory / "0001_init.sql"
    migration.write_text("CREATE TABLE t (id TEXT PRIMARY KEY);", encoding="utf-8")

    settings.ensure_dirs()
    conn = connect(settings.db_path)
    try:
        assert len(migrate(conn, directory)) == 1
        migration.write_text(
            "CREATE TABLE t (id TEXT PRIMARY KEY, extra TEXT);", encoding="utf-8"
        )
        with pytest.raises(MigrationDriftError, match="changed after it was applied"):
            migrate(conn, directory)
    finally:
        conn.close()


def test_a_failed_migration_leaves_no_partial_schema(
    settings: Settings, tmp_path: Path
) -> None:
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "0001_broken.sql").write_text(
        "CREATE TABLE good (id TEXT PRIMARY KEY);\nCREATE TABLE bad (;", encoding="utf-8"
    )

    settings.ensure_dirs()
    conn = connect(settings.db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            migrate(conn, directory)
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "good" not in tables
        assert current_version(conn) == 0
    finally:
        conn.close()


def test_migration_filenames_must_be_ordered(tmp_path: Path) -> None:
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "init.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match=r"NNNN_name\.sql"):
        discover(directory)
