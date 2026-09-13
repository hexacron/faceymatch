"""The request-scoped SQLite connection: pooled per thread, never leaked across requests."""

from __future__ import annotations

import sqlite3
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import ConnDep
from app.config import Settings
from app.db.conn import connect
from app.db.migrate import migrate


def _probe_app(settings: Settings) -> tuple[FastAPI, list[tuple[int, int]]]:
    """An app whose only route reports which thread it ran on and which connection it got."""
    app = FastAPI()
    app.state.settings = settings
    seen: list[tuple[int, int]] = []

    @app.get("/probe")
    def probe(conn: ConnDep) -> dict[str, int]:
        seen.append((threading.get_ident(), id(conn)))
        return {"cases": len(conn.execute("SELECT id FROM cases").fetchall())}

    @app.get("/leak")
    def leak(conn: ConnDep) -> dict[str, bool]:
        # A handler that opens a write transaction and never closes it. Pathological, and
        # exactly why the dependency cannot simply hand the same connection on unchecked.
        conn.execute("BEGIN IMMEDIATE")
        return {"in_transaction": conn.in_transaction}

    return app, seen


def test_a_thread_keeps_one_connection_across_requests(settings: Settings) -> None:
    """Per-request connect() would re-run every PRAGMA and drop the page cache each time."""
    settings.ensure_dirs()
    bootstrap = connect(settings.db_path)
    migrate(bootstrap)
    bootstrap.close()

    app, seen = _probe_app(settings)
    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/probe").status_code == 200

    assert len(seen) == 3
    threads = {ident for ident, _ in seen}
    connections = {handle for _, handle in seen}
    # One connection per thread that served a request — not one per request.
    assert len(connections) == len(threads)


def test_a_leaked_transaction_does_not_poison_the_next_request(settings: Settings) -> None:
    """A pooled connection stuck in BEGIN IMMEDIATE would hold the write lock forever."""
    settings.ensure_dirs()
    bootstrap = connect(settings.db_path)
    migrate(bootstrap)
    bootstrap.close()

    app, _ = _probe_app(settings)
    with TestClient(app) as client:
        assert client.get("/leak").json()["in_transaction"] is True
        assert client.get("/probe").status_code == 200

    # And the write lock really was released: another connection can take it.
    other = connect(settings.db_path)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.execute("ROLLBACK")
    except sqlite3.OperationalError as exc:  # pragma: no cover - the failure this guards
        raise AssertionError(f"write lock was still held: {exc}") from exc
    finally:
        other.close()
