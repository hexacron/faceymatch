"""Operator CLI: `uv run python -m app.cli <command>`."""

from __future__ import annotations

import argparse
import json
import sys

from app import audit, runtime_config
from app.config import get_settings
from app.db.conn import connect
from app.db.migrate import current_version, migrate
from app.jobs import enqueue
from app.main import startup_checks


def _cmd_migrate() -> int:
    settings = get_settings()
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    try:
        applied = migrate(conn)
        for migration in applied:
            print(f"applied {migration.version:04d}_{migration.name}")
        print(f"schema version {current_version(conn)}")
    finally:
        conn.close()
    return 0


def _cmd_check() -> int:
    settings = get_settings()
    lock = startup_checks(settings)
    print(f"startup checks passed; {len(lock.models)} model(s) in models.lock")
    return 0


def _cmd_verify_audit() -> int:
    settings = get_settings()
    conn = connect(settings.db_path)
    try:
        result = audit.verify(conn)
    finally:
        conn.close()
    print(
        json.dumps(
            {
                "ok": result.ok,
                "checked": result.checked,
                "head_seq": result.head_seq,
                "head_hash": result.head_hash,
                "bad_seq": result.bad_seq,
                "reason": result.reason,
            },
            indent=2,
        )
    )
    return 0 if result.ok else 1


def _cmd_enqueue_audit_verify() -> int:
    settings = get_settings()
    conn = connect(settings.db_path)
    try:
        job = enqueue(conn, kind="audit_verify", actor=settings.operator_name)
    finally:
        conn.close()
    print(job.id)
    return 0


def _cmd_enqueue_reembed() -> int:
    """Spec 6.3: a model switch re-embeds templates and tracks, and that is a CLI job too."""
    settings = get_settings()
    conn = connect(settings.db_path)
    try:
        effective = runtime_config.effective(conn, settings)
        job = enqueue(
            conn,
            kind="reembed",
            actor=effective.operator_name,
            params={
                "embedder_model_id": effective.embedder_model,
                "reason": "cli_request",
            },
        )
    finally:
        conn.close()
    print(job.id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply pending migrations")
    sub.add_parser("check", help="run startup checks (schema, models.lock)")
    sub.add_parser("verify-audit", help="recompute the audit hash chain now")
    sub.add_parser("enqueue-audit-verify", help="queue chain verification for the worker")
    sub.add_parser(
        "enqueue-reembed",
        help="queue a re-embed of stored crops, track means and templates (spec 6.3)",
    )

    args = parser.parse_args(argv)
    match args.command:
        case "migrate":
            return _cmd_migrate()
        case "check":
            return _cmd_check()
        case "verify-audit":
            return _cmd_verify_audit()
        case "enqueue-audit-verify":
            return _cmd_enqueue_audit_verify()
        case "enqueue-reembed":
            return _cmd_enqueue_reembed()
        case _:  # pragma: no cover - argparse rejects unknown commands
            parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
