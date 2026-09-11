"""Append-only, hash-chained audit log (invariant 6, spec section 9).

hash = SHA256(prev_hash_ascii || canonical_json(entry))

The canonical form is frozen: UTF-8, keys sorted, separators (",", ":"), allow_nan=False.
Changing it invalidates every stored chain, so it must never drift.

`seq` is assigned inside the writing transaction (not by AUTOINCREMENT) because it is part
of the hashed entry. All writes use BEGIN IMMEDIATE, which makes the head-read and the
append a single exclusive step across the API process and the job worker.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

GENESIS_PREV_HASH = "0" * 64


class AuditChainError(RuntimeError):
    """The stored chain does not verify."""


@dataclass(frozen=True, slots=True)
class AuditEntry:
    seq: int
    ts: str
    actor: str
    case_id: str | None
    action: str
    object_type: str
    object_id: str | None
    payload: dict[str, Any]
    prev_hash: str
    hash: str


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def now_ts() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def entry_digest(
    *,
    seq: int,
    ts: str,
    actor: str,
    case_id: str | None,
    action: str,
    object_type: str,
    object_id: str | None,
    payload: dict[str, Any],
    prev_hash: str,
) -> str:
    body = canonical_json(
        {
            "seq": seq,
            "ts": ts,
            "actor": actor,
            "case_id": case_id,
            "action": action,
            "object_type": object_type,
            "object_id": object_id,
            "payload": payload,
        }
    )
    return hashlib.sha256(prev_hash.encode("ascii") + body).hexdigest()


def head(conn: sqlite3.Connection) -> tuple[int, str]:
    """Return (seq, hash) of the chain head, or (0, GENESIS_PREV_HASH) when empty."""
    row = conn.execute(
        "SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return 0, GENESIS_PREV_HASH
    return int(row["seq"]), str(row["hash"])


def append(
    conn: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    object_type: str,
    object_id: str | None = None,
    case_id: str | None = None,
    payload: dict[str, Any] | None = None,
    ts: str | None = None,
) -> AuditEntry:
    """Append one entry. MUST be called inside an open BEGIN IMMEDIATE transaction.

    Requiring the caller's transaction is deliberate: an audited write and its audit entry
    commit together or not at all.
    """
    if not conn.in_transaction:
        raise RuntimeError(
            "audit.append must run inside db.conn.transaction() so the write and its "
            "audit entry commit atomically"
        )

    prev_seq, prev_hash = head(conn)
    entry_payload = payload if payload is not None else {}
    entry = AuditEntry(
        seq=prev_seq + 1,
        ts=ts if ts is not None else now_ts(),
        actor=actor,
        case_id=case_id,
        action=action,
        object_type=object_type,
        object_id=object_id,
        payload=entry_payload,
        prev_hash=prev_hash,
        hash="",
    )
    digest = entry_digest(
        seq=entry.seq,
        ts=entry.ts,
        actor=entry.actor,
        case_id=entry.case_id,
        action=entry.action,
        object_type=entry.object_type,
        object_id=entry.object_id,
        payload=entry.payload,
        prev_hash=entry.prev_hash,
    )
    conn.execute(
        "INSERT INTO audit_log "
        "(seq, ts, actor, case_id, action, object_type, object_id, payload_json, "
        " prev_hash, hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            entry.seq,
            entry.ts,
            entry.actor,
            entry.case_id,
            entry.action,
            entry.object_type,
            entry.object_id,
            canonical_json(entry.payload).decode("utf-8"),
            entry.prev_hash,
            digest,
        ),
    )
    return AuditEntry(
        seq=entry.seq,
        ts=entry.ts,
        actor=entry.actor,
        case_id=entry.case_id,
        action=entry.action,
        object_type=entry.object_type,
        object_id=entry.object_id,
        payload=entry.payload,
        prev_hash=entry.prev_hash,
        hash=digest,
    )


def row_to_entry(row: sqlite3.Row) -> AuditEntry:
    payload: dict[str, Any] = json.loads(row["payload_json"])
    return AuditEntry(
        seq=int(row["seq"]),
        ts=str(row["ts"]),
        actor=str(row["actor"]),
        case_id=row["case_id"],
        action=str(row["action"]),
        object_type=str(row["object_type"]),
        object_id=row["object_id"],
        payload=payload,
        prev_hash=str(row["prev_hash"]),
        hash=str(row["hash"]),
    )


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    checked: int
    head_seq: int
    head_hash: str | None
    bad_seq: int | None
    reason: str | None


def verify(conn: sqlite3.Connection) -> VerifyResult:
    """Recompute the whole chain from genesis.

    Detects mutated payloads, re-ordered or missing seq values, and spliced entries.
    """
    expected_prev = GENESIS_PREV_HASH
    expected_seq = 1
    checked = 0
    last_hash: str | None = None

    for row in conn.execute(
        "SELECT seq, ts, actor, case_id, action, object_type, object_id, payload_json, "
        "prev_hash, hash FROM audit_log ORDER BY seq"
    ):
        seq = int(row["seq"])
        if seq != expected_seq:
            return VerifyResult(
                ok=False,
                checked=checked,
                head_seq=checked,
                head_hash=last_hash,
                bad_seq=seq,
                reason=f"sequence gap: expected {expected_seq}, found {seq}",
            )
        if str(row["prev_hash"]) != expected_prev:
            return VerifyResult(
                ok=False,
                checked=checked,
                head_seq=checked,
                head_hash=last_hash,
                bad_seq=seq,
                reason="prev_hash does not match the previous entry's hash",
            )
        entry = row_to_entry(row)
        digest = entry_digest(
            seq=entry.seq,
            ts=entry.ts,
            actor=entry.actor,
            case_id=entry.case_id,
            action=entry.action,
            object_type=entry.object_type,
            object_id=entry.object_id,
            payload=entry.payload,
            prev_hash=entry.prev_hash,
        )
        if digest != entry.hash:
            return VerifyResult(
                ok=False,
                checked=checked,
                head_seq=checked,
                head_hash=last_hash,
                bad_seq=seq,
                reason="entry hash does not match its contents",
            )
        expected_prev = entry.hash
        last_hash = entry.hash
        expected_seq += 1
        checked += 1

    return VerifyResult(
        ok=True,
        checked=checked,
        head_seq=checked,
        head_hash=last_hash,
        bad_seq=None,
        reason=None,
    )
