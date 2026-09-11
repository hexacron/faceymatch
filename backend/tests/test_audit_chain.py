"""M0 exit test and the audit chain's tamper-evidence properties (invariant 6)."""

from __future__ import annotations

import sqlite3
from itertools import pairwise

import pytest

from app import audit
from app.db.conn import transaction


def test_chain_verifies_after_1000_writes(conn: sqlite3.Connection) -> None:
    """M0 exit test."""
    for i in range(1000):
        with transaction(conn):
            audit.append(
                conn,
                actor="tester",
                action="test.write",
                object_type="probe",
                object_id=str(i),
                payload={"i": i, "unicode": "ü", "nested": {"b": 1, "a": [1, 2]}},
            )

    result = audit.verify(conn)
    assert result.ok
    assert result.checked == 1000
    assert result.head_seq == 1000
    assert result.bad_seq is None

    first = conn.execute("SELECT prev_hash FROM audit_log WHERE seq = 1").fetchone()
    assert first["prev_hash"] == audit.GENESIS_PREV_HASH


def test_each_entry_links_to_the_previous_hash(conn: sqlite3.Connection) -> None:
    for i in range(5):
        with transaction(conn):
            audit.append(
                conn, actor="tester", action="a", object_type="t", object_id=str(i)
            )

    rows = conn.execute("SELECT seq, prev_hash, hash FROM audit_log ORDER BY seq").fetchall()
    for previous, current in pairwise(rows):
        assert current["prev_hash"] == previous["hash"]


def test_verify_detects_a_mutated_payload(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        audit.append(conn, actor="tester", action="a", object_type="t", payload={"v": 1})
    with transaction(conn):
        audit.append(conn, actor="tester", action="b", object_type="t", payload={"v": 2})

    # Tamper behind the trigger's back: rewrite the row in a copy of the table.
    conn.execute("PRAGMA writable_schema = OFF")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE audit_log SET payload_json = '{\"v\":99}' WHERE seq = 1")

    # Same check with the trigger dropped, proving verify() and not just the trigger
    # is what makes tampering detectable.
    conn.execute("DROP TRIGGER audit_log_no_update")
    conn.execute("UPDATE audit_log SET payload_json = '{\"v\":99}' WHERE seq = 1")

    result = audit.verify(conn)
    assert not result.ok
    assert result.bad_seq == 1
    assert result.reason == "entry hash does not match its contents"


def test_verify_detects_a_deleted_entry(conn: sqlite3.Connection) -> None:
    for i in range(3):
        with transaction(conn):
            audit.append(
                conn, actor="tester", action="a", object_type="t", object_id=str(i)
            )

    conn.execute("DROP TRIGGER audit_log_no_delete")
    conn.execute("DELETE FROM audit_log WHERE seq = 2")

    result = audit.verify(conn)
    assert not result.ok
    assert result.bad_seq == 3
    assert result.reason is not None
    assert "sequence gap" in result.reason


def test_append_requires_an_open_transaction(conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError, match="transaction"):
        audit.append(conn, actor="tester", action="a", object_type="t")


def test_canonical_json_is_key_order_independent() -> None:
    assert audit.canonical_json({"b": 1, "a": 2}) == audit.canonical_json({"a": 2, "b": 1})
    assert audit.canonical_json({"a": "ü"}) == b'{"a":"\xc3\xbc"}'


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range"):
        audit.canonical_json({"score": float("nan")})


def test_head_of_an_empty_chain_is_genesis(conn: sqlite3.Connection) -> None:
    assert audit.head(conn) == (0, audit.GENESIS_PREV_HASH)
