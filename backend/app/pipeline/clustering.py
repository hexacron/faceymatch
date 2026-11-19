"""Cluster unassigned stored track means for operator review (spec 6.7)."""

from __future__ import annotations

import sqlite3
from typing import Literal

import numpy as np
from sklearn.cluster import HDBSCAN

from app import audit
from app.core import vectors
from app.db.conn import transaction
from app.ids import new_id

Scope = Literal["media", "job", "case"]


def cluster_unknowns(
    conn: sqlite3.Connection, *, scope: Scope, scope_id: str, job_id: str,
    embedder_model_id: str, actor: str,
) -> dict[str, int]:
    """Persist groups only; neither identities nor gallery templates are created here."""
    if scope == "job":
        source = conn.execute(
            "SELECT json_extract(params_json, '$.media_id') AS media_id FROM jobs "
            "WHERE id = ? AND kind = 'process' AND status = 'done'", (scope_id,)
        ).fetchone()
        if source is None or source["media_id"] is None:
            raise LookupError("completed process job not found")
        media_id = str(source["media_id"])
    elif scope == "media":
        media_id = scope_id
    else:
        media_id = None
    if scope == "case":
        exists = conn.execute("SELECT 1 FROM cases WHERE id = ?", (scope_id,)).fetchone()
    else:
        exists = conn.execute("SELECT 1 FROM media WHERE id = ?", (media_id,)).fetchone()
    if exists is None:
        raise LookupError(f"{scope} not found")

    dim_row = conn.execute(
        "SELECT dim FROM models WHERE id = ? AND kind = 'embedder'", (embedder_model_id,)
    ).fetchone()
    if dim_row is None or dim_row["dim"] is None:
        raise ValueError("active embedder is not registered")
    scope_key = f"{scope}:{scope_id}"
    # Exclude any accepted identity, explicit rejection, and membership in a labelled group.
    # A rerun may replace only unlabelled groups in this same scope.
    rows = conn.execute(
        "SELECT t.id, t.embedding_mean FROM tracks t JOIN media m ON m.id = t.media_id "
        "LEFT JOIN clusters c ON c.id = t.cluster_id "
        "WHERE t.embedder_model_id = ? AND t.embedding_mean IS NOT NULL "
        "AND (? IS NULL OR t.media_id = ?) AND (? IS NULL OR m.case_id = ?) "
        "AND NOT EXISTS (SELECT 1 FROM identities i WHERE i.track_id = t.id) "
        "AND NOT EXISTS (SELECT 1 FROM identifications i WHERE i.track_id = t.id "
        "AND i.decision = 'reject') "
        "AND (c.id IS NULL OR (c.scope = ? AND c.label_person_id IS NULL)) "
        "ORDER BY t.id",
        (embedder_model_id, media_id, media_id,
         scope_id if scope == "case" else None, scope_id if scope == "case" else None,
         scope_key),
    ).fetchall()
    labels = np.full(len(rows), -1, dtype=np.int64)
    if len(rows) >= 3:
        matrix = vectors.stack_blobs(
            [bytes(row["embedding_mean"]) for row in rows], int(dim_row["dim"])
        )
        # scikit-learn HDBSCAN accepts precomputed cosine distances; float32 unit vectors
        # may put the diagonal slightly above zero, so clamp and reset it explicitly.
        distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
        labels = HDBSCAN(
            min_cluster_size=3, min_samples=1, metric="precomputed",
            allow_single_cluster=True, copy=False,
        ).fit_predict(distances)
    groups: dict[int, list[str]] = {}
    for row, label in zip(rows, labels, strict=True):
        if label >= 0:
            groups.setdefault(int(label), []).append(str(row["id"]))
    with transaction(conn):
        old = conn.execute(
            "SELECT id FROM clusters WHERE scope = ? AND label_person_id IS NULL", (scope_key,)
        ).fetchall()
        for row in old:
            conn.execute("UPDATE tracks SET cluster_id = NULL WHERE cluster_id = ?", (row["id"],))
            conn.execute("DELETE FROM clusters WHERE id = ?", (row["id"],))
        for members in groups.values():
            cluster_id = new_id()
            conn.execute(
                "INSERT INTO clusters (id, job_id, scope, size, created_at) VALUES (?, ?, ?, ?, ?)",
                (cluster_id, job_id, scope_key, len(members), audit.now_ts()),
            )
            conn.executemany(
                "UPDATE tracks SET cluster_id = ? WHERE id = ?",
                [(cluster_id, track_id) for track_id in members],
            )
        count = sum(len(members) for members in groups.values())
        audit.append(
            conn, actor=actor, action="cluster.create", object_type="job", object_id=job_id,
            payload={"scope": scope_key, "clusters": len(groups), "tracks": count},
        )
    return {"clusters": len(groups), "tracks": count, "noise": len(rows) - count}
