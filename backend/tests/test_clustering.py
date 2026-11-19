"""M3: unknowns group without enrollment; explicit assignment rematches older media."""

from __future__ import annotations

import sqlite3

import numpy as np
from fastapi.testclient import TestClient

from app import audit
from app.config import Settings
from app.core.vectors import to_blob
from app.db.conn import connect, transaction
from app.db.migrate import migrate
from app.jobs import enqueue
from app.pipeline.clustering import cluster_unknowns
from app.pipeline.matching import rematch
from app.worker import run_once


def seed(conn: sqlite3.Connection) -> None:
    now = audit.now_ts()
    with transaction(conn):
        conn.execute("INSERT INTO models (id,name,version,kind,sha256,license,commercial_use,dim) "
                     "VALUES ('sface-2021dec','SFace','1','embedder',?,'MIT',1,128)", ("a" * 64,))
        conn.execute("INSERT INTO models (id,name,version,kind,sha256,license,commercial_use) "
                     "VALUES ('det','YuNet','1','detector',?,'MIT',1)", ("b" * 64,))
        conn.execute("INSERT INTO threshold_sets "
                     "(id,model_id,t_strong,t_possible,margin,active,created_at) "
                     "VALUES ('threshold','sface-2021dec',0.8,0.3,0.05,1,?)", (now,))
        conn.execute("INSERT INTO cases (id,name,authorization_basis,created_at,created_by) "
                     "VALUES ('case','Test','fixture consent',?,'tester')", (now,))
        for i in range(4):
            media = f"media-{i}"
            track = f"track-{i}"
            detection = f"detection-{i}"
            conn.execute("INSERT INTO media (id,case_id,sha256,kind,path,ingested_at,status) "
                         "VALUES (?,'case',?,'image',? ,?,'done')",
                         (media, f"{i:064x}", f"media/{media}.jpg", now))
            vec = np.zeros(128, dtype=np.float32)
            vec[0 if i < 3 else 1] = 1.0
            conn.execute("INSERT INTO tracks (id,media_id,start_ms,end_ms,"
                         "embedding_mean,embedder_model_id) VALUES (?, ?,0,0,?,'sface-2021dec')",
                         (track, media, to_blob(vec)))
            conn.execute("INSERT INTO detections (id,media_id,track_id,t_ms,frame_idx,det_idx," 
                         "x,y,w,h,landmarks_json,det_score,quality_json,"
                         "crop_sha256,detector_model_id) "
                         "VALUES (?,?,?,0,0,0,1,1,100,100,'[]',0.99,'{}',?,'det')",
                         (detection, media, track, f"{i+10:064x}"))
            conn.execute("UPDATE tracks SET best_detection_id = ? WHERE id = ?", (detection, track))
            conn.execute("INSERT INTO detection_embeddings "
                         "(detection_id,embedding,embedder_model_id,created_at) "
                         "VALUES (?,?,'sface-2021dec',?)", (detection, to_blob(vec), now))


def test_cluster_assign_rematches_past_media(client: TestClient, settings: Settings) -> None:
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        seed(conn)
        queued = client.post("/api/jobs/cluster", json={"scope": "case", "scope_id": "case"})
        assert queued.status_code == 202, queued.text
        job = run_once(conn, settings)
        assert job is not None and job.kind == "cluster"
        assert client.get(f"/api/jobs/{job.id}").json()["status"] == "done"
        listed = client.get("/api/clusters", params={"scope": "case:case"}).json()["items"]
        assert len(listed) == 1
        group = listed[0]
        assert group["size"] == 3
        assert len(group["tracks"]) == 3
        assert conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0] == 0
        assert client.post(f"/api/clusters/{group['id']}/assign", json={
            "new_name": "Reviewed Person", "exclude_track_ids": [],
            "reason": "reviewed stored crops",
        }).status_code == 200
        # A full re-match, queued by the assignment, rescans older media without
        # overwriting the operator's decisions.
        queued_rematch = conn.execute(
            "SELECT id FROM jobs WHERE kind = 'rematch' AND status = 'queued'"
        ).fetchone()
        assert queued_rematch is not None
        rematch(
            conn, settings, embedder_model_id="sface-2021dec",
            execution_provider="CPUExecutionProvider", actor="tester",
        )
        assert conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0] == 3
        operator_count = conn.execute(
            "SELECT COUNT(*) FROM identities WHERE source = 'operator'"
        ).fetchone()[0]
        assert operator_count == 3
        past_matches = conn.execute(
            "SELECT t.id, m.score FROM tracks t "
            "JOIN matches m ON m.track_id = t.id AND m.rank = 1 "
            "WHERE t.id IN ('track-0', 'track-1', 'track-2')"
        ).fetchall()
        assert len(past_matches) == 3
        assert all(float(row["score"]) > 0.9 for row in past_matches)
        assert client.get("/api/clusters", params={"scope": "case:case"}).json()["items"] == []
        assert client.post(f"/api/clusters/{group['id']}/assign", json={
            "new_name": "Duplicate", "exclude_track_ids": [], "reason": "again",
        }).status_code == 409
        assert audit.verify(conn).ok
    finally:
        conn.close()


def test_clustering_does_not_enroll_or_reassign(conn: sqlite3.Connection) -> None:
    seed(conn)
    first_job = enqueue(conn, kind="cluster", actor="tester")
    first = cluster_unknowns(conn, scope="case", scope_id="case", job_id=first_job.id,
                             embedder_model_id="sface-2021dec", actor="tester")
    assert first["clusters"] == 1 and first["noise"] == 1
    second_job = enqueue(conn, kind="cluster", actor="tester")
    second = cluster_unknowns(conn, scope="case", scope_id="case", job_id=second_job.id,
                              embedder_model_id="sface-2021dec", actor="tester")
    assert second["clusters"] == 1
    assert conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0] == 0
