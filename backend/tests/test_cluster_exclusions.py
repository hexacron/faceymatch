"""Excluding a wrong face must not create a claim or enroll its crop."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.db.conn import connect
from app.pipeline.clustering import cluster_unknowns
from tests.test_clustering import seed


def test_exclude_and_stale_assignment(client: TestClient, settings: Settings) -> None:
    conn = connect(settings.db_path)
    try:
        seed(conn)
        queued = client.post("/api/jobs/cluster", json={"scope": "case", "scope_id": "case"})
        assert queued.status_code == 202
        cluster_unknowns(
            conn, scope="case", scope_id="case", job_id=queued.json()["id"],
            embedder_model_id="sface-2021dec", actor="tester",
        )
        cluster = client.get("/api/clusters", params={"scope": "case:case"}).json()["items"][0]
        excluded = cluster["tracks"][0]["track_id"]
        request = {
            "new_name": "Person", "reason": "reviewed crops",
            "exclude_track_ids": [excluded],
        }
        invalid = client.post(f"/api/clusters/{cluster['id']}/assign", json={
            **request, "exclude_track_ids": ["not-a-member"],
        })
        assert invalid.status_code == 409
        assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0
        assigned = client.post(f"/api/clusters/{cluster['id']}/assign", json=request)
        assert assigned.status_code == 200, assigned.text
        assert assigned.json()["assigned"] == 2
        assert assigned.json()["templates_created"] == 2
        identity = conn.execute(
            "SELECT 1 FROM identities WHERE track_id = ?", (excluded,)
        ).fetchone()
        assert identity is None
        cluster_id = conn.execute(
            "SELECT cluster_id FROM tracks WHERE id = ?", (excluded,)
        ).fetchone()[0]
        assert cluster_id is None
    finally:
        conn.close()
