"""API surface used by the M0 operator shell."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings


def test_healthz_reports_schema_models_and_chain_head(client: TestClient) -> None:
    body = client.get("/api/healthz").json()
    assert body["status"] == "ok"
    assert body["migration_version"] == 1
    assert body["embedder"]["model_id"] == "sface-2021dec"
    # No weights provisioned in a fresh install: C7 banner shows "not provisioned".
    assert body["embedder"]["present"] is False
    assert body["embedder"]["license"] is None
    assert body["allow_noncommercial_models"] is False
    assert body["threshold_set"] is None
    assert body["audit_head_seq"] >= 1  # the migration itself is audited
    assert body["audit_head_hash"] is not None


def test_case_creation_is_audited(client: TestClient) -> None:
    created = client.post(
        "/api/cases", json={"name": "Op Kingfisher", "authorization_basis": "warrant 12/4"}
    )
    assert created.status_code == 201
    case = created.json()

    fetched = client.get(f"/api/cases/{case['id']}").json()
    assert fetched["name"] == "Op Kingfisher"
    assert fetched["authorization_basis"] == "warrant 12/4"

    entries = client.get("/api/audit").json()["entries"]
    case_entries = [e for e in entries if e["action"] == "case.create"]
    assert len(case_entries) == 1
    assert case_entries[0]["object_id"] == case["id"]
    assert case_entries[0]["case_id"] == case["id"]
    assert case_entries[0]["payload"]["name"] == "Op Kingfisher"


def test_case_creation_requires_an_authorization_basis(client: TestClient) -> None:
    response = client.post("/api/cases", json={"name": "No basis"})
    assert response.status_code == 422


def test_missing_case_is_404(client: TestClient) -> None:
    assert client.get("/api/cases/does-not-exist").status_code == 404


def test_audit_pagination_is_inclusive_and_reports_the_head(client: TestClient) -> None:
    for i in range(5):
        client.post(
            "/api/cases", json={"name": f"case {i}", "authorization_basis": "warrant"}
        )

    page = client.get("/api/audit", params={"from_seq": 2, "limit": 2}).json()
    assert [e["seq"] for e in page["entries"]] == [2, 3]
    assert page["next_seq"] == 4
    assert page["head_seq"] >= 6

    tail = client.get("/api/audit", params={"from_seq": page["head_seq"]}).json()
    assert len(tail["entries"]) == 1
    assert tail["next_seq"] is None
    assert tail["entries"][0]["prev_hash"] != tail["entries"][0]["hash"]


def test_audit_limit_is_bounded(client: TestClient) -> None:
    assert client.get("/api/audit", params={"limit": 5000}).status_code == 422


def test_audit_verify_job_can_be_queued_and_read_back(client: TestClient) -> None:
    queued = client.post("/api/jobs/audit_verify")
    assert queued.status_code == 202
    job = queued.json()
    assert job["kind"] == "audit_verify"
    assert job["status"] == "queued"

    fetched = client.get(f"/api/jobs/{job['id']}").json()
    assert fetched["id"] == job["id"]
    assert client.get("/api/jobs/nope").status_code == 404


def test_frontend_build_is_served_at_root_when_present(settings: Settings) -> None:
    settings.frontend_dist.mkdir(parents=True, exist_ok=True)
    (settings.frontend_dist / "index.html").write_text("<!doctype html>ok", encoding="utf-8")

    from app.main import create_app

    with TestClient(create_app(settings)) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "ok" in root.text
        # The API still resolves with the static mount in place.
        assert client.get("/api/healthz").status_code == 200
