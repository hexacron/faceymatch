"""API surface used by the M0 operator shell."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings


def test_healthz_reports_schema_models_and_chain_head(client: TestClient) -> None:
    body = client.get("/api/healthz").json()
    assert body["status"] == "ok"
    assert body["migration_version"] == 2
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


def test_authorization_basis_can_be_corrected_and_the_old_text_stays_in_the_chain(
    client: TestClient,
) -> None:
    """Section 12 makes this field the record justifying biometric processing, so a wrong
    one must be correctable — and the correction must show what it used to say."""
    case = client.post(
        "/api/cases",
        json={"name": "Op Kingfisher", "authorization_basis": "personal test images only"},
    ).json()

    amended = client.patch(
        f"/api/cases/{case['id']}",
        json={
            "authorization_basis": "warrant 12/4, investigative",
            "reason": "original text understated the material",
        },
    )

    assert amended.status_code == 200
    assert amended.json() == {**case, "authorization_basis": "warrant 12/4, investigative"}
    assert (
        client.get(f"/api/cases/{case['id']}").json()["authorization_basis"]
        == "warrant 12/4, investigative"
    )

    entries = client.get("/api/audit").json()["entries"]
    amendments = [e for e in entries if e["action"] == "case.amend_authorization"]
    assert len(amendments) == 1
    assert amendments[0]["object_id"] == case["id"]
    assert amendments[0]["case_id"] == case["id"]
    assert amendments[0]["payload"] == {
        "previous_authorization_basis": "personal test images only",
        "authorization_basis": "warrant 12/4, investigative",
        "reason": "original text understated the material",
    }


def test_amending_an_unknown_case_is_404_and_writes_nothing(client: TestClient) -> None:
    before = client.get("/api/audit").json()["head_seq"]

    response = client.patch(
        "/api/cases/does-not-exist", json={"authorization_basis": "anything"}
    )

    assert response.status_code == 404
    assert client.get("/api/audit").json()["head_seq"] == before


def test_an_amendment_cannot_blank_the_authorization_basis(client: TestClient) -> None:
    case = client.post(
        "/api/cases", json={"name": "Op Kingfisher", "authorization_basis": "warrant 12/4"}
    ).json()

    assert (
        client.patch(f"/api/cases/{case['id']}", json={"authorization_basis": ""}).status_code
        == 422
    )
    assert (
        client.get(f"/api/cases/{case['id']}").json()["authorization_basis"] == "warrant 12/4"
    )


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
