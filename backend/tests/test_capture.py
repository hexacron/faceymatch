"""Screen capture ingest (tier 1). Behavioural: only the process boundary is faked.

The one stubbed seam is `capture.run_capture` — the single place a child process is
spawned — plus the two environment facts `capability()` reads (platform, binary path).
Everything above that runs for real: outcome classification, the temp file, the ingest, the
media row, the `process` job and the audit entry. No test needs a real screen.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sqlite3
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.pipeline import capture


def _png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _screenshot(seed: int = 0) -> bytes:
    """A plausible screenshot: not uniform, so it is not read as a blocked capture."""
    rng = np.random.default_rng(seed)
    return _png(rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8))


BLANK = _png(np.zeros((64, 96, 3), dtype=np.uint8))


@pytest.fixture
def on_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pretend this host is macOS with the capture binary installed."""
    binary = tmp_path / "screencapture"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(capture, "SCREENCAPTURE_BIN", binary)
    return binary


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writes: bytes | None,
    returncode: int = 0,
    stderr: bytes = b"",
) -> list[list[str]]:
    """Fake the process boundary. Returns the argv of every call made."""
    calls: list[list[str]] = []

    def fake_run(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(argv))
        if writes is not None:
            Path(argv[-1]).write_bytes(writes)
        return subprocess.CompletedProcess(list(argv), returncode, b"", stderr)

    monkeypatch.setattr(capture, "run_capture", fake_run)
    return calls


def _case(client: TestClient) -> str:
    created = client.post(
        "/api/cases", json={"name": "Capture", "authorization_basis": "warrant"}
    )
    assert created.status_code == 201
    return str(created.json()["id"])


def _head_seq(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS seq FROM audit_log").fetchone()
    return int(row["seq"])


def test_capture_lands_as_evidence_with_a_job_and_an_audit_entry(
    client: TestClient,
    conn: sqlite3.Connection,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    on_macos: Path,
) -> None:
    pixels = _screenshot()
    _install(monkeypatch, writes=pixels)
    case_id = _case(client)

    response = client.post("/api/capture", json={"case_id": case_id, "mode": "region"})

    assert response.status_code == 201
    body = response.json()
    assert body["sha256"] == hashlib.sha256(pixels).hexdigest()
    assert body["reused"] is False
    assert body["job_id"] is not None

    media = conn.execute("SELECT * FROM media WHERE id = ?", (body["media_id"],)).fetchone()
    assert media is not None
    assert str(media["case_id"]) == case_id
    assert str(media["sha256"]) == body["sha256"]
    assert str(media["status"]) == "new"
    stored = settings.media_dir / str(media["path"])
    assert stored.read_bytes() == pixels

    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (body["job_id"],)).fetchone()
    assert job is not None
    assert str(job["kind"]) == "process"
    assert json.loads(str(job["params_json"]))["media_id"] == body["media_id"]

    entry = conn.execute(
        "SELECT * FROM audit_log WHERE action = 'media.ingest' AND object_id = ?",
        (body["media_id"],),
    ).fetchone()
    assert entry is not None
    payload = json.loads(str(entry["payload_json"]))
    # The capture is distinguishable from an upload in the log.
    assert payload["acquisition"] == "screen_capture"
    assert payload["capture_mode"] == "region"
    assert payload["sha256"] == body["sha256"]


def test_an_upload_records_its_own_acquisition_mode(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    case_id = _case(client)
    pixels = _screenshot(seed=7)

    uploaded = client.post(
        "/api/media",
        data={"case_id": case_id},
        files={"file": ("shot.png", pixels, "image/png")},
    )
    assert uploaded.status_code == 201

    entry = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action = 'media.ingest' AND object_id = ?",
        (uploaded.json()["media_id"],),
    ).fetchone()
    payload = json.loads(str(entry["payload_json"]))
    assert payload["acquisition"] == "upload"
    assert payload["capture_mode"] is None


def test_an_upload_can_declare_that_it_came_off_a_screen(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """The watch helper ingests through `POST /api/media`, so the wire carries the mode.

    Without this the audit log would call a screen frame an operator's file (spec 6.11).
    """
    case_id = _case(client)

    uploaded = client.post(
        "/api/media",
        data={"case_id": case_id, "acquisition": "screen_capture", "capture_mode": "window"},
        files={"file": ("watch.png", _screenshot(seed=11), "image/png")},
    )
    assert uploaded.status_code == 201

    entry = conn.execute(
        "SELECT payload_json FROM audit_log WHERE action = 'media.ingest' AND object_id = ?",
        (uploaded.json()["media_id"],),
    ).fetchone()
    payload = json.loads(str(entry["payload_json"]))
    assert payload["acquisition"] == "screen_capture"
    assert payload["capture_mode"] == "window"


def test_a_capture_mode_without_a_screen_capture_acquisition_is_refused(
    client: TestClient, settings: Settings
) -> None:
    """An incoherent pair is refused before any bytes are spooled, not silently recorded."""
    case_id = _case(client)

    response = client.post(
        "/api/media",
        data={"case_id": case_id, "capture_mode": "window"},
        files={"file": ("shot.png", _screenshot(seed=12), "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "capture_mode only applies to a screen_capture acquisition"
    assert [path for path in settings.media_dir.rglob("*") if path.is_file()] == []


def test_an_upload_cannot_claim_the_folder_import_mode(client: TestClient) -> None:
    """`folder_import` belongs to `POST /api/media/import`; nothing on the wire may claim it."""
    case_id = _case(client)

    response = client.post(
        "/api/media",
        data={"case_id": case_id, "acquisition": "folder_import"},
        files={"file": ("shot.png", _screenshot(seed=13), "image/png")},
    )

    assert response.status_code == 422


def test_modes_select_the_documented_screencapture_behaviour(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, on_macos: Path
) -> None:
    """Region and window are interactive; screen must need no human at all."""
    case_id = _case(client)
    seen: dict[str, list[str]] = {}
    for index, mode in enumerate(("region", "window", "screen")):
        calls = _install(monkeypatch, writes=_screenshot(seed=index + 1))
        assert (
            client.post("/api/capture", json={"case_id": case_id, "mode": mode}).status_code
            == 201
        )
        seen[mode] = calls[0]

    assert "-i" in seen["region"]
    assert "-i" in seen["window"] and "-w" in seen["window"]
    assert "-i" not in seen["screen"] and "-m" in seen["screen"]
    # Never the clipboard, always a file we then delete.
    for argv in seen.values():
        assert "-c" not in argv
        assert argv[-1].endswith(capture.CAPTURE_SUFFIX)


def test_cancelled_capture_writes_nothing(
    client: TestClient,
    conn: sqlite3.Connection,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    on_macos: Path,
) -> None:
    case_id = _case(client)
    before_seq = _head_seq(conn)
    # Esc: screencapture exits non-zero and leaves no file behind.
    calls = _install(monkeypatch, writes=None, returncode=1)

    response = client.post("/api/capture", json={"case_id": case_id, "mode": "region"})

    assert response.status_code == 400
    assert response.json() == {"detail": "capture cancelled"}
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 0
    assert _head_seq(conn) == before_seq
    # No orphan temp file, and no stored object either.
    staged = Path(calls[0][-1])
    assert not staged.exists()
    assert not staged.parent.exists()
    # And nothing reached the content-addressed store either.
    assert [path for path in settings.media_dir.rglob("*") if path.is_file()] == []



def test_capture_is_unavailable_off_macos(
    client: TestClient,
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calls = _install(monkeypatch, writes=_screenshot())
    case_id = _case(client)

    response = client.post("/api/capture", json={"case_id": case_id, "mode": "screen"})

    assert response.status_code == 503
    assert "macOS" in response.json()["detail"]
    assert calls == []  # nothing was ever spawned
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 0


def test_capture_is_unavailable_without_the_binary(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(capture, "SCREENCAPTURE_BIN", tmp_path / "absent")
    calls = _install(monkeypatch, writes=_screenshot())
    case_id = _case(client)

    response = client.post("/api/capture", json={"case_id": case_id, "mode": "screen"})

    assert response.status_code == 503
    assert "not found" in response.json()["detail"]
    assert calls == []


def test_denied_permission_is_reported_as_unavailable_with_instructions(
    client: TestClient, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, on_macos: Path
) -> None:
    case_id = _case(client)
    _install(
        monkeypatch,
        writes=None,
        returncode=1,
        stderr=b"screencapture: cannot run due to missing permission",
    )

    response = client.post("/api/capture", json={"case_id": case_id, "mode": "region"})

    assert response.status_code == 503
    assert "Screen Recording" in response.json()["detail"]
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 0


def test_a_non_interactive_capture_with_no_image_is_not_a_cancellation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, on_macos: Path
) -> None:
    """`screen` needs no human, so an empty result is a failure, not an Esc press."""
    case_id = _case(client)
    _install(monkeypatch, writes=None, returncode=1)

    response = client.post("/api/capture", json={"case_id": case_id, "mode": "screen"})

    assert response.status_code == 503
    assert "Screen Recording" in response.json()["detail"]


def test_a_blank_capture_is_reported_rather_than_ingested(
    client: TestClient, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, on_macos: Path
) -> None:
    """Without permission macOS hands back a blank frame instead of an error."""
    case_id = _case(client)
    _install(monkeypatch, writes=BLANK)

    response = client.post("/api/capture", json={"case_id": case_id, "mode": "screen"})

    assert response.status_code == 503
    assert "blank" in response.json()["detail"]
    assert "Screen Recording" in response.json()["detail"]
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 0


def test_recapturing_identical_bytes_reuses_the_media_row(
    client: TestClient, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, on_macos: Path
) -> None:
    case_id = _case(client)
    _install(monkeypatch, writes=_screenshot(seed=3))

    first = client.post("/api/capture", json={"case_id": case_id, "mode": "screen"}).json()
    second = client.post("/api/capture", json={"case_id": case_id, "mode": "screen"}).json()

    assert second["media_id"] == first["media_id"]
    assert second["reused"] is True
    assert second["job_id"] is None
    assert conn.execute("SELECT COUNT(*) AS n FROM media").fetchone()["n"] == 1
    assert (
        conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE kind = 'process'").fetchone()["n"] == 1
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE action = 'media.ingest'"
        ).fetchone()["n"]
        == 1
    )


def test_unknown_case_is_404_before_the_operator_is_asked_to_select(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, on_macos: Path
) -> None:
    calls = _install(monkeypatch, writes=_screenshot())

    response = client.post("/api/capture", json={"case_id": "no-such-case", "mode": "region"})

    assert response.status_code == 404
    assert calls == []


def test_capture_over_the_upload_limit_is_rejected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, on_macos: Path
) -> None:
    tiny = settings.model_copy(update={"max_upload_bytes": 32})
    with TestClient(create_app(tiny)) as client:
        case_id = _case(client)
        _install(monkeypatch, writes=_screenshot(seed=5))

        response = client.post("/api/capture", json={"case_id": case_id, "mode": "screen"})

        assert response.status_code == 413
        assert client.get("/api/media").json()["items"] == []


def test_healthz_reports_capture_capability(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on_macos: Path
) -> None:
    available = client.get("/api/healthz").json()["capture"]
    assert available == {
        "available": True,
        "platform_supported": True,
        "binary_present": True,
        "reason": None,
    }

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    blocked = client.get("/api/healthz").json()["capture"]
    assert blocked["available"] is False
    assert blocked["platform_supported"] is False
    assert "macOS" in str(blocked["reason"])
