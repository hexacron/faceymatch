"""Launching the watch helper (spec 6.11). Behavioural: only the process boundary is faked.

The one stubbed seam is `watch_launch.spawn` — the single place a child process is started —
plus the three environment facts `capability()` reads (platform, the Qt module name, the uv
binary name). Everything above that runs for real: the refusals, the single-helper guard and
the audit append. No test starts a real GUI; what matters here is the argv and the audit row,
not that a window appeared.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import sys
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import watch_launch
from app.api import watch as watch_api
from app.config import Settings
from app.main import create_app

BACKEND_DIR = Path(watch_launch.__file__).resolve().parents[1]


class FakeChild:
    """Stands in for the `Popen` that `spawn` returns, and for the operator closing it."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.terminated = False
        self._returncode: int | None = None
        self._exited = threading.Event()

    def poll(self) -> int | None:
        return self._returncode

    def wait(self) -> int:
        # What the reaper thread blocks in. A helper nobody has closed never returns.
        self._exited.wait()
        return 0 if self._returncode is None else self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self.exit(-15)

    def exit(self, code: int = 0) -> None:
        """The operator closed the helper's window."""
        self._returncode = code
        self._exited.set()


@pytest.fixture(autouse=True)
def forget_launched_helper() -> Iterator[None]:
    """The single-helper guard is process state, so it must not outlive a test."""
    yield
    watch_launch._child = None


@pytest.fixture
def logged(settings: Settings, tmp_path: Path) -> Settings:
    """The same install, with the helper's log under the test's own directory."""
    return settings.model_copy(update={"logs_dir": tmp_path / "logs"})


@pytest.fixture
def api(logged: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(logged)) as client:
        yield client


@pytest.fixture
def launchable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend this host is a macOS install with the watch extra synced."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(watch_launch, "QT_MODULE", "json")  # a module that certainly imports


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    child: FakeChild | None = None,
    error: OSError | None = None,
) -> list[dict[str, Any]]:
    """Fake the process boundary. Returns what every call was asked to start."""
    calls: list[dict[str, Any]] = []
    started = FakeChild() if child is None else child

    def fake_spawn(argv: Sequence[str], *, cwd: Path, log_path: Path) -> Any:
        calls.append({"argv": list(argv), "cwd": cwd, "log_path": log_path})
        if error is not None:
            raise error
        return started

    monkeypatch.setattr(watch_launch, "spawn", fake_spawn)
    return calls


def _entries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM audit_log WHERE action = 'watch.launch' ORDER BY seq"
    ).fetchall()


def test_a_launch_runs_the_frozen_argv_and_lands_in_the_audit_chain(
    api: TestClient,
    conn: sqlite3.Connection,
    logged: Settings,
    monkeypatch: pytest.MonkeyPatch,
    launchable: None,
) -> None:
    calls = _install(monkeypatch, child=FakeChild(pid=1234))

    response = api.post("/api/watch/launch")

    assert response.status_code == 201
    assert response.json() == {
        "pid": 1234,
        "log_path": str(logged.logs_dir / "watch.log"),
    }
    assert len(calls) == 1
    # The command is this interpreter and nothing else: no shell, no package manager, and
    # no flag a caller could have influenced.
    assert calls[0]["argv"] == [sys.executable, "-m", "watch"]
    assert calls[0]["cwd"] == BACKEND_DIR
    assert calls[0]["log_path"] == logged.logs_dir / "watch.log"

    rows = _entries(conn)
    assert len(rows) == 1
    assert rows[0]["object_type"] == "watch_helper"
    assert rows[0]["object_id"] == "1234"
    assert rows[0]["actor"] == logged.operator_name
    payload: dict[str, Any] = json.loads(str(rows[0]["payload_json"]))
    assert payload["argv"] == list(watch_launch.LAUNCH_ARGV)
    assert payload["log_path"] == str(logged.logs_dir / "watch.log")
    assert "POST /api/live/match" in payload["effect"]


def test_nothing_in_the_request_can_change_what_runs(
    api: TestClient, monkeypatch: pytest.MonkeyPatch, launchable: None
) -> None:
    """The endpoint takes no body, so a body cannot reach the command line."""
    calls = _install(monkeypatch)

    response = api.post(
        "/api/watch/launch",
        json={
            "argv": ["/bin/sh", "-c", "curl http://example.com"],
            "url": "http://example.com",
            "port": 9,
            "fps": "; rm -rf /",
        },
    )

    assert response.status_code == 201
    assert calls[0]["argv"] == list(watch_launch.LAUNCH_ARGV)


def test_a_second_helper_is_refused_while_the_first_is_up(
    api: TestClient, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, launchable: None
) -> None:
    child = FakeChild(pid=777)
    calls = _install(monkeypatch, child=child)
    assert api.post("/api/watch/launch").status_code == 201

    refused = api.post("/api/watch/launch")

    assert refused.status_code == 409
    assert "already running (pid 777)" in refused.json()["detail"]
    assert len(calls) == 1
    assert len(_entries(conn)) == 1


def test_a_closed_helper_can_be_started_again(
    api: TestClient, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, launchable: None
) -> None:
    """The guard is one helper at a time, not one helper per process lifetime."""
    child = FakeChild()
    calls = _install(monkeypatch, child=child)
    assert api.post("/api/watch/launch").status_code == 201
    child.exit()

    assert api.post("/api/watch/launch").status_code == 201
    assert len(calls) == 2
    assert len(_entries(conn)) == 2


def test_the_helper_is_refused_off_macos(
    api: TestClient,
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    calls = _install(monkeypatch)

    response = api.post("/api/watch/launch")

    assert response.status_code == 503
    assert response.json()["detail"] == "the watch helper is macOS only; this host reports Linux"
    assert calls == []
    assert _entries(conn) == []


def test_the_helper_is_refused_without_the_watch_extra(
    api: TestClient,
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(watch_launch, "QT_MODULE", "pyside6_not_installed_here")
    calls = _install(monkeypatch)

    response = api.post("/api/watch/launch")

    assert response.status_code == 503
    assert "uv sync --extra dev --extra watch" in response.json()["detail"]
    assert calls == []
    assert _entries(conn) == []


def test_a_child_that_cannot_be_started_is_reported_and_writes_nothing(
    api: TestClient, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, launchable: None
) -> None:
    _install(monkeypatch, error=OSError("Exec format error"))

    response = api.post("/api/watch/launch")

    assert response.status_code == 503
    assert response.json()["detail"] == "the watch helper did not start: Exec format error"
    assert _entries(conn) == []
    assert api.get("/api/watch").json()["running"] is False


def test_a_helper_whose_launch_cannot_be_audited_is_stopped_again(
    api: TestClient, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, launchable: None
) -> None:
    """Invariant 6, spec 12: nothing that can capture the screen runs unaudited."""
    child = FakeChild()
    _install(monkeypatch, child=child)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("chain head is not readable")

    monkeypatch.setattr(watch_api.audit, "append", refuse)

    with pytest.raises(RuntimeError, match="chain head"):
        api.post("/api/watch/launch")

    assert child.terminated
    assert _entries(conn) == []


def test_status_says_why_the_button_must_not_be_offered(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    status = api.get("/api/watch").json()

    assert status["available"] is False
    assert status["platform_supported"] is False
    assert status["running"] is False
    assert status["pid"] is None
    assert "macOS only" in status["reason"]


def test_status_reports_the_helper_it_started(
    api: TestClient, monkeypatch: pytest.MonkeyPatch, launchable: None
) -> None:
    child = FakeChild(pid=99)
    _install(monkeypatch, child=child)

    before = api.get("/api/watch").json()
    api.post("/api/watch/launch")
    during = api.get("/api/watch").json()
    child.exit()
    after = api.get("/api/watch").json()

    assert (before["available"], before["running"], before["pid"]) == (True, False, None)
    assert (during["available"], during["running"], during["pid"]) == (True, True, 99)
    assert (after["running"], after["pid"]) == (False, None)


def test_the_helpers_output_appends_to_the_shared_log(tmp_path: Path) -> None:
    """Real spawn, trivial child: `run` and this endpoint write one watch.log between them."""
    log_path = tmp_path / "logs" / "watch.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("from ./run --watch\n", encoding="utf-8")

    child = watch_launch.spawn(
        ["/bin/echo", "from the button"], cwd=tmp_path, log_path=log_path
    )
    assert child.wait() == 0

    assert log_path.read_text(encoding="utf-8") == "from ./run --watch\nfrom the button\n"
