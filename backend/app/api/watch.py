"""Starting the watch helper from the web UI (spec 6.11).

The helper is operator-initiated and bounded, and that is what keeps it outside the
monitoring ban in spec section 2. A button is still an operator action, so this endpoint is
built to stay one and nothing more:

- It takes no request body. There is no field to validate because there is no field: the
  argv is the frozen constant in `app.watch_launch`, so the only thing a caller can decide
  is whether a helper starts.
- It starts nothing by itself. There is no start-on-load, no retry, and no poll that
  restarts anything: every helper on this path exists because a person pressed the button.
- Starting the helper is not starting a capture. The process draws its panel and captures
  nothing until somebody at this machine picks a window, display or region in that panel and
  presses Start there.
- It is loopback-only by construction, because the listener is (invariant 11). The endpoint
  adds no binding of its own and needs none.
- It appends `watch.launch` to the hash-chained log. Capturing the operator's own screen is
  an operator action on special-category data (spec 12), so it belongs in the chain beside
  the other audited operator decisions, and a helper whose entry cannot be written is
  stopped again rather than left running.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app import audit, watch_launch
from app.api.deps import ConnDep, SettingsDep
from app.db.conn import transaction

router = APIRouter(prefix="/api/watch", tags=["watch"])


class WatchStatusOut(BaseModel):
    """Whether the button may be offered, and whether a helper is already up.

    `available` is the only field the button needs; `reason` is non-null exactly when
    `available` is false, and the two narrower booleans are diagnostics. `running` covers
    helpers started through this endpoint only — one the operator started from a terminal is
    their own and this process cannot see it.
    """

    available: bool
    platform_supported: bool
    extra_installed: bool
    running: bool
    pid: int | None
    reason: str | None


class WatchLaunchOut(BaseModel):
    """The helper that is now up, and where to read what it says."""

    pid: int
    log_path: str


@router.get("", response_model=WatchStatusOut)
def watch_status() -> WatchStatusOut:
    """Report whether the helper can be started here, and whether one already is."""
    support = watch_launch.capability()
    pid = watch_launch.running_pid()
    return WatchStatusOut(
        available=support.available,
        platform_supported=support.platform_supported,
        extra_installed=support.extra_installed,
        running=pid is not None,
        pid=pid,
        reason=support.reason,
    )


@router.post("/launch", response_model=WatchLaunchOut, status_code=status.HTTP_201_CREATED)
def launch_watch(conn: ConnDep, settings: SettingsDep) -> WatchLaunchOut:
    """Start the watch helper for the operator at this machine (spec 6.11).

    503 when the helper cannot run on this host or install (not macOS, or the `watch` extra
    is not installed) or when the child could not be started at all; 409 when a helper
    started from here is still running, because two always-on-top overlays on one screen
    help nobody.

    The launch is audited as `watch.launch` with the exact argv that ran. If that append
    fails the helper is terminated: an unaudited process that can capture the operator's
    screen is the one outcome worth undoing (invariant 6, spec 12).
    """
    try:
        started = watch_launch.launch(logs_dir=settings.logs_dir)
    except watch_launch.WatchAlreadyRunningError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except watch_launch.WatchLaunchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    try:
        with transaction(conn):
            audit.append(
                conn,
                actor=settings.operator_name,
                action="watch.launch",
                object_type="watch_helper",
                object_id=str(started.pid),
                payload={
                    "argv": list(watch_launch.LAUNCH_ARGV),
                    "cwd": str(watch_launch.LAUNCH_CWD),
                    "log_path": str(started.log_path),
                    "trigger": "POST /api/watch/launch",
                    "effect": (
                        "the operator started the watch helper, which captures one window, "
                        "display or region of this machine's own screen — chosen in the "
                        "helper's own window — and matches each frame through "
                        "POST /api/live/match, storing nothing of its own, until the "
                        "operator stops it (spec 6.11)"
                    ),
                },
            )
    except Exception:
        watch_launch.terminate_launched()
        raise

    return WatchLaunchOut(pid=started.pid, log_path=str(started.log_path))
