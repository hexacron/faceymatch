"""Starting the watch helper (spec 6.11), the one process the API starts on request.

The helper is the PySide6 application in `backend/watch/`. It was terminal-only, which made
the one operator action the spec calls "starts on a click" the only action that needed a
shell. This module is what `POST /api/watch/launch` uses, so the button in the web UI is the
same operator decision reached by a shorter path.

Boundaries, all deliberate:

- `spawn` is the only place a child process is started, and it is the one seam tests fake.
  `LAUNCH_ARGV` is a frozen module constant handed straight to `subprocess.Popen`: no shell,
  and no path, filename, port or flag from a request can reach the command line, because the
  endpoint takes no body at all. A request decides *whether* the helper starts, never what
  runs.
- No package manager in the path. The command is this interpreter — the one already running
  the API, out of the project environment that holds the `watch` extra — so a launch cannot
  resolve, download or change a dependency, and invariant 1 holds by construction rather
  than by an offline flag. It also means the pid the endpoint reports and audits is the
  helper itself, not a launcher that happens to own it.
- Starting the helper is not starting a capture. The process opens its panel and captures
  nothing until somebody at this machine picks a window, display or region in that window
  and presses Start there. That is what keeps the button on the operator-initiated side of
  the section 2 boundary, whatever reaches the loopback listener.
- The child is deliberately *not* put in a new session, so it stays in the process group
  `run` signals: Ctrl-C in the operator's terminal still stops the helper, and no overlay
  outlives the install that drew it (spec 6.11, bounded).
- One helper per API process. A helper the operator started from a terminal is their own and
  this module does not police it; two started from *here* would be two always-on-top
  overlays fighting over one screen.
- macOS only, and only when the `watch` extra resolves. Everywhere else `capability()`
  reports why in a sentence the UI shows instead of offering a dead button.
"""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MACOS = "Darwin"
# What `run` probes before it starts the helper, so the script and the UI agree about what
# "installed" means. The other `watch` extra packages fail into the log with the helper's own
# sentence; Qt is the one whose absence means the extra was never synced at all.
QT_MODULE = "PySide6"
WATCH_MODULE = "watch"

# The whole command line, frozen at import. `sys.executable` is this process's own
# interpreter, which is exactly the environment `capability()` probes; nothing derived from a
# request, a header or a setting is interpolated into it, and there is no shell to
# interpolate into.
LAUNCH_ARGV: tuple[str, ...] = (sys.executable, "-m", WATCH_MODULE)
# `python -m watch` resolves the package from its working directory: `backend/`, which holds
# both the `watch` package and the project environment it was installed into.
LAUNCH_CWD = Path(__file__).resolve().parents[1]
# The same file `run` appends the helper's output to, so one log holds every run of it.
LOG_NAME = "watch.log"


class WatchLaunchError(RuntimeError):
    """A launch that did not happen. The message is the sentence the operator reads."""


class WatchUnavailableError(WatchLaunchError):
    """The helper cannot run here: wrong platform, or the `watch` extra is not installed."""


class WatchAlreadyRunningError(WatchLaunchError):
    """A helper started through this module is still up."""


class WatchSpawnError(WatchLaunchError):
    """The child process could not be started at all."""


@dataclass(frozen=True, slots=True)
class Capability:
    """What the UI needs to know *before* offering the button."""

    available: bool
    platform_supported: bool
    extra_installed: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class Launched:
    pid: int
    log_path: Path


def _module_present(name: str) -> bool:
    """Whether `name` imports in this interpreter, without importing it.

    This interpreter is the one that will run the helper, so it is the only environment the
    question has to be asked about.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - a broken sys.path entry
        return False


def capability() -> Capability:
    """Report whether a launch can be attempted at all, and why not when it cannot."""
    platform_supported = platform.system() == MACOS
    extra_installed = _module_present(QT_MODULE)

    reason: str | None = None
    if not platform_supported:
        reason = (
            "the watch helper is macOS only; this host reports "
            f"{platform.system() or 'an unknown platform'}"
        )
    elif not extra_installed:
        reason = (
            "the watch helper needs the watch extra, which is not installed here: run "
            "`cd backend && uv sync --extra dev --extra watch`, then restart the API so it "
            "sees it"
        )

    return Capability(
        available=platform_supported and extra_installed,
        platform_supported=platform_supported,
        extra_installed=extra_installed,
        reason=reason,
    )


_lock = threading.Lock()
_child: subprocess.Popen[bytes] | None = None


def spawn(argv: Sequence[str], *, cwd: Path, log_path: Path) -> subprocess.Popen[bytes]:
    """Start the helper. The one process boundary in the launch path.

    Output goes to `log_path` in append mode, which is where `run` puts it too. The child
    holds the descriptor, so the helper keeps logging long after the request that started it
    returned, and stdin is closed: a GUI application that blocks on a terminal it does not
    have is a hang nobody can see.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        return subprocess.Popen(  # noqa: S603 - frozen argv, no shell, no request input
            list(argv),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )


def _running_pid_locked() -> int | None:
    global _child
    child = _child
    if child is None:
        return None
    if child.poll() is not None:
        _child = None
        return None
    return child.pid


def running_pid() -> int | None:
    """The pid of a helper started through this module, or None once it has exited."""
    with _lock:
        return _running_pid_locked()


def launch(*, logs_dir: Path) -> Launched:
    """Start one helper, or raise the refusal the operator should read.

    Raises `WatchUnavailableError` when the helper cannot run on this host or install,
    `WatchAlreadyRunningError` when one started from here is still up, and `WatchSpawnError`
    when the child could not be started.
    """
    global _child
    with _lock:
        support = capability()
        if not support.available:
            raise WatchUnavailableError(support.reason or "the watch helper is unavailable here")
        running = _running_pid_locked()
        if running is not None:
            raise WatchAlreadyRunningError(
                f"a watch helper started from here is already running (pid {running}); "
                "stop it from its own window before starting another"
            )
        log_path = logs_dir / LOG_NAME
        try:
            child = spawn(LAUNCH_ARGV, cwd=LAUNCH_CWD, log_path=log_path)
        except OSError as exc:
            raise WatchSpawnError(f"the watch helper did not start: {exc}") from exc
        _child = child

    # Reap where the child dies, not at the next request. The helper is closed from its own
    # window and the next POST may never come, so waiting for one would leave a zombie for
    # as long as the operator keeps working.
    threading.Thread(target=child.wait, name="watch-helper-reaper", daemon=True).start()
    return Launched(pid=child.pid, log_path=log_path)


def terminate_launched() -> None:
    """Stop the helper started through this module, if it is still up.

    This exists for one caller: a launch whose audit entry could not be written. Screen
    capture of the operator's own display is an audited operator action (spec 12), so an
    unaudited helper is undone rather than left running.
    """
    with _lock:
        child = _child
        if child is None or child.poll() is not None:
            return
        child.terminate()
