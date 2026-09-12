"""Screen capture as an ingest source (macOS only).

The operator often has the face on screen in another application: a photo in a browser, a
paused video, Preview. This module turns those pixels into a file on disk. It does nothing
else. The captured file then goes through the *same* `ingest_file` path as an upload
(spec 6.1, invariant 7), so captured pixels are hashed, content-addressed and audited like
any other evidence. There is no second detection path.

Boundaries, all deliberate:

- `run_capture` is the only place a child process is spawned, and it is the one seam tests
  fake. `argv` is a fixed list handed straight to `subprocess.run`: no shell, and no
  operator-supplied string ever reaches a command line (only a closed `CaptureMode`
  vocabulary selects flags).
- Local binary only. Nothing here touches the network (invariant 1).
- Capture goes to a temp file, never the clipboard: a pasteboard round-trip would clobber
  whatever the operator had copied.
- macOS-only. Everywhere else `capability()` reports why, and `capture_image` refuses with
  `CaptureUnavailableError` instead of crashing.
"""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from PIL import Image, UnidentifiedImageError

CaptureMode = Literal["region", "window", "screen"]
CAPTURE_MODES: frozenset[str] = frozenset(get_args(CaptureMode))

MACOS = "Darwin"
SCREENCAPTURE_BIN = Path("/usr/sbin/screencapture")
CAPTURE_SUFFIX = ".png"
_CAPTURE_FORMAT = "png"

# `-x` mutes the shutter sound. Mode flags:
#   region: -i  interactive crosshair selection
#   window: -iw interactive window pick
#   screen: -m  whole main display, no interaction at all
_MODE_FLAGS: dict[str, tuple[str, ...]] = {
    "region": ("-i",),
    "window": ("-i", "-w"),
    "screen": ("-m",),
}
# Only an interactive capture can be cancelled. A non-interactive one that produced no
# image failed, and the usual cause is the permission prompt.
_INTERACTIVE: frozenset[str] = frozenset({"region", "window"})

CANCELLED_DETAIL = "capture cancelled"
PERMISSION_HINT = (
    "Grant Screen Recording to the app running this server (your terminal, or the app "
    "itself) in System Settings > Privacy & Security > Screen Recording, then restart it."
)

# macOS phrases the refusal differently across versions, so match on the shared words.
# `could not create image from display` is what macOS 15/26 actually prints when Screen
# Recording is not granted — observed on this host, rc=1, no file written. Matching it
# matters most for the interactive modes, where an empty result would otherwise be
# indistinguishable from the operator pressing Esc.
_DENIED_MARKERS = (
    "not authorized",
    "not permitted",
    "permission",
    "denied",
    "screen recording",
    "cannot run",
    "could not create image",
)


class CaptureError(RuntimeError):
    """Base class for a capture that did not yield usable pixels."""


class CaptureUnavailableError(CaptureError):
    """Screen capture cannot work here: wrong platform, no binary, or no permission."""


class CaptureCancelledError(CaptureError):
    """The operator dismissed the selection, so there is nothing to ingest."""


@dataclass(frozen=True, slots=True)
class Capability:
    """What the UI needs to know *before* offering the button."""

    available: bool
    platform_supported: bool
    binary_present: bool
    reason: str | None


def capability() -> Capability:
    """Report whether a capture can be attempted at all, and why not when it cannot."""
    platform_supported = platform.system() == MACOS
    binary = SCREENCAPTURE_BIN
    binary_present = binary.is_file() and os.access(binary, os.X_OK)

    reason: str | None = None
    if not platform_supported:
        reason = (
            f"screen capture is macOS-only; this host reports {platform.system() or 'unknown'}"
        )
    elif not binary_present:
        reason = f"macOS screen capture binary not found at {binary}"

    return Capability(
        available=platform_supported and binary_present,
        platform_supported=platform_supported,
        binary_present=binary_present,
        reason=reason,
    )


def run_capture(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    """Spawn `screencapture`. The one process boundary in the capture path."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, no interpolated input
        list(argv),
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def capture_image(*, dest: Path, mode: CaptureMode, timeout: float) -> int:
    """Capture the screen to `dest` and return the byte size written.

    Raises `CaptureUnavailableError` when capture cannot work here (platform, binary,
    permission, or a blank image that means the permission was never granted) and
    `CaptureCancelledError` when the operator selected nothing. On any failure `dest` is
    removed, so a caller never ingests a half-written temp file.
    """
    if mode not in CAPTURE_MODES:  # pragma: no cover - the API validates the vocabulary
        raise ValueError(f"unknown capture mode {mode!r}")

    state = capability()
    if not state.available:
        raise CaptureUnavailableError(state.reason or "screen capture is unavailable")

    argv = [
        str(SCREENCAPTURE_BIN),
        "-x",
        "-t",
        _CAPTURE_FORMAT,
        *_MODE_FLAGS[mode],
        str(dest),
    ]
    try:
        completed = run_capture(argv, timeout)
    except subprocess.TimeoutExpired as exc:
        _discard(dest)
        raise CaptureCancelledError(
            f"capture timed out after {timeout:g}s waiting for a selection"
        ) from exc
    except OSError as exc:  # binary vanished or is not executable after all
        _discard(dest)
        raise CaptureUnavailableError(f"could not run {SCREENCAPTURE_BIN}: {exc}") from exc

    stderr = completed.stderr.decode("utf-8", "replace").strip()
    size = dest.stat().st_size if dest.is_file() else 0

    if _looks_denied(stderr):
        _discard(dest)
        raise CaptureUnavailableError(f"screen capture was refused: {stderr}. {PERMISSION_HINT}")

    if size == 0:
        _discard(dest)
        if mode in _INTERACTIVE:
            raise CaptureCancelledError(CANCELLED_DETAIL)
        raise CaptureUnavailableError(
            f"{SCREENCAPTURE_BIN.name} produced no image"
            f"{f' ({stderr})' if stderr else ''}. {PERMISSION_HINT}"
        )

    if completed.returncode != 0:
        _discard(dest)
        raise CaptureUnavailableError(
            f"{SCREENCAPTURE_BIN.name} exited {completed.returncode}"
            f"{f': {stderr}' if stderr else ''}. {PERMISSION_HINT}"
        )

    if _is_uniform(dest):
        # Without Screen Recording permission macOS hands back a blank frame instead of an
        # error. A single-colour screenshot is never evidence, so say so rather than
        # enrolling a black rectangle.
        _discard(dest)
        raise CaptureUnavailableError(
            f"screen capture came back blank, which means the capture was blocked. "
            f"{PERMISSION_HINT}"
        )

    return size


def _looks_denied(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _DENIED_MARKERS)


def _is_uniform(path: Path) -> bool:
    """True when every pixel has the same luma: the blank-capture failure mode.

    The greyscale histogram is exact and stays in C, so this costs one pass over the
    pixels and 256 ints rather than materialising the whole screenshot as an array.
    """
    try:
        with Image.open(path) as opened:
            grey = opened.convert("L")
            with grey:
                occupied = sum(1 for count in grey.histogram() if count)
    except (UnidentifiedImageError, OSError):
        # Not decodable as an image: let ingest report the real reason on the bytes.
        return False
    return occupied <= 1


def _discard(path: Path) -> None:
    path.unlink(missing_ok=True)
