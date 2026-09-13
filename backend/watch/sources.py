"""Capture and window enumeration, behind one interface.

macOS is the only implementation today. Everything platform-specific lives below
`CaptureBackend`, so Windows and Linux are a new class and a branch in `platform_backend()`
rather than a rewrite of the loop, the overlay or the panel.

Window targeting captures the screen rectangle where the window is, not the window's own
buffer, because per-window pixel readback is the least portable of the three APIs involved.
Anything overlapping the watched window is captured with it — including the helper's own
panel, which is why the panel says so when a window target is chosen.
"""

from __future__ import annotations

import io
import platform
import sys
import threading
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from PIL import Image

from watch.geometry import Rect, capture_size

MACOS = "Darwin"

# The smallest window worth offering: menu shadows, tooltips and status items are all
# smaller than this and none of them ever holds a face.
MIN_WINDOW_EDGE = 64.0

# The self-test grabs the whole target at 64x64 worth of area: enough pixels to tell a real
# window from a denied grant, few enough to cost nothing on Start.
SELF_TEST_MAX_PIXELS = 64 * 64

PERMISSION_HINT = (
    "macOS has not granted Screen Recording to this process. Add it in System Settings > "
    "Privacy & Security > Screen Recording, then restart the helper."
)

TargetKind = Literal["window", "display", "region"]


@dataclass(frozen=True, slots=True)
class WindowInfo:
    window_id: int
    label: str  # "QuickTime Player — clip.mp4"
    rect: Rect


@dataclass(frozen=True, slots=True)
class DisplayInfo:
    index: int
    label: str  # "Display 1 — 1512x982"
    rect: Rect


@dataclass(frozen=True, slots=True)
class Target:
    kind: TargetKind
    window_id: int | None
    display_index: int | None
    rect: Rect  # last known rect; authoritative for display and region
    label: str

    @property
    def capture_mode(self) -> str:
        """The `capture_mode` these pixels are ingested under (spec 6.10)."""
        return "region" if self.kind == "region" else "screen"


@dataclass(frozen=True, slots=True)
class Frame:
    jpeg: bytes
    width: int  # pixels in the JPEG, after the pixel cap
    height: int


class UnsupportedPlatformError(RuntimeError):
    """No capture backend for this platform yet."""


class CaptureError(RuntimeError):
    """The grab failed. Carries text fit for the panel."""


class CapturePermissionError(CaptureError):
    """The platform handed back a blank frame: the grant is missing, not the pixels."""


class CaptureBackend(Protocol):
    def windows(self) -> list[WindowInfo]: ...

    def displays(self) -> list[DisplayInfo]: ...

    def bounds(self, target: Target) -> Rect | None: ...

    def grab(self, rect: Rect, *, max_pixels: int, quality: int) -> Frame: ...

    def self_test(self, target: Target) -> None: ...


def platform_backend() -> CaptureBackend:
    """The backend for this platform, or `UnsupportedPlatformError` naming what is missing."""
    if platform.system() != MACOS:
        raise UnsupportedPlatformError(
            f"the watch helper has no capture backend for {platform.system()} yet; "
            "macOS is the only implementation (spec 6.11)"
        )
    return MacCaptureBackend()


class MacCaptureBackend:
    """Quartz for geometry, `mss` for pixels.

    Window geometry and titles come from the window server through
    `CGWindowListCopyWindowInfo`, which needs no Accessibility grant. Pixels need Screen
    Recording, and macOS answers a missing grant with a blank frame rather than an error —
    hence `self_test`.
    """

    def __init__(self) -> None:
        # `mss` holds CoreGraphics state that is not safe to share: the GUI thread enumerates
        # and self-tests, the session thread grabs, and each gets its own.
        self._local = threading.local()

    # ------------------------------------------------------------- enumeration

    def windows(self) -> list[WindowInfo]:
        quartz = _quartz()
        entries = quartz.CGWindowListCopyWindowInfo(
            quartz.kCGWindowListOptionOnScreenOnly | quartz.kCGWindowListExcludeDesktopElements,
            quartz.kCGNullWindowID,
        )
        found: list[WindowInfo] = []
        for entry in entries or []:
            info = _window_info(entry)
            if info is not None:
                found.append(info)
        found.sort(key=lambda window: window.label.lower())
        return found

    def displays(self) -> list[DisplayInfo]:
        # `mss.monitors[0]` is the union of every display; the real ones start at 1, and
        # their rectangles are the same coordinate space window bounds are reported in.
        monitors = self._mss().monitors[1:]
        return [
            DisplayInfo(
                index=index,
                label=f"Display {index} — {monitor['width']}x{monitor['height']}",
                rect=_monitor_rect(monitor),
            )
            for index, monitor in enumerate(monitors, start=1)
        ]

    def bounds(self, target: Target) -> Rect | None:
        """Where the target is now, or None when a watched window has gone away.

        A window that closed, or moved to another Space, or went full-screen into its own
        Space, all read the same from here: it is no longer on screen, so there is nothing
        to follow and the session ends.
        """
        if target.kind != "window" or target.window_id is None:
            return target.rect
        quartz = _quartz()
        entries = quartz.CGWindowListCopyWindowInfo(
            quartz.kCGWindowListOptionIncludingWindow, target.window_id
        )
        for entry in entries or []:
            rect = _bounds_rect(entry.get("kCGWindowBounds"))
            if rect is not None:
                return rect
        return None

    # ----------------------------------------------------------------- pixels

    def grab(self, rect: Rect, *, max_pixels: int, quality: int) -> Frame:
        image = self._raw(rect, max_pixels=max_pixels)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return Frame(jpeg=buffer.getvalue(), width=image.width, height=image.height)

    def self_test(self, target: Target) -> None:
        """Raise when the platform is handing back blank frames instead of pixels."""
        rect = self.bounds(target) or target.rect
        grey = self._raw(rect, max_pixels=SELF_TEST_MAX_PIXELS).convert("L").tobytes()
        if len(set(grey)) <= 1:
            # Without Screen Recording macOS returns a uniform frame instead of an error,
            # the same signature `app/pipeline/capture.py::_is_uniform` reads. A single
            # colour is never evidence, so refuse here rather than post it as a face-free
            # frame for as long as the operator leaves it running.
            raise CapturePermissionError(PERMISSION_HINT)

    def _raw(self, rect: Rect, *, max_pixels: int) -> Image.Image:
        region = {
            "left": round(rect.x),
            "top": round(rect.y),
            "width": max(1, round(rect.w)),
            "height": max(1, round(rect.h)),
        }
        try:
            shot = self._mss().grab(region)
            image = Image.frombytes("RGB", (shot.width, shot.height), shot.bgra, "raw", "BGRX")
        except CaptureError:
            raise
        except Exception as exc:  # mss raises its own hierarchy; the panel wants a sentence
            raise CaptureError(f"the screen grab failed: {exc}") from exc
        # The one place the device pixel ratio is observed: whatever the display's scale,
        # the grab's own width against the requested width is the truth.
        size = capture_size(rect, shot.width / max(rect.w, 1.0), max_pixels)
        if size != image.size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    def _mss(self) -> Any:
        instance = getattr(self._local, "mss", None)
        if instance is None:
            import mss
            import mss.darwin

            # Drop `kCGWindowImageNominalResolution` from mss's defaults, which is the
            # documented knob (its own comment on `IMAGE_OPTIONS`) for capturing at the
            # display's real resolution. It is worth a full doubling of every linear
            # dimension on a Retina display, and a face is only embedded when its crop is
            # wide enough: at nominal resolution a window-sized photo fails the quality
            # gate on `width_below_min_embed_px` and is never identified at all.
            mss.darwin.IMAGE_OPTIONS = (
                mss.darwin.kCGWindowImageBoundsIgnoreFraming
                | mss.darwin.kCGWindowImageShouldBeOpaque
            )
            instance = mss.MSS()
            self._local.mss = instance
        return instance


def _quartz() -> Any:
    """Imported lazily so `watch.sources` can be imported on any platform."""
    if sys.platform != "darwin":  # pragma: no cover - guarded by platform_backend()
        raise UnsupportedPlatformError("Quartz is macOS only")
    import Quartz

    return Quartz


def _window_info(entry: Any) -> WindowInfo | None:
    """One `CGWindowListCopyWindowInfo` entry, or None when it is not a real window.

    Layer 0 is the ordinary window layer: menus, the dock, the cursor and status items all
    live above it and none of them is ever the thing the operator wants to watch.
    """
    if entry.get("kCGWindowLayer") != 0:
        return None
    owner = str(entry.get("kCGWindowOwnerName") or "").strip()
    if not owner:
        return None
    rect = _bounds_rect(entry.get("kCGWindowBounds"))
    if rect is None or rect.w < MIN_WINDOW_EDGE or rect.h < MIN_WINDOW_EDGE:
        return None
    window_id = entry.get("kCGWindowNumber")
    if not isinstance(window_id, int):
        return None
    title = str(entry.get("kCGWindowName") or "").strip()
    return WindowInfo(
        window_id=window_id,
        label=f"{owner} — {title}" if title else owner,
        rect=rect,
    )


def _bounds_rect(bounds: Any) -> Rect | None:
    if not bounds:
        return None
    try:
        return Rect(
            x=float(bounds["X"]), y=float(bounds["Y"]), w=float(bounds["Width"]),
            h=float(bounds["Height"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _monitor_rect(monitor: Any) -> Rect:
    return Rect(
        x=float(monitor["left"]),
        y=float(monitor["top"]),
        w=float(monitor["width"]),
        h=float(monitor["height"]),
    )
