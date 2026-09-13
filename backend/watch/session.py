"""The sampling loop: grab, match, emit.

A `QObject` owning one worker thread. Qt signals are safe to emit from a worker thread and
are delivered queued on the GUI thread, so the loop never touches a widget and the widgets
never block on a request.

One request is in flight at a time and frames are dropped rather than queued, the same rule
the Live view follows: a backlog of stale frames is worse than a lower rate, because every
one of them describes where the face used to be.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from watch.client import BackendError, Client, MatchResult
from watch.config import (
    BOX_MAX_PIXELS,
    BOX_QUALITY,
    IDENTIFY_MAX_PIXELS,
    IDENTIFY_QUALITY,
    MIN_PERIOD_MS,
)
from watch.geometry import identifies
from watch.sources import CaptureBackend, CaptureError, Target

# How long to wait after a refusal before asking again. A backend that is down, or a case
# that was deleted, does not get hammered at the sample rate.
BACKOFF_S = 1.0

STOP_JOIN_S = 2.0

REASON_STOPPED = "stopped"
REASON_WINDOW_CLOSED = "window closed"


@dataclass(frozen=True, slots=True)
class RetainedFrame:
    """The last identify frame: the only bytes a click may ever store (invariant 13)."""

    jpeg: bytes
    width: int
    height: int


class Session(QObject):
    result = Signal(object)  # MatchResult, either cadence
    identified = Signal(object)  # MatchResult from an identify tick only
    failed = Signal(str)  # human-readable, shown in the panel
    ended = Signal(str)  # REASON_STOPPED | REASON_WINDOW_CLOSED

    def __init__(self, backend: CaptureBackend, client: Client) -> None:
        super().__init__()
        self._backend = backend
        self._client = client
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._retained: RetainedFrame | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, target: Target, case_id: str, fps: float) -> None:
        self.stop()
        stop = threading.Event()
        self._stop = stop
        with self._lock:
            self._retained = None
        self._thread = threading.Thread(
            target=self._run,
            args=(target, case_id, fps, stop),
            name="watch-session",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        self._thread = None
        self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            # A tick already on the wire finishes on its own; the loop checks the flag
            # before it emits anything, so a late answer is discarded rather than drawn.
            thread.join(timeout=STOP_JOIN_S)

    def last_identify_frame(self) -> RetainedFrame | None:
        with self._lock:
            return self._retained

    def _run(self, target: Target, case_id: str, fps: float, stop: threading.Event) -> None:
        reason = REASON_STOPPED
        tick = 0
        while not stop.is_set():
            began = time.monotonic()
            identify = identifies(tick)
            rect = self._backend.bounds(target)
            if rect is None:
                reason = REASON_WINDOW_CLOSED
                break
            try:
                frame = self._backend.grab(
                    rect,
                    max_pixels=IDENTIFY_MAX_PIXELS if identify else BOX_MAX_PIXELS,
                    quality=IDENTIFY_QUALITY if identify else BOX_QUALITY,
                )
                match = self._client.match(frame.jpeg, case_id=case_id, identify=identify)
            except (BackendError, CaptureError) as exc:
                if stop.is_set():
                    break
                self.failed.emit(str(exc))
                stop.wait(BACKOFF_S)
                continue
            if stop.is_set():
                break
            if identify:
                # Only identify frames are retained, because those are the exact bytes the
                # gallery scored and the only ones fit to become evidence (spec 6.10, 6.11).
                with self._lock:
                    self._retained = RetainedFrame(frame.jpeg, frame.width, frame.height)
            self._emit(match)
            tick += 1
            elapsed_ms = (time.monotonic() - began) * 1000.0
            # The measured round trip, not the backend's own figure: encode and wire are
            # most of what a busy machine cannot keep up with, and neither is in `timings`.
            period_ms = max(1000.0 / fps, float(MIN_PERIOD_MS), 2.0 * elapsed_ms)
            stop.wait(max(0.0, period_ms - elapsed_ms) / 1000.0)
        self.ended.emit(reason)

    def _emit(self, match: MatchResult) -> None:
        self.result.emit(match)
        if match.identified:
            self.identified.emit(match)
