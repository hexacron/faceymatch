"""The control window, and the only way to enrol from the helper.

The overlay draws and offers; this decides. A click on a box opens a menu whose one action is
the enrolment path below, the same one the panel's own rows run, so what may be saved has one
rule and one place it happens.

Enrolment is the tier 1 path unchanged (spec 6.10, 6.11, invariant 13): the retained identify
frame becomes hashed, audited evidence through `POST /api/media`, and the decision is then
made in the browser against the stored detection. Nothing here writes an identity.
"""

from __future__ import annotations

import webbrowser
from collections import deque
from datetime import UTC, datetime
from time import monotonic

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from watch.client import BackendError, Client, Face, MatchResult
from watch.config import (
    BAND_COLOR,
    DEFAULT_FPS,
    FPS_CHOICES,
    LABEL_MIN_IOU,
    NO_BAND_COLOR,
    OVERLAY_CLICKS_DEFAULT,
)
from watch.geometry import Rect, best_overlap, normalized
from watch.overlay import Box, Overlay, RegionSelector
from watch.session import REASON_WINDOW_CLOSED, Session
from watch.sources import CaptureBackend, CaptureError, Target

# Faces beyond this many are drawn but not listed: the panel is a control surface, not a
# scrolling report, and a crowd scene is not what the helper is for.
MAX_ROWS = 8

REGION_ITEM = "Region…"

NO_CASE_REFUSAL = "Select a case before saving a frame: evidence has to belong to one."
NO_IDENTIFY_YET = "Waiting for the first identify pass — nothing is stored yet."
NOT_IN_IDENTIFY = (
    "That face was not in the last identify pass, so there is no stored frame showing it. "
    "Give it a moment and click again."
)
OCCLUSION_NOTE = (
    "Window targets capture the screen where the window is, so keep this panel off it: "
    "anything overlapping the window is captured with it."
)


class TargetCombo(QComboBox):
    """A combo that re-reads the window list every time it is opened."""

    about_to_open = Signal()

    def showPopup(self) -> None:  # noqa: N802 - Qt's name
        self.about_to_open.emit()
        super().showPopup()


class Row:
    """One face in the current frame: what it is, and the one action it affords."""

    def __init__(self, index: int, on_save: object) -> None:
        self.widget = QWidget()
        layout = QHBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.button = QPushButton()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.button.clicked.connect(on_save)
        self.index = index
        self.widget.hide()


class Panel(QWidget):
    def __init__(self, client: Client, backend: CaptureBackend, fps: float = DEFAULT_FPS) -> None:
        super().__init__()
        self._client = client
        self._backend = backend
        self._overlay = Overlay()
        self._session = Session(backend, client)
        self._target: Target | None = None
        self._targets: list[Target | None] = []
        self._latest: MatchResult | None = None
        self._identities: MatchResult | None = None
        self._identity_of: list[Face | None] = []
        self._case_name = ""
        # Arrival times of the last few results: the telemetry block reports measured fps.
        self._ticks: deque[float] = deque(maxlen=8)

        self.setWindowTitle("faceymatch watch")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self._build(fps)
        self._overlay.picked.connect(self._on_picked)
        self._overlay.set_interactive(self._clickable.isChecked())

        self._session.result.connect(self._on_result)
        self._session.failed.connect(self._on_failed)
        self._session.ended.connect(self._on_ended)

        self._reload_cases()
        self._reload_targets()

    # ------------------------------------------------------------------ layout

    def _build(self, fps: float) -> None:
        layout = QVBoxLayout(self)

        self._cases = QComboBox()
        self._cases.currentIndexChanged.connect(self._on_case_changed)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._reload_cases)
        layout.addLayout(_labelled("Case", self._cases, refresh))

        self._target_combo = TargetCombo()
        self._target_combo.about_to_open.connect(self._reload_targets)
        self._target_combo.activated.connect(self._on_target_chosen)
        layout.addLayout(_labelled("Watch", self._target_combo))

        self._fps = QComboBox()
        for choice in FPS_CHOICES:
            self._fps.addItem(f"{choice:g} fps")
        self._fps.setCurrentIndex(_nearest_fps(fps))
        layout.addLayout(_labelled("Sample rate", self._fps))

        self._start = QPushButton("Start")
        self._start.clicked.connect(self._on_start)
        self._stop = QPushButton("Stop")
        self._stop.clicked.connect(self._on_stop)
        self._stop.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self._start)
        buttons.addWidget(self._stop)
        layout.addLayout(buttons)

        self._clickable = QCheckBox("Click faces on the overlay")
        self._clickable.setChecked(OVERLAY_CLICKS_DEFAULT)
        self._clickable.setToolTip(
            "Boxes and labels take the click. Turn this off to click straight through to the "
            "window underneath."
        )
        self._clickable.toggled.connect(self._overlay.set_interactive)
        layout.addWidget(self._clickable)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        faces = QGroupBox("Faces in frame")
        self._faces_layout = QVBoxLayout(faces)
        self._empty = QLabel("No frame yet.")
        self._faces_layout.addWidget(self._empty)
        self._rows = [Row(i, _saver(self, i)) for i in range(MAX_ROWS)]
        for row in self._rows:
            self._faces_layout.addWidget(row.widget)
        layout.addWidget(faces)

        self.resize(560, 420)

    # ------------------------------------------------------------- population

    def _reload_cases(self) -> None:
        try:
            cases = self._client.cases()
        except BackendError as exc:
            self._say(str(exc))
            return
        self._cases.clear()
        self._cases.addItem("— select a case —", "")
        for case in cases:
            self._cases.addItem(case.name, case.id)
        self._on_case_changed()

    def _reload_targets(self) -> None:
        chosen = self._target
        self._targets = []
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        try:
            windows = self._backend.windows()
            displays = self._backend.displays()
        except CaptureError as exc:
            self._target_combo.blockSignals(False)
            self._say(str(exc))
            return
        for window in windows:
            self._targets.append(
                Target(
                    kind="window",
                    window_id=window.window_id,
                    display_index=None,
                    rect=window.rect,
                    label=window.label,
                )
            )
            self._target_combo.addItem(window.label)
        for display in displays:
            self._targets.append(
                Target(
                    kind="display",
                    window_id=None,
                    display_index=display.index,
                    rect=display.rect,
                    label=display.label,
                )
            )
            self._target_combo.addItem(display.label)
        if chosen is not None and chosen.kind == "region":
            self._targets.append(chosen)
            self._target_combo.addItem(chosen.label)
        self._targets.append(None)  # the sentinel that runs the drag selector
        self._target_combo.addItem(REGION_ITEM)
        self._select(chosen)
        self._target_combo.blockSignals(False)

    def _select(self, previous: Target | None) -> None:
        """Keep the operator's choice across a refresh, by identity rather than by position."""
        if previous is None:
            self._target = self._targets[0] if self._targets else None
            return
        for index, target in enumerate(self._targets):
            if target is not None and _same_target(target, previous):
                self._target_combo.setCurrentIndex(index)
                # The rect moves between refreshes; the newer one wins.
                self._target = target
                return
        self._target = self._targets[0] if self._targets else None
        self._target_combo.setCurrentIndex(0)

    # ---------------------------------------------------------------- actions

    def _on_case_changed(self) -> None:
        self._case_name = self._cases.currentText() if self._case_id() else ""
        for row in self._rows:
            row.button.setText(self._button_text())

    def _on_target_chosen(self, index: int) -> None:
        if index < 0 or index >= len(self._targets):
            return
        target = self._targets[index]
        if target is not None:
            self._target = target
            self._say(OCCLUSION_NOTE if target.kind == "window" else "")
            return
        selector = RegionSelector()
        if selector.exec() != int(RegionSelector.DialogCode.Accepted) or selector.selection is None:
            self._select(self._target)
            return
        rect = selector.selection
        region = Target(
            kind="region",
            window_id=None,
            display_index=None,
            rect=rect,
            label=f"Region — {round(rect.w)}x{round(rect.h)}",
        )
        self._target = region
        self._reload_targets()

    def _on_start(self) -> None:
        target = self._target
        if target is None:
            self._say("Pick a window, a display or a region first.")
            return
        try:
            self._backend.self_test(target)
        except CaptureError as exc:
            self._say(str(exc))
            return
        self._latest = None
        self._identities = None
        self._identity_of = []
        self._ticks.clear()
        self._render_faces()
        self._overlay.begin(self._backend, target)
        self._session.start(target, self._case_id(), FPS_CHOICES[self._fps.currentIndex()])
        self._start.setEnabled(False)
        self._stop.setEnabled(True)
        self._say(f"Watching {target.label}.")

    def _on_stop(self) -> None:
        self._session.stop()
        self._finish()
        self._say("Stopped.")

    def _finish(self) -> None:
        self._overlay.finish()
        self._start.setEnabled(True)
        self._stop.setEnabled(False)

    # ----------------------------------------------------------- session slots

    def _on_result(self, match: MatchResult) -> None:
        self._latest = match
        if match.identified:
            self._identities = match
        self._identity_of = self._resolve(match)
        self._ticks.append(monotonic())
        self._render_faces()
        self._overlay.set_boxes(self._boxes(match), match.width, match.height)
        self._overlay.set_telemetry(self._telemetry(match))
        self._say(self._status_text(match))

    def _on_failed(self, message: str) -> None:
        self._say(message)

    def _on_ended(self, reason: str) -> None:
        self._finish()
        if reason == REASON_WINDOW_CLOSED:
            self._say("The window closed, so watching stopped.")

    # ------------------------------------------------------------- rendering

    def _resolve(self, match: MatchResult) -> list[Face | None]:
        """The identify-pass face behind each box, by IoU against the last identify result.

        Normalised to fractions of each frame first: the two cadences encode at different
        pixel budgets, so their boxes are only comparable as fractions. Computed once per
        result — the panel and the overlay read the same answer.
        """
        identities = self._identities
        if identities is None:
            return [None] * len(match.faces)
        rects = [
            normalized(_rect(other), identities.width, identities.height)
            for other in identities.faces
        ]
        resolved: list[Face | None] = []
        for face in match.faces:
            target = normalized(_rect(face), match.width, match.height)
            index = best_overlap(rects, target, LABEL_MIN_IOU)
            resolved.append(None if index is None else identities.faces[index])
        return resolved

    def _boxes(self, match: MatchResult) -> list[Box]:
        boxes: list[Box] = []
        pairs = zip(match.faces, self._identity_of, strict=True)
        for index, (face, identity) in enumerate(pairs):
            top = identity.candidates[0] if identity is not None and identity.candidates else None
            colour = BAND_COLOR.get(top.band, NO_BAND_COLOR) if top is not None else NO_BAND_COLOR
            # Dashed means "not embeddable", which the identify pass decides. Reading the
            # boxes-only tick's verdict would dash every box on two ticks in three.
            verdict = identity if identity is not None else face
            boxes.append(
                Box(
                    rect=_rect(face),
                    color=colour,
                    solid=verdict.quality_passed,
                    index=index,
                    label=_chip_text(face, identity),
                )
            )
        return boxes

    def _render_faces(self) -> None:
        faces = self._latest.faces if self._latest is not None else ()
        self._empty.setVisible(not faces)
        for row in self._rows:
            if row.index >= len(faces):
                row.widget.hide()
                continue
            face = faces[row.index]
            identity = self._identity_of[row.index] if row.index < len(self._identity_of) else None
            row.label.setText(_describe(face, identity))
            row.button.setText(self._button_text())
            row.widget.show()

    def _status_text(self, match: MatchResult) -> str:
        parts = [f"{match.gallery_persons} in the gallery", f"{match.elapsed_ms} ms"]
        if not match.auto_accept_allowed:
            reason = match.auto_accept_reason or "the active threshold set is not calibrated"
            parts.append(f"nothing self-confirms: {reason}")
        if self._target is not None and self._target.kind == "window":
            parts.append(OCCLUSION_NOTE)
        return " · ".join(parts)

    def _telemetry(self, match: MatchResult) -> list[str]:
        """The overlay's four lines: cadence, per-stage cost, gallery, and what may happen."""
        cadence = "identify" if match.identified else "boxes only"
        timings = match.timings
        return [
            f"{self._measured_fps():.1f} fps · {cadence} · {match.elapsed_ms} ms",
            f"decode {timings.get('decode', 0.0):.0f} · detect {timings.get('detect', 0.0):.0f}"
            f" · embed {timings.get('embed', 0.0):.0f} · match {timings.get('match', 0.0):.1f} ms",
            f"{len(match.faces)} faces · {match.gallery_persons} in the gallery",
            f"{self._case_name or 'no case selected'} · "
            f"{'auto-accept armed' if match.auto_accept_allowed else 'nothing self-confirms'}",
        ]

    def _measured_fps(self) -> float:
        """Measured arrivals, not the configured rate: the combo already shows that one."""
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0.0:
            return 0.0
        return (len(self._ticks) - 1) / span

    def _say(self, message: str) -> None:
        self._status.setText(message)

    def _button_text(self) -> str:
        return (
            f"Save frame to {self._case_name} and tag this face"
            if self._case_name
            else "Save frame and tag this face"
        )

    def _case_id(self) -> str:
        data = self._cases.currentData()
        return data if isinstance(data, str) else ""

    # ------------------------------------------------------------- enrolment

    def _refusal(self, identity: Face | None) -> str | None:
        """Why this face cannot be saved yet, or None when it can."""
        if not self._case_id():
            return NO_CASE_REFUSAL
        if self._session.last_identify_frame() is None or self._identities is None:
            return NO_IDENTIFY_YET
        if identity is None:
            return NOT_IN_IDENTIFY
        return None

    def save_and_tag(self, index: int) -> None:
        """Store the retained identify frame and open the tag panel on this face.

        The identity comes from `_identity_of`, not from a row: rows stop at `MAX_ROWS`, and
        the overlay draws — and now offers — every face in the frame.
        """
        identity = self._identity_of[index] if index < len(self._identity_of) else None
        refusal = self._refusal(identity)
        if refusal is not None:
            self._say(refusal)
            return
        retained = self._session.last_identify_frame()
        if retained is None or identity is None:  # narrowing; _refusal already covered both
            return
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        try:
            media_id = self._client.ingest(
                retained.jpeg,
                case_id=self._case_id(),
                filename=f"watch-{stamp}.jpg",
                capture_mode=self._target.capture_mode if self._target else "screen",
            )
        except BackendError as exc:
            self._say(f"Could not save the frame: {exc}")
            return
        # The identify face against the identify frame's own dimensions: those are the bytes
        # that were just stored, and a boxes-only tick's rect would point into a frame that
        # does not exist as evidence.
        box = normalized(_rect(identity), retained.width, retained.height)
        focus = ",".join(f"{value:.5f}" for value in (box.x, box.y, box.w, box.h))
        webbrowser.open(f"{self._client.base_url}/#/media/{media_id}/box/{focus}")
        self._say(f"Saved as {media_id}. The tag panel is open in your browser.")

    def _on_picked(self, index: int, position: QPoint) -> None:
        """A click on a box or its label: offer the one action, or say why it is refused."""
        faces = self._latest.faces if self._latest is not None else ()
        if index >= len(faces):
            return
        identity = self._identity_of[index] if index < len(self._identity_of) else None
        menu = QMenu(self)
        # The panel's own long form, so the exact score and the band are one click from the
        # chip's rounded percentage.
        header = menu.addAction(_describe(faces[index], identity))
        header.setEnabled(False)
        menu.addSeparator()
        action = menu.addAction(self._button_text())
        refusal = self._refusal(identity)
        if refusal is not None:
            action.setEnabled(False)
            note = menu.addAction(refusal)
            note.setEnabled(False)
        # A disabled action cannot come back from `exec`, so a hit here needs no second check.
        if menu.exec(position) is action:
            self.save_and_tag(index)

    # ---------------------------------------------------------------- closing

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's name
        self._session.stop()
        self._overlay.finish()
        self._overlay.close()
        super().closeEvent(event)


def _saver(panel: Panel, index: int) -> object:
    """A click handler bound to one row, without capturing the loop variable."""
    return lambda _checked=False: panel.save_and_tag(index)


def _describe(face: Face, identity: Face | None) -> str:
    """What this face is, read off the pass that decided it.

    The quality verdict comes from `identity` — the identify pass — and not from the box on
    screen. A boxes-only tick is encoded at a fourteenth of the identify budget, so nearly
    every face fails `width_below_min_embed_px` there; reading that verdict would flip the
    row between a name and a refusal twice per second and neither reading would be about the
    frame the gallery actually scored.
    """
    verdict = identity if identity is not None else face
    if not verdict.quality_passed:
        reason = verdict.quality_reasons[0] if verdict.quality_reasons else "quality gate"
        return f"Not matched · quality gate: {reason}"
    if identity is None:
        return "Identifying…"
    if not identity.candidates:
        return "Unidentified"
    top = identity.candidates[0]
    return f"{top.name} · {top.band} · {top.score:.3f}"


# The chip is drawn over the operator's own screen, inches from the next face and its own
# chip, so a rejection has to read in a glance and in as little width as the box it labels:
# "quality gate: width_below_min_embed_px" is wider than most faces and buries its
# neighbours. The panel row and the click menu have a column to themselves, so they keep the
# exact reason (see `_describe`) and this is the only place the wording is shortened. Keys are
# every reason `app.pipeline.quality` can emit; they arrive over HTTP as plain strings, which
# is why they are restated here rather than imported.
CHIP_REASON_PHRASE: dict[str, str] = {
    "width_below_min_embed_px": "too small",
    "yaw_above_max": "turned away",
    "sharpness_below_min": "blurred",
    "det_score_below_min": "uncertain",
}

# A reason this build does not know is still a refusal, and saying so beats printing an
# identifier at the operator.
CHIP_REASON_FALLBACK = "low quality"


def _chip_text(face: Face, identity: Face | None) -> str:
    """The box's own caption. Short, and read off the pass that decided it (see `_describe`)."""
    verdict = identity if identity is not None else face
    if not verdict.quality_passed:
        reason = verdict.quality_reasons[0] if verdict.quality_reasons else ""
        return CHIP_REASON_PHRASE.get(reason, CHIP_REASON_FALLBACK)
    if identity is None:
        return "identifying…"
    if not identity.candidates:
        return "no match"
    top = identity.candidates[0]
    # The percentage is the similarity score, not a probability; the exact figure and the
    # band stay on the panel row and in the click menu.
    return f"{top.name} · {top.score * 100:.0f}%"


def _rect(face: Face) -> Rect:
    return Rect(face.x, face.y, face.w, face.h)


def _same_target(a: Target, b: Target) -> bool:
    return (
        a.kind == b.kind and a.window_id == b.window_id and a.display_index == b.display_index
    )


def _nearest_fps(fps: float) -> int:
    return min(range(len(FPS_CHOICES)), key=lambda i: abs(FPS_CHOICES[i] - fps))


def _labelled(text: str, *widgets: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    label = QLabel(text)
    label.setMinimumWidth(90)
    row.addWidget(label)
    for widget in widgets:
        row.addWidget(widget)
    return row
