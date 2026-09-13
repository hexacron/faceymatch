"""Boxes over the watched window, and the drag selector that picks a region.

Both are the same frameless translucent machinery, which is why they live together.

The drawing overlay takes no input at all: `WindowTransparentForInput` keeps the window server
from ever offering it a click, so every pixel of the watched window stays reachable. What takes
a click is one small transparent window per box, `HitWindow`, moved over the box and its label
on every frame. That is the mechanism because a mask is not one: `setMask` clips this window's
painting, but the window server still hands it every click inside its frame (measured), so a
masked overlay would swallow clicks meant for the watched application. A window the window
server knows about cannot be ambiguous that way, and turning clicks off is then just hiding
those windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QDialog, QWidget

from watch.config import (
    CHIP_BG_RGBA,
    CHIP_TEXT,
    FOLLOW_INTERVAL_MS,
    HUD_BG_RGBA,
    HUD_FONT_PT,
    HUD_LINE_GAP,
    HUD_MARGIN,
    HUD_PAD,
    HUD_TEXT,
    LABEL_FONT_PT,
    LABEL_GAP,
    LABEL_PAD_X,
    LABEL_PAD_Y,
    OVERLAY_CLICKS_DEFAULT,
    OVERLAY_HIT_WINDOWS,
)
from watch.geometry import Rect, chip_rect, frame_to_overlay
from watch.sources import CaptureBackend, Target

BOX_WIDTH = 2
SELECTION_COLOR = "#7fbf7f"
SELECTION_FILL = QColor(127, 191, 127, 40)
SHADE = QColor(0, 0, 0, 90)
MIN_REGION_EDGE = 16


@dataclass(frozen=True, slots=True)
class Box:
    """One face box, in captured-frame pixels — the overlay converts."""

    rect: Rect
    color: str
    solid: bool  # dashed when the quality gate rejected the face, as the web overlay draws
    index: int  # this face's position in the result, so a click can name it
    label: str  # "Robert Downey Jr · 90%" | "no match" | "identifying…" | "too small"


class HitWindow(QWidget):
    """A transparent window over one box: the only thing in the helper that takes a click.

    It draws nothing. Its whole frame is its input surface, which is exactly the point: the
    window server decides by window, so a click inside is unambiguously a click on that face
    and a click anywhere else was never ours to begin with.
    """

    pressed = Signal(int, QPoint)  # face index, global position of the click

    def __init__(self, index: int) -> None:
        super().__init__()
        self._index = index
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def place(self, index: int, rect: QRect) -> None:
        """Cover `rect` on behalf of face `index`."""
        self._index = index
        self.setGeometry(rect)
        self.show()
        self.raise_()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's name
        self.pressed.emit(self._index, event.globalPosition().toPoint())
        event.accept()


class Overlay(QWidget):
    """An always-on-top window sized to the target, with a clickable window per box."""

    picked = Signal(int, QPoint)  # face index, global position of the click

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._boxes: tuple[Box, ...] = ()
        self._hud: tuple[str, ...] = ()
        self._frame = (0, 0)
        self._target = Rect(0.0, 0.0, 0.0, 0.0)
        self._backend: CaptureBackend | None = None
        self._watching: Target | None = None
        # Built once and kept: a font per frame is waste at 3 to 8 fps.
        self._label_font = QFont()
        self._label_font.setPointSize(LABEL_FONT_PT)
        self._hud_font = QFont()
        self._hud_font.setPointSize(HUD_FONT_PT)
        self._interactive = OVERLAY_CLICKS_DEFAULT
        self._hits = [HitWindow(index) for index in range(OVERLAY_HIT_WINDOWS)]
        for window in self._hits:
            window.pressed.connect(self.picked.emit)
        self._follow = QTimer(self)
        self._follow.setInterval(FOLLOW_INTERVAL_MS)
        self._follow.timeout.connect(self._reposition)

    def begin(self, backend: CaptureBackend, target: Target) -> None:
        """Show over `target` and keep up with it until `finish()`."""
        self._backend = backend
        self._watching = target
        self._boxes = ()
        self._hud = ()
        self._place(target.rect)
        self.show()
        self._follow.start()

    def finish(self) -> None:
        self._follow.stop()
        self._backend = None
        self._watching = None
        self._boxes = ()
        self._hud = ()
        self._hide_hits(0)
        self.hide()

    def set_boxes(self, boxes: Sequence[Box], frame_w: int, frame_h: int) -> None:
        self._boxes = tuple(boxes)
        self._frame = (frame_w, frame_h)
        self.update()
        self._sync_hits()

    def set_telemetry(self, lines: Sequence[str]) -> None:
        """The telemetry block's lines, drawn in the overlay's top-left corner.

        A readout, not a button: no hit window covers it, so a click there reaches the
        watched application like any other pixel the overlay merely draws on.
        """
        self._hud = tuple(lines)
        self.update()

    def set_interactive(self, interactive: bool) -> None:
        """Whether the boxes and their labels take a click, or everything passes through."""
        self._interactive = interactive
        self._sync_hits()

    # ---------------------------------------------------------------- layout

    def _layout(self) -> tuple[list[tuple[Box, QRect, QRect]], QRect]:
        """Each box as (box, box rect, chip rect) in overlay-local points, plus the HUD rect.

        The paint pass and the hit windows both read this, so what is drawn and what can be
        clicked cannot drift apart.
        """
        frame_w, frame_h = self._frame
        if frame_w <= 0 or frame_h <= 0 or self._target.w <= 0:
            return [], QRect()
        bounds = (float(self.width()), float(self.height()))
        metrics = QFontMetrics(self._label_font)
        laid: list[tuple[Box, QRect, QRect]] = []
        for box in self._boxes:
            local = frame_to_overlay(box.rect, frame_w, frame_h, self._target)
            size = (
                float(metrics.horizontalAdvance(box.label) + 2 * LABEL_PAD_X),
                float(metrics.height() + 2 * LABEL_PAD_Y),
            )
            chip = chip_rect(local, size, bounds, float(LABEL_GAP))
            laid.append((box, _as_qrect(local), _as_qrect(chip)))
        return laid, self._hud_rect()

    def _hud_rect(self) -> QRect:
        if not self._hud:
            return QRect()
        metrics = QFontMetrics(self._hud_font)
        line_h = metrics.height()
        width = max(metrics.horizontalAdvance(line) for line in self._hud) + 2 * HUD_PAD
        height = len(self._hud) * line_h + (len(self._hud) - 1) * HUD_LINE_GAP + 2 * HUD_PAD
        return QRect(HUD_MARGIN, HUD_MARGIN, width, height)

    def _sync_hits(self) -> None:
        """Move one hit window onto each box, largest first so the smallest ends up on top.

        Nested detections are the reason for the order: a face found inside a larger box has
        to stay reachable, and with one window per box that is z-order rather than arithmetic.
        Beyond the pool the boxes are still drawn and still named in the panel; only the
        click is capped, for the same reason the panel lists eight faces and no more.
        """
        if not self._interactive:
            self._hide_hits(0)
            return
        boxes, _hud = self._layout()
        order = sorted(boxes, key=lambda laid: -laid[1].width() * laid[1].height())
        used = 0
        for box, box_rect, chip in order[: len(self._hits)]:
            # The label belongs to the face it names, so it takes the click too. One window
            # over their union: the gap between them is a few points of dead space at worst.
            self._hits[used].place(box.index, box_rect.united(chip))
            used += 1
        self._hide_hits(used)

    def _hide_hits(self, first: int) -> None:
        for window in self._hits[first:]:
            window.hide()

    # --------------------------------------------------------------- drawing

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt's name
        boxes, hud = self._layout()
        if not boxes and hud.isEmpty():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        if not hud.isEmpty():
            self._paint_hud(painter, hud)
        for box, box_rect, chip in boxes:
            pen = QPen(QColor(box.color))
            pen.setWidth(BOX_WIDTH)
            pen.setStyle(Qt.PenStyle.SolidLine if box.solid else Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(box_rect)
            self._paint_chip(painter, chip, box)
        painter.end()

    def _paint_hud(self, painter: QPainter, rect: QRect) -> None:
        painter.fillRect(rect, QColor(*HUD_BG_RGBA))
        painter.setFont(self._hud_font)
        painter.setPen(QPen(QColor(HUD_TEXT)))
        line_h = QFontMetrics(self._hud_font).height()
        flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for number, line in enumerate(self._hud):
            top = rect.y() + HUD_PAD + number * (line_h + HUD_LINE_GAP)
            painter.drawText(
                QRect(rect.x() + HUD_PAD, top, rect.width() - 2 * HUD_PAD, line_h),
                flags,
                line,
            )

    def _paint_chip(self, painter: QPainter, rect: QRect, box: Box) -> None:
        painter.fillRect(rect, QColor(*CHIP_BG_RGBA))
        border = QPen(QColor(box.color))
        border.setWidth(1)
        painter.setPen(border)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.setFont(self._label_font)
        painter.setPen(QPen(QColor(CHIP_TEXT)))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), box.label)

    # -------------------------------------------------------------- position

    def _reposition(self) -> None:
        """Follow the window. A vanished target is the session's business, not the overlay's."""
        if self._backend is None or self._watching is None:
            return
        rect = self._backend.bounds(self._watching)
        if rect is not None and rect != self._target:
            self._place(rect)

    def _place(self, rect: Rect) -> None:
        self._target = rect
        self.setGeometry(
            round(rect.x), round(rect.y), max(1, round(rect.w)), max(1, round(rect.h))
        )
        # The window just changed size, so every box moved with it.
        self._sync_hits()


class RegionSelector(QDialog):
    """Drag a rectangle over the whole desktop. Release captures it, Escape cancels."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.selection: Rect | None = None
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._desktop = _virtual_desktop()
        self.setGeometry(self._desktop)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's name
        self._origin = event.position().toPoint()
        self._current = self._origin
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's name
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt's name
        if self._origin is None:
            self.reject()
            return
        self._current = event.position().toPoint()
        rect = QRect(self._origin, self._current).normalized()
        if rect.width() < MIN_REGION_EDGE or rect.height() < MIN_REGION_EDGE:
            self.reject()
            return
        self.selection = Rect(
            x=float(rect.x() + self._desktop.x()),
            y=float(rect.y() + self._desktop.y()),
            w=float(rect.width()),
            h=float(rect.height()),
        )
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt's name
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.fillRect(self.rect(), SHADE)
        if self._origin is not None and self._current is not None:
            rect = QRect(self._origin, self._current).normalized()
            painter.fillRect(rect, SELECTION_FILL)
            pen = QPen(QColor(SELECTION_COLOR))
            pen.setWidth(BOX_WIDTH)
            painter.setPen(pen)
            painter.drawRect(rect)
        painter.end()


def _virtual_desktop() -> QRect:
    """Every screen's union, in logical points: the same space window bounds are read in."""
    united = QRect()
    for screen in QGuiApplication.screens():
        united = united.united(screen.geometry())
    return united


def _as_qrect(rect: Rect) -> QRect:
    """Rounded to whole points: the overlay draws and places windows in logical points."""
    return QRect(round(rect.x), round(rect.y), round(rect.w), round(rect.h))
