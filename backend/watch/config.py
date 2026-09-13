"""Every number and colour the helper uses, in one place (spec 6.11).

Nothing here is a magic number buried in a loop: sample rates, pixel budgets and the identify
cadence are the knobs that decide whether the helper keeps up with a moving face, so they are
readable in one screen and changeable in one edit.
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("FACEYMATCH_URL", "http://127.0.0.1:8000")

# Invariants 1 and 11: the backend binds loopback, and the helper talks to nothing else.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

DEFAULT_FPS = 3.0
FPS_CHOICES: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0)
MIN_PERIOD_MS = 150

# Every third tick asks for names; the other two ask only for boxes. Same split, and the
# same reasons, as the Live view: embedding is most of a tick and a box that tracks the
# face is what the operator is watching.
IDENTIFY_EVERY = 3
IDENTIFY_MAX_PIXELS = 16_800_000
IDENTIFY_QUALITY = 92
BOX_MAX_PIXELS = 1_000_000
BOX_QUALITY = 70

LABEL_MIN_IOU = 0.3
FOLLOW_INTERVAL_MS = 200  # how often the target's rectangle is re-read
REQUEST_TIMEOUT_S = 10.0

# The values in `frontend/src/lib/display.ts`. They are restated rather than imported
# because the helper shares no code with the bundle, and a divergence would make the overlay
# and the web UI disagree about what a band looks like.
BAND_COLOR = {
    "strong": "#7fbf7f",
    "possible": "#d9b96f",
    "ambiguous": "#c98fd0",
    "unknown": "#8b95a4",
}
NO_BAND_COLOR = "#5a626e"

# Overlay labels and telemetry. The operator is looking at the watched window, not at the
# panel, so the name, the match and the per-stage cost are drawn where they are looking.
LABEL_FONT_PT = 11
LABEL_PAD_X = 6
LABEL_PAD_Y = 3
LABEL_GAP = 4  # points between a chip and the box edge it labels
CHIP_BG_RGBA = (18, 22, 28, 208)
CHIP_TEXT = "#e6e9ee"

# The overlay's only input surface: one square per face, outside its top-left corner. Boxes
# and chips take no clicks at all, so hover-driven controls in the watched window (a video
# player's auto-hiding bar) keep working while the helper draws over them.
HANDLE_PX = 18
HANDLE_GAP = 2
HANDLE_FONT_PT = 9
HANDLE_BORDER = "#12161c"
HANDLE_ALPHA = 208

HUD_FONT_PT = 10
HUD_PAD = 8
HUD_MARGIN = 10
HUD_LINE_GAP = 2
HUD_BG_RGBA = (18, 22, 28, 184)
HUD_TEXT = "#c7ccd4"

# Clicks on the drawn regions are on by default: the ask is to click a face. The panel's
# checkbox turns them off when the operator needs the window underneath instead.
OVERLAY_CLICKS_DEFAULT = True

# How many boxes can be clicked at once: one transparent window each (`watch.overlay`). The
# same figure, and the same reason, as the panel's row limit — a crowd scene is not what the
# helper is for, and beyond this the boxes are still drawn and still named.
OVERLAY_HIT_WINDOWS = 8
