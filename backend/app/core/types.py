"""Shared pipeline types and the model interfaces from spec 6.3.

Conventions fixed here, because every slice depends on them:

- Images move through the pipeline as `np.ndarray` of shape (H, W, 3), dtype uint8, **RGB**,
  already EXIF-corrected. Adapters that need BGR or float scaling convert internally; no
  caller ever has to know a model's colour convention.
- Bounding boxes are `(x, y, w, h)` in original-image pixel coordinates, floats.
- Landmarks are (5, 2) float32 in original-image pixels, ordered
  [right eye, left eye, nose, right mouth corner, left mouth corner] — the ArcFace order.
- Aligned crops are (112, 112, 3) uint8 RGB.
- Embeddings are float32, L2-normalized, shape (dim,) or (N, dim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

# Closed vocabularies. Each one mirrors a CHECK constraint in the schema, so a value the
# database cannot hold is also a value the API cannot serialise: a bad row fails here
# instead of reaching the client as an unrecognised string.
Band = Literal["strong", "possible", "ambiguous", "unknown"]
IdentitySource = Literal["auto", "operator"]
MediaKind = Literal["image", "video"]
MediaStatus = Literal["new", "processing", "done", "failed"]
PersonStatus = Literal["enrolled", "unenrolled"]
TemplateStatus = Literal["active", "revoked"]
# What an operator may ask for through the identification API.
Decision = Literal["confirm", "reject", "reassign", "new"]
# What `identifications.decision` may already hold: `cluster_assign` is written by the
# M4 clustering path, so track history has to be able to report it.
RecordedDecision = Literal["confirm", "reject", "reassign", "new", "cluster_assign"]

CROP_SIZE = 112
LANDMARK_COUNT = 5

# ArcFace 5-point reference template for a 112x112 crop, in
# [right eye, left eye, nose, right mouth, left mouth] order.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected face in original-image pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray  # (5, 2) float32

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def clipped(self, width: int, height: int) -> Detection:
        """Clamp the box to the image; detectors can emit boxes past the border."""
        x0 = max(0.0, self.x)
        y0 = max(0.0, self.y)
        x1 = min(float(width), self.x + self.w)
        y1 = min(float(height), self.y + self.h)
        return Detection(
            x=x0,
            y=y0,
            w=max(0.0, x1 - x0),
            h=max(0.0, y1 - y0),
            score=self.score,
            landmarks=self.landmarks,
        )


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Outcome of the spec 6.2 step 4 gate. `reasons` is empty iff `passed`."""

    passed: bool
    width_px: float
    yaw_deg: float
    sharpness: float
    det_score: float
    reasons: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "width_px": round(self.width_px, 2),
            "yaw_deg": round(self.yaw_deg, 2),
            "sharpness": round(self.sharpness, 2),
            "det_score": round(self.det_score, 4),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class PersonCandidate:
    """A scored gallery person for one probe vector."""

    person_id: str
    score: float
    best_template_id: str
    rank: int


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The numeric part of an active threshold set."""

    t_strong: float
    t_possible: float
    margin: float


@runtime_checkable
class Detector(Protocol):
    model_id: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """image: (H, W, 3) uint8 RGB. Returns detections in image pixels."""
        ...


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, 112, 112, 3) uint8 RGB aligned. Returns (N, dim) float32, L2-normed."""
        ...
