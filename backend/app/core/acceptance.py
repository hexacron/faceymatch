"""Auto-acceptance gate (spec 6.5, C5, invariant 4).

The gate is a value object, not a scattered set of `if` statements: the pipeline builds one
per job from the active threshold set plus the runtime model and execution provider, and
records its `reason` when auto-accept is off. Every blocked reason keeps matches as
candidates rather than failing the job — an uncalibrated install still ingests and matches,
it just never claims an identity by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.types import Band, Thresholds

# Spec 10: open-set false positives grow with gallery size.
GALLERY_WARN_FACTOR = 2.0
GALLERY_BLOCK_FACTOR = 5.0


@dataclass(frozen=True, slots=True)
class ThresholdSet:
    """An active threshold set as the gate needs to see it."""

    id: str
    model_id: str
    thresholds: Thresholds
    calibrated: bool
    gallery_size: int | None
    execution_provider: str | None


@dataclass(frozen=True, slots=True)
class AutoAcceptGate:
    allowed: bool
    reason: str | None
    warning: str | None
    threshold_set_id: str
    thresholds: Thresholds

    def accepts(self, band: Band) -> bool:
        """Only `strong` is auto-accepted (D16), and only when the gate is open."""
        return self.allowed and band == "strong"


def build_gate(
    threshold_set: ThresholdSet,
    *,
    embedder_model_id: str,
    execution_provider: str,
    live_gallery_size: int,
) -> AutoAcceptGate:
    """Decide whether auto-accept may run, and say why not when it may not."""
    reason: str | None = None
    warning: str | None = None

    if not threshold_set.calibrated:
        reason = "active threshold set is not calibrated (C5)"
    elif threshold_set.model_id != embedder_model_id:
        reason = (
            f"threshold set was calibrated for {threshold_set.model_id!r} but the active "
            f"embedder is {embedder_model_id!r} (invariant 2)"
        )
    elif threshold_set.execution_provider != execution_provider:
        reason = (
            f"threshold set was calibrated on {threshold_set.execution_provider!r} but the "
            f"runtime execution provider is {execution_provider!r}; scores are not "
            "reproducible across providers"
        )
    elif threshold_set.gallery_size is not None and threshold_set.gallery_size > 0:
        ratio = live_gallery_size / threshold_set.gallery_size
        if ratio > GALLERY_BLOCK_FACTOR:
            reason = (
                f"live gallery is {live_gallery_size} persons, more than "
                f"{GALLERY_BLOCK_FACTOR:g}x the {threshold_set.gallery_size} it was "
                "calibrated at; recalibrate before auto-accepting"
            )
        elif ratio > GALLERY_WARN_FACTOR:
            warning = (
                f"live gallery is {live_gallery_size} persons, over "
                f"{GALLERY_WARN_FACTOR:g}x the calibrated {threshold_set.gallery_size}; "
                "false positive rate is above the calibrated target"
            )

    return AutoAcceptGate(
        allowed=reason is None,
        reason=reason,
        warning=warning,
        threshold_set_id=threshold_set.id,
        thresholds=threshold_set.thresholds,
    )
