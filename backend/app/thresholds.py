"""Threshold-set bootstrap (spec 10, C5).

An embedder with no threshold set at all has no bands, so nothing can be reviewed and
nothing can be reported. Seeding one uncalibrated set gives the operator a working, honest
starting point: `calibrated = 0` means auto-accept stays off (invariant 4) until a real
calibration run is activated through `POST /api/threshold_sets/{id}/activate`.

The seed is not a calibration and must never be mistaken for one. Its numbers are a safe
default, not a measurement, which is exactly why the calibrated flag is the gate rather
than the presence of a set.
"""

from __future__ import annotations

import sqlite3

from app import audit
from app.ids import new_id

SEED_T_STRONG = 0.55
SEED_T_POSSIBLE = 0.35
SEED_MARGIN = 0.05


def seed_default_threshold_set(
    conn: sqlite3.Connection, *, model_id: str, actor: str
) -> str | None:
    """Create and activate the one uncalibrated bootstrap set when this embedder has none.

    Inside the caller's transaction, so a model switch commits the new configuration and the
    set its bands will come from together. Returns the new id, or None when the model is
    unknown or already has a set of its own.

    Activating it deactivates whatever was active before, which after a model switch is a
    set calibrated for the *previous* embedder. Bands from that set would be a threshold
    measured on one model applied to another model's scores; the previous set is not
    deleted, and the audit entry names it, so switching back is a re-activation away.
    """
    if conn.execute("SELECT 1 FROM models WHERE id = ?", (model_id,)).fetchone() is None:
        return None
    existing = conn.execute(
        "SELECT id FROM threshold_sets WHERE model_id = ? LIMIT 1", (model_id,)
    ).fetchone()
    if existing is not None:
        return None
    previous = conn.execute("SELECT id FROM threshold_sets WHERE active = 1").fetchone()
    threshold_set_id = new_id()
    now = audit.now_ts()
    conn.execute("UPDATE threshold_sets SET active = 0 WHERE active = 1")
    conn.execute(
        "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
        "calibrated, calibrated_at, eval_report_sha256, gallery_size, "
        "execution_provider, active, created_at) "
        "VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, 1, ?)",
        (threshold_set_id, model_id, SEED_T_STRONG, SEED_T_POSSIBLE, SEED_MARGIN, now),
    )
    audit.append(
        conn,
        actor=actor,
        action="threshold_set.seed",
        object_type="threshold_set",
        object_id=threshold_set_id,
        payload={
            "model_id": model_id,
            "t_strong": SEED_T_STRONG,
            "t_possible": SEED_T_POSSIBLE,
            "margin": SEED_MARGIN,
            "calibrated": False,
            "previous_active_id": None if previous is None else str(previous["id"]),
        },
    )
    return threshold_set_id
