"""Threshold set listing and explicit audited activation (spec section 10)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app import audit
from app.api.deps import ConnDep, SettingsDep
from app.db.conn import transaction

router = APIRouter(prefix="/api/threshold_sets", tags=["threshold-sets"])


class ThresholdSetOut(BaseModel):
    id: str
    model_id: str
    t_strong: float
    t_possible: float
    margin: float
    calibrated: bool
    calibrated_at: str | None
    eval_report_sha256: str | None
    gallery_size: int | None
    execution_provider: str | None
    active: bool


class ThresholdSetListOut(BaseModel):
    items: list[ThresholdSetOut]


@router.get("", response_model=ThresholdSetListOut)
def list_threshold_sets(conn: ConnDep) -> ThresholdSetListOut:
    rows = conn.execute(
        "SELECT * FROM threshold_sets ORDER BY active DESC, created_at DESC, id DESC"
    ).fetchall()
    return ThresholdSetListOut(items=[_out(row) for row in rows])


@router.post("/{threshold_set_id}/activate", response_model=ThresholdSetOut)
def activate_threshold_set(
    threshold_set_id: str, conn: ConnDep, settings: SettingsDep
) -> ThresholdSetOut:
    row = conn.execute(
        "SELECT * FROM threshold_sets WHERE id = ?", (threshold_set_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="threshold set not found")
    with transaction(conn):
        previous = conn.execute(
            "SELECT id FROM threshold_sets WHERE active = 1"
        ).fetchone()
        conn.execute("UPDATE threshold_sets SET active = 0 WHERE active = 1")
        conn.execute("UPDATE threshold_sets SET active = 1 WHERE id = ?", (threshold_set_id,))
        audit.append(
            conn,
            actor=settings.operator_name,
            action="threshold_set.activate",
            object_type="threshold_set",
            object_id=threshold_set_id,
            payload={
                "previous_id": None if previous is None else str(previous["id"]),
                "model_id": str(row["model_id"]),
                "calibrated": bool(row["calibrated"]),
            },
        )
    activated = conn.execute(
        "SELECT * FROM threshold_sets WHERE id = ?", (threshold_set_id,)
    ).fetchone()
    if activated is None:
        raise HTTPException(status_code=500, detail="threshold set activation failed")
    return _out(activated)


def _out(row: sqlite3.Row) -> ThresholdSetOut:
    return ThresholdSetOut(
        id=str(row["id"]),
        model_id=str(row["model_id"]),
        t_strong=float(row["t_strong"]),
        t_possible=float(row["t_possible"]),
        margin=float(row["margin"]),
        calibrated=bool(row["calibrated"]),
        calibrated_at=None if row["calibrated_at"] is None else str(row["calibrated_at"]),
        eval_report_sha256=(
            None if row["eval_report_sha256"] is None else str(row["eval_report_sha256"])
        ),
        gallery_size=None if row["gallery_size"] is None else int(row["gallery_size"]),
        execution_provider=(
            None if row["execution_provider"] is None else str(row["execution_provider"])
        ),
        active=bool(row["active"]),
    )
