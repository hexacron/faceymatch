"""Case CRUD (spec section 8).

`authorization_basis` is required at creation: section 12 makes it the record that justifies
processing biometric data for this case. It is also correctable, because a basis that no
longer describes the material is worse than no basis at all — but never silently: the
amendment carries the previous text into the audit chain, so the chain shows what the
justification used to say as well as what it says now.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app import audit
from app.api.deps import ConnDep, SettingsDep
from app.db.conn import transaction
from app.ids import new_id

router = APIRouter(prefix="/api/cases", tags=["cases"])


class CaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    authorization_basis: str = Field(min_length=1, max_length=2000)


class CaseAmendAuthorization(BaseModel):
    authorization_basis: str = Field(min_length=1, max_length=2000)
    reason: str | None = Field(default=None, max_length=2000)


class CaseOut(BaseModel):
    id: str
    name: str
    authorization_basis: str
    created_at: str
    created_by: str

class CaseListOut(BaseModel):
    items: list[CaseOut]


@router.get("", response_model=CaseListOut)
def list_cases(conn: ConnDep) -> CaseListOut:
    rows = conn.execute(
        "SELECT * FROM cases ORDER BY created_at DESC, id DESC"
    ).fetchall()
    return CaseListOut(
        items=[
            CaseOut(
                id=str(row["id"]),
                name=str(row["name"]),
                authorization_basis=str(row["authorization_basis"]),
                created_at=str(row["created_at"]),
                created_by=str(row["created_by"]),
            )
            for row in rows
        ]
    )


@router.post("", response_model=CaseOut, status_code=status.HTTP_201_CREATED)
def create_case(body: CaseCreate, conn: ConnDep, settings: SettingsDep) -> CaseOut:
    case_id = new_id()
    now = audit.now_ts()
    with transaction(conn):
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, body.name, body.authorization_basis, now, settings.operator_name),
        )
        audit.append(
            conn,
            actor=settings.operator_name,
            action="case.create",
            object_type="case",
            object_id=case_id,
            case_id=case_id,
            payload={"name": body.name, "authorization_basis": body.authorization_basis},
        )
    return CaseOut(
        id=case_id,
        name=body.name,
        authorization_basis=body.authorization_basis,
        created_at=now,
        created_by=settings.operator_name,
    )


@router.get("/{case_id}", response_model=CaseOut)
def get_case(case_id: str, conn: ConnDep) -> CaseOut:
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case not found")
    return CaseOut(
        id=str(row["id"]),
        name=str(row["name"]),
        authorization_basis=str(row["authorization_basis"]),
        created_at=str(row["created_at"]),
        created_by=str(row["created_by"]),
    )


@router.patch("/{case_id}", response_model=CaseOut)
def amend_authorization_basis(
    case_id: str, body: CaseAmendAuthorization, conn: ConnDep, settings: SettingsDep
) -> CaseOut:
    """Correct a case's `authorization_basis`. The correction is itself evidence.

    The previous text, the new text and the operator's reason all go into one audit entry
    in the same transaction as the update (invariant 6), so nothing about the change is
    recoverable only from a backup.
    """
    with transaction(conn):
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case not found")
        previous = str(row["authorization_basis"])
        conn.execute(
            "UPDATE cases SET authorization_basis = ? WHERE id = ?",
            (body.authorization_basis, case_id),
        )
        audit.append(
            conn,
            actor=settings.operator_name,
            action="case.amend_authorization",
            object_type="case",
            object_id=case_id,
            case_id=case_id,
            payload={
                "previous_authorization_basis": previous,
                "authorization_basis": body.authorization_basis,
                "reason": body.reason,
            },
        )
    return CaseOut(
        id=case_id,
        name=str(row["name"]),
        authorization_basis=body.authorization_basis,
        created_at=str(row["created_at"]),
        created_by=str(row["created_by"]),
    )
