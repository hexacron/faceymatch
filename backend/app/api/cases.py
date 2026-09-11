"""Case CRUD (spec section 8).

`authorization_basis` is required at creation: section 12 makes it the record that justifies
processing biometric data for this case.
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


class CaseOut(BaseModel):
    id: str
    name: str
    authorization_basis: str
    created_at: str
    created_by: str


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
