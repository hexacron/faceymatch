"""Review and explicitly assign groups of stored unknown faces (spec 6.7)."""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from app import audit
from app.api.deps import ConnDep, SettingsDep
from app.db.conn import transaction
from app.enrollment import EnrollmentError, create_person, create_template_from_detection
from app.ids import new_id
from app.jobs import insert

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


class ClusterTrackOut(BaseModel):
    track_id: str
    media_id: str
    case_id: str
    crop_sha256: str | None


class ClusterOut(BaseModel):
    id: str
    scope: str
    size: int
    label_person_id: str | None
    tracks: list[ClusterTrackOut]


class ClusterListOut(BaseModel):
    items: list[ClusterOut]


class ClusterAssignIn(BaseModel):
    person_id: str | None = None
    new_name: str | None = Field(default=None, min_length=1, max_length=200)
    exclude_track_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def one_target(self) -> ClusterAssignIn:
        if (self.person_id is None) == (self.new_name is None):
            raise ValueError("choose exactly one of person_id or new_name")
        return self


class ClusterAssignOut(BaseModel):
    person_id: str
    assigned: int
    excluded: int
    templates_created: int
    rematch_job_id: str


def _members(conn: sqlite3.Connection, cluster_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT t.id AS track_id, t.media_id, m.case_id, t.best_detection_id, "
        "t.embedder_model_id, d.crop_sha256 FROM tracks t "
        "JOIN media m ON m.id = t.media_id "
        "LEFT JOIN detections d ON d.id = t.best_detection_id "
        "WHERE t.cluster_id = ? ORDER BY t.id", (cluster_id,),
    ).fetchall()


@router.get("", response_model=ClusterListOut)
def list_clusters(conn: ConnDep, scope: Annotated[str | None, Query()] = None) -> ClusterListOut:
    rows = conn.execute(
        "SELECT id, scope, size, label_person_id FROM clusters "
        "WHERE (? IS NULL OR scope = ?) AND label_person_id IS NULL "
        "ORDER BY created_at DESC, id DESC", (scope, scope),
    ).fetchall()
    return ClusterListOut(items=[
        ClusterOut(
            id=str(row["id"]), scope=str(row["scope"]), size=int(row["size"]),
            label_person_id=None,
            tracks=[ClusterTrackOut(track_id=str(member["track_id"]),
                                    media_id=str(member["media_id"]),
                                    case_id=str(member["case_id"]),
                                    crop_sha256=member["crop_sha256"])
                    for member in _members(conn, str(row["id"]))],
        ) for row in rows
    ])


@router.post("/{cluster_id}/assign", response_model=ClusterAssignOut)
def assign_cluster(
    cluster_id: str, body: ClusterAssignIn, conn: ConnDep, settings: SettingsDep,
) -> ClusterAssignOut:
    try:
        with transaction(conn):
            cluster = conn.execute(
                "SELECT id, label_person_id FROM clusters WHERE id = ?", (cluster_id,)
            ).fetchone()
            if cluster is None:
                raise LookupError("cluster not found")
            if cluster["label_person_id"] is not None:
                raise ValueError("cluster has already been assigned")
            members = _members(conn, cluster_id)
            ids = {str(row["track_id"]) for row in members}
            excluded = set(body.exclude_track_ids)
            if len(excluded) != len(body.exclude_track_ids) or not excluded <= ids:
                raise ValueError("excluded tracks must be distinct members of this cluster")
            chosen = [row for row in members if row["track_id"] not in excluded]
            if not chosen:
                raise ValueError("at least one track must be assigned")
            for row in chosen:
                if row["embedder_model_id"] != settings.embedder_model:
                    raise ValueError("cluster model differs from active embedder; re-cluster")
                identified = conn.execute(
                    "SELECT 1 FROM identities WHERE track_id = ?", (row["track_id"],)
                ).fetchone()
                if identified:
                    raise ValueError("cluster contains a track already identified; re-cluster")
                if conn.execute(
                    "SELECT 1 FROM identifications WHERE track_id = ? AND decision = 'reject'",
                    (row["track_id"],),
                ).fetchone():
                    raise ValueError("cluster contains a rejected track; re-cluster")
            if body.new_name is not None:
                person_id = create_person(
                    conn, display_name=body.new_name, case_id=str(chosen[0]["case_id"]),
                    actor=settings.operator_name,
                    audit_payload={"display_name": body.new_name, "from_cluster_id": cluster_id},
                )
            else:
                person_id = str(body.person_id)
                person = conn.execute(
                    "SELECT do_not_enroll FROM persons WHERE id = ?", (person_id,)
                ).fetchone()
                if person is None:
                    raise LookupError("person not found")
                if person["do_not_enroll"]:
                    raise ValueError("person is marked do_not_enroll")
            now = audit.now_ts()
            templates = 0
            for row in chosen:
                track_id = str(row["track_id"])
                detection_id = row["best_detection_id"]
                if detection_id is None:
                    raise ValueError("cluster track has no stored detection")
                created = create_template_from_detection(
                    conn, person_id=person_id, detection_id=str(detection_id),
                    embedder_model_id=settings.embedder_model, actor=settings.operator_name,
                )
                templates += created is not None
                match = conn.execute(
                    "SELECT id, threshold_set_id FROM matches WHERE track_id = ? AND person_id = ? "
                    "ORDER BY rank LIMIT 1", (track_id, person_id),
                ).fetchone()
                conn.execute(
                    "INSERT INTO identities "
                    "(track_id, person_id, source, match_id, threshold_set_id, updated_at) "
                    "VALUES (?, ?, 'operator', ?, ?, ?)",
                    (track_id, person_id, match["id"] if match else None,
                     match["threshold_set_id"] if match else None, now),
                )
                identification_id = new_id()
                conn.execute(
                    "INSERT INTO identifications "
                    "(id, track_id, person_id, decision, operator, note, created_at) "
                    "VALUES (?, ?, ?, 'cluster_assign', ?, ?, ?)",
                    (identification_id, track_id, person_id,
                     settings.operator_name, body.reason, now),
                )
                audit.append(
                    conn, actor=settings.operator_name, case_id=str(row["case_id"]),
                    action="identification.cluster_assign", object_type="track", object_id=track_id,
                    payload={
                        "identification_id": identification_id,
                        "person_id": person_id, "cluster_id": cluster_id,
                        "reason": body.reason, "template_created": created is not None,
                    },
                )
            for track_id in excluded:
                conn.execute("UPDATE tracks SET cluster_id = NULL WHERE id = ?", (track_id,))
            conn.execute(
                "UPDATE clusters SET label_person_id = ?, size = ? WHERE id = ?",
                (person_id, len(chosen), cluster_id),
            )
            job_id = insert(
                conn, kind="rematch", actor=settings.operator_name,
                params={"reason": "cluster_enrollment", "cluster_id": cluster_id},
            )
            return ClusterAssignOut(
                person_id=person_id, assigned=len(chosen), excluded=len(excluded),
                templates_created=templates, rematch_job_id=job_id,
            )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, EnrollmentError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
