-- 0001_init: full v1 schema (spec section 7).
-- Invariants that SQLite can enforce are enforced here with CHECK constraints and
-- triggers, so a future code path cannot violate them by omission.

CREATE TABLE cases (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    authorization_basis TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    created_by          TEXT NOT NULL
);

CREATE TABLE models (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    version        TEXT NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('detector', 'embedder')),
    sha256         TEXT NOT NULL,
    license        TEXT NOT NULL,
    commercial_use INTEGER NOT NULL CHECK (commercial_use IN (0, 1)),
    dim            INTEGER,
    CHECK (kind = 'detector' OR dim IS NOT NULL)
);

-- calibrated = 1 is the auto-accept gate (C5, invariant 4). A calibrated set must carry
-- the evidence that justifies it: when, from which eval report, at what gallery size, and
-- on which execution provider (CoreML and CPU EPs do not produce bit-identical scores).
CREATE TABLE threshold_sets (
    id                  TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL REFERENCES models (id),
    t_strong            REAL NOT NULL,
    t_possible          REAL NOT NULL,
    margin              REAL NOT NULL CHECK (margin >= 0.0),
    calibrated          INTEGER NOT NULL DEFAULT 0 CHECK (calibrated IN (0, 1)),
    calibrated_at       TEXT,
    eval_report_sha256  TEXT,
    gallery_size        INTEGER,
    execution_provider  TEXT,
    active              INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1)),
    created_at          TEXT NOT NULL,
    CHECK (t_strong >= t_possible),
    CHECK (
        calibrated = 0
        OR (calibrated_at IS NOT NULL
            AND eval_report_sha256 IS NOT NULL
            AND gallery_size IS NOT NULL
            AND execution_provider IS NOT NULL)
    )
);

-- At most one active threshold set.
CREATE UNIQUE INDEX threshold_sets_single_active
    ON threshold_sets (active) WHERE active = 1;

CREATE TABLE media (
    id          TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL REFERENCES cases (id),
    sha256      TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('image', 'video')),
    path        TEXT NOT NULL,
    source_url  TEXT,
    acquired_at TEXT,
    width       INTEGER,
    height      INTEGER,
    duration_ms INTEGER,
    fps         REAL,
    ingested_at TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('new', 'processing', 'done', 'failed'))
);

CREATE INDEX media_case ON media (case_id);
CREATE INDEX media_sha256 ON media (sha256);
CREATE UNIQUE INDEX media_case_sha256 ON media (case_id, sha256);

-- Persons are global (D3). status = 'unenrolled' means every template was purged with its
-- source case (section 12): the row survives for audit and graph history, but the person is
-- excluded from matching.
CREATE TABLE persons (
    id                 TEXT PRIMARY KEY,
    display_name       TEXT NOT NULL,
    notes              TEXT,
    do_not_enroll      INTEGER NOT NULL DEFAULT 0 CHECK (do_not_enroll IN (0, 1)),
    status             TEXT NOT NULL DEFAULT 'enrolled'
                       CHECK (status IN ('enrolled', 'unenrolled')),
    enrolled_in_case_id TEXT REFERENCES cases (id),
    created_at         TEXT NOT NULL,
    created_by         TEXT NOT NULL
);

-- A still image produces exactly one detection row per face with frame_idx = 0 and t_ms = 0.
-- (media_id, frame_idx, det_idx) is the idempotency key that makes a resumed job safe.
CREATE TABLE detections (
    id                TEXT PRIMARY KEY,
    media_id          TEXT NOT NULL REFERENCES media (id),
    track_id          TEXT REFERENCES tracks (id),
    t_ms              INTEGER NOT NULL,
    frame_idx         INTEGER NOT NULL,
    det_idx           INTEGER NOT NULL,
    x                 REAL NOT NULL,
    y                 REAL NOT NULL,
    w                 REAL NOT NULL,
    h                 REAL NOT NULL,
    landmarks_json    TEXT NOT NULL,
    det_score         REAL NOT NULL,
    quality_json      TEXT NOT NULL,
    crop_sha256       TEXT,
    detector_model_id TEXT NOT NULL REFERENCES models (id)
);

CREATE UNIQUE INDEX detections_frame_slot ON detections (media_id, frame_idx, det_idx);
CREATE INDEX detections_track ON detections (track_id);
CREATE INDEX detections_media_time ON detections (media_id, t_ms);

-- One track per still-image detection (start_ms = end_ms = 0) so images and video share one
-- matching and identity code path.
CREATE TABLE tracks (
    id                TEXT PRIMARY KEY,
    media_id          TEXT NOT NULL REFERENCES media (id),
    start_ms          INTEGER NOT NULL,
    end_ms            INTEGER NOT NULL,
    best_detection_id TEXT REFERENCES detections (id),
    cluster_id        TEXT REFERENCES clusters (id),
    embedding_mean    BLOB,
    embedder_model_id TEXT REFERENCES models (id),
    CHECK (end_ms >= start_ms),
    CHECK ((embedding_mean IS NULL) = (embedder_model_id IS NULL))
);

CREATE INDEX tracks_media ON tracks (media_id);
CREATE INDEX tracks_cluster ON tracks (cluster_id);

-- Per-crop embeddings (spec 6.2 step 6). Enrollment from a stored detection reads this, and
-- a model switch re-embeds from the stored aligned crop.
CREATE TABLE detection_embeddings (
    detection_id      TEXT NOT NULL REFERENCES detections (id),
    embedding         BLOB NOT NULL,
    embedder_model_id TEXT NOT NULL REFERENCES models (id),
    created_at        TEXT NOT NULL,
    PRIMARY KEY (detection_id, embedder_model_id)
);

CREATE TABLE templates (
    id                TEXT PRIMARY KEY,
    person_id         TEXT NOT NULL REFERENCES persons (id),
    detection_id      TEXT REFERENCES detections (id),
    source_case_id    TEXT REFERENCES cases (id),
    embedding         BLOB NOT NULL,
    embedder_model_id TEXT NOT NULL REFERENCES models (id),
    quality           REAL,
    status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
    created_at        TEXT NOT NULL,
    created_by        TEXT NOT NULL
);

CREATE INDEX templates_person ON templates (person_id, status);
CREATE INDEX templates_model ON templates (embedder_model_id, status);

-- best_template_id and threshold_set_id are NOT NULL: invariant 12.
CREATE TABLE matches (
    id                TEXT PRIMARY KEY,
    track_id          TEXT NOT NULL REFERENCES tracks (id),
    person_id         TEXT NOT NULL REFERENCES persons (id),
    rank              INTEGER NOT NULL CHECK (rank >= 1),
    score             REAL NOT NULL,
    band              TEXT NOT NULL CHECK (band IN ('strong', 'possible', 'ambiguous', 'unknown')),
    best_template_id  TEXT NOT NULL REFERENCES templates (id),
    threshold_set_id  TEXT NOT NULL REFERENCES threshold_sets (id),
    embedder_model_id TEXT NOT NULL REFERENCES models (id),
    created_at        TEXT NOT NULL
);

CREATE UNIQUE INDEX matches_track_rank ON matches (track_id, rank);
CREATE INDEX matches_person ON matches (person_id);

-- Current answer per track. identifications below holds operator history.
CREATE TABLE identities (
    track_id         TEXT PRIMARY KEY REFERENCES tracks (id),
    person_id        TEXT NOT NULL REFERENCES persons (id),
    source           TEXT NOT NULL CHECK (source IN ('auto', 'operator')),
    match_id         TEXT REFERENCES matches (id),
    threshold_set_id TEXT NOT NULL REFERENCES threshold_sets (id),
    updated_at       TEXT NOT NULL
);

CREATE INDEX identities_person ON identities (person_id);

CREATE TABLE identifications (
    id         TEXT PRIMARY KEY,
    track_id   TEXT NOT NULL REFERENCES tracks (id),
    person_id  TEXT REFERENCES persons (id),
    decision   TEXT NOT NULL CHECK (
                   decision IN ('confirm', 'reject', 'reassign', 'new', 'cluster_assign')
               ),
    operator   TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX identifications_track ON identifications (track_id, created_at);

CREATE TABLE clusters (
    id              TEXT PRIMARY KEY,
    job_id          TEXT REFERENCES jobs (id),
    scope           TEXT NOT NULL,
    size            INTEGER NOT NULL,
    label_person_id TEXT REFERENCES persons (id),
    created_at      TEXT NOT NULL
);

CREATE TABLE jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL CHECK (
                   kind IN ('ingest', 'process', 'rematch', 'reembed', 'cluster',
                            'export', 'audit_verify')
               ),
    status     TEXT NOT NULL CHECK (
                   status IN ('queued', 'running', 'done', 'failed', 'cancelled')
               ),
    params_json TEXT NOT NULL DEFAULT '{}',
    progress    TEXT NOT NULL DEFAULT '{}',
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX jobs_queue ON jobs (status, created_at);

-- Append-only, hash-chained (invariant 6). seq is assigned inside the writing transaction,
-- never by AUTOINCREMENT, because seq is part of the hashed entry.
CREATE TABLE audit_log (
    seq          INTEGER PRIMARY KEY,
    ts           TEXT NOT NULL,
    actor        TEXT NOT NULL,
    case_id      TEXT,
    action       TEXT NOT NULL,
    object_type  TEXT NOT NULL,
    object_id    TEXT,
    payload_json TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    hash         TEXT NOT NULL UNIQUE
);

-- Invariant 6: the audit log is append-only. No code path may rewrite history.
CREATE TRIGGER audit_log_no_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
END;

CREATE TRIGGER audit_log_no_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
END;

-- Invariant 5: an operator decision always wins. A re-match may not downgrade it.
CREATE TRIGGER identities_operator_wins
BEFORE UPDATE ON identities
WHEN OLD.source = 'operator' AND NEW.source <> 'operator'
BEGIN
    SELECT RAISE(ABORT, 'operator identity cannot be overwritten by an auto result');
END;

-- Invariant 4 / C5: auto-accept only under a calibrated threshold set.
CREATE TRIGGER identities_auto_requires_calibrated_insert
BEFORE INSERT ON identities
WHEN NEW.source = 'auto'
    AND (SELECT calibrated FROM threshold_sets WHERE id = NEW.threshold_set_id) IS NOT 1
BEGIN
    SELECT RAISE(ABORT, 'auto identity requires a calibrated threshold set');
END;

CREATE TRIGGER identities_auto_requires_calibrated_update
BEFORE UPDATE ON identities
WHEN NEW.source = 'auto'
    AND (SELECT calibrated FROM threshold_sets WHERE id = NEW.threshold_set_id) IS NOT 1
BEGIN
    SELECT RAISE(ABORT, 'auto identity requires a calibrated threshold set');
END;

-- Section 12: an unenrolled person is out of the gallery, so nothing may auto-accept to them.
CREATE TRIGGER identities_auto_requires_enrolled
BEFORE INSERT ON identities
WHEN NEW.source = 'auto'
    AND (SELECT status FROM persons WHERE id = NEW.person_id) <> 'enrolled'
BEGIN
    SELECT RAISE(ABORT, 'auto identity requires an enrolled person');
END;

-- Invariant 2: never compare embeddings across model_id values. A match row must agree with
-- both the track it scored and the template that produced the score.
CREATE TRIGGER matches_same_model
BEFORE INSERT ON matches
WHEN (NEW.embedder_model_id
          IS NOT (SELECT embedder_model_id FROM tracks WHERE id = NEW.track_id))
    OR (NEW.best_template_id IS NOT NULL
        AND NEW.embedder_model_id
            IS NOT (SELECT embedder_model_id FROM templates WHERE id = NEW.best_template_id))
BEGIN
    SELECT RAISE(ABORT, 'match crosses embedder_model_id boundary');
END;

-- Section 12: do_not_enroll blocks template creation for that person.
CREATE TRIGGER templates_respect_do_not_enroll
BEFORE INSERT ON templates
WHEN (SELECT do_not_enroll FROM persons WHERE id = NEW.person_id) = 1
BEGIN
    SELECT RAISE(ABORT, 'person is marked do_not_enroll');
END;

-- Section 7: co-occurrence is a view, never a stored edge table. Two identified people in the
-- same frame of the same media.
CREATE VIEW v_co_occurrence AS
SELECT
    d1.media_id                             AS media_id,
    d1.frame_idx                            AS frame_idx,
    d1.t_ms                                 AS t_ms,
    i1.person_id                            AS person_a,
    i2.person_id                            AS person_b,
    i1.source                               AS source_a,
    i2.source                               AS source_b,
    d1.id                                   AS detection_a,
    d2.id                                   AS detection_b
FROM detections d1
JOIN identities i1 ON i1.track_id = d1.track_id
JOIN detections d2
     ON d2.media_id = d1.media_id
    AND d2.frame_idx = d1.frame_idx
JOIN identities i2 ON i2.track_id = d2.track_id
WHERE i1.person_id < i2.person_id;
