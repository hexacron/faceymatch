-- 0002: an operator decision must not require a threshold set.
--
-- 0001 made identities.threshold_set_id NOT NULL, which is right for auto-accepted rows
-- (they are only legal under a calibrated set) but wrong for operator decisions: before
-- calibration exists there is no set to reference, and tagging must still work. SQLite
-- cannot relax a NOT NULL in place, so the table is rebuilt.
--
-- The CHECK keeps the invariant that matters: source = 'auto' still requires provenance.

DROP VIEW v_co_occurrence;

CREATE TABLE identities_new (
    track_id         TEXT PRIMARY KEY REFERENCES tracks (id),
    person_id        TEXT NOT NULL REFERENCES persons (id),
    source           TEXT NOT NULL CHECK (source IN ('auto', 'operator')),
    match_id         TEXT REFERENCES matches (id),
    threshold_set_id TEXT REFERENCES threshold_sets (id),
    updated_at       TEXT NOT NULL,
    CHECK (source <> 'auto' OR threshold_set_id IS NOT NULL)
);

INSERT INTO identities_new (track_id, person_id, source, match_id, threshold_set_id, updated_at)
SELECT track_id, person_id, source, match_id, threshold_set_id, updated_at FROM identities;

-- Dropping the old table drops its triggers, so they are recreated below verbatim.
DROP TABLE identities;
ALTER TABLE identities_new RENAME TO identities;

CREATE INDEX identities_person ON identities (person_id);

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

CREATE VIEW v_co_occurrence AS
SELECT
    d1.media_id AS media_id,
    d1.frame_idx AS frame_idx,
    d1.t_ms AS t_ms,
    i1.person_id AS person_a,
    i2.person_id AS person_b,
    i1.source AS source_a,
    i2.source AS source_b,
    d1.id AS detection_a,
    d2.id AS detection_b
FROM detections d1
JOIN identities i1 ON i1.track_id = d1.track_id
JOIN detections d2
     ON d2.media_id = d1.media_id
    AND d2.frame_idx = d1.frame_idx
JOIN identities i2 ON i2.track_id = d2.track_id
WHERE i1.person_id < i2.person_id;
