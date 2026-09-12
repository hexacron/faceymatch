-- 0003: durable runtime overrides for the editable subset of Settings.
--
-- Settings come from the environment and .env, which a running process cannot change and
-- an operator cannot audit. A model switch is an operator decision with evidentiary weight
-- (spec 6.3, 9), so the value that decided it has to live in the database next to the audit
-- entry that records it, not in a file nobody hashed.
--
-- One row per overridden key. Absent key = the environment value stands, so this table is a
-- sparse overlay and never a second copy of the whole configuration.
--
-- Only the keys in app/runtime_config.py EDITABLE_FIELDS are read back. A hand-edited row
-- for anything else is ignored, which is what keeps allow_noncommercial_models env-only
-- (invariant 9): a licensing gate that a database write can flip is not a gate.

CREATE TABLE runtime_config (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL
);
