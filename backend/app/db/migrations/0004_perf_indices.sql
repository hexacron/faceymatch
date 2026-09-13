-- 0004_perf_indices: indices for the three queries that were scanning whole tables.
-- No column, constraint or trigger changes: the schema means exactly what it meant before.

-- Rematch selects every track embedded under the active model. Partial on the same
-- predicate the query carries, so the index holds only rows a rematch can score.
CREATE INDEX tracks_embedder ON tracks (embedder_model_id) WHERE embedding_mean IS NOT NULL;

-- GET /api/review filters rank-1 rows by band and orders by score descending. Without this
-- the planner scans matches and builds a temp b-tree for the ORDER BY.
CREATE INDEX matches_review ON matches (band, score DESC) WHERE rank = 1;

-- The media list joins the latest job per media through json_extract; nothing indexed it,
-- so every list and every detail page partitioned a full scan of jobs.
CREATE INDEX jobs_media ON jobs (json_extract(params_json, '$.media_id'));

ANALYZE;
