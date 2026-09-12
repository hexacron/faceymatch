# AGENTS.md

Local face match system. Full spec: `docs/spec.md`. Read it before any work.

## How to work

- Build one milestone at a time (spec section 14).
- A milestone is done only when its exit test passes. Do not start the next one before that.
- Plan first. List the files you will change. Then build.
- If the spec is unclear or wrong, stop and ask. Do not guess. Do not change the spec without approval.
- Keep diffs small. One concern per commit.

## Invariants (never break)

1. No outbound network calls at runtime. No telemetry. No dependency that phones home.
2. Never compare embeddings with different `model_id` values.
3. Only operator actions create templates. Auto-matches never create templates.
4. Auto-accept runs only when the active threshold set has `calibrated = true`. Otherwise all matches stay candidates.
5. An operator decision always wins. Re-match never overwrites a row with `source = operator`.
6. Every write path appends to the audit log. The log is append-only and hash-chained.
7. Hash every ingested file (SHA-256) before processing.
8. Verify model files against `models.lock` at start. Refuse to start on mismatch.
9. Load non-commercial models only when `ALLOW_NONCOMMERCIAL_MODELS=true`.
10. No age, gender, emotion, or attribute models. Do not load them from the buffalo_l pack.
11. Bind the server to `127.0.0.1` only.
12. Every match stores `best_template_id` and `threshold_set_id`.

## Stack

- Backend: Python 3.12, FastAPI, Uvicorn, Pydantic v2, onnxruntime, PyAV, scikit-learn, SQLite + sqlite-vec.
- Frontend: React + Vite + TypeScript (strict), Sigma.js + graphology.
- Package managers: `uv` (Python), `bun` (frontend).
- No Redis, no Postgres, no Node runtime in production. The backend serves the frontend build.

## Commands

```
cd backend
uv sync --extra dev              # backend deps (uv-managed CPython: sqlite needs loadable extensions)
uv run pytest                    # backend tests
uv run ruff check . && uv run mypy app
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
uv run python -m app.worker      # job worker, separate process
uv run python -m app.cli check   # startup checks: schema, models.lock
uv run python -m app.cli verify-audit

# weights: build-time only, digest-pinned, never fetched at runtime (C1)
uv run --python 3.12 --no-project python tools/fetch_models.py       # from repo root
uv run --python 3.12 --no-project python tools/fetch_models.py --allow-noncommercial

cd frontend && bun install && bun run build   # output frontend/dist, served by the backend

docker compose up                # x86 Beelink only; Apple Silicon runs natively (CoreML EP)
```

Update this section when commands change.

## Code rules

- Python: full type hints, Pydantic models at API edges, `ruff` and `mypy --strict` clean.
- TypeScript: `strict: true`, no `any` without a comment that says why.
- All model code goes through the `Detector` and `Embedder` interfaces in `app/core/`. No direct model calls elsewhere.
- Thresholds, sample rate, and quality gates come from config. No magic numbers in pipeline code.
- DB changes go through migrations in `app/db/migrations/`.

## Tests

- CI runs with the SFace adapter, so the buffalo_l swap stays a config change.
- Test each invariant above with at least one test.
- Do not commit face images. Test fixtures load from the path in `FIXTURES_DIR` (gitignored).
- Do not commit model files. They load from `models/` (gitignored) and match `models.lock`.

## Out of scope

- Webcam or live input.
- Web or third-party face search.
- Mobile clients.
