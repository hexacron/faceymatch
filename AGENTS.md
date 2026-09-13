# AGENTS.md

Local face match system, built for investigations. Full spec: `docs/spec.md`. Read it before
any work. Biometric material here is special-category data whatever its provenance: spec
section 12 applies in full, and investigative use raises that bar rather than lowering it.

## How to work

- Build one milestone at a time (spec section 14).
- Done: M0, M1, M1.5 (screen acquisition and live match, spec 6.10), M2 (video: decode, track, match, player overlay). Next: M3 (clustering).
- A milestone is done only when its exit test passes. Do not start the next one before that.
- Plan first. List the files you will change. Then build.
- If the spec is unclear or wrong, stop and ask. Do not guess. Do not change the spec without approval.
- Keep diffs small. One concern per commit.

## Invariants (never break)

1. No outbound network calls at runtime. No telemetry. No dependency that phones home.
2. Never compare embeddings with different `model_id` values.
3. Only operator actions create templates. Auto-matches never create templates. Tagging is not
   enrolling: `confirm` and `reassign` create a template only when the request sets
   `enroll: true`; `decision = "new"` bootstraps exactly one (D17, spec 6.6).
4. Auto-accept runs only when the active threshold set has `calibrated = true`. Otherwise all matches stay candidates.
5. An operator decision always wins. Re-match never overwrites a row with `source = operator`.
6. Every write path appends to the audit log. The log is append-only and hash-chained.
7. Hash every ingested file (SHA-256) before processing.
8. Verify model files against `models.lock` at start. Refuse to start on mismatch.
9. Load non-commercial models only when `allow_noncommercial_models` is true. It ships false
   and is an audited operator decision (`PATCH /api/config`, reason required to enable), not
   an env-only flag. The license stays visible whether it is on or off. `models.lock`
   verification (invariant 8) is unaffected: that is integrity, not licensing.
10. No age, gender, emotion, or attribute models. Do not load them from the buffalo_l pack.
11. Bind the server to `127.0.0.1` only.
12. Every match stores `best_template_id` and `threshold_set_id`.
13. No identity or template may derive from transient pixels. Match-only paths persist nothing;
    enrolment and tagging act only on stored detections (spec 6.10, 12).

## Stack

- Backend: Python 3.12, FastAPI, Uvicorn, Pydantic v2, onnxruntime, PyAV, scikit-learn, SQLite + sqlite-vec.
- Frontend: React + Vite + TypeScript (strict), Sigma.js + graphology.
- Package managers: `uv` (Python), `bun` (frontend).
- No Redis, no Postgres, no Node runtime in production. The backend serves the frontend build.

## Commands

```
./run                            # everything: startup checks, API, worker; Ctrl-C stops all
./run --watch                    # ... plus the watch helper (spec 6.11, macOS only)
                                 # logs in data/logs; --port and --no-ui-build also exist

cd backend
uv sync --extra dev              # backend deps (uv-managed CPython: sqlite needs loadable extensions)
uv run pytest                    # backend tests
uv run ruff check . ../eval ../tools && uv run mypy   # one ruff.toml at the repo root; mypy covers app + watch + eval + tools
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
uv run python -m app.worker      # job worker, separate process
uv run python -m app.cli check   # startup checks: schema, models.lock
uv run python -m app.cli verify-audit
uv run python -m app.cli enqueue-reembed   # re-embed stored crops under the active embedder
                                           # (spec 6.3). PATCH /api/config queues it too.

# watch helper (spec 6.11): operator-started desktop app, macOS only. Qt and the capture
# libraries live in the `watch` extra, so a plain `uv sync` never pulls them.
uv sync --extra dev --extra watch
uv run --extra watch python -m watch           # --url, --fps; needs the API running

# minimal calibration (spec 10): writes the report and an inactive calibrated threshold_set.
# Activate it with POST /api/threshold_sets/{id}/activate; nothing auto-accepts before that.
uv run python ../eval/run.py ../fixtures --output ../eval/report.json

# performance: medians over a scratch DB, a synthetic gallery and fixture faces. Never
# touches data/facematch.db. Quote its JSON before and after any change that claims a speedup.
uv run python ../tools/bench_perf.py

# weights: build-time only, digest-pinned, never fetched at runtime (C1)
uv run --python 3.12 --no-project python tools/fetch_models.py       # from repo root
uv run --python 3.12 --no-project python tools/fetch_models.py --allow-noncommercial

cd frontend && bun install && bun run typecheck && bun run build   # output frontend/dist, served by the backend

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
- Do not commit face images. Test fixtures load from `FIXTURES_DIR` (gitignored). Tests that need
  a real face skip when that path holds none, so a fresh clone still passes.
- Do not commit model files. They load from `models/` (gitignored) and match `models.lock`.

## Out of scope

- Camera or sensor capture (webcam, phone, capture card, network stream).
- Unattended or continuous monitoring, and alerting. Operator-initiated screen capture and
  live match of the operator's own display are in scope (spec 6.10), including the watch
  helper that follows one chosen window until the operator stops it (spec 6.11).
- Web or third-party face search.
- Mobile clients.
