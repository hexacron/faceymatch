# faceymatch

A local face match system for investigations. It ingests images, detects and embeds faces,
matches them against a gallery the operator built by hand, and shows how people connect
through shared appearances. It runs entirely on the operator's machine.

The design is specified in [`docs/spec.md`](docs/spec.md). That document is authoritative;
this file is how to run the thing. `AGENTS.md` holds the working rules for changing it.

Biometric material is special-category data under GDPR Art. 9 whatever its provenance, and
investigative use raises that bar rather than lowering it. Spec section 12 applies in full.
Get legal review before any product or case use.

Status: milestones M0 (skeleton, schema, audit chain, job worker), M1 (images end to end
with a minimal calibration) and M1.5 (screen acquisition and live match) are done. M2,
video, is next. Clustering, the graph view and exports come after it.

## What it does not do

- No camera or sensor capture: no webcam, phone, capture card or network stream. Operator-
  initiated capture of the operator's own display is in scope; unattended monitoring is not.
- No unattended or continuous monitoring, and no alerting.
- No web or third-party face search. Nothing leaves the machine.
- No age, gender, emotion or attribute estimation. Those models are not loaded even when
  they ship inside a pack we use.

## Invariants

These are enforced by code, schema constraints and tests, not by convention. The full list
with rationale is in `AGENTS.md`; the ones that shape how you operate the system:

- **No outbound network calls at runtime.** Weights are provisioned at build time by a
  standalone script that the runtime package does not import.
- **The server binds `127.0.0.1` only.** Remote access is a `tailscale serve` proxy onto
  loopback, never a tailnet bind. There is no app-level authentication in v1.
- **Only operator actions create templates.** An automatic match never enrolls anybody.
- **Auto-accept runs only under a calibrated, activated threshold set.** Without one, every
  match stays a candidate awaiting a decision.
- **An operator decision always wins.** Re-matching never overwrites it.
- **Every write appends to an append-only, hash-chained audit log.**
- **Every ingested file is SHA-256 hashed before it is processed**, and every model file is
  verified against `models.lock` before it is run.
- **Embeddings from different models are never compared**, in code or in SQL.
- **Nothing identifiable derives from transient pixels.** The live match path stores nothing
  and therefore cannot tag or enroll; acting on a face means storing the frame as evidence
  first.

## Stack

Python 3.12, FastAPI, Uvicorn, Pydantic v2, onnxruntime and SQLite on the backend; React +
Vite + TypeScript on the front end. `uv` and `bun` are the package managers. No Redis, no
Postgres, and no Node runtime in production — the backend serves the built frontend.

Two dependencies named in the spec are not installed yet because the milestones that need
them have not landed: PyAV (video decode, M2) and scikit-learn HDBSCAN plus Sigma.js and
graphology (clustering and the graph view, M3 and M4). `sqlite-vec` is installed but not
yet used — matching is a blocked numpy matmul over one in-memory gallery matrix.

Detection is YuNet; embedding is SFace by default (Apache-2.0). buffalo_l / ArcFace r50 is
available but non-commercially licensed and stays off until an operator turns it on with a
recorded reason.

## Setup

```sh
# 1. Weights. Build-time only, digest-pinned, never fetched at runtime.
uv run --python 3.12 --no-project python tools/fetch_models.py
uv run --python 3.12 --no-project python tools/fetch_models.py --allow-noncommercial  # adds buffalo_l
uv run --python 3.12 --no-project python tools/fetch_models.py --verify               # check what is here

# 2. Backend deps. The uv-managed CPython is required: the macOS system Python's sqlite3
#    has no loadable-extension support.
cd backend && uv sync --extra dev

# 3. Frontend build. Output lands in frontend/dist and the backend serves it.
cd ../frontend && bun install && bun run build
```

Configuration that must be true before the process starts lives in a gitignored `.env` at
the repo root. On Apple Silicon that is one line, because a threshold set is only
reproducible on the execution provider it was calibrated on:

```sh
EXECUTION_PROVIDER=CoreMLExecutionProvider
```

Everything an operator can decide while the system is running lives in the `runtime_config`
table instead, changed through `PATCH /api/config` and audited — including the
non-commercial model licence gate, which requires a reason.

## Running

```sh
cd backend
uv run python -m app.cli check                 # schema and models.lock, before anything else
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
uv run python -m app.worker                    # separate process, one worker
```

Open <http://127.0.0.1:8000>. One uvicorn worker and one job worker, deliberately: the audit
chain has a single writer.

Other operator commands:

```sh
uv run python -m app.cli migrate               # apply pending migrations
uv run python -m app.cli verify-audit          # walk the hash chain
uv run python -m app.cli enqueue-reembed       # re-embed stored crops under the active embedder
```

## Calibration

Auto-accept is gated on calibration and nothing self-confirms without it.

```sh
cd backend
uv run python ../eval/run.py ../fixtures --output ../eval/report.json
```

That writes a report and an **inactive** calibrated threshold set. Activate it deliberately
with `POST /api/threshold_sets/{id}/activate`; until then every match is a candidate. A
threshold set records the gallery size and execution provider it was measured at, and the
acceptance gate refuses to open if either has moved.

## Working on it

```sh
cd backend
uv run pytest
uv run ruff check . ../eval ../tools && uv run mypy      # one ruff.toml at the repo root
uv run python ../tools/bench_perf.py                     # performance, scratch DB, medians

cd ../frontend
bun run typecheck && bun run build
bun run dev                                              # Vite on 5173, /api proxied to 8000
```

Tests never commit face images or model files: fixtures load from `FIXTURES_DIR` and weights
from `models/`, both gitignored, and tests that need a real face or a real graph skip
cleanly when they are absent, so a fresh clone passes. The suite is written against the
SFace adapter so the buffalo_l swap stays a configuration change rather than a code change.
There is no CI runner yet; the four commands above are the gate.

`tools/bench_perf.py` builds a scratch database in a temp directory, seeds a synthetic
1000-person gallery, loads the real weights and prints medians as one JSON object. It never
touches `data/facematch.db`. A change that claims a speedup quotes its output before and
after. It is not a numerics check: for that, re-run `eval/run.py` against a scratch database
and compare `t_strong`, `t_possible` and `margin` to the active threshold set — a change to
the execution provider, provider options, graph optimisation level or a weight file moves
scores while the active calibrated set still looks valid.

## Layout

```
backend/app/api/         routers, request-scoped dependencies
backend/app/core/        model registry, scoring, acceptance, vectors, storage
backend/app/adapters/    yunet.py, sface.py, arcface_r50.py
backend/app/pipeline/    ingest, decode, quality, align, process, matching, reembed,
                         capture, live
backend/app/db/          conn.py, migrate.py, migrations/  (the only way the schema changes)
frontend/src/            views/, components/, lib/, api/
models/                  ONNX weights (gitignored) + models.lock (tracked)
data/                    gitignored: facematch.db, media/, crops/, logs/
eval/                    calibration harness (spec section 10)
tools/                   fetch_models.py, bench_perf.py
docs/spec.md             the design spec
```

## Deployment

`docker compose up` targets the x86 Beelink. Apple Silicon runs natively so onnxruntime can
use the CoreML execution provider; the container cannot.

## License

Our code is MIT (`LICENSE`). Model weights keep their own licenses, which `models.lock`
records and the UI displays. SFace is Apache-2.0. buffalo_l is non-commercial research only
and refuses to load until `allow_noncommercial_models` is turned on, which is an audited
operator decision requiring a reason — not an environment flag.
