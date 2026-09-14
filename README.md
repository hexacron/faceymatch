# faceymatch

A local face match system for investigations. It ingests images and video, detects and
embeds faces, matches them against a gallery the operator built by hand, and shows how
people connect through shared appearances. It runs entirely on the operator's machine.

The design is specified in [`docs/spec.md`](docs/spec.md). That document is authoritative;
this file is how to run the thing. `AGENTS.md` holds the working rules for changing it.

Biometric material is special-category data under GDPR Art. 9 whatever its provenance, and
investigative use raises that bar rather than lowering it. Spec section 12 applies in full.
Get legal review before any product or case use.

Status: milestones M0 (skeleton, schema, audit chain, job worker), M1 (images end to end
with a minimal calibration), M1.5 (screen acquisition and live match) and M2 (video decode,
tracking, matching and a player overlay) are done. M3, clustering, is next. The graph view
and exports come after it.

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

Python 3.12, FastAPI, Uvicorn, Pydantic v2, onnxruntime, PyAV and SQLite on the backend;
React + Vite + TypeScript on the front end. `uv` and `bun` are the package managers. No
Redis, no Postgres, and no Node runtime in production — the backend serves the built
frontend.

The dependencies named in the spec that are not installed yet are the ones whose milestones
have not landed: scikit-learn HDBSCAN (clustering, M3) and Sigma.js plus graphology
(the graph view, M4). `sqlite-vec` is loaded onto every connection but nothing queries it
yet — matching is a blocked numpy matmul over one in-memory gallery matrix.

Detection is YuNet and embedding is SFace by default, both permissively licensed (MIT and
Apache-2.0). Three heavier models ship as options and all three are non-commercial research
only: the buffalo_l pack's SCRFD-10GF detector and ArcFace r50 embedder, and antelopev2's
glintr100 embedder. They refuse to load until an operator turns the licence gate on with a
recorded reason.

## Setup

```sh
# 1. Weights. Build-time only, digest-pinned, never fetched at runtime.
uv run --python 3.12 --no-project python tools/fetch_models.py
uv run --python 3.12 --no-project python tools/fetch_models.py --allow-noncommercial  # + the gated packs
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
./run                                          # startup checks, API and worker; Ctrl-C stops both
./run --watch                                  # ... plus the watch helper (macOS only, spec 6.11)
```

Or run the two processes yourself:

```sh
cd backend
uv run python -m app.cli check                 # schema and models.lock, before anything else
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
uv run python -m app.worker                    # separate process, one worker
```

Open <http://127.0.0.1:8000>. One uvicorn worker and one job worker, deliberately: the audit
chain has a single writer. Logs from `./run` land in `data/logs/`.

The watch helper is a desktop app the operator starts, points at one window of their own
machine and stops. Qt and the capture libraries live in a separate extra so a plain
`uv sync` never pulls them onto a machine that only serves the API:

```sh
cd backend && uv sync --extra dev --extra watch
uv run --extra watch python -m watch           # needs the API running
```

Other operator commands:

```sh
uv run python -m app.cli migrate               # apply pending migrations
uv run python -m app.cli verify-audit          # walk the hash chain
uv run python -m app.cli enqueue-reembed       # re-embed stored crops under the active embedder
```

## Getting evidence in

Drop or paste an image into the Media tab, capture a region of your own screen, or import a
folder. A folder import is recursive and registers one file per `process` job.

An import can be asked to **keep only the files a face was found in** — the "Only keep files
with a face" checkbox, or `faces_only` on `POST /api/media/import`. Every file is still
hashed and registered first, because that is the only way to find out what is in it; the
worker then purges the ones the detector found nothing in, and records `no faces at import`
as the reason on the `media.purge` audit entry. "Has a face" means at least one detection,
whatever the quality gate said about it: a small, blurred or off-angle face cannot be
enrolled but it is still a face in the picture.

For a folder already imported without that, the library's **Faces** filter lists the files
that hold no face (`GET /api/media?has_faces=false`, processed files only — one still queued
has not been asked yet). Select all, delete the selection, and the mixed import is cleaned
up. Deletion is a section 12 purge: the row, the derived rows, and the bytes once no other
case points at them.

A curated `Person Name/*.jpg` tree becomes persons and templates in one audited request with
a required reason — "Enrol from folder names", or `POST /api/persons/enroll_folder`. It reads
the stored detections, never the files, so only an imported file can be enrolled from.

## Calibration

Auto-accept is gated on calibration and nothing self-confirms without it.

```sh
cd backend
uv run python ../eval/run.py ../fixtures --output ../eval/report.json
```

That writes a report and an **inactive** calibrated threshold set. Activate it deliberately
with `POST /api/threshold_sets/{id}/activate`; until then every match is a candidate.
Activation does not rescore anything, so follow it with `POST /api/jobs/rematch` to move
existing tracks onto the new thresholds — operator decisions survive that untouched.

A threshold set belongs to one embedder, one execution provider and one gallery size, all
recorded on the row. Switching the embedder seeds a fresh uncalibrated set and closes the
gate until you calibrate the new one; the gate also refuses to open on a different provider,
warns past twice the calibrated gallery size and blocks past five times it.

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
backend/app/adapters/    yunet.py, scrfd.py, sface.py, arcface.py, boxes.py
backend/app/pipeline/    ingest, decode, quality, align, process, matching, reembed,
                         capture, live, video, tracking
backend/app/db/          conn.py, migrate.py, migrations/  (the only way the schema changes)
backend/watch/           the operator-started watch helper (spec 6.11)
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
records and the UI displays. YuNet is MIT and SFace is Apache-2.0. The buffalo_l and
antelopev2 weights are non-commercial research only and refuse to load until
`allow_noncommercial_models` is turned on, which is an audited operator decision requiring a
reason — not an environment flag.
