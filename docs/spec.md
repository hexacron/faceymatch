# Local Face Match System: Design Spec

Version: 0.4
Owner: Brock
Status: All core decisions closed. Proposed items can change during build.

Changes from 0.3:

- D4, D6 through D13 closed. AGENTS.md already hard-codes them as the stack.
- Default embedder is SFace (MIT). buffalo_l is opt-in behind `ALLOW_NONCOMMERCIAL_MODELS=true`, so a fresh clone starts in a permissively licensed state.
- The `insightface` Python package must never be imported at runtime: its model loader fetches weights over the network, breaking C1. Adapters load raw `.onnx` files through onnxruntime.
- New table `detection_embeddings`. 6.2 stores an embedding per quality-passing crop, so enrollment from a stored detection and re-embed on model switch both read from it.
- `matches` gains `embedder_model_id`. `threshold_sets` gains `gallery_size` and `execution_provider`. `models` gains `kind`. `persons` gains `status`. New `schema_migrations`.
- Auto-accept also requires the runtime execution provider to equal the active threshold set's `execution_provider`.
- A still image produces exactly one track per detection (`start_ms = end_ms = 0`), so images and video share one `tracks`/`identities` code path.
- Band assignment is deterministic: strong, then ambiguous, then possible, then unknown. `margin` compares the top two persons, and `top2 = 0.0` below two persons in the gallery.
- Every quality-passing aligned 112x112 crop is written content-addressed and hashed, which is what makes re-embed possible without re-decoding media.
- Job resume uses a structured checkpoint in `jobs.progress`. Detection writes are idempotent on `(media_id, frame_idx)`.
- `jobs.kind` gains `audit_verify`: section 9 chain verification runs as a job, which is the M0 exit test made runnable and auditable.
- Audit canonical JSON frozen: UTF-8, sorted keys, separators `(",", ":")`, `allow_nan=false`. Single writer under `BEGIN IMMEDIATE`, one uvicorn worker. Genesis `prev_hash` is 64 zero characters.
- Re-match scores one gallery matrix with a blocked matmul, not row-by-row vector-store queries.
- New endpoints: `GET /api/persons`, `GET /api/models`, `GET /api/threshold_sets`, `POST /api/threshold_sets/{id}/activate`, `GET /api/healthz`.
- M1 now includes a minimal calibration run and an audited calibrated threshold set. M5 replaces it with the full study.
- Layout deviation from 0.3: migrations are the single source of truth. `app/db/` holds `conn.py, migrate.py, migrations/, vec.py`. There is no `schema.sql`, because two copies of the schema drift.
- Section 12 rewritten for invariant 11: bind `127.0.0.1` only, remote access by `tailscale serve` proxy to loopback. No app-level auth in v1 is recorded as an accepted risk.
- Zero-template purge outcome specified: the person row survives as `status = 'unenrolled'`.

Changes from 0.2:

- D2 closed: buffalo_l for lab prototype only. SFace adapter stays tested for the swap.
- D3 closed: one global gallery across all cases. Storage moves to one DB.

Changes from 0.1:

- No webcam or live input. Input is collected images and video only (D1 closed).
- All detection runs on the backend. The browser draws stored tracks during playback.
- Automatic identification is allowed. Operator review is optional, not required.
- Only operator actions create templates. Auto-matches never enroll.

---

## 1. Purpose

The system finds face matches in collected images and video. It runs fully local. The operator enrolls known people. The system identifies them automatically in new media, draws boxes and scores over the media, and shows how people connect through shared appearances.

## 2. Scope

In scope:

- Ingest of images and video files.
- Face detection, tracking, and embedding on the backend.
- Enrollment of people from images or from detected faces.
- 1:N matching with open-set "unknown" results.
- Automatic acceptance of matches above a calibrated threshold.
- Overlay during playback: bounding box, name, score band, track ID.
- Click-to-tag on any detected face, to correct or add identities.
- Clustering of unknown faces for bulk enrollment.
- Co-occurrence graph of people, with proof for each edge.
- Evidence hashing, append-only audit log, export.

Out of scope for v1:

- Webcam or live stream input.
- Age, gender, emotion, or any attribute estimation. Do not ship these models.
- Face search against the open web or third-party services.
- Mobile clients. Multi-site sync.

## 3. Constraints

- C1. Local-first. No outbound network calls at runtime. No telemetry.
- C2. Our code is MIT. Model weights keep their own licenses. Record the license of each model file.
- C3. No discrete GPU. Targets: Beelink SEi (x86 CPU) and Apple Silicon Macs.
- C4. Defensible by default. Every match records how it was accepted (auto or operator), with model and threshold versions.
- C5. Auto-acceptance needs an active, calibrated threshold set. Without one, all matches stay candidates.
- C6. Only operator actions create templates.
- C7. Non-commercial models load only when `ALLOW_NONCOMMERCIAL_MODELS=true`. The UI and every export state the active model license.

## 4. Architecture

```
+---------------- Browser (operator UI) ----------------+
|  Media library, upload                                |
|  Player: <video>/<img> + canvas overlay               |
|     draws stored tracks, synced to currentTime        |
|  Tag panel, cluster review, graph view                |
+--------------------------|----------------------------+
                           | HTTP to 127.0.0.1 (tailscale serve for remote)
+--------------------------v----------------------------+
|  Backend (FastAPI + 1 job worker)                     |
|  Ingest -> hash -> decode -> detect -> track          |
|     -> quality gate -> embed -> match -> accept       |
|     -> cluster unknowns                               |
|  Graph builder, export, audit log                     |
+--------------------------|----------------------------+
                           v
          SQLite + sqlite-vec (one DB, rows tagged by case)
          Media store (content-addressed by SHA-256)
```

## 5. Decision register

| ID | Decision | Choice | Status | Reason |
|----|----------|--------|--------|--------|
| D1 | Input | Collected images and video only | Closed | No webcam need. |
| D2 | Recognition model | SFace (MIT) is the shipped default. buffalo_l (w600k_r50, 512-d) is opt-in behind `ALLOW_NONCOMMERCIAL_MODELS=true`, lab prototype only | Closed | A fresh clone starts permissively licensed. License or swap buffalo_l before product or case use (C7). |
| D3 | Gallery scope | One global gallery across all cases | Closed | A person enrolled once is found in all media. |
| D4 | Detector | YuNet (MIT). SCRFD-10GF adapter if licensed | Closed | Fast on CPU, 5 landmarks, permissive. |
| D5 | Where detection runs | Backend only | Closed | No live input. One code path. |
| D6 | Tracker | ByteTrack-lite in Python | Closed | Simple, robust to blur, no extra deps. |
| D7 | Frontend | React + Vite + TypeScript, static build | Closed | Backend serves the build. No Node runtime in production. |
| D8 | Backend | Python 3.12, FastAPI, Uvicorn, Pydantic v2, onnxruntime | Closed | Matches existing FastAPI work. CoreML EP on Mac. |
| D9 | Vector store | One SQLite DB + sqlite-vec, rows tagged by case | Closed | A global gallery needs one store. Case bundles come from filtered export. Move track vectors to Qdrant behind the same interface if tracks pass about 1M. |
| D10 | Job queue | SQLite `jobs` table + one worker process | Closed | No Redis. |
| D11 | Video decode | PyAV (ffmpeg bindings) | Closed | Frame-accurate timestamps. |
| D12 | Clustering | scikit-learn HDBSCAN, cosine | Closed | No cluster count needed. Noise maps to unknown. |
| D13 | Graph UI | Sigma.js + graphology | Closed | WebGL, scales past 10k nodes. |
| D14 | Encryption at rest | OS full-disk encryption | Proposed | SQLCipher optional later. |
| D15 | Operator model | Single operator per install, named in audit log | Proposed | Revisit for shared instances. |
| D16 | Identity acceptance | Auto-accept `strong` band. Operator may override any result | Closed | No mandatory human review. |
| D17 | Template creation | Operator action only | Closed | Stops gallery drift from wrong auto-matches. |

## 6. Components

### 6.1 Ingest

- Accept images (JPEG, PNG, WebP, HEIC) and video (anything PyAV decodes).
- Hash each file (SHA-256) before processing. Store content-addressed. Reuse stored bytes when the same file appears in another case.
- Apply EXIF orientation for images before detection. Store the original unchanged.
- Folder import: recursive, with a job per file.
- A still image produces exactly one track per detection, with `start_ms = end_ms = 0`. Images and video therefore share one `tracks`/`identities` code path.

### 6.2 Processing pipeline (job worker)

Per file:

1. Decode. Images: one frame. Video: sample at `SAMPLE_FPS` (default 3). Keep true timestamps.
2. Detect with the active `Detector`. Letterbox, score threshold, NMS, 5 landmarks.
3. Track (video). Kalman filter, IoU association, high-score pass then low-score pass. States: tentative (3 hits), confirmed, lost (1 s), deleted.
4. Quality gate per detection:
   - Width >= `MIN_EMBED_PX` (start 80, tune in eval).
   - Yaw from landmark symmetry below `MAX_YAW`.
   - Laplacian variance above `MIN_SHARPNESS`.
   - Detector score above `MIN_DET_SCORE`.
5. Align passing crops to the 112x112 ArcFace 5-point template. Write the aligned crop of every quality-passing detection to the content-addressed store and record its SHA-256 in `detections.crop_sha256`.
6. Embed every stored crop and write one row per crop to `detection_embeddings`. Store the L2-normalized mean of the best K crops per track (default K = 5) in `tracks.embedding_mean`. Keeping every crop and its embedding is what lets a model switch re-embed without re-decoding media.
7. Match each track mean against the gallery (6.4).
8. Apply acceptance rules (6.5).
9. Cluster tracks that end as `unknown` (6.7).
10. Write results. Emit progress events.

Still images: one track per detection, `start_ms = end_ms = 0`. Step 3 is skipped; steps 7 to 10 are identical to video, so there is one `tracks`/`identities` code path.

Job resume: after each committed batch the worker writes a structured checkpoint into `jobs.progress` (last decoded `frame_idx`, last `t_ms`, per-stage counts). Resume reads that checkpoint rather than scanning for the last written row. Detection writes are idempotent on `(media_id, frame_idx)` under a unique index, so a job resumed inside the batch it died in cannot double-insert.

Re-match job: when the gallery changes (new person or template), re-match stored track embeddings. No re-decode needed. Run it on demand or after each enrollment.

Re-match loads the whole gallery for the active `embedder_model_id` as one `(templates, dim)` float32 matrix and scores blocks of track means against it with a matmul, then reduces per person. It does not issue row-by-row vector-store queries. That is what makes the section 11 target of 100k tracks against 1k persons in under 60 s reachable.

### 6.3 Model interfaces

```python
class Detector(Protocol):
    model_id: str
    def detect(self, image: np.ndarray) -> list[Detection]: ...
    # Detection: bbox (x, y, w, h), score, landmarks (5x2)

class Embedder(Protocol):
    model_id: str          # e.g. "sface-2021dec" or "buffalo_l-w600k_r50"
    dim: int
    def embed(self, crops: np.ndarray) -> np.ndarray: ...
    # crops: (N, 112, 112, 3) aligned. Returns (N, dim), L2-normalized.
```

Rules:

- Never compare embeddings from different `model_id` values.
- Store `model_id` with every embedding, template, and match.
- A model switch re-embeds templates and tracks. Provide a CLI job.
- Verify each model file SHA-256 against `models.lock` at start. Refuse to start on mismatch.
- `models.lock` records the license of each file. Refuse to load a non-commercial model unless `ALLOW_NONCOMMERCIAL_MODELS=true` (C7).
- CI runs the full test suite with SFace, so the swap from buffalo_l stays a config change.
- SFace is the shipped default embedder. buffalo_l loads only when `ALLOW_NONCOMMERCIAL_MODELS=true`, so a fresh clone starts in a permissively licensed state.
- Never import the `insightface` Python package at runtime. Its model loader fetches weights over the network, which breaks C1. Adapters load raw `.onnx` files from `models/` through onnxruntime directly.
- Every match stores the `embedder_model_id` that produced its scores (`matches.embedder_model_id`).

### 6.4 Matching and scoring

- Similarity: cosine on L2-normalized vectors.
- Person score: max over that person's active templates. Option: mean of top 3 when a person has 5 or more templates.
- Return top K = 3 candidates.
- Record the template that gave the top score (`best_template_id`). With a global gallery, this shows which case and media the match came from.
- Bands come from the active `threshold_set`. `top1` is the best person score and `top2` the second best *person* score, not the second best template of the same person. With fewer than two persons in the gallery, `top2 = 0.0`. Evaluate in this fixed order and take the first rule that holds:
  1. `strong`: `top1 >= t_strong` AND `(top1 - top2) >= margin`
  2. `ambiguous`: `top1 >= t_possible` AND `(top1 - top2) < margin`
  3. `possible`: `top1 >= t_possible`
  4. `unknown`: otherwise
- UI shows band first, raw score second. Do not label the score as a probability.

### 6.5 Acceptance rules

| Band | Result | Graph |
|------|--------|-------|
| `strong` | Auto-accepted identity | Yes |
| `possible` | Candidate. Listed in optional review queue | Only if filter allows candidates |
| `ambiguous` | Candidate. Top 2 shown | No |
| `unknown` | No identity. Goes to clustering | No |

- Auto-accept only runs when the active threshold set has `calibrated = true` (C5), its `model_id` matches the active embedder, and the runtime execution provider equals its `execution_provider` (section 10). Any miss leaves every match a candidate.
- Each accepted identity records `source = auto | operator`.
- An operator decision always wins over an auto result. A later re-match never overwrites an operator decision.
- The review queue is optional. It sorts `possible` and `ambiguous` tracks by score for fast bulk confirm or reject.

### 6.6 Enrollment and tagging

- Enroll from an uploaded image, a stored detection, or a whole cluster.
- Target 5 or more templates per person across pose and lighting.
- Only operator actions create templates (C6). Auto-accepted tracks never become templates.
- Tag actions on any track: confirm, reject, reassign, create new person, mark "do not enroll".
- Tagging a cluster assigns all its tracks. The operator can exclude single tracks.
- Each action writes an `identifications` row and an audit entry.

### 6.7 Clustering

- Input: `unknown` track means from one media file, a job batch, or the whole case.
- HDBSCAN with cosine metric. `min_cluster_size` default 3.
- Output: clusters with a representative thumbnail grid. Noise tracks stay unknown.
- Operator enrolls a cluster as a new person, or assigns it to an existing one.

### 6.8 Browser client

Player overlay:

- `<video>` or `<img>` with a `<canvas>` on top.
- On load and on seek, fetch tracks for a time window around `currentTime`.
- Draw with `requestVideoFrameCallback`. Interpolate boxes linearly between stored samples, so boxes stay smooth at 3 fps sampling.
- Sync canvas size with `ResizeObserver` and `devicePixelRatio`. Map through the `object-fit` letterbox of the media element.
- Box color by band. Label: name, band, score, track ID. Mark operator-confirmed identities with a distinct badge.

Tag panel:

- Click hit-tests the drawn boxes. Video pauses on click.
- Shows top 3 candidates, source (auto or operator), and the best crops of the track.

Other views:

- Media library with processing status.
- Person page: templates, appearances, timeline.
- Cluster review grid.
- Optional review queue.
- Graph view (6.9).

### 6.9 Graph

- Nodes: persons, media. Optional: cases, locations.
- Edge (person to person): both appear in the same frame (strong) or same media (weak). Weight = count of co-occurring frames.
- Edge style by weakest identity on the edge:
  - Solid: both operator-confirmed.
  - Solid, lighter: at least one auto-accepted.
  - Dashed: at least one candidate (hidden by default).
- Every edge links to its proof: media, timestamps, detection IDs, thumbnails, scores.
- Filters: case (one, several, or all), min weight, source (operator only, auto + operator, include candidates), date range, media set.
- Cross-case edges are allowed. Each edge lists the cases its proof comes from.
- Exports: GraphML, CSV, Maltego import. The Maltego format is open (section 15).

## 7. Data model

One DB for all cases. IDs are UUIDv7. Persons and templates are global. Media and all rows derived from media carry a case.

```
cases(id, name, authorization_basis, created_at, created_by)

models(id, name, version, kind[detector|embedder], sha256, license, dim)

threshold_sets(id, model_id, t_strong, t_possible, margin, calibrated,
               calibrated_at, eval_report_sha256, gallery_size,
               execution_provider, active)

media(id, case_id, sha256, kind[image|video], path, source_url,
      acquired_at, width, height, duration_ms, fps, ingested_at, status)

persons(id, display_name, notes, do_not_enroll, enrolled_in_case_id,
        status[enrolled|unenrolled], created_at, created_by)

detections(id, media_id, track_id, t_ms, frame_idx,
           x, y, w, h, landmarks_json, det_score, quality_json,
           crop_sha256, detector_model_id)

detection_embeddings(detection_id PK, embedding BLOB, embedder_model_id,
                     created_at)

tracks(id, media_id, start_ms, end_ms, best_detection_id,
       cluster_id, embedding_mean BLOB, embedder_model_id)

templates(id, person_id, detection_id, source_case_id, embedding BLOB,
          embedder_model_id, quality, status[active|revoked],
          created_at, created_by)

matches(id, track_id, person_id, rank, score, band,
        best_template_id, threshold_set_id, embedder_model_id, created_at)

identities(track_id PK, person_id, source[auto|operator],
           match_id, threshold_set_id, updated_at)

identifications(id, track_id, person_id,
                decision[confirm|reject|reassign|new|cluster_assign],
                operator, note, created_at)

clusters(id, job_id, scope, size, label_person_id)

jobs(id, kind[ingest|process|rematch|reembed|cluster|export|audit_verify],
     status, params_json, progress, error, created_at, updated_at)

schema_migrations(version PK, applied_at, sha256)

audit_log(seq, ts, actor, case_id, action, object_type, object_id,
          payload_json, prev_hash, hash)
```

- `identities` holds the current answer per track. `identifications` holds operator history.
- Vector index: `sqlite-vec` virtual tables over `templates.embedding` and `tracks.embedding_mean`, one per `embedder_model_id`.
- Co-occurrence is a SQL view over `detections` + `identities`. Do not store edges as a separate source of truth.
- `detection_embeddings` exists because 6.2 step 6 stores an embedding for every quality-passing crop, not just the track mean. Enrollment from a stored detection reads that vector instead of re-running the embedder, and a model switch re-embeds the stored crops into a fresh row set keyed by the new `embedder_model_id`.
- `matches.embedder_model_id` is required by 6.3: every match records the model that produced its scores. It is the audit anchor for invariant 2, so a reviewer can prove both compared vectors came from the same model.
- `threshold_sets.gallery_size` is the gallery size the set was calibrated at, which section 10 needs for the 2x warn and 5x block rule. `threshold_sets.execution_provider` is recorded because CoreML and CPU execution providers produce slightly different scores, so a threshold set is only reproducible on the EP it was calibrated on.
- `models.kind` separates detectors from embedders, so `models.lock` verification and the C7 license banner can address each class on its own.
- `persons.status` is `enrolled` normally and `unenrolled` once a purge strands the person at zero active templates (section 12). `unenrolled` persons are excluded from the matching gallery.
- `schema_migrations` is written by `app/db/migrate.py`. Migrations are the single source of truth for the schema; there is no `schema.sql`.
- `jobs.progress` holds a JSON object, not a scalar percentage. It carries the structured checkpoint from 6.2 (last `frame_idx`, last `t_ms`, per-stage counts) for pipeline jobs, and the verification outcome for an `audit_verify` job.
- Chain verification runs as an `audit_verify` job rather than a script, so its result is itself appended to the chain. M0 ships the job worker with `audit_verify` as its one registered handler, before any media pipeline exists.

## 8. API contract (v1)

```
POST   /api/cases
GET    /api/cases/{case_id}

POST   /api/media                      multipart upload -> {media_id, sha256, job_id}
POST   /api/media/import               {folder_path} -> {job_ids[]}
GET    /api/media?status=
GET    /api/media/{id}
GET    /api/media/{id}/file            range requests for video
GET    /api/media/{id}/tracks?from_ms=&to_ms=
  resp: [{track_id, person_id, name, band, score, source,
          samples: [{t_ms, x, y, w, h}]}]

GET    /api/tracks/{id}                candidates, crops, history
POST   /api/identifications            confirm | reject | reassign | new

GET    /api/persons?q=&status=
POST   /api/persons
GET    /api/persons/{id}
POST   /api/persons/{id}/templates     from image upload or detection_id

GET    /api/models                     active detector and embedder, license per file (C7 banner)
GET    /api/threshold_sets
POST   /api/threshold_sets/{id}/activate   explicit audited activation (section 10)

GET    /api/review?band=possible|ambiguous&limit=
POST   /api/review/bulk                [{track_id, decision, person_id?}]

POST   /api/jobs/rematch
POST   /api/jobs/cluster               {scope}
GET    /api/jobs/{id}
WS     /api/jobs/events

GET    /api/clusters?scope=
POST   /api/clusters/{id}/assign       {person_id | new_name, exclude_track_ids[]}

GET    /api/graph?case_ids=&min_weight=&source=&from=&to=
GET    /api/export?case_id=&format=bundle|graphml|csv|maltego

DELETE /api/cases/{id}                purge case (section 12)
DELETE /api/persons/{id}              purge person across all cases
GET    /api/audit?from_seq=
GET    /api/healthz                    db, models.lock, active threshold set, execution provider
```

## 9. Evidence and audit

- Hash every ingested file before processing. Store content-addressed.
- Write the aligned 112x112 crop of every quality-passing detection to the content-addressed store and record its SHA-256 in `detections.crop_sha256`. Stored crops are what let a model switch re-embed templates and tracks without re-decoding the original media.
- Audit log is append-only and hash-chained: `hash = SHA256(prev_hash || canonical_json(entry))`.
- Canonical JSON is frozen: UTF-8 bytes, object keys sorted, separators `(",", ":")`, `allow_nan=false`. Changing this form invalidates every existing chain, so it is not a tunable.
- The genesis entry uses a `prev_hash` of 64 `0` characters.
- A single writer appends to the chain. Read the current head and append the new row inside one `BEGIN IMMEDIATE` transaction, and serve the API from a single uvicorn worker, so two writers cannot fork the chain.
- Log auto-acceptance events with `match_id` and `threshold_set_id`. A reviewer can rerun any match and get the same result.
- Reports and exports state the source of each identity: auto or operator.
- Case export bundle: a filtered DB with the case media and derived rows, plus the persons and templates its identities depend on (with their source cases). Add media, `models.lock`, active threshold set, eval report, and audit log head hash.
- Later: hand the bundle to the Glasshouse custody ledger (C2PA manifest, RFC 3161 timestamp).

## 10. Evaluation and calibration

Calibration is now a hard gate. Auto-accept does not run without it (C5).

Labelled set from our own data type:

- 50 or more identities, 5 or more images each. Include poor quality: small, blurred, off-angle.
- Tag pairs with quality buckets (size, yaw). Tag cohort where known, for bias checks.

Metrics:

- Verification: FNMR at FMR = 1e-3 and 1e-4.
- Open-set identification: FNIR vs FPIR at gallery sizes 100, 1k, 10k.
- Report per quality bucket and per cohort.

Threshold policy:

- Set `t_strong` for a false positive identification rate at or below `FPIR_TARGET` (start 1e-3) at the expected gallery size.
- Set `t_possible` for review recall.
- If one cohort misses the FPIR target, raise `t_strong` until all cohorts meet it.
- Open-set false positives grow with gallery size. Store the gallery size in each threshold set. Warn when the live gallery passes 2x that size. Block auto-accept past 5x until recalibrated.
- Record the `execution_provider` the set was calibrated on. CoreML and CPU execution providers produce slightly different scores, so a threshold set is only reproducible on its own EP. Auto-accept requires the runtime execution provider to equal the active threshold set's `execution_provider`; otherwise every match stays a candidate until the set is recalibrated on that EP.

Outputs:

- `eval/run.py --model <id>` writes a report and a candidate `threshold_set`.
- The operator activates a threshold set by an explicit, audited action through `POST /api/threshold_sets/{id}/activate`.
- M1 runs this over about 10 identities to get a calibrated set early. M5 runs the full study and replaces it.
- Run the same eval on SFace and buffalo_l. The gap decides whether buffalo_l is worth licensing (D2); SFace stays the shipped default either way.

## 11. Performance targets

Targets, not measurements. Verify in M1 and M2.

| Path | Target |
|------|--------|
| Image ingest to matched, Beelink CPU | < 1 s per image with 1 to 5 faces |
| Video, 3 fps sampling, 1080p, Beelink | >= 1x real time |
| Re-match of 100k tracks against 1k persons | < 60 s |
| Overlay redraw during playback | Display rate |
| Track fetch per 10 s window | < 50 ms |

## 12. Security and privacy

- Bind the server to `127.0.0.1` only (invariant 11). Remote access is a `tailscale serve` proxy in front of that loopback listener. Never bind a tailnet IP, and never bind `0.0.0.0`.
- Accepted risk: there is no app-level authentication in v1. Anyone who can reach the proxy is treated as the single operator named in `OPERATOR_NAME`, so `identifications.operator` is an install-level claim, not an authenticated one.
- No outbound calls. Block container egress at the network level.
- Full-disk encryption on all hosts (D14).
- The global gallery removes case compartments. Any enrolled person can match in any case. Record the enrolling case and its `authorization_basis` on each person.
- Case purge removes the case media, crops, detections, tracks, and all templates sourced from that case. Then it re-matches affected persons in other cases.
- A case purge that strands a person at zero active templates sets `persons.status = 'unenrolled'`. The person row survives, so audit and identification history stay readable. An `unenrolled` person is excluded from matching, and the following re-match reverts that person's auto-accepted identities to unknown. Operator-confirmed identities survive (invariant 5).
- Person purge removes the person, all templates, and all identities in every case.
- The audit log keeps purge entries with hashes only.
- `do_not_enroll` on a person blocks template creation and auto-acceptance for that person.
- Biometric data is special-category data under GDPR Art. 9, and Canadian regulators have acted on facial recognition misuse. Automated identification without review raises the bar. Record `authorization_basis` on each case. Get legal review before any product use.

## 13. Repo layout

```
face-match/
  frontend/            React + Vite + TS
    src/player/        overlay, interpolation, hit-test
    src/tagging/
    src/review/
    src/clusters/
    src/graph/
  backend/
    app/api/           routers
    app/core/          interfaces, scoring, acceptance, config
    app/adapters/      yunet.py, sface.py, arcface_r50.py
    app/pipeline/      ingest.py, decode.py, track.py, quality.py,
                       align.py, cluster.py, rematch.py
    app/db/            conn.py, migrate.py, migrations/, vec.py
    app/audit.py
    app/worker.py
  models/              ONNX files (gitignored) + models.lock
  eval/
  docs/spec.md
  docker-compose.yml
  LICENSE              MIT
```

## 14. Milestones

| M | Deliverable | Exit test |
|---|-------------|-----------|
| M0 | Repo, compose, FastAPI, static frontend, schema, audit chain, job worker | Audit chain verifies after 1000 writes |
| M1 | Images: ingest, detect, enroll, match, auto-accept, overlay, tag, plus a minimal calibration run (about 10 identities) that writes an audited `threshold_set` with `calibrated = true` | Enroll 10 people, activate the calibrated set, auto-identify on new photos end to end, overrides logged |
| M2 | Video: decode, track, embed, match, player overlay | 30 min video processed, resumed after kill, boxes stay aligned on seek |
| M3 | Clustering + cluster enrollment + re-match job | New person enrolled from cluster appears in all past media |
| M4 | Graph + exports | Every edge opens its proof frames |
| M5 | Eval + calibration, buffalo_l vs SFace report | Full threshold set activated from report and replacing M1's minimal set, license gap measured |

Note: M0 has no calibrated threshold set and therefore no auto-accept. Every match stays a candidate in that state (C5). M1 closes the gap with a minimal calibration run over about 10 identities, activated by an explicit audited action. M5 replaces M1's minimal set with the full study: 50 or more identities, quality buckets and cohorts, buffalo_l vs SFace.

## 15. Open items

- Commercial license for buffalo_l, or swap to SFace, before any product or case use.
- Maltego export format: MTGX file vs CSV for the import wizard vs a local transform.
- Whether Hermes Agent gets an MCP tool for read-only graph queries.
- Authenticated multi-operator use on one instance is deferred past v1 (D15). Until then `identifications.operator` is an install-level claim (section 12).
