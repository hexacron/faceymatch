# Local Face Match System: Design Spec

Version: 0.8
Owner: Brock
Status: All core decisions closed. Proposed items can change during build.

Changes from 0.7:

- The watch helper's overlay takes a click on one small numbered square per face instead of on the box and its label, so a hover-driven control in the watched window — a video player's auto-hiding bar — keeps working while the helper draws over it (6.11). The panel opens compact with its face list behind a disclosure, and remembers its size and that disclosure between runs.
- Folder enrolment: `POST /api/persons/enroll_folder` turns a `Person Name/*.jpg` tree that has already been imported into persons and templates in one audited operator request with a required reason (6.6). It reads stored detections only, never the files on disk (invariant 13), and reports per file what it refused to guess rather than guessing.
- The media library has two layouts: a gallery of previews and the existing list of rows. Both select files with a checkbox and delete them, one at a time with `DELETE /api/media/{id}` or as a selection with `POST /api/media/bulk_delete` (6.1, 12). A preview is a derived downscale served by `GET /api/media/{id}/thumbnail`; nothing is enrolled or tagged from it (invariant 13).
- Video (M2): `app/pipeline/video.py` samples a container on a fixed time grid and `app/pipeline/tracking.py` tracks faces across those samples, both behind the existing `process` job, so a video and a still reach matching by the same path. The media detail view plays the file with a box overlay synchronised to the playhead, and a video's library preview is its first frame.

Changes from 0.6:

- The SCRFD-10GF detector adapter (`buffalo_l-det_10g`, `app/adapters/scrfd.py`) and the ArcFace glintr100 embedder (`antelopev2-glintr100`, 512-d, on the same adapter as w600k_r50) land as gated options. Both come out of InsightFace packs the fetch tool already pins by digest, both are non-commercial, and both therefore load only once `allow_noncommercial_models` is on (C7, invariant 9). The shipped defaults do not move: a fresh clone still runs YuNet plus SFace and is still permissively licensed. Switching the embedder carries the usual consequences (re-embed, re-match, auto-accept closed until a threshold set calibrated for the new `model_id` is activated); switching the detector re-runs nothing on already-stored media.

Changes from 0.5:

- New section 6.11: a local watch helper (`backend/watch/`) that the operator starts, points at one window, display or dragged region of their own machine, and stops. It matches through the same `POST /api/live/match` as the Live view and stores nothing of its own; enrolment goes through `POST /api/media` and the existing tag panel. It is not monitoring: there is no alerting, no recording, no schedule, and it dies with its window. The operator starts it from a terminal or with `POST /api/watch/launch`, the audited, fixed-argv, macOS-only endpoint behind the button in the Live view.
- `POST /api/live/match` takes an optional `identify` form field (default true). `identify=false` returns boxes and the quality verdict only — no alignment, no embedding, no gallery scoring — so a client can track faces at detector latency and ask for names less often. The response echoes it as `identified`, because an empty candidate list otherwise cannot be told from a frame nobody was asked to identify. Persisting nothing is unchanged, and invariant 13 is untouched: a boxes-only frame is even further from evidence than an identify frame.
- `models.lock` verification is once per process per model configuration, not once per job (6.3). Invariant 8 is about refusing to run weights whose bytes moved; re-hashing the same files between two jobs of the same model proves nothing the first verification did not.
- Performance work across the pipeline, all of it numerics-preserving and none of it touching the execution provider, graph optimisation level or any weight file. Measured by `tools/bench_perf.py` (section 11).
- Section 7 records the SQLite connection policy (pooled per thread, PRAGMAs) and migration `0004_perf_indices`.
- The live view samples at two cadences: every third tick identifies at the evidence resolution, the rest ask for boxes only at 1 MP. Only identify frames are retained, so a click still stores the exact bytes the gallery scored (invariant 13, 6.10). Labels carry across boxes-only ticks by IoU against the last identify pass.
- Overlay labels are drawn only for the hovered box and the selected box (6.8). A caption over every box hides the faces underneath it.
- Person purge is implemented (section 12, `DELETE /api/persons/{id}`): the person, their templates, and every identity, identification and match naming them, in every case. The evidence stays. A bin icon on the persons library, one click to confirm; the audit entry hashes the name.
- The backend gzips its responses and the production bundle ships without a sourcemap.
- Section 11 distinguishes targets from measurements and names the harness that produces the measurements.

Changes from 0.4:

- Intended use is investigations, not lab exercise. Section 12's requirements all stand and are tightened, not relaxed, by that: biometric material handled here is special-category data whatever its provenance.
- New section 6.10: two tiers for screen material. `POST /api/capture` ingests screen pixels as evidence through the existing ingest; `POST /api/live/match` matches one frame and persists nothing (D18).
- New invariant 13: no identity, identification or template may derive from transient pixels. Enrollment and tagging act only on a stored detection whose source media was hashed at ingest.
- `media.ingest` payload records `acquisition` (`upload`, `folder_import`, `screen_capture`) and `capture_mode`. One `media.ingest` entry per media row still holds, whatever the source.
- `GET /api/healthz` reports screen-capture capability so the UI can refuse before the operator is asked to select.
- Quality gate step 4: sharpness is measured on the detection box resampled to the 112x112 embed size. `MIN_SHARPNESS` stays 40.0 but bounds a different quantity, so values in `detections.quality_json` from before the change are not comparable with values after it.
- `PATCH /api/cases/{id}` corrects `authorization_basis`, audited as `case.amend_authorization` carrying the previous text, the new text and a reason (D19).
- Out of scope reworded: the ban is on camera and sensor capture and on unattended monitoring, not on live input as such. Operator-initiated capture and match of the operator's own display are in scope.
- D1 and D5 reworded to match: input includes screen acquisition; detection is still backend only.

Changes from 0.3:

- D4, D6 through D13 closed. AGENTS.md already hard-codes them as the stack.
- Default embedder is SFace (MIT). buffalo_l is opt-in behind `allow_noncommercial_models`, which starts false, so a fresh clone starts in a permissively licensed state. Since 0.6 that setting is an audited operator decision rather than a deployment-time env flag (C7, 6.3).
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
- Screen acquisition on the operator workstation: capture what is on screen in another application as evidence, and match a screen frame without storing it (6.10).
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

- Camera and sensor capture: webcams, phone cameras, capture cards, RTSP and other network streams. No video capture device is ever a source.
- Unattended or continuous monitoring, and alerting. No always-on watcher, and no matching of a stream the operator is not actively looking at. Operator-initiated screen capture and operator-initiated live match of the operator's own display are in scope (6.10): each is one explicit action on pixels already on that operator's screen. The watch helper (6.11) is in scope on the same terms — the operator starts it, points it at one surface of their own machine, and stops it — and it is bounded by that: no alert, no recording, no schedule, and no life beyond the window it follows.
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
- C7. Non-commercial models load only when `allow_noncommercial_models` is true. It ships false and is an audited operator decision through `PATCH /api/config`, not an environment-only constant: the operator who owns the installation is the one entitled to accept a licence, and making them edit a file and restart made the decision less visible rather than more considered. Enabling it requires a reason and appends `config.change` with the previous value, the new value, the actor and that reason. The UI and every export state the active model license.

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
| D1 | Input | Collected images and video files, plus screen acquisition on the operator workstation (D18) | Closed | No camera or sensor capture. The screen is not a new collection channel: the operator is already looking at those pixels. |
| D2 | Recognition model | SFace (MIT) is the shipped default. buffalo_l (w600k_r50, 512-d) and antelopev2 glintr100 (ArcFace R100, 512-d) are opt-in behind `allow_noncommercial_models`, lab prototype only | Closed | A fresh clone starts permissively licensed. License or swap the InsightFace weights before product or case use (C7); glintr100 sits on the same licence footing as w600k_r50 and adds a stronger backbone, not a freer one. Making the flag operator-settable (C7, 0.6) changed who records the decision and when; it did not change what the InsightFace licence permits, and this row is a statement about the licence. |
| D3 | Gallery scope | One global gallery across all cases | Closed | A person enrolled once is found in all media. |
| D4 | Detector | YuNet (MIT) is the shipped default. SCRFD-10GF (`det_10g`) is implemented and selectable behind `allow_noncommercial_models` | Closed | YuNet is fast on CPU, 5 landmarks, permissive. SCRFD-10GF is the heavier, stronger option on the same 5-landmark contract, so it is a config change and not a code path (0.7). |
| D5 | Where detection runs | Backend only | Closed | One code path, and the browser never runs a model. Screen frames are posted to the backend like any other pixels (6.10). |
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
| D18 | Screen as a source | Two tiers: `POST /api/capture` ingests screen pixels as evidence; `POST /api/live/match` matches a frame and persists nothing. No identity or template may derive from transient pixels (6.10, invariant 13) | Closed | The operator's material is often already on screen in another application. One path had to be either fully audited evidence or fully transient; a middle path would put an un-auditable image behind a biometric claim. Matching at browse speed needs no writes, and enrolling needs a hashed file, so the split follows the requirement rather than the transport. |
| D19 | Authorization basis | Correctable through `PATCH /api/cases/{id}`, audited as `case.amend_authorization` with the previous text, the new text and a reason | Closed | Section 12 makes the basis the record justifying biometric processing. An uncorrectable field goes stale and stops describing the data; a silently overwritten one destroys the record of what processing was justified under. Both are worse than an audited amendment. |

## 6. Components

### 6.1 Ingest

- Accept images (JPEG, PNG, WebP, HEIC) and video. The video containers accepted are listed in `video.SUPPORTED_VIDEO_SUFFIXES` (mp4, m4v, mov, avi, mkv, webm, mpg, mpeg, wmv): PyAV decodes more, but ingest has to decide `media.kind` from the suffix before anything opens the bytes, because probing an untrusted file would decode it ahead of the hash invariant 7 requires to come first.
- Hash each file (SHA-256) before processing. Store content-addressed. Reuse stored bytes when the same file appears in another case.
- Apply EXIF orientation for images before detection. Store the original unchanged.
- Folder import: recursive, with a job per file.
- A still image produces exactly one track per detection, with `start_ms = end_ms = 0`. Images and video therefore share one `tracks`/`identities` code path.
- The library lists a case's files in two layouts, and the choice is remembered per browser: a gallery of previews, and a list carrying the hash, the dimensions and the job's progress. A preview is `GET /api/media/{id}/thumbnail`, a downscale derived on request from the stored original and never stored; it is a picture to recognise a file by, not evidence, and every decision is still made against the stored detection (invariant 13). Video has no preview until M2 decodes a frame for one.
- A file can be deleted from the library, one at a time (`DELETE /api/media/{id}`) or as a selection (`POST /api/media/bulk_delete`), both section 12.

### 6.2 Processing pipeline (job worker)

Per file:

1. Decode. Images: one frame. Video: sample at `SAMPLE_FPS` (default 3). Keep true timestamps.
2. Detect with the active `Detector`. Letterbox, score threshold, NMS, 5 landmarks.
3. Track (video). Kalman filter, IoU association, high-score pass then low-score pass. States: tentative (3 hits), confirmed, lost (1 s), deleted.
4. Quality gate per detection:
   - Width >= `MIN_EMBED_PX` (start 80, tune in eval).
   - Yaw from landmark symmetry below `MAX_YAW`.
   - Laplacian variance above `MIN_SHARPNESS`, measured on the detection box resampled to the 112x112 embed size, not on native pixels. Laplacian variance tracks sampling density as well as focus, so on native pixels the same face scores an order of magnitude lower when it arrives as an interpolated upscale, which is what a screen capture of a photo on a high-DPI display is. Judging the pixels at the size the embedder consumes makes the criterion a focus test at any rendering scale, and is why `MIN_SHARPNESS` is one number rather than a function of face size.
   - Detector score above `MIN_DET_SCORE`.
5. Align passing crops to the 112x112 ArcFace 5-point template. Write the aligned crop of every quality-passing detection to the content-addressed store and record its SHA-256 in `detections.crop_sha256`.
6. Embed every stored crop and write one row per crop to `detection_embeddings`. Store the L2-normalized mean of the best K crops per track (default K = 5) in `tracks.embedding_mean`. Keeping every crop and its embedding is what lets a model switch re-embed without re-decoding media.
7. Match each track mean against the gallery (6.4).
8. Apply acceptance rules (6.5).
9. Cluster tracks that end as `unknown` (6.7).
10. Write results. Emit progress events.

Still images: one track per detection, `start_ms = end_ms = 0`. Step 3 is skipped; steps 7 to 10 are identical to video, so there is one `tracks`/`identities` code path.

Measurement domain change: `MIN_SHARPNESS` stays 40.0, but the quantity it bounds changed with the step 4 wording above. Sharpness values recorded in `detections.quality_json` before that change are in the old native-pixel domain and are not comparable with values recorded after it. Both remain a faithful record of the decision made at the time; anything that compares them across the boundary is comparing different units.

Job resume: after each committed batch the worker writes a structured checkpoint into `jobs.progress` (last decoded `frame_idx`, last `t_ms`, per-stage counts, carried forward rather than restarted). Resume reads that checkpoint rather than scanning for the last written row, and seeks to the preceding keyframe so the decoder has its references. Detection writes are idempotent on `(media_id, frame_idx, det_idx)` under a unique index, and `frame_idx` is the sample grid's index — a function of the timestamp alone, never a decode ordinal — so the same frame occupies the same slot whether the job started at the beginning or resumed into the middle, and a job resumed inside the batch it died in cannot double-insert. What a resume does not carry is the tracker's own state: a face on screen at the moment of the kill becomes two tracks, one either side of the checkpoint. A track holding fewer detections than `track_min_hits` is swept at end of file, which is also what removes the tentative tracks a killed run left behind.

Re-match job: when the gallery changes (new person or template), re-match stored track embeddings. No re-decode needed. Run it on demand or after each enrollment.

Re-match loads the whole gallery for the active `embedder_model_id` as one `(templates, dim)` float32 matrix and scores blocks of track means against it with a matmul, then reduces per person. It does not issue row-by-row vector-store queries. That is what makes the section 11 target of 100k tracks against 1k persons in under 60 s reachable.

### 6.3 Model interfaces

```python
class Detector(Protocol):
    model_id: str
    def detect(self, image: np.ndarray) -> list[Detection]: ...
    # Detection: bbox (x, y, w, h), score, landmarks (5x2)

class Embedder(Protocol):
    model_id: str          # e.g. "sface-2021dec", "buffalo_l-w600k_r50", "antelopev2-glintr100"
    dim: int
    def embed(self, crops: np.ndarray) -> np.ndarray: ...
    # crops: (N, 112, 112, 3) aligned. Returns (N, dim), L2-normalized.
```

Rules:

- Never compare embeddings from different `model_id` values.
- Store `model_id` with every embedding, template, and match.
- A model switch re-embeds templates and tracks. Provide a CLI job.
- Verify each model file SHA-256 against `models.lock` at start. Refuse to start on mismatch. The API process verifies in `startup_checks`; the worker verifies on its first job and caches the result per `(models_dir, detector, embedder)`, so a configuration change re-verifies before the new model is used but two jobs of the same model do not re-hash 203 MB between them.
- The read path that reports whether a model *could* be selected (`GET /api/config`, `runtime_config.blocked_reason`) memoises digests on `(path, size, mtime_ns)`. Selection itself keeps the uncached hash: `PATCH /api/config` is the moment the bytes are proved, and a stat tuple is not proof.
- `models.lock` records the license of each file. Refuse to load a non-commercial model unless `allow_noncommercial_models` is true (C7). That setting is an audited operator decision, not an environment-only flag: turning it on requires a reason and is recorded in the audit chain, and the license text stays visible in `GET /api/models`, `GET /api/config` and the C7 banner whether it is on or off. `models.lock` verification itself does not move (invariant 8) — unknown or SHA-mismatched weights are refused regardless of any setting, because that is integrity, not licensing.
- Turning `allow_noncommercial_models` off while a non-commercial model is the active detector or embedder is refused (409), naming the active model and the remedy. Forcing the embedder back would re-embed the whole gallery as a side effect of a checkbox; one `PATCH` carrying both keys does it deliberately, with the ordinary model-switch consequences (6.2 re-embed, re-match, auto-accept off until recalibration).
- SFace is the shipped default embedder. buffalo_l loads only once `allow_noncommercial_models` is turned on, so a fresh clone starts in a permissively licensed state. The same holds for glintr100 and for the SCRFD-10GF detector.
- A detector must return five landmarks in the ArcFace order (`[right eye, left eye, nose, right mouth, left mouth]`, as `ARCFACE_TEMPLATE` is laid out), because alignment (6.2 step 5) is a similarity transform onto that template. A detector that emits no keypoints, or emits them in another order, cannot be adopted however good its boxes are: the crop would be warped wrongly and every embedding taken from it would be silently off.
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
- A folder tree already imported through `POST /api/media/import` can be enrolled in one request: one person per immediate subfolder, named verbatim by that folder, with one template per image from the detection already stored for it. It is an operator action with a required reason, it hashes each file to find the media row those bytes were registered as, and it creates nothing from pixels that were never ingested (invariant 13).
- That request refuses to guess, and says so per file: a file loose in the root has no person folder, an image with more than one embeddable face does not say which face is the subject, a folder name that already belongs to two persons is not a decidable target (the whole request stops there), and a person marked `do_not_enroll` is skipped. Everything else in the folder still enrols, and a second run over the same folder creates nothing.
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
- Box color by band. Mark operator-confirmed identities with a distinct badge.
- Labels are on demand, not always on. The caption (name, band, score, track ID) is drawn only for the box under the cursor and for the selected box; the rest are outlines. A frame with 24 detections is 24 captions over the faces the operator is trying to look at, and the live view's quality-gate reasons are long enough to cover a face on their own. The track list beside the media carries the same text for every box at once, so nothing is hidden — it is moved off the picture. Hovering also thickens the box, so the pointer's target is unambiguous before a click selects it.

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

### 6.10 Screen acquisition and live match

The operator's source is often a face already on screen in another application: a photo in a browser, a paused video, Preview. Two tiers serve that. The split between them is a privacy boundary, not an implementation convenience.

Tier 1, evidence. `POST /api/capture` (macOS only) captures the screen with the local `screencapture` binary into a temp file and hands those bytes to the same ingest as `POST /api/media`: hash first (invariant 7), content-addressed store, one `media` row in the case, one `process` job, one `media.ingest` audit entry. The payload records `acquisition` (`upload`, `folder_import` or `screen_capture`) and, for a capture, the selection `capture_mode` (`region`, `window`, `screen`). One action per media row, whatever the source. The temp file is deleted after ingest and the clipboard is never used, so the operator's pasteboard is never disturbed. `GET /api/healthz` reports `capture: {available, platform_supported, binary_present, reason}` so the UI refuses before asking the operator to select. Wrong platform, missing binary, denied Screen Recording permission and a blank frame all fail closed with a reason; a cancelled selection writes nothing.

Tier 2, live match. `POST /api/live/match` takes one frame and runs detect, quality gate, align, embed, gallery match and band assignment through the same `Detector`, `Embedder`, active threshold set and band rules as 6.2 and 6.4. It persists nothing: no `media`, `detections`, `detection_embeddings`, `tracks`, `matches`, `identities`, `identifications` or `templates` row, no stored crop, no per-frame audit entry. It is a read-only query against the gallery, safe to call repeatedly while the operator browses. The answer is advisory. The auto-accept gate is reported so the UI can say why nothing self-confirms, but no live face is ever accepted: auto-acceptance is a property of stored matches (invariant 4, D16).

The optional `identify` field (default true) stops the chain after the quality gate. With `identify=false` no crop is warped, nothing is embedded and no gallery row is scored; the response carries the same boxes and the same quality verdicts with empty candidate lists, `identified: false`, and `embed` and `match` timings of 0.0. The gallery and the auto-accept gate are still read — both are cached — so `gallery_persons` and `auto_accept_allowed` mean what they always mean. This exists because a box that tracks a moving face is worth more to the operator than a name that arrives a third of a second late, and embedding is most of that third of a second. It weakens nothing: a boxes-only frame does strictly less than an identify frame and still stores nothing.

Because tier 2 stores nothing, nothing in a live frame can be tagged or enrolled (invariant 13). Acting on a face the operator sees requires tier 1 first: capture it as evidence, then decide on the resulting detection.

The tier 2 gallery loads once per `embedder_model_id` as one `(templates, dim)` matrix, cached and keyed on the audit chain head hash. Invariant 6 makes that head a total version counter for the database, so a cached matrix cannot outlive a template, person-status or embedding change, and the freshness check costs one indexed read per frame instead of a gallery reload.

Measured on an M5 with CoreML by `tools/bench_perf.py` (section 11), 1080p frames against a 1000-person, 5000-template gallery: 18 ms with no face, 31 ms with one, 38 ms with three. Decode and detect dominate an empty frame. Embedding is about 11 ms for one quality-passing face and about 17 ms for three: the SFace graph has a fixed batch of 1, so one Run per crop is forced, but `Session.run` releases the GIL and the adapter overlaps the Runs on a pool bounded at four, which is worth 2.0x at three crops and above. Gallery scoring is 0.35 ms at three faces. Loopback HTTP adds 1 to 3 ms. The response carries per-stage timings so the client sets its own sampling interval from measurement; one request in flight, frames dropped rather than queued.

The live view runs two cadences over one serial loop. Every third tick is an identify pass encoded at the evidence budget (16.8 MP cap, JPEG q0.92); the other two ask for boxes only at 1 MP and q0.7, which is roughly 13 ms per megapixel of encode and wire against roughly 103 ms. Boxes therefore track the face at detector latency while names refresh at fps/3. Only identify frames are retained for a click, because those are the exact bytes the gallery scored and the only ones fit to become evidence; a box with no identity behind it renders as `identifying…` and refuses to persist. A box inherits the label of the identify face it overlaps by IoU >= 0.3, the same measure the live-to-stored handoff uses, so a name never moves onto a different face. The HUD shows the per-stage timings and which cadence produced the current boxes, and the loop sizes its period from the measured round trip rather than the backend's own figure — encode and wire are most of what a slow machine cannot keep up with.

### 6.11 Watch helper

The Live view can only match what the operator is looking at, because a hidden tab throttles its sampling loop to nothing. The watch helper is the same tier 2 match with a different surface: a small PySide6 application the operator starts, which captures one chosen window, display or dragged region at the configured sample rate, posts each frame to `POST /api/live/match`, and draws the returned boxes, each labelled with its match, on an always-on-top overlay sized to the target. The overlay window itself takes no input. What takes a click is one small numbered square per face, drawn just outside its box and covered by a transparent window of exactly that size; the boxes, the labels and the telemetry block are paint and nothing else, so every other pixel of the watched window keeps both its clicks and its hover — a video player's auto-hiding control bar still appears and still responds under a box. The click surface is small rather than masked because the window server decides by window: a masked overlay is still handed every click inside its frame, and would swallow them. It holds no database handle and no model session — it is an HTTP client of the local backend and nothing else — so every rule the endpoint enforces still holds, and a watched frame is as transient as a live one.

Enrolment from the helper is the tier 1 path, unchanged: the operator clicks a face — on its numbered handle on the overlay, or on its row in the panel's face list — the retained identify frame is posted to `POST /api/media` as `acquisition = screen_capture`, and the browser opens on `#/media/{id}/box/x,y,w,h` so the decision is made against the stored detection (invariant 13). The overlay offers the action and nothing else: no identity is written from a transient frame, and a click on a box with no identity behind it refuses with the reason instead.

It is operator-initiated and bounded, which is what keeps it outside the monitoring ban in section 2: it starts on a click, watches exactly one surface the operator picked, raises no alert, records no video, and stops when the operator stops it or the window it was following closes.

The operator starts it in one of two places, and both are the same action. `./run --watch` starts it with the install, and `POST /api/watch/launch` starts it from the Live view, which is where the operator already is at the moment the browser's own sampling loop is the thing in the way. The endpoint takes no request body, because nothing a caller sends may reach a command line: the argv is a frozen constant — the interpreter already serving the API, `-m watch`, no shell, run from `backend/` — so the only thing a request decides is whether a helper starts. Launching through that interpreter rather than through a package manager is what makes C1 structural rather than incidental, since a launch that could resolve a missing Qt would be an outbound call on an operator's click; it also means the pid reported and audited is the helper itself and not a launcher that owns it. The child's output appends to `data/logs/watch.log`, the file `run` already writes, and it takes the backend URL from the environment `run` exports, so not even a port is interpolated into anything. Refusals are sentences: 503 off macOS, 503 when the `watch` extra is not installed here, and 409 when a helper started this way is already running, because two always-on-top overlays over one screen help nobody. `GET /api/watch` reports the same availability, so the UI shows the reason instead of offering a dead button, and the running helper it reports is one this backend started — a helper the operator ran from a terminal is their own and the endpoint does not police it.

Each launch appends `watch.launch` to the hash-chained log with the exact argv, the working directory, the log path and the pid it started, because beginning to capture the operator's own display is an operator decision about special-category data and belongs in the chain beside the others (invariant 6, section 12). A launch whose entry cannot be written terminates the helper rather than leave an unaudited process that can read a screen. None of this widens section 2, because pressing the button starts an application and not a capture: the helper draws its panel and captures nothing until somebody at that machine picks a window, display or region in it and presses Start there. Nothing else may start it — no start-on-load, no retry after a refusal, and no status poll that restarts anything — and it is loopback-only by construction, because the listener is (invariant 11).

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
- Connection policy (`app/db/conn.py`): WAL, `synchronous = FULL`, foreign keys on, a 10 s busy timeout, 64 MiB page cache, 256 MiB mmap and in-memory temp store. `synchronous` stays FULL — the audit log is the evidence, and its durability is not tradeable for write speed. The API holds one connection per (thread, database) for the life of the process rather than opening one per request, because opening one re-runs every PRAGMA and discards the page cache it just declared. Writes still serialise through `BEGIN IMMEDIATE`, so the single-writer property section 9 depends on is unchanged, and a request that returns with a transaction still open has it rolled back before the connection is reused.
- Migration `0004_perf_indices` adds three indices for queries that were scanning whole tables: `tracks (embedder_model_id) WHERE embedding_mean IS NOT NULL` for re-match, `matches (band, score DESC) WHERE rank = 1` for the review queue, and `jobs (json_extract(params_json, '$.media_id'))` for the latest-job-per-media join in the media list. A re-match scoped to specific tracks filters them in SQL, chunked at 500 bound ids, rather than loading every track and filtering in Python.

## 8. API contract (v1)

```
POST   /api/cases
GET    /api/cases/{case_id}
PATCH  /api/cases/{case_id}            {authorization_basis, reason?} -> CaseOut
                                       audited correction, keeps the old text (section 12)

POST   /api/media                      multipart upload -> {media_id, sha256, job_id}
POST   /api/media/import               {case_id, folder_path} ->
                                       {job_ids[], media_ids[], reused}
POST   /api/capture                    {case_id, mode, source_url} ->
                                       {media_id, sha256, job_id, reused}
                                       macOS screen capture, ingested as evidence (6.10)
POST   /api/live/match                 multipart frame + optional case_id + optional
                                       identify (default true) -> boxes and, when
                                       identify is true, ranked candidates. The response
                                       echoes `identified`. Advisory, persists nothing (6.10)
GET    /api/watch                      {available, platform_supported, extra_installed,
                                       running, pid, reason}
POST   /api/watch/launch               no body -> {pid, log_path}. Starts the watch helper
                                       on this machine from a frozen argv, audited as
                                       `watch.launch`. 503 off macOS or without the `watch`
                                       extra, 409 when one started this way runs (6.11)
GET    /api/media?status=
GET    /api/media/{id}
GET    /api/media/{id}/file            range requests for video
GET    /api/media/{id}/thumbnail?size= derived JPEG preview: a downscale for an image, the
                                       first decodable frame for a video. ETag on the source
                                       digest; 410 when the object is missing from the store,
                                       415 when it will not decode (6.1)
DELETE /api/media/{id}                 purge one file (section 12). No body; 404 when it is
                                       already gone
  resp: {media_id, detections, tracks, templates, identities, identifications,
         matches, persons_unenrolled, object_removed, crops_removed, rematch_job_id}
POST   /api/media/bulk_delete          {media_ids[]} -> {deleted[], errors[], detections,
                                       tracks, templates, persons_unenrolled,
                                       objects_removed, rematch_job_id}. One purge per
                                       file; a file already gone is reported, not fatal
GET    /api/media/{id}/tracks?from_ms=&to_ms=
  resp: [{track_id, person_id, name, band, score, source,
          samples: [{t_ms, x, y, w, h}]}]

GET    /api/tracks/{id}                candidates, crops, history
POST   /api/identifications            confirm | reject | reassign | new

GET    /api/persons?q=&status=
POST   /api/persons
GET    /api/persons/{id}
POST   /api/persons/{id}/templates     from image upload or detection_id
POST   /api/persons/enroll_folder      {case_id, folder_path, reason} -> persons[],
                                       templates_created, skipped[], files_seen,
                                       rematch_job_id. Bulk enrol from an imported
                                       folder tree, one person per subfolder (6.6)

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
DELETE /api/persons/{id}              purge person across all cases (section 12). No body;
                                      404 when they are already gone
  resp: {person_id, templates, identities, identifications, matches,
         clusters_unlabelled, rematch_job_id}
GET    /api/audit?from_seq=
GET    /api/healthz                    db, models.lock, active threshold set, execution
                                       provider, screen-capture capability
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

Measurements come from `tools/bench_perf.py`, which builds a scratch database in a temp directory, seeds a synthetic 1000-person gallery, loads the real weights through the ordinary registry, and reports medians as one JSON object: per-frame stage timings at three resolutions and three face counts, embed at 1/3/8 crops, one `process_image`, one full `rematch`, `models_lock.verify`, and `runtime_config.blocked_reason`. It never opens `data/facematch.db`. A change that claims a speedup quotes its output before and after.

A measurement is only comparable against another run of that script on the same machine. It is also not a numerics check: any change to the execution provider, provider options, graph optimisation level or a re-exported weight file moves scores while the active calibrated threshold set still looks valid, because `acceptance.build_gate` compares only the provider name. Re-running `eval/run.py` against a scratch DB and comparing `t_strong`, `t_possible` and `margin` to the active set is the check that catches that.

## 12. Security and privacy

- Bind the server to `127.0.0.1` only (invariant 11). Remote access is a `tailscale serve` proxy in front of that loopback listener. Never bind a tailnet IP, and never bind `0.0.0.0`.
- Accepted risk: there is no app-level authentication in v1. Anyone who can reach the proxy is treated as the single operator named in `OPERATOR_NAME`, so `identifications.operator` is an install-level claim, not an authenticated one.
- No outbound calls. Block container egress at the network level.
- Full-disk encryption on all hosts (D14).
- The global gallery removes case compartments. Any enrolled person can match in any case. Record the enrolling case and its `authorization_basis` on each person.
- Case purge removes the case media, crops, detections, tracks, and all templates sourced from that case. Then it re-matches affected persons in other cases.
- A case purge that strands a person at zero active templates sets `persons.status = 'unenrolled'`. The person row survives, so audit and identification history stay readable. An `unenrolled` person is excluded from matching, and the following re-match reverts that person's auto-accepted identities to unknown. Operator-confirmed identities survive (invariant 5).
- Media purge (`DELETE /api/media/{id}`) is case purge at file granularity, and removes the same classes of row: the media, its detections and their stored crops, its tracks, and every template enrolled from one of its faces, plus the identities, identifications and matches that rest on either. A person left at zero active templates becomes `unenrolled` exactly as a case purge leaves them, and an operator decision on another file keeps its person and loses only the match it was scored against (invariant 5). The stored object and each crop are content-addressed and may belong to more than one case, so they are unlinked only once no row anywhere still points at them. The response reports what went, including whether the bytes were removed. A delete that removed a template queues a re-match; one that removed none leaves the gallery, and therefore every stored score, exactly as it was.
- Person purge (`DELETE /api/persons/{id}`) removes the person, all their templates — revoked ones included — and every identity, identification and match naming them, in every case. It does not touch the evidence those claims were made from: the media, the detections and the stored crops stay, because they record what was in the picture rather than who it was. Re-processing the same file afterwards finds the same faces and names nobody, which is the correct end state. A cluster labelled with the person keeps its membership and loses the label.
- A purge is irreversible, and the guard is a single confirm click on the bin icon in the persons library — not a typed name and not a written justification. A form that has to be argued with gets clicked through rather than read, and the operator clearing their own gallery is the ordinary case rather than the dangerous one. The purge queues a re-match, since the gallery it removed a person from is the one every stored auto score was measured against. Deleting a person who is already gone is a 404 that writes nothing.
- The audit log keeps purge entries with hashes only. A `person.purge` entry carries the person id, the counts of what was removed, and the SHA-256 of the display name — never the name. A log that reprints the personal data it just deleted has not deleted it.
- `do_not_enroll` on a person blocks template creation and auto-acceptance for that person.
- No identity, identification or template may derive from transient pixels (invariant 13). The live match path (6.10) stores nothing and therefore cannot enroll or tag. Enrollment and tagging act only on a stored detection whose source media was hashed and content-addressed at ingest, so a biometric claim stays re-checkable against the bytes it was made from.
- `authorization_basis` is correctable and every correction is audited. `PATCH /api/cases/{id}` writes the new text and appends `case.amend_authorization` in the same transaction, carrying the previous text, the new text and the operator's reason. A basis that no longer describes the material is worse than no basis, and a silent overwrite would destroy the record of what processing was justified under. The correction is evidence too.
- Biometric data is special-category data under GDPR Art. 9, and Canadian regulators have acted on facial recognition misuse. Automated identification without review raises the bar. Record `authorization_basis` on each case. Get legal review before any product use.

## 13. Repo layout

```
face-match/
  frontend/            React + Vite + TS, built to frontend/dist and served by the backend
    src/views/         one file per route: media library, media detail, live,
                       persons, person detail, review, status, config
    src/components/    shared pieces (band pill, loading, case basis, notices)
    src/lib/           router, resource polling, geometry, live frame plumbing
    src/api/           client.ts, types.ts (the wire contract, hand-written)
  backend/
    app/api/           routers + deps.py (connection pool, effective settings)
    app/core/          interfaces, registry, scoring, acceptance, vectors, storage
    app/adapters/      yunet.py, scrfd.py, sface.py, arcface.py, boxes.py
    app/pipeline/      ingest.py, decode.py, video.py, quality.py, align.py,
                       tracking.py, process.py, matching.py, reembed.py,
                       capture.py, live.py
    app/db/            conn.py, migrate.py, migrations/
    app/audit.py, app/worker.py, app/jobs.py, app/config.py, app/purge.py,
    app/runtime_config.py, app/models_lock.py, app/thresholds.py, app/cli.py,
    app/watch_launch.py
    tests/
  models/              ONNX files (gitignored) + models.lock (tracked)
  tools/fetch_models.py  build-time weight provisioning, pinned by SHA-256.
                       Standalone: imports nothing from app/, so the runtime
                       package contains no download code at all (C1).
  tools/bench_perf.py    performance harness (section 11). Imports app/, runs
                       against a scratch DB, never touches data/.
  eval/                run.py, metrics.py (section 10)
  data/                gitignored: facematch.db, media/, crops/, logs/
  fixtures/            gitignored face images for tests and calibration
  docs/spec.md
  README.md
  docker-compose.yml
  LICENSE              MIT
```

## 14. Milestones

| M | Deliverable | Exit test |
|---|-------------|-----------|
| M0 | Repo, compose, FastAPI, static frontend, schema, audit chain, job worker | Audit chain verifies after 1000 writes |
| M1 | Images: ingest, detect, enroll, match, auto-accept, overlay, tag, plus a minimal calibration run (about 10 identities) that writes an audited `threshold_set` with `calibrated = true` | Enroll 10 people, activate the calibrated set, auto-identify on new photos end to end, overrides logged |
| M1.5 | Screen acquisition and live match (6.10): `POST /api/capture`, `POST /api/live/match`, capture capability in `GET /api/healthz` | Capture a face shown in another application, worker processes it, tracks appear; a live frame returns boxes and candidates while writing no row and no audit entry |
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
