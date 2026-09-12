/**
 * Wire types for the backend API (spec section 8).
 *
 * These mirror the Pydantic response models in `backend/app/api/`. Keep them in
 * step with that package: the two ends change together. Field names are exactly
 * the JSON keys, so snake_case here is deliberate.
 */

export type Band = "strong" | "possible" | "ambiguous" | "unknown";

export const BANDS: readonly Band[] = ["strong", "possible", "ambiguous", "unknown"];

/** `identities.source`. Absent when nothing has been accepted for the track. */
export type IdentitySource = "auto" | "operator";

/** What an operator may ask for through the identification API. */
export type Decision = "confirm" | "reject" | "reassign" | "new";

/**
 * What `identifications.decision` may already hold. `cluster_assign` is written
 * by the clustering path, never requested by an operator, so track history can
 * report it but `Decision` cannot express it.
 */
export type RecordedDecision = Decision | "cluster_assign";

/* ------------------------------------------------------------------ health */

export type ModelStatus = {
  model_id: string;
  /** From `models.lock`. Null until the weight file is provisioned (C2, C7). */
  license: string | null;
  present: boolean;
  /** Embedding dimensionality. Null for the detector and for missing weights. */
  dim: number | null;
};

export type ThresholdSetStatus = {
  id: string;
  calibrated: boolean;
  /** Gallery size the set was calibrated against (section 10). */
  gallery_size: number | null;
  execution_provider: string | null;
};

/**
 * Whether `POST /api/capture` can run here. `reason` is non-null exactly when
 * `available` is false; the two narrower booleans are diagnostics.
 */
export type CaptureStatus = {
  available: boolean;
  platform_supported: boolean;
  binary_present: boolean;
  reason: string | null;
};

/**
 * The C5 gate, decided by the backend. `reason` is non-null exactly when
 * `allowed` is false; `warning` is a caveat that stands even when it is true
 * (a set calibrated against a much smaller gallery, say). Read it rather than
 * re-deriving it from the threshold set: the gate also depends on state the
 * threshold set does not carry, such as a re-embed in flight.
 */
export type AutoAcceptState = {
  allowed: boolean;
  reason: string | null;
  warning: string | null;
};

export type Health = {
  status: string;
  version: string;
  db_path: string;
  migration_version: number;
  embedder: ModelStatus;
  detector: ModelStatus;
  execution_provider: string;
  allow_noncommercial_models: boolean;
  /** Null until a threshold set is activated. No auto-accept without one (C5). */
  threshold_set: ThresholdSetStatus | null;
  /** Whether anything may be auto-accepted right now (C5, invariant 4). */
  auto_accept: AutoAcceptState;
  /** Screen-capture support on this machine. */
  capture: CaptureStatus;
  audit_head_seq: number;
  /** Null on an empty chain. */
  audit_head_hash: string | null;
};

/* ------------------------------------------------------------------- audit */

export type AuditEntry = {
  seq: number;
  /** ISO 8601 UTC with microseconds, trailing `Z`. */
  ts: string;
  actor: string;
  case_id: string | null;
  action: string;
  object_type: string;
  object_id: string | null;
  payload: Record<string, unknown>;
  /** 64 ASCII zeros for the genesis entry, never null. */
  prev_hash: string;
  hash: string;
};

export type AuditPage = {
  entries: AuditEntry[];
  /** Seq to request for the next page, or null at the head of the chain. */
  next_seq: number | null;
  head_seq: number;
};

/* ------------------------------------------------------------------- cases */

export type Case = {
  id: string;
  name: string;
  authorization_basis: string;
  created_at: string;
  created_by: string;
};

export type CaseList = {
  items: Case[];
};

/**
 * `PATCH /api/cases/{case_id}`: correct the record that justifies processing
 * everything already in the case. The response is the amended `Case`; the
 * amendment is audited with the old and the new text, so the previous basis
 * stays part of the record rather than being overwritten.
 */
export type CaseAmendment = {
  /** The corrected basis, 1..2000 characters. */
  authorization_basis: string;
  /** Why the correction was made, recorded alongside it; null when unstated. */
  reason: string | null;
};

/* -------------------------------------------------------------------- jobs */

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

/** Registered worker handlers; the backend rejects any other kind at enqueue. */
export type JobKind =
  | "ingest"
  | "process"
  | "rematch"
  | "reembed"
  | "cluster"
  | "export"
  | "audit_verify";

export type Job = {
  id: string;
  kind: JobKind;
  status: JobStatus;
  params: Record<string, unknown>;
  progress: Record<string, unknown>;
  error: string | null;
  created_at: string;
  updated_at: string;
};

/* ------------------------------------------------------------------- media */

export type MediaStatus = "new" | "processing" | "done" | "failed";

export type Media = {
  id: string;
  case_id: string;
  sha256: string;
  kind: "image" | "video";
  source_url: string | null;
  acquired_at: string | null;
  /** Null until the pipeline has decoded the file. */
  width: number | null;
  height: number | null;
  duration_ms: number | null;
  fps: number | null;
  ingested_at: string;
  status: MediaStatus;
  /** Latest pipeline job for this media, so the library can show live state. */
  job: MediaJob | null;
  /** Detections stored so far. */
  detection_count: number;
};

export type MediaJob = {
  id: string;
  kind: "process" | "rematch";
  status: JobStatus;
  error: string | null;
  progress: Record<string, unknown>;
  updated_at: string;
};

export type MediaList = {
  items: Media[];
};

export type MediaUpload = {
  media_id: string;
  sha256: string;
  /** Null exactly when `reused`: identical bytes were already ingested. */
  job_id: string | null;
  reused: boolean;
};

/** `POST /api/media/import` of a folder of stills. */
export type MediaImport = {
  job_ids: string[];
  media_ids: string[];
  reused: number;
};

/* ----------------------------------------------------------------- capture */

/** What the host screen-capture tool is asked to grab. */
export type CaptureMode = "region" | "window" | "screen";

/**
 * `POST /api/capture`: the host grabs the screen and ingests the result, so the
 * response is a plain `MediaUpload`. 400 `capture cancelled` means the operator
 * pressed Esc; 503 means capture is unavailable on this machine.
 */
export type CaptureRequest = {
  case_id: string;
  mode: CaptureMode;
  /** Provenance to record with the grab; null when there is none. */
  source_url: string | null;
};

/* -------------------------------------------------------------------- live */

/**
 * Tier 2 live match (`POST /api/live/match`).
 *
 * Advisory only: the endpoint stores no row and appends no audit entry, so a
 * live face can never be tagged or enrolled. Acting on one means persisting
 * the frame first through `POST /api/media`, then deciding on the stored
 * detection.
 */
export type LiveCandidate = {
  person_id: string;
  name: string;
  rank: number;
  score: number;
  band: Band;
  best_template_id: string;
};

/** Box in frame pixels, as analysed at the resolution the frame was posted in. */
export type LiveFace = {
  x: number;
  y: number;
  w: number;
  h: number;
  det_score: number;
  /** False when the crop failed the gate: it was never embedded or matched. */
  quality_passed: boolean;
  quality_reasons: string[];
  candidates: LiveCandidate[];
};

/** Per-stage wall time in ms, so the UI can size its own sampling interval. */
export type LiveTimings = {
  decode: number;
  detect: number;
  quality_align: number;
  embed: number;
  match: number;
};

export type LiveMatchResult = {
  width: number;
  height: number;
  faces: LiveFace[];
  /** Null when no threshold set is active. */
  threshold_set_id: string | null;
  /** The gate for the stored path; no live face is ever auto-accepted (C4). */
  auto_accept_allowed: boolean;
  auto_accept_reason: string | null;
  gallery_persons: number;
  elapsed_ms: number;
  timings: LiveTimings;
};

/* ------------------------------------------------------------------ tracks */

/** One stored box sample. Still images have exactly one, at `t_ms = 0`. */
export type TrackSample = {
  t_ms: number;
  x: number;
  y: number;
  w: number;
  h: number;
  detection_id: string;
  crop_sha256: string | null;
};

/** Overlay row from `GET /api/media/{id}/tracks`. */
export type TrackOverlay = {
  track_id: string;
  /** Current identity, or null when the track is unidentified. */
  person_id: string | null;
  name: string | null;
  /** Band of the rank-1 match, or null when the track was never matched. */
  band: Band | null;
  /** Raw cosine similarity of the rank-1 match. Never a probability (6.4). */
  score: number | null;
  source: IdentitySource | null;
  samples: TrackSample[];
  /** Best crop for the track; null when the quality gate rejected every crop. */
  crop_sha256: string | null;
};

export type MediaTracks = {
  media_id: string;
  /** Original-image pixel size the boxes are expressed in. */
  width: number | null;
  height: number | null;
  tracks: TrackOverlay[];
};

export type Candidate = {
  person_id: string;
  name: string;
  rank: number;
  score: number;
  band: Band;
  best_template_id: string;
};

export type TrackIdentity = {
  person_id: string;
  name: string;
  source: IdentitySource;
  updated_at: string;
};

export type TrackHistoryEntry = {
  id: string;
  decision: RecordedDecision;
  person_id: string | null;
  name: string | null;
  operator: string;
  note: string | null;
  created_at: string;
};

/** `GET /api/tracks/{id}`: candidates, crops, history (spec section 8). */
export type TrackDetail = {
  track_id: string;
  media_id: string;
  case_id: string;
  start_ms: number;
  end_ms: number;
  identity: TrackIdentity | null;
  candidates: Candidate[];
  /** SHA-256 of the track's crops, best first; fetch at `/api/crops/{sha256}`. */
  crops: string[];
  /** Detection ids of the track, aligned with `crops` where a crop exists. */
  detection_ids: string[];
  history: TrackHistoryEntry[];
};

/* --------------------------------------------------------- identifications */

export type IdentificationRequest = {
  track_id: string;
  decision: Decision;
  /** Required for `confirm` and `reassign`; omitted for `reject` and `new`. */
  person_id?: string;
  /** Required for `new`: the display name of the person to create. */
  new_name?: string;
  note?: string;
  /**
   * Opt-in enrollment (D17, spec 6.6): tagging says who a track is, enrolling
   * donates its crop to the gallery. Honoured for `confirm` and `reassign`;
   * `new` always bootstraps one template regardless, and sending it with
   * `reject` is a 422.
   */
  enroll?: boolean;
};

export type Identification = {
  id: string;
  track_id: string;
  person_id: string | null;
  /** Echo of the requested decision; never `cluster_assign` (operator path). */
  decision: Decision;
  operator: string;
  note: string | null;
  /**
   * Whether THIS request created a template. Tagging is not enrolling (D17),
   * and `new` bootstraps one only when the track has a quality-passing crop to
   * enrol from, so a saved decision does not imply the person can ever be
   * matched.
   */
  template_created: boolean;
  created_at: string;
};

/* ----------------------------------------------------------------- persons */

export type PersonStatus = "enrolled" | "unenrolled";

export type Person = {
  id: string;
  display_name: string;
  notes: string | null;
  do_not_enroll: boolean;
  status: PersonStatus;
  template_count: number;
  /**
   * Crop of the person's best active template, for a gallery thumbnail; null
   * when nothing is enrolled. Fetch at `/api/crops/{sha256}`.
   */
  crop_sha256: string | null;
  created_at: string;
  created_by: string;
};

export type PersonList = {
  items: Person[];
};

export type PersonCreate = {
  display_name: string;
  notes?: string;
};

/** Body of `PATCH /api/persons/{person_id}`. */
export type PersonUpdate = {
  do_not_enroll: boolean;
};

export type Template = {
  id: string;
  detection_id: string | null;
  source_case_id: string | null;
  embedder_model_id: string;
  quality: number | null;
  status: "active" | "revoked";
  crop_sha256: string | null;
  created_at: string;
  created_by: string;
};

/**
 * Body of `POST /api/persons/{person_id}/templates/{template_id}/revoke`.
 *
 * Answers with the `Template` in its revoked state. 404 when the person or the
 * template is unknown, or when that template belongs to someone else; 409 when
 * it was already revoked, which is a race rather than a failure.
 */
export type TemplateRevoke = {
  /** Why it is leaving the gallery, for the audit log. Null when none was given. */
  reason: string | null;
};

/** One track where this person appears (spec 6.8 person page). */
export type Appearance = {
  track_id: string;
  media_id: string;
  case_id: string;
  source: IdentitySource;
  score: number | null;
  band: Band | null;
  t_ms: number;
  crop_sha256: string | null;
  /** Media ingest timestamp: the timeline axis for stills. */
  ingested_at: string;
};

export type PersonDetail = {
  person: Person;
  templates: Template[];
  appearances: Appearance[];
};

/* ------------------------------------------------------------------ review */

export type ReviewItem = {
  track_id: string;
  media_id: string;
  case_id: string;
  band: Band;
  score: number;
  person_id: string;
  name: string;
  crop_sha256: string | null;
  t_ms: number;
};

export type ReviewList = {
  items: ReviewItem[];
};

export type ReviewBulkDecision = {
  track_id: string;
  decision: Decision;
  /** Required for `confirm` and `reassign`; omitted for `reject` and `new`. */
  person_id?: string;
  /** Required for `new`: the display name of the person to create. */
  new_name?: string;
  /** Opt-in enrollment, same rule as `IdentificationRequest.enroll` (D17). */
  enroll?: boolean;
};

export type ReviewBulkRequest = {
  decisions: ReviewBulkDecision[];
};

export type ReviewBulkResult = {
  applied: number;
  /** Per-track failures, e.g. a trigger refusing to downgrade an operator row. */
  errors: { track_id: string; error: string }[];
};

/* --------------------------------------------------------- threshold sets */

export type ThresholdSet = {
  id: string;
  model_id: string;
  t_strong: number;
  t_possible: number;
  margin: number;
  calibrated: boolean;
  calibrated_at: string | null;
  eval_report_sha256: string | null;
  gallery_size: number | null;
  execution_provider: string | null;
  active: boolean;
};

export type ThresholdSetList = {
  items: ThresholdSet[];
};

/* ------------------------------------------------------------------ models */

export type ModelKind = "detector" | "embedder";

export type ModelInfo = {
  id: string;
  name: string;
  version: string;
  kind: ModelKind;
  sha256: string;
  /** From `models.lock`. Null until the weight file is provisioned (C2, C7). */
  license: string | null;
  dim: number | null;
  active: boolean;
  present: boolean;
};

export type ModelsInfo = {
  items: ModelInfo[];
  execution_provider: string;
  allow_noncommercial_models: boolean;
};

/* ------------------------------------------------------------------ config */

/** Spec 6.4: `max` is one template, `mean_top3` needs five or more. */
export type PersonScoreMode = "max" | "mean_top3";

/**
 * The settings `PATCH /api/config` will take. Everything else in `Settings` is
 * environment-only and arrives under `readonly`.
 */
export type ConfigEditable = {
  detector_model: string;
  embedder_model: string;
  min_embed_px: number;
  max_yaw: number;
  min_sharpness: number;
  min_det_score: number;
  sample_fps: number;
  top_k: number;
  person_score_mode: PersonScoreMode;
};

/** Facts about this deployment. Changing any of them means editing `.env` and restarting. */
export type ConfigReadonly = {
  execution_provider: string;
  allow_noncommercial_models: boolean;
  operator_name: string;
  db_path: string;
  models_dir: string;
  max_upload_bytes: number;
  fpir_target: number;
  embed_k: number;
  rematch_block_size: number;
};

/**
 * A model the operator may pick. Distinct from {@link ModelInfo}: this one
 * carries `commercial_use`, which with `allow_noncommercial_models` decides
 * whether the backend would load it at all (invariant 9, C7), and drops the
 * digest, which belongs on the license surface rather than on a picker.
 */
export type ConfigModel = {
  id: string;
  name: string;
  version: string;
  kind: ModelKind;
  license: string | null;
  commercial_use: boolean;
  dim: number | null;
  present: boolean;
  active: boolean;
};

/**
 * The job that has to finish before another model change is accepted. Only the
 * two model-switch kinds can hold this slot, and only while unfinished: a
 * settled job leaves `pending_job` null.
 */
export type ConfigPendingJob = {
  id: string;
  kind: "reembed" | "rematch";
  status: "queued" | "running";
};

export type Config = {
  editable: ConfigEditable;
  readonly: ConfigReadonly;
  models: ConfigModel[];
  pending_job: ConfigPendingJob | null;
};

/**
 * Body of `PATCH /api/config`. Only the keys that actually changed go in
 * `changes`; an unknown or unchangeable key is a 400. `reason` is the audit
 * justification and is required for a model change by the UI.
 */
export type ConfigPatchRequest = {
  changes: Partial<ConfigEditable>;
  reason: string | null;
};

/**
 * The 200 of a PATCH: the new config, plus the jobs the change enqueued, in
 * the order the worker will run them. An embedder change yields two, the
 * re-embed then the re-match; a detector change yields none, because detection
 * is never re-run on media that is already stored.
 */
export type ConfigPatchResult = Config & {
  jobs_enqueued: string[];
};
