/**
 * Bounds, labels and the change diff for the runtime settings a operator may edit.
 *
 * The backend is the authority and answers 422, but these knobs are the
 * pipeline: `min_embed_px` at 8 embeds noise, `min_det_score` at 0.99 finds
 * nothing, and either mistake is discovered a whole re-process later. The
 * bounds here are the range in which the value still means what its name says,
 * so a typo is caught in the field that made it rather than in a round trip.
 *
 * Every draft field is held as a string. A number input mid-edit is "", "0."
 * or "-", none of which is a number, and coercing on each keystroke would
 * rewrite what the operator is typing.
 */

import type { ConfigEditable, PersonScoreMode } from "../api/types";

export type NumericKey =
  | "min_embed_px"
  | "max_yaw"
  | "min_sharpness"
  | "min_det_score"
  | "sample_fps"
  | "top_k";

export type NumericField = {
  key: NumericKey;
  label: string;
  /** Shown after the value; null when the number is dimensionless. */
  unit: string | null;
  help: string;
  min: number;
  max: number;
  step: number;
  integer: boolean;
};

/** Quality gate, spec 6.2 step 4: what a detection must clear to be embedded. */
export const QUALITY_FIELDS: readonly NumericField[] = [
  {
    key: "min_embed_px",
    label: "Minimum face width",
    unit: "px",
    // The aligned crop is 112x112, so anything below that is upscaled into the
    // embedder; 16 px is already past the point of carrying an identity.
    help: "Detections narrower than this never reach the embedder. The aligned crop is 112 px, so below that the face is an upscale.",
    min: 16,
    max: 1024,
    step: 1,
    integer: true,
  },
  {
    key: "max_yaw",
    label: "Maximum yaw",
    unit: "\u00b0",
    help: "Head turn estimated from landmark symmetry. A profile past this angle is rejected rather than embedded badly.",
    min: 0,
    max: 90,
    step: 1,
    integer: false,
  },
  {
    key: "min_sharpness",
    label: "Minimum sharpness",
    unit: null,
    help: "Laplacian variance of the box resampled to 112 px, so it measures focus and not resolution. Values recorded before that change are in a different domain.",
    min: 0,
    max: 10000,
    step: 1,
    integer: false,
  },
  {
    key: "min_det_score",
    label: "Minimum detector score",
    unit: null,
    help: "Detector confidence, 0 to 1. Raising it drops weak boxes before the rest of the gate sees them.",
    min: 0,
    max: 1,
    step: 0.01,
    integer: false,
  },
];

/** Matching, spec 6.4: how many candidates a track is scored against. */
export const TOP_K_FIELD: NumericField = {
  key: "top_k",
  label: "Candidates per track",
  unit: null,
  help: "How many persons the matcher returns for one track, best first.",
  min: 1,
  max: 50,
  step: 1,
  integer: true,
};

/** Sampling, spec 6.2 step 1. Stills are one frame whatever this says. */
export const SAMPLE_FPS_FIELD: NumericField = {
  key: "sample_fps",
  label: "Video sample rate",
  unit: "fps",
  help: "Frames decoded per second of video. Higher costs processing time; lower risks missing a short appearance.",
  min: 0.1,
  max: 60,
  step: 0.1,
  integer: false,
};

export const NUMERIC_FIELDS: readonly NumericField[] = [
  ...QUALITY_FIELDS,
  TOP_K_FIELD,
  SAMPLE_FPS_FIELD,
];

export const FIELD_LABELS: Record<keyof ConfigEditable, string> = {
  detector_model: "Detector model",
  embedder_model: "Embedder model",
  allow_noncommercial_models: "Non-commercial models",
  min_embed_px: "Minimum face width",
  max_yaw: "Maximum yaw",
  min_sharpness: "Minimum sharpness",
  min_det_score: "Minimum detector score",
  sample_fps: "Video sample rate",
  top_k: "Candidates per track",
  person_score_mode: "Person score mode",
};

export const PERSON_SCORE_MODES: readonly { value: PersonScoreMode; label: string }[] = [
  // Short enough to read inside the select; the caveats are in the help text.
  { value: "max", label: "max \u2014 best template" },
  { value: "mean_top3", label: "mean_top3 \u2014 mean of best 3" },
];

/**
 * Every editable setting as text, which is what the controls bind to. The
 * license flag rides as "true"/"false" rather than splitting the draft into
 * two shapes for one checkbox; {@link diffDraft} narrows it back.
 */
export type Draft = Record<keyof ConfigEditable, string>;

export function draftFrom(editable: ConfigEditable): Draft {
  return {
    detector_model: editable.detector_model,
    embedder_model: editable.embedder_model,
    allow_noncommercial_models: editable.allow_noncommercial_models ? "true" : "false",
    min_embed_px: String(editable.min_embed_px),
    max_yaw: String(editable.max_yaw),
    min_sharpness: String(editable.min_sharpness),
    min_det_score: String(editable.min_det_score),
    sample_fps: String(editable.sample_fps),
    top_k: String(editable.top_k),
    person_score_mode: editable.person_score_mode,
  };
}

/** Null when the raw text is a usable value for this field. */
export function fieldError(field: NumericField, raw: string): string | null {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return "Required.";
  }
  const value = Number(trimmed);
  // `Number("")` is 0 and `Number("12px")` is NaN: the empty case is handled
  // above, so this only rejects text that is not a number at all.
  if (!Number.isFinite(value)) {
    return "Must be a number.";
  }
  if (field.integer && !Number.isInteger(value)) {
    return "Must be a whole number.";
  }
  if (value < field.min || value > field.max) {
    return `Must be between ${String(field.min)} and ${String(field.max)}.`;
  }
  return null;
}

export type FieldErrors = Partial<Record<NumericKey, string>>;

export function draftErrors(draft: Draft): FieldErrors {
  const errors: FieldErrors = {};
  for (const field of NUMERIC_FIELDS) {
    const message = fieldError(field, draft[field.key]);
    if (message !== null) {
      errors[field.key] = message;
    }
  }
  return errors;
}

/** One pending edit, in the words the confirm step and the notice both use. */
export type Change = {
  key: keyof ConfigEditable;
  label: string;
  from: string;
  to: string;
};

export type Diff = {
  changes: Partial<ConfigEditable>;
  list: Change[];
};

function record<K extends keyof ConfigEditable>(
  diff: Diff,
  key: K,
  from: ConfigEditable[K],
  to: ConfigEditable[K],
): void {
  if (from === to) {
    return;
  }
  diff.changes[key] = to;
  diff.list.push({ key, label: FIELD_LABELS[key], from: String(from), to: String(to) });
}

/**
 * What this draft would send. A field that fails validation is left out
 * entirely: an unparseable box is not a change, it is an unfinished one, and
 * putting NaN in the summary would say otherwise.
 */
export function diffDraft(draft: Draft, current: ConfigEditable): Diff {
  const diff: Diff = { changes: {}, list: [] };
  const errors = draftErrors(draft);
  record(diff, "detector_model", current.detector_model, draft.detector_model);
  record(diff, "embedder_model", current.embedder_model, draft.embedder_model);
  // Spelled out rather than "false → true": the summary is read as a sentence
  // about what the install will do, and a bare boolean does not say it.
  const allow = draft.allow_noncommercial_models === "true";
  if (allow !== current.allow_noncommercial_models) {
    diff.changes.allow_noncommercial_models = allow;
    diff.list.push({
      key: "allow_noncommercial_models",
      label: FIELD_LABELS.allow_noncommercial_models,
      from: current.allow_noncommercial_models ? "allowed" : "blocked",
      to: allow ? "allowed" : "blocked",
    });
  }
  for (const field of NUMERIC_FIELDS) {
    if (errors[field.key] !== undefined) {
      continue;
    }
    record(diff, field.key, current[field.key], Number(draft[field.key].trim()));
  }
  // The select only ever holds one of the two literals; the ternary is what
  // narrows the draft string back to the union.
  const mode: PersonScoreMode = draft.person_score_mode === "mean_top3" ? "mean_top3" : "max";
  record(diff, "person_score_mode", current.person_score_mode, mode);
  return diff;
}
