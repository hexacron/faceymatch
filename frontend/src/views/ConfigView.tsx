import { useCallback, useEffect, useId, useMemo, useRef, useState, type FormEvent } from "react";

import { ApiError, errorMessage, patchJson, postJson } from "../api/client";
import type {
  AutoAcceptState,
  Config,
  ConfigModel,
  ConfigPatchRequest,
  ConfigPatchResult,
  Health,
  Job,
  ModelKind,
} from "../api/types";
import { Loaded } from "../components/Loading";
import { formatBytes, truncateHash } from "../lib/display";
import {
  diffDraft,
  draftErrors,
  draftFrom,
  NUMERIC_FIELDS,
  PERSON_SCORE_MODES,
  QUALITY_FIELDS,
  SAMPLE_FPS_FIELD,
  TOP_K_FIELD,
  type Draft,
  type FieldErrors,
  type NumericField,
} from "../lib/configFields";
import { useResource } from "../lib/useResource";

/**
 * Runtime settings the operator can change, and the ones they cannot.
 *
 * Status is the read-only health surface; this page is where a value moves.
 * Two of these settings are not preferences:
 *
 * The embedder decides what every stored vector means. Embeddings from two
 * different models are never compared (invariant 2), so switching it leaves the
 * whole gallery unmatched until every template and track is re-embedded, and
 * the thresholds calibrated for the old model do not carry over — auto-accept
 * stays off until a set calibrated for the new one is active (invariant 4).
 * That is a confirm step with the consequences spelled out, not a dropdown.
 *
 * The detector only decides what gets found next time: media already processed
 * keeps the detections it has, and no job is enqueued at all. Saying so is the
 * point, because the two controls sit next to each other and look alike.
 *
 * Nothing here needs a restart. Operators of local tools assume the opposite,
 * so the copy says it where the change is made.
 */

type Notice = { tone: "error" | "attention" | "success"; text: string };

const TERMINAL: readonly Job["status"][] = ["done", "failed", "cancelled"];

const NO_RESTART = "The change is live immediately — the backend does not need a restart.";

export default function ConfigView() {
  const config = useResource<Config>("/api/config");
  const health = useResource<Health>("/api/healthz");

  const reloadAll = useCallback(() => {
    config.reload();
    health.reload();
  }, [config, health]);

  return (
    <>
      <div className="view-heading">
        <div>
          <h1>Config</h1>
          <p className="tagline">
            Runtime settings for this backend. Every change is an audited write, a model change
            carries its reason, and nothing here needs a restart.
          </p>
        </div>
        <button type="button" onClick={reloadAll}>Refresh</button>
      </div>
      <Loaded state={health.state} label="backend health">
        {(healthData) => <AutoAcceptBanner state={healthData.auto_accept} />}
      </Loaded>
      <Loaded state={config.state} label="the runtime config">
        {(data) => <ConfigEditor config={data} onReload={reloadAll} />}
      </Loaded>
    </>
  );
}

/**
 * The C5 gate as the backend reports it. It belongs on this page because the
 * commonest way to turn auto-accept off is to change the embedder here.
 */
function AutoAcceptBanner({ state }: { state: AutoAcceptState }) {
  if (!state.allowed) {
    return (
      <div className="notice error banner">
        <strong>Auto-accept is off.</strong>{" "}
        {state.reason ?? "The backend did not give a reason."} Every match stays a candidate until
        that is resolved.
      </div>
    );
  }
  if (state.warning !== null) {
    return (
      <div className="notice attention banner">
        <strong>Auto-accept is on.</strong> {state.warning}
      </div>
    );
  }
  return (
    <div className="notice success banner">
      <strong>Auto-accept is on.</strong> Changing the embedder below turns it off until a
      threshold set calibrated for the new model is active.
    </div>
  );
}

function ConfigEditor({ config, onReload }: { config: Config; onReload: () => void }) {
  const [draft, setDraft] = useState<Draft>(() => draftFrom(config.editable));
  const [reason, setReason] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  /** Enqueued jobs in run order; only the head is actually running. */
  const [queue, setQueue] = useState<readonly string[]>(
    config.pending_job === null ? [] : [config.pending_job.id],
  );
  /** A re-embed that did not complete, which is the one case worth re-running. */
  const [failedReembed, setFailedReembed] = useState<Job | null>(null);
  // A job that already reported a terminal status must not be picked up again
  // from a `pending_job` read before the reload landed.
  const settled = useRef<Set<string>>(new Set());
  const reasonId = useId();
  const summaryId = useId();

  const pending = config.pending_job;
  useEffect(() => {
    if (pending === null || settled.current.has(pending.id)) {
      return;
    }
    setQueue((current) => (current.includes(pending.id) ? current : [...current, pending.id]));
  }, [pending]);

  const onSettled = useCallback(
    (job: Job) => {
      settled.current.add(job.id);
      setQueue((current) => current.filter((id) => id !== job.id));
      if (job.status === "done") {
        setNotice({
          tone: "success",
          text: `The ${job.kind} job ${truncateHash(job.id)} finished.${job.kind === "reembed" ? " Every template and track is embedded under the active model again; auto-accept still needs a threshold set calibrated for it." : ""}`,
        });
      } else {
        if (job.kind === "reembed") {
          setFailedReembed(job);
        }
        setNotice({
          tone: "error",
          text: `The ${job.kind} job ${truncateHash(job.id)} ended ${job.status}${job.error === null ? "" : `: ${job.error}`}. The gallery may be part-embedded under two models, so nothing should be matched until it completes.`,
        });
      }
      onReload();
    },
    [onReload],
  );

  const errors: FieldErrors = useMemo(() => draftErrors(draft), [draft]);
  const diff = useMemo(() => diffDraft(draft, config.editable), [draft, config.editable]);
  const invalid = NUMERIC_FIELDS.some((field) => errors[field.key] !== undefined);
  const embedderChange = diff.changes.embedder_model !== undefined;
  const detectorChange = diff.changes.detector_model !== undefined;
  const modelChange = embedderChange || detectorChange;
  const reasonText = reason.trim();
  const reasonMissing = modelChange && reasonText === "";
  const modelsLocked = queue.length > 0 || pending !== null;
  const showConfirm = confirming && embedderChange;
  const head = queue[0] ?? null;

  const setField = (key: keyof Draft, value: string): void => {
    setDraft((current) => ({ ...current, [key]: value }));
    setConfirming(false);
  };

  function appliedText(result: ConfigPatchResult): Notice {
    const applied = diff.list.map((change) => `${change.label} ${change.from} \u2192 ${change.to}`);
    const head = `Applied: ${applied.join("; ")}. ${NO_RESTART}`;
    if (result.jobs_enqueued.length > 0) {
      const ids = result.jobs_enqueued.map(truncateHash).join(", then ");
      return {
        tone: "attention",
        text: `${head} Enqueued ${String(result.jobs_enqueued.length)} job(s), run in this order: ${ids}. Until they finish the gallery is part-embedded and cannot be matched, and auto-accept stays off until a threshold set calibrated for the new model is active.`,
      };
    }
    if (detectorChange) {
      return {
        tone: "attention",
        text: `${head} No job was enqueued: changing the detector does not re-run detection on media that is already stored. Existing detections, crops and templates are unchanged, and only media processed from now on uses the new detector.`,
      };
    }
    return { tone: "success", text: head };
  }

  async function apply(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (invalid || diff.list.length === 0 || reasonMissing) {
      return;
    }
    // Two-step for the embedder: the first submit opens the consequences, the
    // second one sends them.
    if (embedderChange && !confirming) {
      setConfirming(true);
      return;
    }
    const body: ConfigPatchRequest = {
      changes: diff.changes,
      reason: reasonText === "" ? null : reasonText,
    };
    setBusy(true);
    try {
      const result = await patchJson<ConfigPatchResult>("/api/config", body);
      setNotice(appliedText(result));
      setDraft(draftFrom(result.editable));
      setReason("");
      setConfirming(false);
      setFailedReembed(null);
      if (result.jobs_enqueued.length > 0) {
        for (const id of result.jobs_enqueued) {
          settled.current.delete(id);
        }
        setQueue(result.jobs_enqueued);
      }
      onReload();
    } catch (failure) {
      setConfirming(false);
      if (failure instanceof ApiError && failure.status === 409) {
        // Not a failure of this request: a re-embed the operator may not have
        // seen is still running, and the backend is protecting the gallery.
        setNotice({
          tone: "attention",
          text: "A re-embed is already running, so the backend refused this change and nothing was written. Reloaded from the server \u2014 wait for that job to finish, then submit again.",
        });
      } else if (failure instanceof ApiError && failure.status === 422) {
        // The pickers already exclude every model the backend would reject, so
        // a 422 here means this page and the backend disagree. Verbatim.
        setNotice({
          tone: "error",
          text: `The backend rejected this change as invalid, which should not happen from this page \u2014 report it with this text: ${failure.detail}`,
        });
      } else if (failure instanceof ApiError && failure.status === 400) {
        setNotice({ tone: "error", text: `The backend refused this change: ${failure.detail}` });
      } else {
        setNotice({ tone: "error", text: `Could not apply the change: ${errorMessage(failure)}` });
      }
      onReload();
    } finally {
      setBusy(false);
    }
  }

  async function rerunReembed(): Promise<void> {
    setBusy(true);
    try {
      const job = await postJson<Job>("/api/jobs/reembed", {});
      settled.current.delete(job.id);
      setQueue((current) => [...current, job.id]);
      setFailedReembed(null);
      setNotice({
        tone: "attention",
        text: `Re-embed ${truncateHash(job.id)} queued. It reads the crops already on disk, so no media is decoded again.`,
      });
      onReload();
    } catch (failure) {
      setNotice({ tone: "error", text: `Could not queue the re-embed: ${errorMessage(failure)}` });
    } finally {
      setBusy(false);
    }
  }

  const submitLabel = busy
    ? "Applying\u2026"
    : embedderChange && !confirming
      ? "Review embedder change\u2026"
      : embedderChange
        ? "Change embedder and re-embed the gallery"
        : "Apply changes";

  return (
    <>
      {notice !== null && (
        <div className={`notice ${notice.tone} banner`} role="status">
          {notice.text}
        </div>
      )}

      {head !== null && (
        <JobWatch
          key={head}
          jobId={head}
          remaining={queue.length - 1}
          onSettled={onSettled}
        />
      )}

      {head === null && failedReembed !== null && (
        <div className="notice error banner">
          <p className="compact">
            The last re-embed ended {failedReembed.status}, so part of the gallery is still embedded
            under the previous model and nothing should be matched against it. Re-running resumes
            from the crops already on disk; no media is decoded again.
          </p>
          <p className="compact">
            <button type="button" disabled={busy} onClick={() => void rerunReembed()}>
              Re-run the re-embed
            </button>
          </p>
        </div>
      )}

      <form className="config-form" onSubmit={(event) => void apply(event)}>
        <h2>Models</h2>
        {modelsLocked && (
          <p className="notice attention">
            A model job is running, so the pickers are locked. The backend answers 409 to a model
            change while the gallery is being re-embedded, and a second switch mid-flight would
            leave templates written under two models.
          </p>
        )}
        <ModelPicker
          kind="detector"
          legend="Detector model"
          note="Changing the detector enqueues nothing and does not re-run detection on media that is already stored: existing detections, crops and templates stay exactly as they are. Only media processed after the change uses the new detector, so the library will hold boxes from both until you re-process."
          models={config.models}
          allowNoncommercial={config.readonly.allow_noncommercial_models}
          value={draft.detector_model}
          locked={modelsLocked}
          onPick={(id) => setField("detector_model", id)}
        />
        <ModelPicker
          kind="embedder"
          legend="Embedder model"
          note="Changing the embedder invalidates every stored vector: the backend enqueues a re-embed of all templates and tracks, then a re-match, and nothing can be matched until both finish. You will be asked to confirm the consequences before this is sent."
          models={config.models}
          allowNoncommercial={config.readonly.allow_noncommercial_models}
          value={draft.embedder_model}
          locked={modelsLocked}
          onPick={(id) => setField("embedder_model", id)}
        />

        <fieldset className="config-group">
          <legend>Quality gate</legend>
          <p className="field-help">
            What a detection must clear before it is aligned, stored and embedded (spec 6.2 step 4).
            {" "}
            {NO_RESTART} It applies to media processed from now on; nothing already stored is
            re-gated.
          </p>
          <div className="field-grid">
            {QUALITY_FIELDS.map((field) => (
              <NumberField
                key={field.key}
                field={field}
                value={draft[field.key]}
                error={errors[field.key] ?? null}
                onChange={(value) => setField(field.key, value)}
              />
            ))}
          </div>
        </fieldset>

        <fieldset className="config-group">
          <legend>Matching</legend>
          <div className="field-grid">
            <NumberField
              field={TOP_K_FIELD}
              value={draft.top_k}
              error={errors.top_k ?? null}
              onChange={(value) => setField("top_k", value)}
            />
            <ScoreModeField
              value={draft.person_score_mode}
              onChange={(value) => setField("person_score_mode", value)}
            />
          </div>
        </fieldset>

        <fieldset className="config-group">
          <legend>Sampling</legend>
          <div className="field-grid">
            <NumberField
              field={SAMPLE_FPS_FIELD}
              value={draft.sample_fps}
              error={errors.sample_fps ?? null}
              onChange={(value) => setField("sample_fps", value)}
            />
          </div>
        </fieldset>

        <fieldset className="config-group">
          <legend>Pending changes</legend>
          {diff.list.length === 0 ? (
            <p className="field-help" id={summaryId}>
              Nothing is changed. Edit a value above and the change is listed here before it is
              sent.
            </p>
          ) : (
            <ul className="change-list" id={summaryId}>
              {diff.list.map((change) => (
                <li key={change.key}>
                  {change.label}: <span className="mono">{change.from}</span> {"\u2192"}{" "}
                  <span className="mono">{change.to}</span>
                </li>
              ))}
            </ul>
          )}
          {invalid && (
            <p className="field-error">
              Some values are out of range. They are left out of the list above, and nothing is sent
              until they are fixed.
            </p>
          )}
          {detectorChange && !embedderChange && (
            <p className="field-help">
              This enqueues no job. Stored media keeps the detections it already has.
            </p>
          )}

          <label className="field" htmlFor={reasonId}>
            Reason{" "}
            <span className="muted">
              {modelChange ? "(required for a model change)" : "(optional)"}
            </span>
          </label>
          <textarea
            id={reasonId}
            rows={2}
            maxLength={2000}
            value={reason}
            required={modelChange}
            aria-describedby={summaryId}
            placeholder={
              modelChange
                ? "Why this model is being changed \u2014 this is what the audit log records"
                : "Optional note recorded with this change"
            }
            onChange={(event) => setReason(event.currentTarget.value)}
          />
          {reasonMissing && (
            <p className="field-error">A model change is only accepted with a stated reason.</p>
          )}

          {showConfirm && (
            <EmbedderConfirm
              from={config.editable.embedder_model}
              to={draft.embedder_model}
              detectorChange={detectorChange}
            />
          )}

          <div className="config-actions">
            <button
              type="submit"
              className={embedderChange ? "danger" : undefined}
              disabled={busy || invalid || diff.list.length === 0 || reasonMissing}
            >
              {submitLabel}
            </button>
            <button
              type="button"
              disabled={busy || diff.list.length === 0}
              onClick={() => {
                setDraft(draftFrom(config.editable));
                setReason("");
                setConfirming(false);
              }}
            >
              Discard changes
            </button>
          </div>
        </fieldset>
      </form>

      <h2>Fixed for this deployment</h2>
      <ReadonlyFacts config={config} />
    </>
  );
}

/**
 * Live state of the job at the head of the queue. Polled rather than pushed:
 * the backend has no event channel, and a re-embed the operator cannot watch is
 * indistinguishable from one that died.
 */
function JobWatch({
  jobId,
  remaining,
  onSettled,
}: {
  jobId: string;
  remaining: number;
  onSettled: (job: Job) => void;
}) {
  const job = useResource<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, 1_000);
  const reported = useRef(false);

  useEffect(() => {
    if (job.state.phase !== "ready" || reported.current) {
      return;
    }
    const data = job.state.data;
    if (TERMINAL.includes(data.status)) {
      reported.current = true;
      onSettled(data);
    }
  }, [job.state, onSettled]);

  if (job.state.phase === "error") {
    return (
      <div className="notice error banner" role="status">
        Could not read job <span className="mono">{truncateHash(jobId)}</span>:{" "}
        <span className="mono">{job.state.message}</span>
      </div>
    );
  }
  if (job.state.phase === "loading") {
    return (
      <p className="notice banner" aria-live="polite">
        Reading job <span className="mono">{truncateHash(jobId)}</span>&hellip;
      </p>
    );
  }

  const data = job.state.data;
  const steps = Object.entries(data.progress)
    .filter(([, value]) => typeof value === "string" || typeof value === "number")
    .map(([key, value]) => `${key.replaceAll("_", " ")}: ${String(value)}`);

  return (
    <div className="notice attention banner" role="status" aria-live="polite">
      <p className="compact">
        <strong>
          {data.kind} job {data.status}
        </strong>{" "}
        <span className="mono">{truncateHash(data.id)}</span>
        {remaining > 0 && (
          <span className="muted"> {"\u00b7"} {String(remaining)} more queued behind it</span>
        )}
      </p>
      <p className="job-progress">
        {steps.length > 0 ? steps.join(" \u00b7 ") : "No progress reported yet."}
      </p>
      <p className="job-progress">
        Model changes are blocked until this finishes, and the gallery is not matchable while it
        runs.
      </p>
    </div>
  );
}

/** Whether this model can be picked here, and if not, the reason in words. */
function pickability(model: ConfigModel, allowNoncommercial: boolean): string | null {
  if (model.active) {
    return null;
  }
  if (!model.present) {
    return "Weights are not provisioned, so the backend cannot load this model. Fetch them with tools/fetch_models.py.";
  }
  if (!model.commercial_use && !allowNoncommercial) {
    return "Non-commercial license, and ALLOW_NONCOMMERCIAL_MODELS is false, so the backend refuses to load it (invariant 9). That flag is set in .env and does need a restart.";
  }
  return null;
}

function ModelPicker({
  kind,
  legend,
  note,
  models,
  allowNoncommercial,
  value,
  locked,
  onPick,
}: {
  kind: ModelKind;
  legend: string;
  note: string;
  models: ConfigModel[];
  allowNoncommercial: boolean;
  value: string;
  locked: boolean;
  onPick: (id: string) => void;
}) {
  const groupId = useId();
  const options = models.filter((model) => model.kind === kind);

  return (
    <fieldset className="config-group">
      <legend>{legend}</legend>
      <p className="field-help">{note}</p>
      {options.length === 0 ? (
        <p className="field-error">
          The backend lists no {kind} at all, so there is nothing to choose between.
        </p>
      ) : (
        <div className="model-choices">
          {options.map((model, index) => {
            const blocked = pickability(model, allowNoncommercial);
            const describedBy = `${groupId}-${String(index)}-desc`;
            const checked = value === model.id;
            return (
              <label
                key={model.id}
                className={`model-choice${checked ? " checked" : ""}${blocked === null ? "" : " blocked"}`}
              >
                <input
                  type="radio"
                  name={`${groupId}-${kind}`}
                  value={model.id}
                  checked={checked}
                  disabled={locked || blocked !== null}
                  aria-describedby={describedBy}
                  onChange={() => onPick(model.id)}
                />
                <span>
                  <span className="model-choice-facts">
                    <strong>{model.name}</strong>
                    <span className="mono">{model.id}</span>
                    <span className="pill">v{model.version}</span>
                    <span className="pill">license: {model.license ?? "not provisioned"}</span>
                    <span className="pill">
                      dim: {model.dim === null ? "n/a" : String(model.dim)}
                    </span>
                    {model.active && <span className="badge-operator">active</span>}
                    {!model.present && (
                      <span className="status-chip chip-attention">no weights</span>
                    )}
                  </span>
                  <span className="field-help" id={describedBy}>
                    {blocked ??
                      (model.commercial_use
                        ? "Commercial use permitted by its license."
                        : "Non-commercial license: selectable only because ALLOW_NONCOMMERCIAL_MODELS is true, and every export has to state it (C7).")}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
      )}
    </fieldset>
  );
}

function NumberField({
  field,
  value,
  error,
  onChange,
}: {
  field: NumericField;
  value: string;
  error: string | null;
  onChange: (value: string) => void;
}) {
  const inputId = useId();
  const helpId = `${inputId}-help`;
  const errorId = `${inputId}-error`;

  return (
    <div className="field">
      <label htmlFor={inputId}>
        {field.label}
        {field.unit === null ? "" : ` (${field.unit})`}
      </label>
      <input
        id={inputId}
        type="number"
        inputMode="decimal"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        aria-invalid={error !== null}
        aria-describedby={error === null ? helpId : `${errorId} ${helpId}`}
        onChange={(event) => onChange(event.currentTarget.value)}
      />
      {error !== null && (
        <p className="field-error" id={errorId}>
          {error}
        </p>
      )}
      <p className="field-help" id={helpId}>
        {field.help} Allowed {String(field.min)} to {String(field.max)}
        {field.integer ? ", whole numbers only." : "."}
      </p>
    </div>
  );
}

function ScoreModeField({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  const inputId = useId();
  const helpId = `${inputId}-help`;

  return (
    <div className="field">
      <label htmlFor={inputId}>Person score mode</label>
      <select
        id={inputId}
        value={value}
        aria-describedby={helpId}
        onChange={(event) => onChange(event.currentTarget.value)}
      >
        {PERSON_SCORE_MODES.map((mode) => (
          <option key={mode.value} value={mode.value}>
            {mode.label}
          </option>
        ))}
      </select>
      <p className="field-help" id={helpId}>
        How a person&apos;s score is formed from their templates (spec 6.4).{" "}
        <span className="mono">mean_top3</span> only means anything for persons with five or more
        templates. Calibration scores <span className="mono">max</span>, so a threshold set
        calibrated under one mode does not describe the other.
      </p>
    </div>
  );
}

/** The consequences of an embedder switch, in the order they will happen. */
function EmbedderConfirm({
  from,
  to,
  detectorChange,
}: {
  from: string;
  to: string;
  detectorChange: boolean;
}) {
  return (
    <div className="notice error confirm-panel" role="alert">
      <strong>
        Switching the embedder from <span className="mono">{from}</span> to{" "}
        <span className="mono">{to}</span> invalidates the gallery.
      </strong>
      <ul>
        <li>
          Every stored template and every track has to be re-embedded under{" "}
          <span className="mono">{to}</span>. Confirming enqueues that job now, and a re-match after
          it. The re-embed reads the crops already on disk, so no media is decoded again, but both
          have to finish before the gallery means anything.
        </li>
        <li>
          Nothing can be matched across two models: embeddings from different models are never
          compared. Until the re-embed completes the gallery is part{" "}
          <span className="mono">{from}</span> and part <span className="mono">{to}</span>, and
          matching against it is off.
        </li>
        <li>
          Auto-accept stays off. The active threshold set was calibrated against{" "}
          <span className="mono">{from}</span> and its numbers do not carry over, so every match
          stays a candidate until you run a calibration for <span className="mono">{to}</span> and
          activate its threshold set.
        </li>
        <li>
          Nothing is deleted. Detections, crops, persons and operator decisions are untouched; only
          the vectors are recomputed. The switch itself takes effect immediately, with no restart.
        </li>
        {detectorChange && (
          <li>
            The detector change in this submission does <strong>not</strong> re-run detection on
            stored media and enqueues nothing. Existing boxes stay as they are; only media processed
            after this uses the new detector.
          </li>
        )}
      </ul>
      <p className="compact">
        The reason above is written to the audit log with this change. Press the button again to
        send it, or discard the change to back out.
      </p>
    </div>
  );
}

function ReadonlyFacts({ config }: { config: Config }) {
  const fixed = config.readonly;
  return (
    <div className="panel">
      <p className="field-help">
        Read-only here, unlike everything above. These come from the environment the backend started
        with: change them in <span className="mono">.env</span> and restart, and remember that a
        restart re-verifies every weight file against <span className="mono">models.lock</span>.
      </p>
      <dl className="facts">
        <dt>Operator</dt>
        <dd>{fixed.operator_name}</dd>

        <dt>Execution provider</dt>
        <dd className="mono">{fixed.execution_provider}</dd>

        <dt>Non-commercial models</dt>
        <dd className={fixed.allow_noncommercial_models ? "status-bad" : ""}>
          {fixed.allow_noncommercial_models
            ? "allowed \u2014 exports must state the active model license (C7)"
            : "blocked \u2014 non-commercial models cannot be selected above"}
        </dd>

        <dt>Database</dt>
        <dd className="mono">{fixed.db_path}</dd>

        <dt>Models directory</dt>
        <dd className="mono">{fixed.models_dir}</dd>

        <dt>Maximum upload</dt>
        <dd>{formatBytes(fixed.max_upload_bytes)}</dd>

        <dt>FPIR target</dt>
        <dd className="mono">{fixed.fpir_target}</dd>

        <dt>Crops per track mean</dt>
        <dd>
          best {fixed.embed_k} <span className="muted">(embed_k)</span>
        </dd>

        <dt>Re-match block size</dt>
        <dd>
          {fixed.rematch_block_size} <span className="muted">(rows per block)</span>
        </dd>
      </dl>
    </div>
  );
}
