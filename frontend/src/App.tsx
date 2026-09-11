import { useCallback, useEffect, useState } from "react";

/**
 * Wire types for the M0 backend. These mirror the JSON emitted by
 * `GET /api/healthz` and `GET /api/audit`. Keep them in step with
 * `backend/app/api` — later milestones extend both ends together.
 */

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

export type Health = {
  status: string;
  version: string;
  db_path: string;
  migration_version: number;
  embedder: ModelStatus;
  detector: ModelStatus;
  execution_provider: string;
  allow_noncommercial_models: boolean;
  /** Null until a threshold set is activated (M1). No auto-accept without one (C5). */
  threshold_set: ThresholdSetStatus | null;
  audit_head_seq: number;
  /** Null on an empty chain. */
  audit_head_hash: string | null;
};

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

const AUDIT_PAGE_SIZE = 20;

/**
 * The backend serves this bundle at `/` and the API under `/api`, so every
 * request stays relative. Never hardcode an origin: in dev Vite proxies `/api`
 * to 127.0.0.1:8000, and in production there is no second origin to reach.
 */
async function getJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`${path} responded ${response.status} ${response.statusText}`);
  }
  // `Response.json()` is typed `Promise<any>` by the DOM lib; we assert the
  // documented wire type here so the rest of the app stays fully typed.
  return (await response.json()) as T;
}

function truncateHash(hash: string): string {
  return hash.length > 12 ? `${hash.slice(0, 12)}\u2026` : hash;
}

/** Trim microseconds to milliseconds; audit timestamps stay in UTC. */
function formatTs(ts: string): string {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?/.exec(ts);
  if (match === null) {
    return ts;
  }
  const [, date, time, fraction] = match;
  const millis = (fraction ?? "").padEnd(3, "0").slice(0, 3);
  return `${date} ${time}.${millis}Z`;
}

type LoadState =
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "ready"; health: Health; audit: AuditPage };

export default function App() {
  const [state, setState] = useState<LoadState>({ phase: "loading" });
  const [reloadToken, setReloadToken] = useState(0);

  const reload = useCallback(() => {
    setState({ phase: "loading" });
    setReloadToken((token) => token + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    void (async () => {
      try {
        const [health, audit] = await Promise.all([
          getJson<Health>("/api/healthz", controller.signal),
          getJson<AuditPage>(
            `/api/audit?from_seq=0&limit=${String(AUDIT_PAGE_SIZE)}`,
            controller.signal,
          ),
        ]);
        setState({ phase: "ready", health, audit });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setState({
          phase: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      }
    })();

    return () => {
      controller.abort();
    };
  }, [reloadToken]);

  return (
    <main className="shell">
      <header>
        <h1>faceymatch</h1>
        <p className="tagline">Local face match system &mdash; operator console</p>
      </header>

      {state.phase === "loading" && (
        <>
          <h2>Backend</h2>
          <p className="notice">Contacting the backend&hellip;</p>
        </>
      )}

      {state.phase === "error" && (
        <>
          <h2>Backend</h2>
          <div className="notice error">
            <p>
              Could not reach the backend: <span className="mono">{state.message}</span>
            </p>
            <p>
              Start it with <span className="mono">uv run uvicorn app.main:app</span> from{" "}
              <span className="mono">backend/</span>, then retry.
            </p>
            <button type="button" onClick={reload}>
              Retry
            </button>
          </div>
        </>
      )}

      {state.phase === "ready" && (
        <>
          <h2>Backend</h2>
          <BackendPanel health={state.health} onReload={reload} />
          <h2>Audit log</h2>
          <AuditTable audit={state.audit} />
        </>
      )}
    </main>
  );
}

function BackendPanel({
  health,
  onReload,
}: {
  health: Health;
  onReload: () => void;
}) {
  const healthy = health.status === "ok";
  const { embedder, detector, threshold_set: thresholdSet } = health;

  return (
    <div className="panel">
      <dl className="facts">
        <dt>Status</dt>
        <dd className={healthy ? "status-ok" : "status-bad"}>
          {health.status} <span className="pill">v{health.version}</span>
        </dd>

        <dt>Database</dt>
        <dd className="mono">{health.db_path}</dd>

        <dt>Migration</dt>
        <dd>applied version {health.migration_version}</dd>

        <dt>Embedder</dt>
        <dd>
          <span className="mono">{embedder.model_id}</span>{" "}
          <span className="pill">
            license: {embedder.license ?? "not provisioned"}
          </span>{" "}
          <span className="pill">
            dim: {embedder.dim === null ? "unknown" : embedder.dim}
          </span>{" "}
          <span className={embedder.present ? "status-ok" : "status-bad"}>
            {embedder.present ? "weights present" : "weights missing"}
          </span>
        </dd>

        <dt>Detector</dt>
        <dd>
          <span className="mono">{detector.model_id}</span>{" "}
          <span className="pill">
            license: {detector.license ?? "not provisioned"}
          </span>{" "}
          <span className={detector.present ? "status-ok" : "status-bad"}>
            {detector.present ? "weights present" : "weights missing"}
          </span>
        </dd>

        <dt>Execution provider</dt>
        <dd className="mono">{health.execution_provider}</dd>

        <dt>Non-commercial models</dt>
        <dd className={health.allow_noncommercial_models ? "status-bad" : ""}>
          {health.allow_noncommercial_models
            ? "ALLOWED \u2014 exports must state the active model license (C7)"
            : "blocked"}
        </dd>

        <dt>Threshold set</dt>
        <dd>
          {thresholdSet === null ? (
            <span className="status-bad">
              none active &mdash; auto-accept is off, all matches stay candidates (C5)
            </span>
          ) : (
            <>
              <span className="mono">{thresholdSet.id}</span>{" "}
              <span className={thresholdSet.calibrated ? "status-ok" : "status-bad"}>
                {thresholdSet.calibrated ? "calibrated" : "uncalibrated"}
              </span>{" "}
              <span className="pill">
                gallery size: {thresholdSet.gallery_size ?? "unrecorded"}
              </span>
            </>
          )}
        </dd>

        <dt>Audit head</dt>
        <dd>
          seq {health.audit_head_seq}{" "}
          <span className="mono">
            {health.audit_head_hash === null
              ? "(empty chain)"
              : truncateHash(health.audit_head_hash)}
          </span>
        </dd>
      </dl>
      <p>
        <button type="button" onClick={onReload}>
          Refresh
        </button>
      </p>
    </div>
  );
}

function AuditTable({ audit }: { audit: AuditPage }) {
  if (audit.entries.length === 0) {
    return <p className="notice">The audit chain is empty. No writes yet.</p>;
  }

  return (
    <table>
      <caption>
        Oldest {audit.entries.length} of {audit.head_seq} entries, hash-chained and
        append-only.
      </caption>
      <thead>
        <tr>
          <th className="num">seq</th>
          <th>ts</th>
          <th>actor</th>
          <th>action</th>
          <th>object_type</th>
          <th>hash</th>
        </tr>
      </thead>
      <tbody>
        {audit.entries.map((entry) => (
          <tr key={entry.seq}>
            <td className="num">{entry.seq}</td>
            <td className="mono">{formatTs(entry.ts)}</td>
            <td>{entry.actor}</td>
            <td className="mono">{entry.action}</td>
            <td>{entry.object_type}</td>
            <td className="mono" title={entry.hash}>
              {truncateHash(entry.hash)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
