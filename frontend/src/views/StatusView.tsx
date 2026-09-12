import { useCallback, useState } from "react";

import { errorMessage, postJson } from "../api/client";
import type {
  AuditPage,
  Health,
  ModelsInfo,
  ThresholdSet,
  ThresholdSetList,
} from "../api/types";
import { Loaded } from "../components/Loading";
import { formatScore, formatTs, truncateHash } from "../lib/display";
import { useResource } from "../lib/useResource";

const AUDIT_PAGE_SIZE = 20;

/**
 * Backend, license and audit status.
 *
 * Two compliance surfaces live here and must stay visible:
 * C7, the license of the active model files, which every export has to state;
 * and C5, the calibrated-threshold-set gate — with no calibrated active set,
 * auto-accept is off and every match stays a candidate.
 */
export default function StatusView() {
  const health = useResource<Health>("/api/healthz");
  const models = useResource<ModelsInfo>("/api/models");
  const thresholdSets = useResource<ThresholdSetList>("/api/threshold_sets");
  const audit = useResource<AuditPage>(`/api/audit?from_seq=0&limit=${String(AUDIT_PAGE_SIZE)}`);

  const reloadAll = useCallback(() => {
    health.reload();
    models.reload();
    thresholdSets.reload();
    audit.reload();
  }, [health, models, thresholdSets, audit]);

  return (
    <>
      <h2>Backend</h2>
      <Loaded state={health.state} label="backend health">
        {(data) => <HealthPanel health={data} onReload={reloadAll} />}
      </Loaded>

      <h2>Model licenses (C7)</h2>
      <Loaded state={models.state} label="models">
        {(data) => <ModelTable models={data} />}
      </Loaded>

      <h2>Threshold sets</h2>
      <Loaded state={thresholdSets.state} label="threshold sets">
        {(data) => <ThresholdSetTable sets={data.items} onChanged={reloadAll} />}
      </Loaded>

      <h2>Audit log</h2>
      <Loaded state={audit.state} label="the audit log">
        {(data) => <AuditTable audit={data} />}
      </Loaded>
    </>
  );
}

function HealthPanel({ health, onReload }: { health: Health; onReload: () => void }) {
  const { embedder, detector, threshold_set: thresholdSet, auto_accept: autoAccept } = health;

  return (
    <div className="panel">
      {/* The backend decides the C5 gate; re-deriving it from the threshold set
          alone would miss the other states that close it, such as a re-embed in
          flight after a model change. */}
      {!autoAccept.allowed && (
        <div className="notice error banner">
          <strong>Auto-accept is off.</strong>{" "}
          {autoAccept.reason ?? "The backend did not give a reason"}, so every match stays a
          candidate and nothing is accepted automatically (C5). Run a calibration, then activate its
          threshold set below.
        </div>
      )}
      {autoAccept.allowed && autoAccept.warning !== null && (
        <div className="notice attention banner">
          <strong>Auto-accept is on.</strong> {autoAccept.warning}
        </div>
      )}
      <dl className="facts">
        <dt>Status</dt>
        <dd className={health.status === "ok" ? "status-ok" : "status-bad"}>
          {health.status} <span className="pill">v{health.version}</span>
        </dd>

        <dt>Database</dt>
        <dd className="mono">{health.db_path}</dd>

        <dt>Migration</dt>
        <dd>applied version {health.migration_version}</dd>

        <dt>Embedder</dt>
        <dd>
          <span className="mono">{embedder.model_id}</span>{" "}
          <span className="pill">license: {embedder.license ?? "not provisioned"}</span>{" "}
          <span className="pill">dim: {embedder.dim === null ? "unknown" : embedder.dim}</span>{" "}
          <span className={embedder.present ? "status-ok" : "status-bad"}>
            {embedder.present ? "weights present" : "weights missing"}
          </span>
        </dd>

        <dt>Detector</dt>
        <dd>
          <span className="mono">{detector.model_id}</span>{" "}
          <span className="pill">license: {detector.license ?? "not provisioned"}</span>{" "}
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
            : "blocked right now"}
        </dd>

        <dt>Threshold set</dt>
        <dd>
          {thresholdSet === null ? (
            <span className="status-bad">none active</span>
          ) : (
            <>
              <span className="mono">{thresholdSet.id}</span>{" "}
              <span className={thresholdSet.calibrated ? "status-ok" : "status-bad"}>
                {thresholdSet.calibrated ? "calibrated" : "uncalibrated"}
              </span>{" "}
              <span className="pill">gallery size: {thresholdSet.gallery_size ?? "unrecorded"}</span>{" "}
              <span className="pill">EP: {thresholdSet.execution_provider ?? "unrecorded"}</span>
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

function ModelTable({ models }: { models: ModelsInfo }) {
  return (
    <table>
      <caption>
        Weight files verified against <span className="mono">models.lock</span> at startup.
        Execution provider <span className="mono">{models.execution_provider}</span>. Non-commercial
        models {models.allow_noncommercial_models ? "allowed" : "blocked"}.
      </caption>
      <thead>
        <tr>
          <th>model_id</th>
          <th>kind</th>
          <th>license</th>
          <th className="num">dim</th>
          <th>sha256</th>
          <th>state</th>
        </tr>
      </thead>
      <tbody>
        {models.items.map((model) => (
          <tr key={model.id}>
            <td className="mono">{model.id}</td>
            <td>{model.kind}</td>
            <td className={(model.license ?? "").toLowerCase().includes("non-commercial") ? "status-bad" : ""}>
              {model.license ?? "not provisioned"}
            </td>
            <td className="num">{model.dim ?? "\u2014"}</td>
            <td className="mono" title={model.sha256}>
              {truncateHash(model.sha256)}
            </td>
            <td>
              {model.active && <span className="badge-operator">active</span>}{" "}
              {!model.present && <span className="status-bad">missing</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ThresholdSetTable({
  sets,
  onChanged,
}: {
  sets: ThresholdSet[];
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activate = async (id: string): Promise<void> => {
    setBusy(id);
    setError(null);
    try {
      await postJson<ThresholdSet>(`/api/threshold_sets/${encodeURIComponent(id)}/activate`, {});
      onChanged();
    } catch (activateError) {
      setError(errorMessage(activateError));
    } finally {
      setBusy(null);
    }
  };

  if (sets.length === 0) {
    return (
      <p className="notice">
        No threshold sets yet. Run a calibration (section 10) to produce one; until one is active and
        calibrated, auto-accept stays off.
      </p>
    );
  }

  return (
    <>
      {error !== null && <div className="notice error">{error}</div>}
      <table>
        <caption>Activation is an explicit audited action (section 10).</caption>
        <thead>
          <tr>
            <th>id</th>
            <th>model_id</th>
            <th className="num">t_strong</th>
            <th className="num">t_possible</th>
            <th className="num">margin</th>
            <th className="num">gallery</th>
            <th>EP</th>
            <th>state</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {sets.map((set) => (
            <tr key={set.id}>
              <td className="mono" title={set.id}>
                {truncateHash(set.id)}
              </td>
              <td className="mono">{set.model_id}</td>
              <td className="num mono">{formatScore(set.t_strong)}</td>
              <td className="num mono">{formatScore(set.t_possible)}</td>
              <td className="num mono">{formatScore(set.margin)}</td>
              <td className="num">{set.gallery_size ?? "\u2014"}</td>
              <td className="mono">{set.execution_provider ?? "\u2014"}</td>
              <td>
                <span className={set.calibrated ? "status-ok" : "status-bad"}>
                  {set.calibrated ? "calibrated" : "uncalibrated"}
                </span>{" "}
                {set.active && <span className="badge-operator">active</span>}
              </td>
              <td>
                <button
                  type="button"
                  disabled={set.active || busy !== null}
                  onClick={() => void activate(set.id)}
                >
                  {busy === set.id ? "Activating\u2026" : "Activate"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function AuditTable({ audit }: { audit: AuditPage }) {
  if (audit.entries.length === 0) {
    return <p className="notice">The audit chain is empty. No writes yet.</p>;
  }

  return (
    <table>
      <caption>
        Oldest {audit.entries.length} of {audit.head_seq} entries, hash-chained and append-only.
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
