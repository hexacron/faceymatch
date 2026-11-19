import { useCallback, useEffect, useState } from "react";

import { errorMessage, postJson } from "../api/client";
import type { CaseList, Cluster, ClusterAssignResult, ClusterList, Job, PersonList } from "../api/types";
import { Loaded } from "../components/Loading";
import { hrefFor } from "../lib/router";
import { useResource } from "../lib/useResource";

function ClusterCard({ cluster, people, onAssigned }: {
  cluster: Cluster; people: PersonList; onAssigned: () => void;
}) {
  const [excluded, setExcluded] = useState<Set<string>>(() => new Set());
  const [personId, setPersonId] = useState("");
  const [name, setName] = useState("");
  const [reason, setReason] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function assign(): Promise<void> {
    setPending(true);
    setMessage(null);
    try {
      const result = await postJson<ClusterAssignResult>(`/api/clusters/${encodeURIComponent(cluster.id)}/assign`, {
        person_id: personId || null,
        new_name: personId ? null : name.trim() || null,
        exclude_track_ids: [...excluded],
        reason: reason.trim(),
      });
      setMessage(`Assigned ${result.assigned} faces; queued re-match ${result.rematch_job_id}.`);
      onAssigned();
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setPending(false);
    }
  }

  return <section className="panel cluster-panel">
    <h2>Unknown group · {cluster.size} faces</h2>
    <p className="muted">Review each stored face before assigning a person. Uncheck a face to exclude it.</p>
    <div className="review-grid">
      {cluster.tracks.map((track) => <label className="review-card cluster-face" key={track.track_id}>
        <span className="cluster-check"><input type="checkbox" checked={!excluded.has(track.track_id)}
          onChange={(event) => setExcluded((previous) => {
            const next = new Set(previous);
            if (event.currentTarget.checked) next.delete(track.track_id);
            else next.add(track.track_id);
            return next;
          })} /> Include face</span>
        {track.crop_sha256 ? <img src={`/api/crops/${encodeURIComponent(track.crop_sha256)}`} alt="Stored face crop" />
          : <span className="crop-missing">No crop</span>}
        <span className="review-card-body"><a href={hrefFor({ view: "viewer", mediaId: track.media_id })}>Open source media</a></span>
      </label>)}
    </div>
    <div className="field-grid">
      <label>Existing person
        <select value={personId} onChange={(event) => setPersonId(event.currentTarget.value)}>
          <option value="">Create a new person</option>
          {people.items.filter((person) => !person.do_not_enroll).map((person) =>
            <option key={person.id} value={person.id}>{person.display_name}</option>)}
        </select>
      </label>
      {!personId && <label>New person's name<input value={name} onChange={(event) => setName(event.currentTarget.value)} /></label>}
      <label>Reason for assignment<input value={reason} onChange={(event) => setReason(event.currentTarget.value)} /></label>
    </div>
    <p><button type="button" disabled={pending || excluded.size === cluster.tracks.length || !reason.trim() || (!personId && !name.trim())}
      onClick={() => void assign()}>Assign {cluster.tracks.length - excluded.size} faces and enroll</button></p>
    {message && <p role="status" className="notice">{message}</p>}
  </section>;
}

function ClusterJob({ id, onFinished }: { id: string; onFinished: (status: Job["status"]) => void }) {
  const job = useResource<Job>(`/api/jobs/${encodeURIComponent(id)}`, 1_000,
    { stopWhen: (value) => value.status === "done" || value.status === "failed" });
  useEffect(() => {
    if (job.state.phase === "ready" && (job.state.data.status === "done" || job.state.data.status === "failed")) {
      onFinished(job.state.data.status);
    }
  }, [job.state, onFinished]);
  return <Loaded state={job.state} label="clustering job">{(value) => <p className="notice" role="status">
    Clustering: {value.status}{value.error ? ` — ${value.error}` : ""}
    {value.status === "done" ? ` · ${String(value.progress.clusters ?? 0)} groups; ${String(value.progress.noise ?? 0)} noise faces` : ""}
  </p>}</Loaded>;
}

export default function ClustersView() {
  const cases = useResource<CaseList>("/api/cases");
  const people = useResource<PersonList>("/api/persons");
  const [caseId, setCaseId] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [running, setRunning] = useState(false);
  const clusters = useResource<ClusterList>(`/api/clusters?scope=${encodeURIComponent(`case:${caseId}`)}`);

  async function start(): Promise<void> {
    setPending(true);
    setRunning(true);
    setFailure(null);
    setJobId(null);
    try {
      const job = await postJson<Job>("/api/jobs/cluster", { scope: "case", scope_id: caseId });
      setJobId(job.id);
    } catch (error) {
      setFailure(errorMessage(error));
      setRunning(false);
    } finally {
      setPending(false);
    }
  }

  const onFinished = useCallback((status: Job["status"]) => {
    setRunning(false);
    if (status === "done") clusters.reload();
  }, [clusters.reload]);

  return <>
    <div className="view-heading"><div><h1>Unknown face clusters</h1>
      <p className="tagline">Group stored unknown faces; only your explicit assignment adds gallery templates.</p></div>
      <button type="button" onClick={clusters.reload}>Refresh</button></div>
    <div className="toolbar panel"><label>Case
      <select value={caseId} onChange={(event) => { setCaseId(event.currentTarget.value); setJobId(null); setFailure(null); }}>
        <option value="">Choose a case</option>
        {cases.state.phase === "ready" && cases.state.data.items.map((item) =>
          <option key={item.id} value={item.id}>{item.name}</option>)}
      </select></label>
      <button type="button" disabled={!caseId || pending || running} onClick={() => void start()}>Cluster unknown faces</button>
    </div>
    {cases.state.phase === "error" && <p className="notice error">{cases.state.message}</p>}
    {failure && <p className="notice error" role="alert">{failure}</p>}
    {jobId && <ClusterJob id={jobId} onFinished={onFinished} />}
    {caseId && <Loaded state={clusters.state} label="clusters">{(list) =>
      list.items.length === 0 ? <p className="notice">No groups found. A cluster needs at least three quality-passing unknown faces.</p> :
      <Loaded state={people.state} label="persons">{(persons) => <div className="cluster-list">{list.items.map((cluster) =>
        <ClusterCard key={cluster.id} cluster={cluster} people={persons} onAssigned={() => { clusters.reload(); people.reload(); }} />)}</div>}</Loaded>
    }</Loaded>}
  </>;
}
