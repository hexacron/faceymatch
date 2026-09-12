import { useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent } from "react";

import type { CaseList, Health, Media, MediaList } from "../api/types";
import { CaseBasis } from "../components/CaseBasis";
import { Loaded } from "../components/Loading";
import { formatTs, truncateHash } from "../lib/display";
import {
  captureFromScreen,
  collectImages,
  ingestFiles,
  noCaseRefusal,
  reconcileSelectedCase,
  setSelectedCase,
  syncCaptureCapability,
  useCaptureCapability,
  useSelectedCase,
} from "../lib/ingest";
import { hrefFor } from "../lib/router";
import { useResource } from "../lib/useResource";

function progressText(media: Media): string {
  const job = media.job;
  if (job === null) {
    return media.status === "done" ? `${String(media.detection_count)} detections` : "No active job";
  }
  const parts = Object.entries(job.progress)
    .filter(([, value]) => typeof value === "string" || typeof value === "number")
    .slice(0, 3)
    .map(([key, value]) => `${key.replaceAll("_", " ")}: ${String(value)}`);
  if (job.error !== null) {
    parts.push(job.error);
  }
  return parts.length > 0 ? parts.join(" · ") : job.status;
}

export default function MediaLibraryView() {
  const cases = useResource<CaseList>("/api/cases");
  const health = useResource<Health>("/api/healthz");
  const caseId = useSelectedCase();
  const mediaPath = useMemo(
    () => (caseId === "" ? "/api/media" : `/api/media?case_id=${encodeURIComponent(caseId)}`),
    [caseId],
  );
  const media = useResource<MediaList>(mediaPath, 1_000);
  const fileRef = useRef<HTMLInputElement>(null);
  const [sourceUrl, setSourceUrl] = useState("");
  const [busy, setBusy] = useState<"upload" | "capture" | null>(null);
  const [dragging, setDragging] = useState(false);
  // dragenter/dragleave fire for every child element, so count the depth
  // instead of clearing the highlight the first time the pointer crosses one.
  const dragDepth = useRef(0);
  const capture = useCaptureCapability();

  // No default: a stored selection is kept only while the backend still lists
  // it, and nothing picks a case on the operator's behalf.
  useEffect(() => {
    if (cases.state.phase === "ready") {
      reconcileSelectedCase(cases.state.data.items);
    }
  }, [cases.state]);

  useEffect(() => {
    if (health.state.phase === "ready") {
      syncCaptureCapability(health.state.data);
    }
  }, [health.state]);

  async function uploadPicked(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const picked = fileRef.current?.files;
    if (caseId === "" || picked === null || picked === undefined || picked.length === 0) {
      return;
    }
    setBusy("upload");
    try {
      await ingestFiles(caseId, { files: Array.from(picked), rejected: [] }, sourceUrl);
      if (fileRef.current !== null) {
        fileRef.current.value = "";
      }
      media.reload();
    } finally {
      setBusy(null);
    }
  }

  async function runCapture(): Promise<void> {
    if (caseId === "") {
      return;
    }
    setBusy("capture");
    try {
      await captureFromScreen(caseId, "region");
      media.reload();
    } finally {
      setBusy(null);
    }
  }

  function onDragEnter(event: DragEvent<HTMLDivElement>): void {
    if (!event.dataTransfer.types.includes("Files")) {
      return;
    }
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  }

  function onDragOver(event: DragEvent<HTMLDivElement>): void {
    if (!event.dataTransfer.types.includes("Files")) {
      return;
    }
    // Without this the browser navigates away to the dropped file.
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }

  function onDragLeave(): void {
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) {
      setDragging(false);
    }
  }

  async function onDrop(event: DragEvent<HTMLDivElement>): Promise<void> {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const payload = collectImages(event.dataTransfer, null);
    if (payload.files.length === 0 && payload.rejected.length === 0) {
      return;
    }
    setBusy("upload");
    try {
      await ingestFiles(caseId, payload, null);
      media.reload();
    } finally {
      setBusy(null);
    }
  }

  const noCase = caseId === "";
  // The destination case itself: its name goes next to every control that can
  // write, and its authorization basis is shown beside the picker, because
  // that claim is what the evidence about to be added rests on.
  const selected =
    cases.state.phase === "ready"
      ? (cases.state.data.items.find((item) => item.id === caseId) ?? null)
      : null;
  const caseName = selected?.name ?? null;
  const captureBlocked = capture !== null && !capture.supported;
  // A missing case blocks first: it is the operator's decision to make, and
  // capture support is irrelevant until there is somewhere to file the grab.
  const captureReason = noCase
    ? noCaseRefusal("capturing the screen")
    : captureBlocked
      ? (capture?.reason ?? "unavailable on this machine")
      : null;

  return (
    <div
      className="ingest-surface"
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={(event) => void onDrop(event)}
    >
      <div className="view-heading">
        <div>
          <h1>Media library</h1>
          <p className="tagline">Paste, drop, capture, or pick a file; every path lands on the faces.</p>
        </div>
        <button type="button" onClick={media.reload}>Refresh</button>
      </div>

      <section className="panel upload-panel" aria-labelledby="upload-heading">
        <h2 id="upload-heading">Add media</h2>
        <Loaded state={cases.state} label="cases">
          {(caseList) =>
            caseList.items.length === 0 ? (
              <p className="notice">No cases exist. Create a case through the API before uploading evidence.</p>
            ) : (
              <>
                <form className="form-grid" onSubmit={(event) => void uploadPicked(event)}>
                  <label>
                    Case
                    <select
                      value={caseId}
                      onChange={(event) => setSelectedCase(event.currentTarget.value)}
                      required
                    >
                      <option value="">Choose a case…</option>
                      {caseList.items.map((item) => (
                        <option key={item.id} value={item.id}>{item.name}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Image
                    <input
                      ref={fileRef}
                      name="file"
                      type="file"
                      accept="image/jpeg,image/png,image/webp,image/heic,image/heif"
                      multiple
                      required
                    />
                  </label>
                  <label>
                    Source URL <span className="muted">(optional)</span>
                    <input
                      name="source_url"
                      type="url"
                      placeholder="https://source.example/item"
                      value={sourceUrl}
                      onChange={(event) => setSourceUrl(event.currentTarget.value)}
                    />
                  </label>
                  <button className="primary" type="submit" disabled={busy !== null || noCase}>
                    {busy === "upload" ? "Uploading…" : "Upload and process"}
                  </button>
                </form>

                {selected !== null && (
                  <CaseBasis key={selected.id} record={selected} onAmended={cases.reload} />
                )}

                <div className={dragging ? "dropzone drag-over" : "dropzone"}>
                  {noCase ? (
                    <p className="compact">
                      Choose a case above, then drop image files here or press{" "}
                      <kbd className="kbd">⌘V</kbd> to ingest one from the clipboard.
                    </p>
                  ) : (
                    <p className="compact">
                      Drop image files here, or press <kbd className="kbd">⌘V</kbd> anywhere to
                      ingest an image into <strong>{caseName ?? "the selected case"}</strong>.
                    </p>
                  )}
                  <p className="compact muted">
                    Copy a photo in Preview, Finder, or a browser, then paste — no saved file needed.
                  </p>
                </div>

                <div className="ingest-actions">
                  <button
                    type="button"
                    onClick={() => void runCapture()}
                    disabled={busy !== null || noCase || captureBlocked}
                    {...(captureReason === null ? {} : { title: captureReason })}
                    aria-describedby="capture-destination capture-reason"
                  >
                    {busy === "capture" ? "Waiting for the screen grab…" : "Capture from screen"}
                  </button>
                  <span className="case-destination" id="capture-destination">
                    {noCase ? (
                      "No case selected"
                    ) : (
                      <>
                        Files to <strong>{caseName ?? "the selected case"}</strong>
                      </>
                    )}
                  </span>
                  <span className="muted" id="capture-reason">
                    {captureReason ??
                      "macOS draws a crosshair over every app: drag a box around the face on screen, or press Esc to cancel."}
                  </span>
                </div>
              </>
            )
          }
        </Loaded>
      </section>

      <section aria-labelledby="library-heading">
        <h2 id="library-heading">Files</h2>
        <Loaded state={media.state} label="media">
          {(list) =>
            list.items.length === 0 ? (
              <p className="notice">This case has no media yet.</p>
            ) : (
              <div className="card-list">
                {list.items.map((item) => (
                  <article className="media-row panel" key={item.id}>
                    <div className="media-row-main">
                      <a className="title-link mono" href={hrefFor({ view: "viewer", mediaId: item.id })}>
                        {truncateHash(item.sha256)}
                      </a>
                      <span className={`status-chip status-${item.status}`}>{item.status}</span>
                      <span className="pill">{item.kind}</span>
                    </div>
                    <dl className="inline-facts">
                      <div><dt>Ingested</dt><dd>{formatTs(item.ingested_at)}</dd></div>
                      <div><dt>Dimensions</dt><dd>{item.width === null ? "pending" : `${String(item.width)} × ${String(item.height)}`}</dd></div>
                      <div><dt>Faces</dt><dd>{item.detection_count}</dd></div>
                    </dl>
                    <p className="job-progress" aria-live="polite">{progressText(item)}</p>
                    <a className="button-link" href={hrefFor({ view: "viewer", mediaId: item.id })}>Open media</a>
                  </article>
                ))}
              </div>
            )
          }
        </Loaded>
      </section>
    </div>
  );
}
