import { useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent } from "react";

import { ApiError, errorMessage, postJson } from "../api/client";
import type { CaseList, Health, Media, MediaBulkDelete, MediaList } from "../api/types";
import { CaseBasis } from "../components/CaseBasis";
import { FolderEnroll } from "../components/FolderEnroll";
import type { GalleryNotice } from "../components/GalleryControls";
import { Loaded } from "../components/Loading";
import { DeleteMediaButton, MediaCard } from "../components/MediaCard";
import { NewCase } from "../components/NewCase";
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
import { setMediaLayout, useMediaLayout } from "../lib/layout";
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

/**
 * What a bulk delete actually did, in the order it matters: the files, then the
 * gallery, then the bytes. The template and unenrolment clauses are conditional
 * because they are usually zero, and a sentence that recites zeroes stops being
 * read; the errors clause is last because it is the exception.
 */
function bulkDeleteText(result: MediaBulkDelete): string {
  const files = `${String(result.deleted.length)} file${result.deleted.length === 1 ? "" : "s"}`;
  const parts = [
    `${files} deleted: ${String(result.detections)} detection(s) and ${String(result.tracks)} track(s) with them`,
  ];
  if (result.templates > 0) {
    parts.push(
      `${String(result.templates)} template(s) left the gallery` +
        (result.persons_unenrolled > 0
          ? `, and ${String(result.persons_unenrolled)} person(s) are now unenrolled`
          : ""),
    );
  }
  parts.push(
    `${String(result.objects_removed)} stored object(s) removed` +
      (result.objects_removed < result.deleted.length
        ? "; the rest stay, because another case holds the same bytes"
        : ""),
  );
  if (result.errors.length > 0) {
    parts.push(`${String(result.errors.length)} could not be deleted and were left alone`);
  }
  return `${parts.join("; ")}.`;
}

export default function MediaLibraryView() {
  const cases = useResource<CaseList>("/api/cases");
  const health = useResource<Health>("/api/healthz");
  const caseId = useSelectedCase();
  const [faces, setFaces] = useState<"" | "true" | "false">("");
  const mediaPath = useMemo(() => {
    const params = new URLSearchParams();
    if (caseId !== "") {
      params.set("case_id", caseId);
    }
    if (faces !== "") {
      params.set("has_faces", faces);
    }
    const suffix = params.toString();
    return suffix === "" ? "/api/media" : `/api/media?${suffix}`;
  }, [caseId, faces]);
  const media = useResource<MediaList>(mediaPath, 1_000);
  const fileRef = useRef<HTMLInputElement>(null);
  const [sourceUrl, setSourceUrl] = useState("");
  const [busy, setBusy] = useState<"upload" | "capture" | "delete" | null>(null);
  const [dragging, setDragging] = useState(false);
  const layout = useMediaLayout();
  /** Which files the bulk action applies to, by media id. */
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [confirmingBulk, setConfirmingBulk] = useState(false);
  /** The outcome of a delete: the one thing in the library that destroys. */
  const [notice, setNotice] = useState<GalleryNotice | null>(null);
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

  // A selection is about the list on screen: switching case or filter must not carry ids
  // that are no longer visible into a delete.
  useEffect(() => {
    setSelected(new Set());
    setConfirmingBulk(false);
  }, [caseId, faces]);

  function toggleSelected(mediaId: string): void {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(mediaId)) {
        next.delete(mediaId);
      } else {
        next.add(mediaId);
      }
      return next;
    });
  }

  async function deleteSelected(mediaIds: readonly string[]): Promise<void> {
    if (mediaIds.length === 0) {
      return;
    }
    setBusy("delete");
    try {
      const result = await postJson<MediaBulkDelete>("/api/media/bulk_delete", {
        media_ids: mediaIds,
      });
      setNotice({
        tone: result.errors.length > 0 ? "attention" : "success",
        text: bulkDeleteText(result),
      });
      setSelected(new Set());
      media.reload();
    } catch (failure) {
      setNotice({
        tone: "error",
        text:
          failure instanceof ApiError
            ? `Could not delete the selection (${String(failure.status)}): ${failure.detail}`
            : `Could not delete the selection: ${errorMessage(failure)}`,
      });
    } finally {
      setBusy(null);
      setConfirmingBulk(false);
    }
  }

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
  const selectedCase =
    cases.state.phase === "ready"
      ? (cases.state.data.items.find((item) => item.id === caseId) ?? null)
      : null;
  const caseName = selectedCase?.name ?? null;
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
              <>
                <p className="notice">
                  No cases exist. Evidence belongs to a case, so create one before adding any.
                </p>
                <NewCase onCreated={cases.reload} />
              </>
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
                    Image or video
                    <input
                      ref={fileRef}
                      name="file"
                      type="file"
                      accept="image/jpeg,image/png,image/webp,image/heic,image/heif,video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-msvideo"
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

                {/* Outside the upload form: a form cannot nest inside another one. */}
                <NewCase onCreated={cases.reload} />

                {selectedCase !== null && (
                  <CaseBasis
                    key={selectedCase.id}
                    record={selectedCase}
                    onAmended={cases.reload}
                  />
                )}

                <FolderEnroll caseId={caseId} onImported={media.reload} />

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
        <div className="view-heading library-heading">
          <h2 id="library-heading">Files</h2>
          <label>
            Layout
            <select
              value={layout}
              onChange={(event) =>
                setMediaLayout(event.currentTarget.value === "list" ? "list" : "tiles")
              }
            >
              <option value="tiles">Gallery</option>
              <option value="list">List</option>
            </select>
          </label>
          <label>
            Faces
            <select
              value={faces}
              onChange={(event) =>
                setFaces(event.currentTarget.value as "" | "true" | "false")
              }
            >
              <option value="">All files</option>
              <option value="true">With faces</option>
              <option value="false">Without faces</option>
            </select>
          </label>
        </div>
        {notice !== null && (
          <p
            className={`notice ${notice.tone}`}
            role={notice.tone === "error" ? "alert" : "status"}
          >
            {notice.text}
          </p>
        )}
        <Loaded state={media.state} label="media">
          {(list) => {
            const chosen = list.items.filter((item) => selected.has(item.id));
            return list.items.length === 0 ? (
              <p className="notice">
                {faces === "" ? "This case has no media yet." : "No files match that filter."}
              </p>
            ) : (
              <>
                <div className="bulk-bar panel">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={chosen.length === list.items.length}
                      onChange={(event) =>
                        setSelected(
                          event.currentTarget.checked
                            ? new Set(list.items.map((item) => item.id))
                            : new Set(),
                        )
                      }
                    />
                    Select all ({chosen.length} selected)
                  </label>
                  {confirmingBulk ? (
                    <span className="delete-confirm">
                      <button
                        className="danger"
                        type="button"
                        disabled={busy === "delete"}
                        onClick={() => void deleteSelected(chosen.map((item) => item.id))}
                      >
                        {busy === "delete"
                          ? "Deleting…"
                          : `Delete ${String(chosen.length)} file${chosen.length === 1 ? "" : "s"}`}
                      </button>
                      <button
                        type="button"
                        disabled={busy === "delete"}
                        onClick={() => setConfirmingBulk(false)}
                      >
                        Cancel
                      </button>
                    </span>
                  ) : (
                    <button
                      className="danger"
                      type="button"
                      disabled={busy !== null || chosen.length === 0}
                      onClick={() => setConfirmingBulk(true)}
                    >
                      Delete selected
                    </button>
                  )}
                </div>
                {layout === "tiles" ? (
                  <div className="media-grid">
                    {list.items.map((item) => (
                      <MediaCard
                        key={item.id}
                        media={item}
                        selected={selected.has(item.id)}
                        onSelect={toggleSelected}
                        onOutcome={setNotice}
                        onDeleted={media.reload}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="card-list">
                    {list.items.map((item) => (
                      <article
                        className={selected.has(item.id) ? "media-row panel selected" : "media-row panel"}
                        key={item.id}
                      >
                        <div className="media-row-main">
                          <input
                            type="checkbox"
                            checked={selected.has(item.id)}
                            aria-label={`Select ${truncateHash(item.sha256)}`}
                            onChange={() => toggleSelected(item.id)}
                          />
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
                        <div className="row-actions">
                          <a className="button-link" href={hrefFor({ view: "viewer", mediaId: item.id })}>Open media</a>
                          <DeleteMediaButton
                            media={item}
                            onOutcome={setNotice}
                            onDeleted={media.reload}
                          />
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </>
            );
          }}
        </Loaded>
      </section>
    </div>
  );
}
