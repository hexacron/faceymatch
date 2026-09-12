import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
} from "react";

import { errorMessage, postJson } from "../api/client";
import type {
  Candidate,
  Identification,
  IdentificationRequest,
  Media,
  MediaTracks,
  PersonList,
  TrackDetail,
  TrackOverlay,
  TrackSample,
} from "../api/types";
import { BandPill, SourceBadge } from "../components/BandPill";
import {
  NO_CROP_DONATED,
  NO_TEMPLATE_EXPLANATION,
  NO_TEMPLATE_REMEDY,
} from "../components/GalleryState";
import { Loaded } from "../components/Loading";
import { BAND_COLOR, formatScore, NO_BAND_COLOR, SELECTED_COLOR, truncateHash } from "../lib/display";
import {
  bestOverlapRect,
  containLetterbox,
  hitTestImageRects,
  imageRectToCss,
  normalizeRect,
  prepareCanvas,
  type Rect,
} from "../lib/geometry";
import { hrefFor } from "../lib/router";
import { useResource } from "../lib/useResource";

type DrawableTrack = { track: TrackOverlay; sample: TrackSample };

function drawableTracks(tracks: readonly TrackOverlay[]): DrawableTrack[] {
  const result: DrawableTrack[] = [];
  for (const track of tracks) {
    const sample = track.samples[0];
    if (sample !== undefined) {
      result.push({ track, sample });
    }
  }
  return result;
}

function sampleRect(sample: TrackSample): Rect {
  return { x: sample.x, y: sample.y, w: sample.w, h: sample.h };
}

function ImageOverlay({
  media,
  tracks,
  selectedTrackId,
  onSelect,
}: {
  media: Media;
  tracks: MediaTracks;
  selectedTrackId: string | null;
  onSelect: (trackId: string) => void;
}) {
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawables = useMemo(() => drawableTracks(tracks.tracks), [tracks.tracks]);

  const draw = useCallback(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (image === null || canvas === null) {
      return;
    }
    const cssWidth = image.clientWidth;
    const cssHeight = image.clientHeight;
    const context = prepareCanvas(canvas, cssWidth, cssHeight, window.devicePixelRatio || 1);
    if (context === null) {
      return;
    }
    context.clearRect(0, 0, cssWidth, cssHeight);
    const naturalWidth = tracks.width ?? media.width ?? image.naturalWidth;
    const naturalHeight = tracks.height ?? media.height ?? image.naturalHeight;
    const box = containLetterbox(naturalWidth, naturalHeight, cssWidth, cssHeight);
    if (box.scale <= 0) {
      return;
    }

    for (const { track, sample } of drawables) {
      const rect = imageRectToCss(sampleRect(sample), box);
      const selected = track.track_id === selectedTrackId;
      const color = selected
        ? SELECTED_COLOR
        : track.band === null
          ? NO_BAND_COLOR
          : BAND_COLOR[track.band];
      context.strokeStyle = color;
      context.lineWidth = selected ? 3 : 2;
      context.strokeRect(rect.x, rect.y, rect.w, rect.h);

      const label = `${track.name ?? "unidentified"} · ${track.band ?? "no match"} ${formatScore(track.score)} · #${track.track_id.slice(0, 8)}${track.source === "operator" ? " · OP" : ""}`;
      context.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      const labelWidth = Math.min(context.measureText(label).width + 10, cssWidth - rect.x);
      const labelY = Math.max(0, rect.y - 21);
      context.fillStyle = "rgba(12, 14, 17, 0.88)";
      context.fillRect(rect.x, labelY, labelWidth, 21);
      context.fillStyle = color;
      context.fillText(label, rect.x + 5, labelY + 14, Math.max(0, labelWidth - 10));
    }
  }, [drawables, media.height, media.width, selectedTrackId, tracks.height, tracks.width]);

  useEffect(() => {
    const image = imageRef.current;
    if (image === null) {
      return;
    }
    const observer = new ResizeObserver(draw);
    observer.observe(image);
    image.addEventListener("load", draw);
    window.addEventListener("resize", draw);
    draw();
    return () => {
      observer.disconnect();
      image.removeEventListener("load", draw);
      window.removeEventListener("resize", draw);
    };
  }, [draw]);

  function hitTest(event: MouseEvent<HTMLCanvasElement>): void {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (image === null || canvas === null) {
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    const box = containLetterbox(
      tracks.width ?? media.width ?? image.naturalWidth,
      tracks.height ?? media.height ?? image.naturalHeight,
      image.clientWidth,
      image.clientHeight,
    );
    const hit = hitTestImageRects(
      drawables.map(({ sample }) => sampleRect(sample)),
      event.clientX - bounds.left,
      event.clientY - bounds.top,
      box,
    );
    const selected = hit === null ? undefined : drawables[hit];
    if (selected !== undefined) {
      onSelect(selected.track.track_id);
    }
  }

  return (
    <>
      <div className="image-stage">
        <img ref={imageRef} src={`/api/media/${encodeURIComponent(media.id)}/file`} alt={`Evidence media ${media.id}`} />
        <canvas
          ref={canvasRef}
          onClick={hitTest}
          aria-label="Face detection overlay. Select a face using the buttons below."
        />
      </div>
      {drawables.length === 0 ? (
        media.status === "new" || media.status === "processing" ? (
          // A fresh paste, drop, or capture lands here before the worker has
          // run, so say so instead of showing a bare image with no boxes.
          <p className="notice" role="status">
            Processing this image{media.status === "processing" ? "" : " shortly"}; face boxes appear
            here as soon as the worker finishes. This page refreshes itself.
          </p>
        ) : (
          <p className="notice">No face tracks are available for this image.</p>
        )
      ) : (
        <div className="track-picker" aria-label="Detected faces">
          {drawables.map(({ track }) => (
            <button
              key={track.track_id}
              type="button"
              className={track.track_id === selectedTrackId ? "selected" : ""}
              onClick={() => onSelect(track.track_id)}
            >
              {track.name ?? "Unidentified"} <BandPill band={track.band} score={track.score} />
            </button>
          ))}
        </div>
      )}
    </>
  );
}

function CandidateRow({
  candidate,
  disabled,
  onConfirm,
}: {
  candidate: Candidate;
  disabled: boolean;
  onConfirm: (candidate: Candidate) => void;
}) {
  return (
    <li className="candidate-row">
      <div>
        <strong>{candidate.name}</strong>
        <span className="muted"> rank {candidate.rank}</span>
      </div>
      <BandPill band={candidate.band} score={candidate.score} />
      <button type="button" disabled={disabled} onClick={() => onConfirm(candidate)}>Confirm</button>
    </li>
  );
}

function TagPanel({ trackId, onChanged }: { trackId: string; onChanged: () => void }) {
  const detail = useResource<TrackDetail>(`/api/tracks/${encodeURIComponent(trackId)}`);
  const persons = useResource<PersonList>("/api/persons");
  const [reassignId, setReassignId] = useState("");
  const [newName, setNewName] = useState("");
  const [note, setNote] = useState("");
  const [enroll, setEnroll] = useState(false);
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  /**
   * Set when a decision that was meant to enrol did not produce a template.
   * Separate from `message` because it is the outcome the operator would
   * otherwise never learn: "saved" and "in the gallery" are different things.
   */
  const [enrolGap, setEnrolGap] = useState<string | null>(null);

  useEffect(() => {
    setMessage(null);
    setFailure(null);
    setEnrolGap(null);
    setReassignId("");
    setNewName("");
    setNote("");
    setEnroll(false);
  }, [trackId]);

  async function decide(request: IdentificationRequest): Promise<void> {
    setPending(true);
    setFailure(null);
    setMessage(null);
    setEnrolGap(null);
    try {
      const result = await postJson<Identification>("/api/identifications", request);
      detail.reload();
      persons.reload();
      onChanged();
      const meantToEnrol =
        request.decision === "new" ||
        ((request.decision === "confirm" || request.decision === "reassign") &&
          request.enroll === true);
      // A backend that does not report the outcome gets no claim made on its
      // behalf: announcing an enrolment nobody verified is how three persons
      // ended up in the gallery with no template and nobody noticed. The
      // runtime check is deliberate — the field is required by the contract,
      // and a bundle can outlive the backend that serves it.
      const created: boolean | null =
        typeof result.template_created === "boolean" ? result.template_created : null;
      if (!meantToEnrol || created === null) {
        setMessage("Operator decision saved.");
        return;
      }
      if (created) {
        setMessage("Operator decision saved, and this crop is now a gallery template.");
        return;
      }
      setMessage("Operator decision saved: the face is tagged.");
      setEnrolGap(
        request.decision === "new"
          ? `${NO_TEMPLATE_EXPLANATION} ${NO_TEMPLATE_REMEDY}`
          : `${NO_CROP_DONATED} ${NO_TEMPLATE_REMEDY}`,
      );
    } catch (error) {
      setFailure(errorMessage(error));
    } finally {
      setPending(false);
    }
  }

  return (
    <aside className="tag-panel panel" aria-labelledby="tag-heading">
      <h2 id="tag-heading">Tag face</h2>
      <p className="mono selectable-id">{trackId}</p>
      <Loaded state={detail.state} label="track details">
        {(track) => (
          <>
            <div className="identity-summary">
              <span className="muted">Current identity</span>
              <strong>{track.identity?.name ?? "Unidentified"}</strong>
              <SourceBadge source={track.identity?.source ?? null} />
            </div>

            <h3>Best crops</h3>
            {track.crops.length === 0 ? (
              <p className="muted">No crop passed the quality gate.</p>
            ) : (
              <div className="crop-strip">
                {track.crops.slice(0, 4).map((sha, index) => (
                  // Two detections in one track can share byte-identical crop
                  // bytes, so the content hash is not a unique key here.
                  // `detection_ids` is aligned with `crops` and is unique.
                  <img
                    key={track.detection_ids[index] ?? `${sha}-${String(index)}`}
                    src={`/api/crops/${encodeURIComponent(sha)}`}
                    alt={`Face crop ${truncateHash(sha)}`}
                  />
                ))}
              </div>
            )}

            <h3>Top candidates</h3>
            {track.candidates.length === 0 ? (
              <p className="muted">No gallery candidates.</p>
            ) : (
              <ol className="candidate-list">
                {track.candidates.slice(0, 3).map((candidate) => (
                  <CandidateRow
                    key={candidate.person_id}
                    candidate={candidate}
                    disabled={pending}
                    onConfirm={(value) => void decide({
                      track_id: trackId,
                      decision: "confirm",
                      person_id: value.person_id,
                      enroll,
                      ...(note.trim() === "" ? {} : { note: note.trim() }),
                    })}
                  />
                ))}
              </ol>
            )}

            <label>
              Decision note <span className="muted">(optional)</span>
              <input value={note} onChange={(event) => setNote(event.currentTarget.value)} />
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={enroll}
                disabled={pending}
                onChange={(event) => setEnroll(event.currentTarget.checked)}
              />
              Also add this crop as a template
            </label>
            <p className="muted compact">
              Tagging says who this is; enrolling donates the crop to the gallery (D17). Applies
              to Confirm and Reassign only — a new person always gets a first template, and
              rejecting never enrolls.
            </p>
            <div className="decision-block">
              <button
                className="danger"
                type="button"
                disabled={pending}
                onClick={() => void decide({
                  track_id: trackId,
                  decision: "reject",
                  ...(note.trim() === "" ? {} : { note: note.trim() }),
                })}
              >
                Reject match
              </button>
            </div>

            <h3>Reassign</h3>
            <Loaded state={persons.state} label="persons">
              {(personList) => (
                <div className="inline-action">
                  <select value={reassignId} onChange={(event) => setReassignId(event.currentTarget.value)} aria-label="Person for reassignment">
                    <option value="">Choose a person</option>
                    {personList.items.map((person) => (
                      <option key={person.id} value={person.id}>{person.display_name}</option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={pending || reassignId === ""}
                    onClick={() => void decide({
                      track_id: trackId,
                      decision: "reassign",
                      person_id: reassignId,
                      enroll,
                      ...(note.trim() === "" ? {} : { note: note.trim() }),
                    })}
                  >Reassign</button>
                </div>
              )}
            </Loaded>

            <h3>Create new person</h3>
            <div className="inline-action">
              <input
                value={newName}
                onChange={(event) => setNewName(event.currentTarget.value)}
                placeholder="Display name"
                aria-label="New person display name"
              />
              <button
                type="button"
                disabled={pending || newName.trim() === ""}
                onClick={() => void decide({
                  track_id: trackId,
                  decision: "new",
                  new_name: newName.trim(),
                  ...(note.trim() === "" ? {} : { note: note.trim() }),
                })}
              >Create and assign</button>
            </div>
          </>
        )}
      </Loaded>
      {failure !== null && <p className="notice error" role="alert">{failure}</p>}
      {message !== null && <p className="notice success" role="status">{message}</p>}
      {enrolGap !== null && <p className="notice attention" role="status">{enrolGap}</p>}
    </aside>
  );
}

/**
 * `focus` is the box the operator clicked in live mode, in fractions of the
 * live frame. The stored file is the same picture, so the detection that
 * overlaps it is the face they meant, and the tag panel opens on that face
 * instead of on whichever track happens to be first.
 */
export default function MediaDetailView({
  mediaId,
  focus,
}: {
  mediaId: string;
  focus: Rect | null;
}) {
  const media = useResource<Media>(`/api/media/${encodeURIComponent(mediaId)}`, 1_000);
  const tracks = useResource<MediaTracks>(`/api/media/${encodeURIComponent(mediaId)}/tracks`, 1_000);
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [focusNote, setFocusNote] = useState<string | null>(null);
  // One resolution per handoff: once it lands, the operator's own clicks win.
  const [resolvedFocusKey, setResolvedFocusKey] = useState<string | null>(null);

  useEffect(() => {
    if (focus === null) {
      return;
    }
    const key = `${mediaId}|${focus.x.toFixed(5)},${focus.y.toFixed(5)}`;
    if (resolvedFocusKey === key || tracks.state.phase !== "ready") {
      return;
    }
    const data = tracks.state.data;
    const width = data.width;
    const height = data.height;
    const drawables = drawableTracks(data.tracks);
    if (drawables.length === 0 || width === null || height === null) {
      // The worker may still be running; polling re-runs this on the next tick.
      const status = media.state.phase === "ready" ? media.state.data.status : "processing";
      if (status === "new" || status === "processing") {
        return;
      }
      setResolvedFocusKey(key);
      setFocusNote(
        "This frame stored no detections, so the face you clicked cannot be tagged here. Live matching is advisory: the stored pipeline runs its own detector and quality gate, and can reject a face live mode still drew a box around.",
      );
      return;
    }
    const index = bestOverlapRect(
      drawables.map(({ sample }) => normalizeRect(sampleRect(sample), width, height)),
      focus,
      0.2,
    );
    setResolvedFocusKey(key);
    const picked = index === null ? undefined : drawables[index];
    if (picked === undefined) {
      setFocusNote(
        "No stored detection overlaps the face you clicked, so the tag panel is on the nearest stored face instead. The quality gate or the stored detector rejected the one you picked.",
      );
      return;
    }
    setSelectedTrackId(picked.track.track_id);
    setFocusNote(null);
  }, [focus, media.state, mediaId, resolvedFocusKey, tracks.state]);

  // Functional update on purpose: the focus handoff above may have just set a
  // track in this same commit, and a stale closure would clobber it.
  useEffect(() => {
    if (tracks.state.phase !== "ready") {
      return;
    }
    const available = tracks.state.data.tracks;
    setSelectedTrackId((current) =>
      current !== null && available.some((track) => track.track_id === current)
        ? current
        : (available[0]?.track_id ?? null),
    );
  }, [tracks.state]);

  const reload = useCallback(() => {
    media.reload();
    tracks.reload();
  }, [media, tracks]);

  return (
    <>
      <div className="view-heading">
        <div>
          <a className="back-link" href={hrefFor({ view: "media" })}>← Media library</a>
          <h1>Media detail</h1>
          <p className="tagline mono">{mediaId}</p>
        </div>
        <button type="button" onClick={reload}>Refresh</button>
      </div>
      <Loaded state={media.state} label="media">
        {(item) => (
          <>
            <div className="media-meta panel">
              <span className={`status-chip status-${item.status}`}>{item.status}</span>
              <span>{item.width === null ? "Dimensions pending" : `${String(item.width)} × ${String(item.height)}`}</span>
              <span>{item.detection_count} detections</span>
              <span className="mono">SHA-256 {truncateHash(item.sha256)}</span>
              {item.job?.error !== null && item.job?.error !== undefined && <span className="status-bad">{item.job.error}</span>}
            </div>
            {focus !== null && resolvedFocusKey === null && (
              <p className="notice" role="status">
                Waiting for the stored detections so the face you clicked can be selected&hellip;
              </p>
            )}
            {focusNote !== null && <p className="notice error" role="alert">{focusNote}</p>}
            {item.kind !== "image" ? (
              <p className="notice">This M1 viewer handles still images. The stored video remains available from the media API.</p>
            ) : (
              <div className="viewer-layout">
                <main className="viewer-main">
                  <Loaded state={tracks.state} label="face tracks">
                    {(trackData) => (
                      <ImageOverlay
                        media={item}
                        tracks={trackData}
                        selectedTrackId={selectedTrackId}
                        onSelect={setSelectedTrackId}
                      />
                    )}
                  </Loaded>
                </main>
                {selectedTrackId === null ? (
                  <aside className="tag-panel panel"><p className="muted">Select a detected face to review its identity.</p></aside>
                ) : (
                  <TagPanel trackId={selectedTrackId} onChanged={reload} />
                )}
              </div>
            )}
          </>
        )}
      </Loaded>
    </>
  );
}
