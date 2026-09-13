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

/**
 * Boxes are sampled along a video at about 3 fps, so the sample nearest the
 * playhead is normally within ~167 ms of it. Beyond this the nearest sample
 * describes a different moment — the detector lost the face, the track ended,
 * or the operator has seeked into a gap — and a box frozen over a face that
 * has since moved is worse than no box at all, so nothing is drawn.
 */
const SAMPLE_TOLERANCE_MS = 250;

/**
 * How much of a video's timeline one `/tracks` request covers. A half-hour
 * file at 3 fps stores thousands of samples per track and the overlay draws
 * exactly one of them per instant, so the whole file is never worth fetching.
 * A minute keeps a response to a couple of hundred samples per track while
 * being long enough that watching straight through refetches once a minute.
 */
const TRACK_WINDOW_MS = 60_000;

/**
 * Fetched either side of the window, so boxes do not blink out while the next
 * window is in flight and a short scrub over a boundary stays covered.
 */
const TRACK_WINDOW_MARGIN_MS = 5_000;

/**
 * The sample closest to `tMs`, by bisection: samples arrive ordered by time,
 * and a window holds a few hundred of them per track, which the overlay would
 * otherwise rescan for every track on every animation frame.
 */
function nearestSample(samples: readonly TrackSample[], tMs: number): TrackSample | undefined {
  let low = 0;
  let high = samples.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    const sample = samples[mid];
    if (sample !== undefined && sample.t_ms < tMs) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  const after = samples[low];
  const before = low > 0 ? samples[low - 1] : undefined;
  if (after === undefined) {
    return before;
  }
  if (before === undefined) {
    return after;
  }
  return tMs - before.t_ms <= after.t_ms - tMs ? before : after;
}

/** The boxes to draw at `tMs`: one per track, and only while it is current. */
function tracksAt(tracks: readonly TrackOverlay[], tMs: number): DrawableTrack[] {
  const result: DrawableTrack[] = [];
  for (const track of tracks) {
    const sample = nearestSample(track.samples, tMs);
    if (sample !== undefined && Math.abs(sample.t_ms - tMs) <= SAMPLE_TOLERANCE_MS) {
      result.push({ track, sample });
    }
  }
  return result;
}

/**
 * One box, captioned only when `labelled`. Shared by the still and the video
 * overlay so a box means the same thing in both: same colours, same emphasis
 * for the face under the pointer, same caption text.
 */
function paintTrackBox(
  context: CanvasRenderingContext2D,
  track: TrackOverlay,
  rect: Rect,
  color: string,
  labelled: boolean,
  cssWidth: number,
): void {
  context.strokeStyle = color;
  context.lineWidth = labelled ? 3 : 2;
  context.strokeRect(rect.x, rect.y, rect.w, rect.h);
  if (!labelled) {
    return;
  }
  const label = `${track.name ?? "unidentified"} · ${track.band ?? "no match"} ${formatScore(track.score)} · #${track.track_id.slice(0, 8)}${track.source === "operator" ? " · OP" : ""}`;
  context.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  const labelWidth = Math.min(context.measureText(label).width + 10, cssWidth - rect.x);
  const labelY = Math.max(0, rect.y - 21);
  context.fillStyle = "rgba(12, 14, 17, 0.88)";
  context.fillRect(rect.x, labelY, labelWidth, 21);
  context.fillStyle = color;
  context.fillText(label, rect.x + 5, labelY + 14, Math.max(0, labelWidth - 10));
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

  /**
   * Cursor position in CSS pixels relative to the overlay, or null. A ref, not
   * state: the pointer moves at the display rate and the handler redraws the
   * canvas directly rather than re-rendering the view for every move.
   */
  const pointerRef = useRef<{ x: number; y: number } | null>(null);

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
    // A caption over every box hides the faces underneath it. The label belongs
    // to the box the operator is pointing at, or the one they have selected;
    // the track list carries the same text for all of them at once.
    const pointer = pointerRef.current;
    const hovered =
      pointer === null
        ? null
        : hitTestImageRects(
            drawables.map(({ sample }) => sampleRect(sample)),
            pointer.x,
            pointer.y,
            box,
          );

    for (const [index, { track, sample }] of drawables.entries()) {
      const selected = track.track_id === selectedTrackId;
      paintTrackBox(
        context,
        track,
        imageRectToCss(sampleRect(sample), box),
        selected ? SELECTED_COLOR : track.band === null ? NO_BAND_COLOR : BAND_COLOR[track.band],
        selected || index === hovered,
        cssWidth,
      );
    }
  }, [drawables, media.height, media.width, selectedTrackId, tracks.height, tracks.width]);

  // The observer only ever needs to call the latest `draw`, and `draw` changes
  // identity on every poll response. Keying the effect on it disconnected and
  // rebuilt the observer once a second for a picture that had not moved.
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    drawRef.current();
  }, [draw]);

  useEffect(() => {
    const image = imageRef.current;
    if (image === null) {
      return;
    }
    const redraw = (): void => {
      drawRef.current();
    };
    const observer = new ResizeObserver(redraw);
    observer.observe(image);
    image.addEventListener("load", redraw);
    window.addEventListener("resize", redraw);
    redraw();
    return () => {
      observer.disconnect();
      image.removeEventListener("load", redraw);
      window.removeEventListener("resize", redraw);
    };
  }, []);

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

  function onPointerMove(event: MouseEvent<HTMLCanvasElement>): void {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    pointerRef.current = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    drawRef.current();
  }

  function onPointerLeave(): void {
    pointerRef.current = null;
    drawRef.current();
  }

  return (
    <>
      <div className="image-stage">
        <img ref={imageRef} src={`/api/media/${encodeURIComponent(media.id)}/file`} alt={`Evidence media ${media.id}`} />
        <canvas
          ref={canvasRef}
          onClick={hitTest}
          onMouseMove={onPointerMove}
          onMouseLeave={onPointerLeave}
          aria-label="Face detection overlay. Hover a box to read its label. Select a face using the buttons below."
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

/**
 * The same overlay over a playing file (spec 6.2). Three things differ from
 * the still path:
 *
 *   - a track has many samples, and only the one at the playhead may be drawn;
 *   - the boxes have to follow playback, so the canvas is redrawn from an
 *     animation frame rather than only when React re-renders;
 *   - the picture underneath has native controls, which a canvas that takes
 *     input would swallow whole — the scrub bar cannot be reached through a
 *     covering element, and neither can the hover that reveals it. So this
 *     canvas is pass-through by default and takes a click only while the
 *     pointer is over a box.
 */
function VideoOverlay({
  media,
  tracks,
  selectedTrackId,
  onSelect,
  onWindowChange,
}: {
  media: Media;
  tracks: MediaTracks;
  selectedTrackId: string | null;
  onSelect: (trackId: string) => void;
  /** Playback has entered another track window; fetch the samples for it. */
  onWindowChange: (startMs: number) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pointerRef = useRef<{ x: number; y: number } | null>(null);

  /**
   * The playhead, the boxes last drawn for it, and the window already asked
   * for. Refs, not state, for the reason `pointerRef` is one in the still
   * overlay: these change at the display rate, only the canvas and the hit
   * test read them, and holding them in state would re-render this subtree
   * sixty times a second to produce identical DOM.
   */
  const timeMsRef = useRef(0);
  const drawnRef = useRef<DrawableTrack[]>([]);
  const windowRef = useRef<number | null>(null);

  // The animation loop is mounted once and must see the current props without
  // being torn down and rebuilt on every poll response.
  const tracksRef = useRef(tracks.tracks);
  tracksRef.current = tracks.tracks;
  const windowChangeRef = useRef(onWindowChange);
  windowChangeRef.current = onWindowChange;

  const draw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video === null || canvas === null) {
      return;
    }
    const cssWidth = video.clientWidth;
    const cssHeight = video.clientHeight;
    const context = prepareCanvas(canvas, cssWidth, cssHeight, window.devicePixelRatio || 1);
    if (context === null) {
      return;
    }
    context.clearRect(0, 0, cssWidth, cssHeight);
    // The letterbox has to be computed from the frame size the browser is
    // actually painting, which is why this reads the element and not the
    // media row; the stored dimensions only stand in until metadata loads,
    // and are null themselves until the pipeline has decoded the file.
    const box = containLetterbox(
      video.videoWidth || media.width || 0,
      video.videoHeight || media.height || 0,
      cssWidth,
      cssHeight,
    );
    if (box.scale <= 0) {
      return;
    }
    const drawn = tracksAt(tracksRef.current, timeMsRef.current);
    // What the click handler hit-tests against: exactly the boxes on screen.
    drawnRef.current = drawn;
    const pointer = pointerRef.current;
    const hovered =
      pointer === null
        ? null
        : hitTestImageRects(
            drawn.map(({ sample }) => sampleRect(sample)),
            pointer.x,
            pointer.y,
            box,
          );
    // The whole input story of this overlay: a box is clickable, everything
    // else belongs to the player. A face sitting over the control bar is the
    // one place the two compete, and the face list below is the way out.
    canvas.style.pointerEvents = hovered === null ? "none" : "auto";

    for (const [index, { track, sample }] of drawn.entries()) {
      const selected = track.track_id === selectedTrackId;
      paintTrackBox(
        context,
        track,
        imageRectToCss(sampleRect(sample), box),
        selected ? SELECTED_COLOR : track.band === null ? NO_BAND_COLOR : BAND_COLOR[track.band],
        selected || index === hovered,
        cssWidth,
      );
    }
  }, [media.height, media.width, selectedTrackId]);

  const drawRef = useRef(draw);
  drawRef.current = draw;

  // Repaint on a new window of samples as well as on a new `draw`: while the
  // file is paused, nothing else asks for one, and the boxes for the time the
  // operator seeked to arrive after the seek that asked for them.
  useEffect(() => {
    drawRef.current();
  }, [draw, tracks.tracks]);

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) {
      return;
    }
    let frame = 0;

    /** Read the playhead, tell the parent if the window moved, repaint. */
    const sync = (): void => {
      timeMsRef.current = video.currentTime * 1000;
      // Whole buckets: a window that slid with the playhead would change its
      // URL on every frame, and refetching is the cost windowing avoids.
      const start = Math.floor(timeMsRef.current / TRACK_WINDOW_MS) * TRACK_WINDOW_MS;
      if (windowRef.current !== start) {
        windowRef.current = start;
        windowChangeRef.current(start);
      }
      drawRef.current();
    };

    // `timeupdate` fires about four times a second, which leaves a box a
    // quarter of a second behind the face it belongs to. While the file plays
    // the playhead is read once per displayed frame instead.
    const tick = (): void => {
      frame = requestAnimationFrame(tick);
      sync();
    };
    const startLoop = (): void => {
      if (frame === 0) {
        frame = requestAnimationFrame(tick);
      }
      sync();
    };
    const stopLoop = (): void => {
      if (frame !== 0) {
        cancelAnimationFrame(frame);
        frame = 0;
      }
      sync();
    };

    const redraw = (): void => {
      drawRef.current();
    };
    const observer = new ResizeObserver(redraw);
    observer.observe(video);
    window.addEventListener("resize", redraw);
    // A seek, a pause and a stopped file all land on one exact time, and each
    // is handled in its own listener rather than left to the next animation
    // frame: the redraw then happens before the browser paints the frame the
    // operator asked for, instead of one frame behind it.
    const played = ["play", "playing"];
    const stopped = ["pause", "ended"];
    const moved = ["seeking", "seeked", "timeupdate", "loadedmetadata"];
    for (const event of played) {
      video.addEventListener(event, startLoop);
    }
    for (const event of stopped) {
      video.addEventListener(event, stopLoop);
    }
    for (const event of moved) {
      video.addEventListener(event, sync);
    }
    sync();

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", redraw);
      for (const event of played) {
        video.removeEventListener(event, startLoop);
      }
      for (const event of stopped) {
        video.removeEventListener(event, stopLoop);
      }
      for (const event of moved) {
        video.removeEventListener(event, sync);
      }
      if (frame !== 0) {
        cancelAnimationFrame(frame);
      }
    };
  }, []);

  /**
   * Pointer position in CSS pixels relative to the overlay. Read from the
   * stage rather than the canvas: the canvas is pass-through most of the time,
   * so these events arrive from the video element underneath it.
   */
  function pointerAt(event: MouseEvent<HTMLDivElement>): { x: number; y: number } | null {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return null;
    }
    const bounds = canvas.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  function hitTest(event: MouseEvent<HTMLDivElement>): void {
    const video = videoRef.current;
    const pointer = pointerAt(event);
    if (video === null || pointer === null) {
      return;
    }
    const box = containLetterbox(
      video.videoWidth || media.width || 0,
      video.videoHeight || media.height || 0,
      video.clientWidth,
      video.clientHeight,
    );
    const drawn = drawnRef.current;
    const hit = hitTestImageRects(
      drawn.map(({ sample }) => sampleRect(sample)),
      pointer.x,
      pointer.y,
      box,
    );
    const selected = hit === null ? undefined : drawn[hit];
    if (selected !== undefined) {
      onSelect(selected.track.track_id);
    }
  }

  function onPointerMove(event: MouseEvent<HTMLDivElement>): void {
    pointerRef.current = pointerAt(event);
    drawRef.current();
  }

  function onPointerLeave(): void {
    pointerRef.current = null;
    drawRef.current();
  }

  return (
    <>
      {/* The pointer handlers sit on the stage, not on the canvas: most of the
          time the canvas is pass-through and the events come from the video
          element under it. Everything they drive is also on a real button in
          the face list below, so nothing here is keyboard-only reachable. */}
      <div
        className="image-stage video-stage"
        onClick={hitTest}
        onMouseMove={onPointerMove}
        onMouseLeave={onPointerLeave}
      >
        <video
          ref={videoRef}
          controls
          playsInline
          preload="metadata"
          src={`/api/media/${encodeURIComponent(media.id)}/file`}
          aria-label={`Evidence video ${media.id}`}
        />
        <canvas
          ref={canvasRef}
          aria-label="Face detection overlay. Hover a box to read its label. Select a face using the buttons below."
        />
      </div>
      {tracks.tracks.length === 0 ? (
        media.status === "new" || media.status === "processing" ? (
          <p className="notice" role="status">
            Processing this video{media.status === "processing" ? "" : " shortly"}; face boxes
            appear here as soon as the worker finishes. This page refreshes itself.
          </p>
        ) : (
          // Not "no faces in this video": only one window of it was asked for.
          <p className="notice">No face tracks are stored for this part of the video.</p>
        )
      ) : (
        <div className="track-picker" aria-label="Detected faces in this part of the video">
          {tracks.tracks.map((track) => (
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
  /**
   * Polling stops when the pipeline does.
   *
   * `done` and `failed` are terminal for a media row: the worker will not touch
   * it again, so every further request returns the same bytes. This page used
   * to poll two endpoints a second for as long as it stayed open, competing
   * with the live loop for the one uvicorn worker.
   */
  const media = useResource<Media>(`/api/media/${encodeURIComponent(mediaId)}`, 1_000, {
    stopWhen: (row) => row.status === "done" || row.status === "failed",
  });
  const settled =
    media.state.phase === "ready" &&
    (media.state.data.status === "done" || media.state.data.status === "failed");
  const kind = media.state.phase === "ready" ? media.state.data.kind : null;
  const [windowStartMs, setWindowStartMs] = useState(0);
  /**
   * Tracks have no status of their own; they stop changing when the media does.
   *
   * The request always carries a window, including for a still, and that is
   * deliberate: the media row has not arrived on the first render, so a URL
   * that depended on `kind` would fetch the whole file for a video before
   * finding out it was one — the thousands of samples this windowing exists
   * to avoid — and would then refetch a still under a second URL. A still's
   * one sample per track sits at `t_ms = 0`, inside the first window, so it
   * gets the same payload it always did from one request.
   */
  const tracksPath = useMemo(() => {
    const from = Math.max(0, windowStartMs - TRACK_WINDOW_MARGIN_MS);
    const to = windowStartMs + TRACK_WINDOW_MS + TRACK_WINDOW_MARGIN_MS;
    return `/api/media/${encodeURIComponent(mediaId)}/tracks?from_ms=${String(from)}&to_ms=${String(to)}`;
  }, [mediaId, windowStartMs]);
  const tracks = useResource<MediaTracks>(tracksPath, settled ? 0 : 1_000);
  /**
   * The last payload the video overlay saw. `Loaded` replaces its children
   * with a placeholder while a request is in flight, and the video's URL
   * changes every time playback crosses a window — unmounting the `<video>`
   * to do that would stop playback dead and lose the playhead. So the player
   * is never wrapped in `Loaded`: it holds the previous window's boxes, which
   * the fetch margin keeps valid, until the next one lands.
   */
  const [videoTracks, setVideoTracks] = useState<MediaTracks | null>(null);
  useEffect(() => {
    if (kind === "video" && tracks.state.phase === "ready") {
      setVideoTracks(tracks.state.data);
    }
  }, [kind, tracks.state]);
  // What the player renders before its first window arrives: itself, and no
  // boxes. A video whose processing has not finished stays in this state.
  const emptyTracks = useMemo<MediaTracks>(
    () => ({ media_id: mediaId, width: null, height: null, tracks: [] }),
    [mediaId],
  );
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
  //
  // A video's list is one window of the file, so a track that has scrolled out
  // of it is still the face the operator is tagging: the panel reads the track
  // by id, not from this list, and only a still may reset a stale selection.
  useEffect(() => {
    if (tracks.state.phase !== "ready") {
      return;
    }
    const available = tracks.state.data.tracks;
    setSelectedTrackId((current) =>
      current !== null && (kind === "video" || available.some((track) => track.track_id === current))
        ? current
        : (available[0]?.track_id ?? null),
    );
  }, [kind, tracks.state]);

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
            {item.kind === "video" && tracks.state.phase === "error" && (
              // The player is outside `Loaded`, so a failing tracks request has
              // nowhere else to surface: the video would just play boxless.
              <p className="notice error" role="alert">
                Could not load face tracks: <span className="mono">{tracks.state.message}</span>
              </p>
            )}
            <div className="viewer-layout">
              <main className="viewer-main">
                {item.kind === "video" ? (
                  <VideoOverlay
                    media={item}
                    tracks={videoTracks ?? emptyTracks}
                    selectedTrackId={selectedTrackId}
                    onSelect={setSelectedTrackId}
                    onWindowChange={setWindowStartMs}
                  />
                ) : (
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
                )}
              </main>
              {selectedTrackId === null ? (
                <aside className="tag-panel panel"><p className="muted">Select a detected face to review its identity.</p></aside>
              ) : (
                <TagPanel trackId={selectedTrackId} onChanged={reload} />
              )}
            </div>
          </>
        )}
      </Loaded>
    </>
  );
}
