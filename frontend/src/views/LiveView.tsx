import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";

import { ApiError, errorMessage } from "../api/client";
import type { CaseList, LiveFace, LiveMatchResult, LiveTimings } from "../api/types";
import { BandPill } from "../components/BandPill";
import { CaseBasis } from "../components/CaseBasis";
import { Loaded } from "../components/Loading";
import { WatchHelper } from "../components/WatchHelper";
import { BAND_COLOR, formatScore, NO_BAND_COLOR } from "../lib/display";
import {
  bestOverlapRect,
  containLetterbox,
  hitTestImageRects,
  imageRectToCss,
  normalizeRect,
  prepareCanvas,
  type Rect,
} from "../lib/geometry";
import {
  noCaseRefusal,
  reconcileSelectedCase,
  setSelectedCase,
  useSelectedCase,
} from "../lib/ingest";
import {
  BOX_MAX_PIXELS,
  BOX_QUALITY,
  encodeFrame,
  FRAME_MAX_PIXELS,
  matchFrame,
  persistFrame,
  sleep,
  type Frame,
} from "../lib/liveMatch";
import { navigate } from "../lib/router";
import { useResource } from "../lib/useResource";

/** Sampling rates worth offering: below 1 fps feels broken, above 8 fps just drops frames. */
const FPS_CHOICES: readonly number[] = [1, 2, 3, 5, 8];
const DEFAULT_FPS = 3;
/** Floor the backend asked for, whatever rate the operator picks. */
const MIN_PERIOD_MS = 150;
/**
 * Every third tick asks for names; the other two ask only for boxes.
 *
 * Identification is most of a tick — embedding three faces costs more than
 * decode, detect and the gallery matmul together — and a box that tracks the
 * face is what the operator is actually watching. Names refresh at fps/3,
 * which for the default 3 fps is once a second: faster than anyone reads them.
 */
const IDENTIFY_EVERY = 3;
/**
 * How much a box from a boxes-only tick must overlap a box from the last
 * identify tick before it inherits its label. Same measure the stored-media
 * handoff uses; 0.3 tolerates a face moving between the two ticks and still
 * refuses to move a name onto a different face.
 */
const LABEL_MIN_IOU = 0.3;

/** Backend stage timings, in the order the pipeline runs them. */
const STAGES: readonly (readonly [keyof LiveTimings, string])[] = [
  ["decode", "Decode"],
  ["detect", "Detect"],
  ["quality_align", "Align"],
  ["embed", "Embed"],
  ["match", "Match"],
];

type Phase = "idle" | "starting" | "running" | "hidden" | "stopped";

type Metrics = {
  /** Backend-reported processing time for the last frame. */
  elapsedMs: number;
  /** Wall time from encode to parsed response, so the operator sees their own cost. */
  roundTripMs: number;
  /** Frames actually matched per second, measured over the last few ticks. */
  effectiveFps: number;
  /** Ticks skipped because the previous frame was still in flight. */
  dropped: number;
  /** Per-stage backend time for the last frame, and which cadence produced it. */
  timings: LiveTimings | null;
  identified: boolean;
};

const NO_METRICS: Metrics = {
  elapsedMs: 0,
  roundTripMs: 0,
  effectiveFps: 0,
  dropped: 0,
  timings: null,
  identified: false,
};

/**
 * The working resolution of the last frame, which is also the resolution a
 * click stores, because both are the same encode. `source*` is the share's own
 * size, so the view can say when the pixel budget had to shrink it.
 */
type FrameGeometry = {
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
};

function faceRect(face: LiveFace): Rect {
  return { x: face.x, y: face.y, w: face.w, h: face.h };
}

function faceLabel(face: LiveFace, identity: LiveFace | null): string {
  if (!face.quality_passed) {
    const reason = face.quality_reasons[0] ?? "quality gate";
    return `quality gate: ${reason}`;
  }
  if (identity === null) {
    // A box from a cadence that did not ask, or a face the last identify pass
    // did not see. Either way the honest answer is "not yet", not "no match".
    return "identifying…";
  }
  const top =
    identity.candidates.find((candidate) => candidate.rank === 1) ?? identity.candidates[0];
  if (top === undefined) {
    return "unidentified · no match";
  }
  return `${top.name} · ${top.band} ${formatScore(top.score)}`;
}

function faceColor(face: LiveFace, identity: LiveFace | null): string {
  if (!face.quality_passed || identity === null) {
    return NO_BAND_COLOR;
  }
  const top =
    identity.candidates.find((candidate) => candidate.rank === 1) ?? identity.candidates[0];
  return top === undefined ? NO_BAND_COLOR : BAND_COLOR[top.band];
}

/**
 * The face in the last identify pass that this box is, or null.
 *
 * The two cadences encode at different resolutions, so both sides are
 * normalised to fractions of their own frame before they are compared — the
 * same trick the live-to-stored handoff uses.
 */
function identityFor(
  face: LiveFace,
  boxes: LiveMatchResult,
  identities: LiveMatchResult | null,
): LiveFace | null {
  if (identities === null) {
    return null;
  }
  const target = normalizeRect(faceRect(face), boxes.width, boxes.height);
  const candidates = identities.faces.map((other) =>
    normalizeRect(faceRect(other), identities.width, identities.height),
  );
  const hit = bestOverlapRect(candidates, target, LABEL_MIN_IOU);
  return hit === null ? null : (identities.faces[hit] ?? null);
}

export default function LiveView() {
  const cases = useResource<CaseList>("/api/cases");
  const caseId = useSelectedCase();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  /**
   * The last *identify* frame, and the only thing a click may persist.
   *
   * A boxes-only frame is a 1 MP q0.7 encode that no gallery ever saw; storing
   * it as evidence would put a crop in the template store that the quality
   * gate was never run against at that resolution. So it is never retained.
   */
  const frameRef = useRef<Frame | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [fps, setFps] = useState(DEFAULT_FPS);
  /** Latest result of either cadence: where the boxes on screen come from. */
  const [boxes, setBoxes] = useState<LiveMatchResult | null>(null);
  /** Latest identify result: where the labels come from, and what a click acts on. */
  const [identities, setIdentities] = useState<LiveMatchResult | null>(null);
  const [metrics, setMetrics] = useState<Metrics>(NO_METRICS);
  /**
   * Two error channels on purpose.
   *
   * `matchError` belongs to the sampling loop and clears on the next good
   * frame. `actionError` belongs to whatever the operator just did — a refused
   * save, a failed upload — and must survive; folding them into one state let
   * the loop wipe a refusal about 300 ms after the click, so the operator
   * clicked a face and saw nothing happen.
   */
  const [matchError, setMatchError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [persisting, setPersisting] = useState(false);
  const [geometry, setGeometry] = useState<FrameGeometry | null>(null);

  // A stored selection survives only while the backend still lists it; a
  // deleted case must never silently absorb the next saved frame.
  useEffect(() => {
    if (cases.state.phase === "ready") {
      reconcileSelectedCase(cases.state.data.items);
    }
  }, [cases.state]);

  const stop = useCallback(() => {
    const stream = streamRef.current;
    if (stream !== null) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video !== null) {
      video.srcObject = null;
    }
    // The view holds the last frame so a click can store the bytes the boxes
    // came from; a stopped session has no click left to serve.
    frameRef.current = null;
    setPhase((current) => (current === "idle" ? "idle" : "stopped"));
  }, []);

  async function start(): Promise<void> {
    setMatchError(null);
    setActionError(null);
    setPhase("starting");
    try {
      // The operator picks screen, window, or tab in the browser's own dialog.
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video !== null) {
        video.srcObject = stream;
        await video.play();
      }
      const track = stream.getVideoTracks()[0];
      if (track !== undefined) {
        // The operator can end the share from the browser's own control.
        track.addEventListener("ended", () => {
          setActionError("Screen share ended from the browser control.");
          stop();
        });
      }
      setBoxes(null);
      setIdentities(null);
      setMetrics(NO_METRICS);
      setGeometry(null);
      frameRef.current = null;
      // Start already paused when the tab is not visible: a hidden tab throttles
      // canvas encoding to about one frame a second, so sampling there would
      // only produce stale frames and burn battery.
      setPhase(document.hidden ? "hidden" : "running");
    } catch (failure) {
      streamRef.current = null;
      setPhase("idle");
      const name = failure instanceof DOMException ? failure.name : "";
      setActionError(
        name === "NotAllowedError"
          ? "Screen share permission denied. Click Start and choose a screen, window, or tab to watch."
          : name === "NotFoundError"
            ? "No screen source is available to share on this machine."
            : `Could not start the screen share: ${errorMessage(failure)}`,
      );
    }
  }

  // Pause sampling while the tab is hidden: a background tab must not burn CPU
  // or keep posting frames the operator cannot see.
  useEffect(() => {
    const onVisibility = (): void => {
      setPhase((current) => {
        if (document.hidden) {
          return current === "running" ? "hidden" : current;
        }
        return current === "hidden" ? "running" : current;
      });
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  // Release the share when the view unmounts, including the navigation a click
  // on a face performs.
  useEffect(() => stop, [stop]);

  // The sampling loop. One request in flight; a tick that overruns its budget
  // drops frames instead of queueing them, because a stale frame is worse than
  // a missing one while the operator drags a window around.
  useEffect(() => {
    if (phase !== "running") {
      return;
    }
    const controller = new AbortController();
    const canvas = new OffscreenCanvas(1, 1);
    const targetPeriod = 1_000 / fps;
    let stopped = false;
    let dropped = 0;
    let matched = 0;
    let tick = 0;
    /**
     * The true round trip of the last tick, not the backend's own figure.
     *
     * The backoff exists so a machine that cannot keep up stops queueing work,
     * and encode plus wire is most of what it cannot keep up with — sizing the
     * period off `elapsed_ms` made the loop blind to exactly the cost the two
     * cadences were introduced to cut.
     */
    let lastElapsed = 0;
    let windowStart = performance.now();

    const loop = async (): Promise<void> => {
      while (!stopped) {
        // The chosen rate is a ceiling; the measured round trip is the floor, so
        // the loop backs off on its own when the machine is busy.
        const period = Math.max(targetPeriod, MIN_PERIOD_MS, 2 * lastElapsed);
        const video = videoRef.current;
        if (video === null || video.readyState < 2) {
          await sleep(period);
          continue;
        }
        const tickStart = performance.now();
        try {
          const identify = tick % IDENTIFY_EVERY === 0;
          tick += 1;
          const frame = await encodeFrame(
            video,
            canvas,
            identify ? {} : { maxPixels: BOX_MAX_PIXELS, quality: BOX_QUALITY },
          );
          if (frame !== null) {
            const matchResult = await matchFrame(frame, caseId, controller.signal, { identify });
            if (stopped) {
              return;
            }
            setBoxes(matchResult);
            if (matchResult.identified) {
              // Only an identify frame is evidence, and only its geometry
              // describes what a click would store.
              frameRef.current = frame;
              setIdentities(matchResult);
              setGeometry({
                width: frame.width,
                height: frame.height,
                sourceWidth: frame.sourceWidth,
                sourceHeight: frame.sourceHeight,
              });
            }
            matched += 1;
            const spent = performance.now() - tickStart;
            lastElapsed = spent;
            // Ticks the requested rate wanted during this frame's round trip.
            dropped += Math.max(0, Math.floor(spent / targetPeriod) - 1);
            const sinceWindow = performance.now() - windowStart;
            setMetrics({
              elapsedMs: matchResult.elapsed_ms,
              roundTripMs: Math.round(spent),
              effectiveFps: sinceWindow > 0 ? (matched * 1_000) / sinceWindow : 0,
              dropped,
              timings: matchResult.timings,
              identified: matchResult.identified,
            });
            if (sinceWindow > 4_000) {
              // Short measurement window so the reading tracks the current rate.
              matched = 0;
              windowStart = performance.now();
            }
            setMatchError(null);
          }
        } catch (failure) {
          if (controller.signal.aborted || stopped) {
            return;
          }
          setMatchError(
            failure instanceof ApiError
              ? `Live match failed (${String(failure.status)}): ${failure.detail}`
              : `Live match failed: ${errorMessage(failure)}`,
          );
          // Back off after a failure rather than hammering a broken endpoint.
          await sleep(1_000);
        }
        const remaining = period - (performance.now() - tickStart);
        if (remaining > 0) {
          await sleep(remaining);
        }
      }
    };

    void loop();
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [caseId, fps, phase]);

  /**
   * Cursor position in CSS pixels relative to the overlay, or null when the
   * pointer is elsewhere. A ref rather than state: the pointer moves at the
   * display rate and every move would otherwise re-render the whole view; the
   * handler redraws the canvas directly instead.
   */
  const pointerRef = useRef<{ x: number; y: number } | null>(null);

  // Draw the overlay with the same letterbox and DPR mapping the stored media
  // overlay uses, so a box means the same thing in both views.
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
    if (boxes === null) {
      return;
    }
    const box = containLetterbox(boxes.width, boxes.height, cssWidth, cssHeight);
    if (box.scale <= 0) {
      return;
    }
    // Labels are on demand, not always on. A permanent caption over every box
    // covers the faces the operator is trying to look at, and the reasons are
    // long ("quality gate: width_below_min_embed"). The list beside the video
    // carries the same text for every face at once; the overlay captions only
    // the box under the cursor.
    const pointer = pointerRef.current;
    const hovered =
      pointer === null
        ? null
        : hitTestImageRects(boxes.faces.map(faceRect), pointer.x, pointer.y, box);

    for (const [index, face] of boxes.faces.entries()) {
      const identity = identityFor(face, boxes, identities);
      const rect = imageRectToCss(faceRect(face), box);
      const color = faceColor(face, identity);
      context.strokeStyle = color;
      context.lineWidth = index === hovered ? 3 : 2;
      // A face the gate rejected is dashed and muted: it is not being matched,
      // and the operator has to be able to see that at a glance.
      context.setLineDash(face.quality_passed ? [] : [6, 4]);
      context.strokeRect(rect.x, rect.y, rect.w, rect.h);
      context.setLineDash([]);
      if (index !== hovered) {
        continue;
      }

      const label = faceLabel(face, identity);
      context.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      const labelWidth = Math.min(context.measureText(label).width + 10, cssWidth - rect.x);
      const labelY = Math.max(0, rect.y - 21);
      context.fillStyle = "rgba(12, 14, 17, 0.88)";
      context.fillRect(rect.x, labelY, labelWidth, 21);
      context.fillStyle = color;
      context.fillText(label, rect.x + 5, labelY + 14, Math.max(0, labelWidth - 10));
    }
  }, [boxes, identities]);

  // `draw` changes identity on every result, and rebuilding the observer with
  // it disconnected and re-registered one per tick. The observer only ever
  // needs to call the latest `draw`, so it reads it from a ref and is
  // registered once per mount.
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    drawRef.current();
  }, [draw]);

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) {
      return;
    }
    const redraw = (): void => {
      drawRef.current();
    };
    const observer = new ResizeObserver(redraw);
    observer.observe(video);
    window.addEventListener("resize", redraw);
    redraw();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", redraw);
    };
  }, []);

  /**
   * Persist the clicked face, then hand off to the tag panel.
   *
   * This is the only route from a live face to an identity, because the live
   * endpoint stores nothing: the frame becomes evidence first, and the stored
   * detection is what gets tagged.
   *
   * It acts on the *identify* frame and the face the identify pass found there
   * — never on a boxes-only tick. Those bytes were never stored and the box
   * would be pointing into a frame that does not exist as evidence.
   */
  async function enrollFace(index: number): Promise<void> {
    const frame = frameRef.current;
    const face = boxes?.faces[index];
    if (boxes === null || face === undefined) {
      return;
    }
    if (frame === null || identities === null) {
      setActionError("Waiting for the first identify pass — nothing is stored yet.");
      return;
    }
    const identity = identityFor(face, boxes, identities);
    if (identity === null) {
      setActionError(
        "That face was not in the last identify pass, so there is no stored frame showing " +
          "it. Give it a moment and click again.",
      );
      return;
    }
    if (caseId === "") {
      setActionError(`${noCaseRefusal("saving a frame")} The picker is at the top of this view.`);
      return;
    }
    setPersisting(true);
    setActionError(null);
    try {
      const upload = await persistFrame(frame, caseId);
      navigate({
        view: "viewer",
        mediaId: upload.media_id,
        // The identify frame's own geometry: that is the media being stored.
        focus: normalizeRect(faceRect(identity), identities.width, identities.height),
      });
    } catch (failure) {
      setActionError(
        failure instanceof ApiError
          ? `Could not save the frame (${String(failure.status)}): ${failure.detail}`
          : `Could not save the frame: ${errorMessage(failure)}`,
      );
    } finally {
      setPersisting(false);
    }
  }

  function onCanvasClick(event: MouseEvent<HTMLCanvasElement>): void {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video === null || canvas === null || boxes === null) {
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    const box = containLetterbox(boxes.width, boxes.height, video.clientWidth, video.clientHeight);
    const hit = hitTestImageRects(
      boxes.faces.map(faceRect),
      event.clientX - bounds.left,
      event.clientY - bounds.top,
      box,
    );
    if (hit !== null) {
      void enrollFace(hit);
    }
  }

  function onCanvasPointerMove(event: MouseEvent<HTMLCanvasElement>): void {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    pointerRef.current = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    // Redraw straight from the handler: the pointer lives in a ref precisely so
    // moving it does not re-render the view at the display rate.
    drawRef.current();
  }

  function onCanvasPointerLeave(): void {
    pointerRef.current = null;
    drawRef.current();
  }

  const live = phase === "running" || phase === "hidden";
  const uncalibrated = boxes !== null && !boxes.auto_accept_allowed;
  const noCase = caseId === "";
  // Matching needs no case (it writes nothing); persisting does, so only the
  // persisting controls are gated and the destination is named on them.
  const selected =
    cases.state.phase === "ready"
      ? (cases.state.data.items.find((item) => item.id === caseId) ?? null)
      : null;
  const caseName = selected?.name ?? null;

  return (
    <>
      <div className="view-heading">
        <div>
          <h1>Live match</h1>
          <p className="tagline">
            Watch a screen, window, or tab and see who the gallery thinks is on it.
          </p>
        </div>
        <button type="button" onClick={live ? stop : () => void start()} className={live ? "" : "primary"}>
          {phase === "starting" ? "Waiting for the picker…" : live ? "Stop" : "Start"}
        </button>
      </div>

      <p className="notice live-safety">
        Live frames are never stored, never audited, and nothing can be identified or enrolled from
        them. Clicking a face saves that exact frame as evidence first, then opens it for tagging:{" "}
        {noCase
          ? "choose a case below before clicking, because nothing is saved without one."
          : caseName === null
            ? "it is filed to the selected case."
            : `it is filed to case “${caseName}”.`}
      </p>

      <WatchHelper />

      {selected !== null && (
        <CaseBasis key={selected.id} record={selected} onAmended={cases.reload} />
      )}

      <div className="live-controls panel">
        <Loaded state={cases.state} label="cases">
          {(caseList) => (
            <label>
              Case for saved frames
              <select
                value={caseId}
                onChange={(event) => {
                  setSelectedCase(event.currentTarget.value);
                  // The refusal they are reading is about to be untrue.
                  setActionError(null);
                }}
                disabled={caseList.items.length === 0}
              >
                <option value="">
                  {caseList.items.length === 0 ? "No cases exist" : "Choose a case…"}
                </option>
                {caseList.items.map((item) => (
                  <option key={item.id} value={item.id}>{item.name}</option>
                ))}
              </select>
            </label>
          )}
        </Loaded>
        <label>
          Sample rate
          <select value={fps} onChange={(event) => setFps(Number(event.currentTarget.value))}>
            {FPS_CHOICES.map((choice) => (
              <option key={choice} value={choice}>{choice} fps</option>
            ))}
          </select>
        </label>
        <dl className="live-hud" aria-label="Live sampling metrics" aria-live="off">
          <div><dt>Backend</dt><dd className="mono">{metrics.elapsedMs} ms</dd></div>
          <div><dt>Round trip</dt><dd className="mono">{metrics.roundTripMs} ms</dd></div>
          <div><dt>Effective</dt><dd className="mono">{metrics.effectiveFps.toFixed(1)} fps</dd></div>
          <div><dt>Dropped</dt><dd className="mono">{metrics.dropped}</dd></div>
          <div><dt>Gallery</dt><dd className="mono">{boxes?.gallery_persons ?? "—"}</dd></div>
          <div>
            <dt>Frame</dt>
            <dd className="mono">
              {geometry === null ? "—" : `${String(geometry.width)}×${String(geometry.height)}`}
            </dd>
          </div>
          {/* Which cadence produced the boxes on screen, and what each backend stage
              cost on it. Without this the operator cannot see which stage hurts, and
              a boxes-only tick reading 0 ms of embed is the whole point of the split. */}
          <div>
            <dt>Pass</dt>
            <dd className="mono">{metrics.identified ? "identify" : "boxes"}</dd>
          </div>
          {STAGES.map(([key, label]) => (
            <div key={key}>
              <dt>{label}</dt>
              <dd className="mono">
                {metrics.timings === null ? "—" : `${metrics.timings[key].toFixed(1)} ms`}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      {geometry !== null && geometry.width !== geometry.sourceWidth && (
        <p className="notice" role="status">
          Identify frames are capped at {(FRAME_MAX_PIXELS / 1_000_000).toFixed(1)} megapixels
          to keep sampling interactive, and the shared surface is {geometry.sourceWidth}×
          {geometry.sourceHeight}, so both the match and the stored evidence use{" "}
          {geometry.width}×{geometry.height} — the same pixels either way. Boxes-only
          frames are smaller still, and are never stored.
        </p>
      )}
      {phase === "hidden" && (
        <p className="notice" role="status">Sampling paused while this tab is in the background.</p>
      )}
      {uncalibrated && (
        <p className="notice" role="status">
          No calibrated threshold set is active
          {boxes?.auto_accept_reason === null ? "" : `: ${boxes?.auto_accept_reason ?? ""}`}. Every
          match here is a candidate for an operator decision, never an acceptance.
        </p>
      )}
      {actionError !== null && <p className="notice error" role="alert">{actionError}</p>}
      {matchError !== null && <p className="notice error" role="alert">{matchError}</p>}
      {persisting && <p className="notice" role="status">Saving the frame as evidence…</p>}

      <div className="viewer-layout">
        <main className="viewer-main">
          <div className="image-stage">
            <video ref={videoRef} muted playsInline aria-label="Shared screen being matched" />
            <canvas
              ref={canvasRef}
              onClick={onCanvasClick}
              onMouseMove={onCanvasPointerMove}
              onMouseLeave={onCanvasPointerLeave}
              aria-label="Live face overlay. Hover a box to read its label. Use the list beside the video to save and tag a face."
            />
          </div>
          {phase === "idle" && (
            <p className="notice">
              Click Start, then choose the screen, window, or tab showing the face you want to
              enroll.
            </p>
          )}
        </main>

        <aside className="tag-panel panel">
          <h2>Faces in frame</h2>
          {noCase ? (
            <p className="notice" role="status">
              {noCaseRefusal("saving a frame")} The picker is above. Matching keeps running either
              way — it stores nothing.
            </p>
          ) : (
            <p className="muted compact">
              Saved frames are filed to <strong>{caseName ?? "the selected case"}</strong>.
            </p>
          )}
          {boxes === null ? (
            <p className="muted">No frame has been matched yet.</p>
          ) : boxes.faces.length === 0 ? (
            <p className="muted">No face detected in the current frame.</p>
          ) : (
            <ul className="candidate-list">
              {boxes.faces.map((face, index) => {
                const identity = identityFor(face, boxes, identities);
                const top = identity?.candidates.find((candidate) => candidate.rank === 1);
                return (
                  <li className="live-face" key={`${String(index)}-${String(Math.round(face.x))}`}>
                    <div className="live-face-head">
                      {!face.quality_passed ? (
                        <span className="muted">Not matched · quality gate</span>
                      ) : identity === null ? (
                        <span className="muted">Identifying…</span>
                      ) : top === undefined ? (
                        <span>Unidentified</span>
                      ) : (
                        <>
                          <span>{top.name}</span>
                          <BandPill band={top.band} score={top.score} />
                        </>
                      )}
                    </div>
                    <p className="compact muted">
                      det {formatScore(face.det_score)}
                      {face.quality_reasons.length > 0 && ` · ${face.quality_reasons.join(", ")}`}
                    </p>
                    <button
                      type="button"
                      disabled={persisting || noCase || identity === null}
                      {...(noCase ? { title: noCaseRefusal("saving a frame") } : {})}
                      onClick={() => void enrollFace(index)}
                    >
                      {noCase
                        ? "Save frame and tag this face"
                        : `Save frame to ${caseName ?? "the selected case"} and tag this face`}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </aside>
      </div>
    </>
  );
}
