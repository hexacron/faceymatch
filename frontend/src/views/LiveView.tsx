import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";

import { ApiError, errorMessage } from "../api/client";
import type { CaseList, LiveFace, LiveMatchResult } from "../api/types";
import { BandPill } from "../components/BandPill";
import { CaseBasis } from "../components/CaseBasis";
import { Loaded } from "../components/Loading";
import { BAND_COLOR, formatScore, NO_BAND_COLOR } from "../lib/display";
import {
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
};

const NO_METRICS: Metrics = { elapsedMs: 0, roundTripMs: 0, effectiveFps: 0, dropped: 0 };

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

function faceLabel(face: LiveFace): string {
  if (!face.quality_passed) {
    const reason = face.quality_reasons[0] ?? "quality gate";
    return `quality gate: ${reason}`;
  }
  const top = face.candidates.find((candidate) => candidate.rank === 1) ?? face.candidates[0];
  if (top === undefined) {
    return "unidentified · no match";
  }
  return `${top.name} · ${top.band} ${formatScore(top.score)}`;
}

function faceColor(face: LiveFace): string {
  if (!face.quality_passed) {
    return NO_BAND_COLOR;
  }
  const top = face.candidates.find((candidate) => candidate.rank === 1) ?? face.candidates[0];
  return top === undefined ? NO_BAND_COLOR : BAND_COLOR[top.band];
}

export default function LiveView() {
  const cases = useResource<CaseList>("/api/cases");
  const caseId = useSelectedCase();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  /** The frame the boxes on screen came from: what a click persists. */
  const frameRef = useRef<Frame | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [fps, setFps] = useState(DEFAULT_FPS);
  const [result, setResult] = useState<LiveMatchResult | null>(null);
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
      setResult(null);
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
    let lastElapsed = 0;
    let windowStart = performance.now();

    const loop = async (): Promise<void> => {
      while (!stopped) {
        // The chosen rate is a ceiling; measured backend time is the floor, so
        // the loop backs off on its own when the machine is busy (the backend
        // measures ~9 ms for an empty frame and ~40 ms for three faces).
        const period = Math.max(targetPeriod, MIN_PERIOD_MS, 2 * lastElapsed);
        const video = videoRef.current;
        if (video === null || video.readyState < 2) {
          await sleep(period);
          continue;
        }
        const tickStart = performance.now();
        try {
          const frame = await encodeFrame(video, canvas);
          if (frame !== null) {
            const matchResult = await matchFrame(frame, caseId, controller.signal);
            if (stopped) {
              return;
            }
            frameRef.current = frame;
            setResult(matchResult);
            setGeometry({
              width: frame.width,
              height: frame.height,
              sourceWidth: frame.sourceWidth,
              sourceHeight: frame.sourceHeight,
            });
            matched += 1;
            lastElapsed = matchResult.elapsed_ms;
            const spent = performance.now() - tickStart;
            // Ticks the requested rate wanted during this frame's round trip.
            dropped += Math.max(0, Math.floor(spent / targetPeriod) - 1);
            const sinceWindow = performance.now() - windowStart;
            setMetrics({
              elapsedMs: matchResult.elapsed_ms,
              roundTripMs: Math.round(spent),
              effectiveFps: sinceWindow > 0 ? (matched * 1_000) / sinceWindow : 0,
              dropped,
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
    if (result === null) {
      return;
    }
    const box = containLetterbox(result.width, result.height, cssWidth, cssHeight);
    if (box.scale <= 0) {
      return;
    }
    for (const face of result.faces) {
      const rect = imageRectToCss(faceRect(face), box);
      const color = faceColor(face);
      context.strokeStyle = color;
      context.lineWidth = 2;
      // A face the gate rejected is dashed and muted: it is not being matched,
      // and the operator has to be able to see that at a glance.
      context.setLineDash(face.quality_passed ? [] : [6, 4]);
      context.strokeRect(rect.x, rect.y, rect.w, rect.h);
      context.setLineDash([]);

      const label = faceLabel(face);
      context.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      const labelWidth = Math.min(context.measureText(label).width + 10, cssWidth - rect.x);
      const labelY = Math.max(0, rect.y - 21);
      context.fillStyle = "rgba(12, 14, 17, 0.88)";
      context.fillRect(rect.x, labelY, labelWidth, 21);
      context.fillStyle = color;
      context.fillText(label, rect.x + 5, labelY + 14, Math.max(0, labelWidth - 10));
    }
  }, [result]);

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) {
      return;
    }
    const observer = new ResizeObserver(draw);
    observer.observe(video);
    window.addEventListener("resize", draw);
    draw();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", draw);
    };
  }, [draw]);

  /**
   * Persist the clicked face, then hand off to the tag panel.
   *
   * This is the only route from a live face to an identity, because the live
   * endpoint stores nothing: the frame becomes evidence first, and the stored
   * detection is what gets tagged.
   */
  async function enrollFace(index: number): Promise<void> {
    const frame = frameRef.current;
    const current = result;
    const face = current?.faces[index];
    if (frame === null || current === null || face === undefined) {
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
        focus: normalizeRect(faceRect(face), current.width, current.height),
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
    if (video === null || canvas === null || result === null) {
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    const box = containLetterbox(result.width, result.height, video.clientWidth, video.clientHeight);
    const hit = hitTestImageRects(
      result.faces.map(faceRect),
      event.clientX - bounds.left,
      event.clientY - bounds.top,
      box,
    );
    if (hit !== null) {
      void enrollFace(hit);
    }
  }

  const live = phase === "running" || phase === "hidden";
  const uncalibrated = result !== null && !result.auto_accept_allowed;
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
          <div><dt>Gallery</dt><dd className="mono">{result?.gallery_persons ?? "—"}</dd></div>
          <div>
            <dt>Frame</dt>
            <dd className="mono">
              {geometry === null ? "—" : `${String(geometry.width)}×${String(geometry.height)}`}
            </dd>
          </div>
        </dl>
      </div>

      {geometry !== null && geometry.width !== geometry.sourceWidth && (
        <p className="notice" role="status">
          Frames are capped at {(FRAME_MAX_PIXELS / 1_000_000).toFixed(1)} megapixels to keep
          sampling interactive, and the shared surface is {geometry.sourceWidth}×
          {geometry.sourceHeight}, so both the match and the stored evidence use{" "}
          {geometry.width}×{geometry.height} — the same pixels either way.
        </p>
      )}
      {phase === "hidden" && (
        <p className="notice" role="status">Sampling paused while this tab is in the background.</p>
      )}
      {uncalibrated && (
        <p className="notice" role="status">
          No calibrated threshold set is active
          {result?.auto_accept_reason === null ? "" : `: ${result?.auto_accept_reason ?? ""}`}. Every
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
              aria-label="Live face overlay. Use the list beside the video to save and tag a face."
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
          {result === null ? (
            <p className="muted">No frame has been matched yet.</p>
          ) : result.faces.length === 0 ? (
            <p className="muted">No face detected in the current frame.</p>
          ) : (
            <ul className="candidate-list">
              {result.faces.map((face, index) => {
                const top = face.candidates.find((candidate) => candidate.rank === 1);
                return (
                  <li className="live-face" key={`${String(index)}-${String(Math.round(face.x))}`}>
                    <div className="live-face-head">
                      {face.quality_passed ? (
                        top === undefined ? (
                          <span>Unidentified</span>
                        ) : (
                          <>
                            <span>{top.name}</span>
                            <BandPill band={top.band} score={top.score} />
                          </>
                        )
                      ) : (
                        <span className="muted">Not matched · quality gate</span>
                      )}
                    </div>
                    <p className="compact muted">
                      det {formatScore(face.det_score)}
                      {face.quality_reasons.length > 0 && ` · ${face.quality_reasons.join(", ")}`}
                    </p>
                    <button
                      type="button"
                      disabled={persisting || noCase}
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
