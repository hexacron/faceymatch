/**
 * Frame plumbing for live mode.
 *
 * Live matching is advisory: `POST /api/live/match` stores no row and appends
 * no audit entry, so nothing seen here can be tagged or enrolled. The only
 * bridge to identity is `persistFrame`, which uploads the exact bytes that
 * were matched through the ordinary ingest path, turning them into hashed,
 * audited evidence the tag panel can act on.
 *
 * ONE ENCODE PER TICK, ONE RESOLUTION. The matched frame and the stored frame
 * are the same blob, so anything the gate accepted while the overlay was drawn
 * is still accepted once it is evidence. Two encodes at two resolutions put a
 * green box on a face whose stored crop then failed the gate.
 *
 * The sampler deliberately holds no queue. One request is in flight at a time
 * and a frame that arrives while the previous one is still out is dropped, not
 * buffered: a stale frame is worse than a missing one when the operator is
 * dragging a window around.
 */

import { postForm } from "../api/client";
import type { LiveMatchResult, MediaUpload } from "../api/types";
import { ingestFile } from "./ingest";

/**
 * Pixel budget for one frame, matched and stored.
 *
 * There is exactly one encode per tick and both consumers get the same bytes,
 * so this number is the only resolution in live mode. It is set from the
 * interactive budget: a 3 fps tick allows 333 ms, and measured on this machine
 * (M5, CoreML, three faces, JPEG q0.92) a frame costs encode + round trip of
 * 133 ms at 5.6 MP, 145 ms at 7.35 MP and 271 ms at 16.1 MP. 16.8 MP is
 * therefore the largest surface that still samples at 3 fps, which covers
 * every built-in and 5K display; only a pathological multi-monitor surface is
 * downscaled, and the view says so when it happens.
 *
 * Higher is better evidence: the crop the gallery template is built from is a
 * real face at screen resolution rather than a 105 px thumbnail.
 */
export const FRAME_MAX_PIXELS = 16_800_000;

/**
 * JPEG, not PNG, and this is the evidence quality too.
 *
 * Lossless at full resolution costs 218 ms to encode and 128 ms on the wire
 * (measured, 2750x1692), which blows the 333 ms tick on its own; q0.92 costs
 * 34 ms and 69 ms for the same pixels and matches identically (top candidate
 * 0.9648 vs 0.9630). Storing the same bytes that were matched is worth more
 * than the last fraction of a percent of encoder fidelity.
 */
export const FRAME_QUALITY = 0.92;

/** Pause the sampling loop. Named because the loop awaits it in three places. */
export function sleep(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

/**
 * One frame, one encode, two consumers.
 *
 * `blob` is posted to the match endpoint and, if the operator clicks a face,
 * uploaded byte-for-byte as evidence. `width`/`height` are the space the boxes
 * come back in AND the dimensions of the stored media, so a box drawn over the
 * live overlay lands on the same pixels once stored.
 */
export type Frame = {
  blob: Blob;
  width: number;
  height: number;
  /** Capture size, to report a frame the pixel budget had to shrink. */
  sourceWidth: number;
  sourceHeight: number;
};

/**
 * Encode the video's current frame once, at the working resolution.
 *
 * The advisory match and the stored evidence MUST be the same pixels: two
 * encodes at two resolutions let the overlay draw a face as quality-passing
 * and the stored crop then fail the same gate, which is the UI promising
 * something the evidence path cannot honour.
 *
 * The canvas is passed in and reused for every tick: allocating one per frame
 * at 3 fps would churn a few hundred MB of backing store a minute.
 */
export async function encodeFrame(
  video: HTMLVideoElement,
  canvas: OffscreenCanvas,
): Promise<Frame | null> {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (sourceWidth === 0 || sourceHeight === 0) {
    return null;
  }
  const context = canvas.getContext("2d");
  if (context === null) {
    return null;
  }
  // Area, not long edge: cost tracks pixels, and a two-monitor surface is wide
  // rather than tall.
  const scale = Math.min(1, Math.sqrt(FRAME_MAX_PIXELS / (sourceWidth * sourceHeight)));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  // Straight from the video: no intermediate bitmap to retain, because the
  // encoded blob itself is what a click stores.
  context.drawImage(video, 0, 0, width, height);
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: FRAME_QUALITY });
  return { blob, width, height, sourceWidth, sourceHeight };
}

/**
 * Match one frame. `caseId` does not scope the gallery (persons and templates
 * are global); it is sent so a stale case id surfaces here as a 404 instead of
 * mattering nowhere.
 */
export async function matchFrame(
  frame: Frame,
  caseId: string,
  signal: AbortSignal,
): Promise<LiveMatchResult> {
  const form = new FormData();
  form.set("frame", frame.blob, "frame.jpg");
  if (caseId !== "") {
    form.set("case_id", caseId);
  }
  return postForm<LiveMatchResult>("/api/live/match", form, signal);
}

/**
 * Persist the frame the operator clicked, so the face they saw becomes
 * evidence: content-addressed, audited, and taggable.
 *
 * The same bytes that were matched, re-uploaded, not re-encoded: the stored
 * media therefore hashes what the overlay was drawn from, decodes to the same
 * pixels the quality gate already accepted, and carries the box coordinates
 * the live result reported, so a face the operator saw pass cannot fail once
 * stored.
 */
export async function persistFrame(frame: Frame, caseId: string): Promise<MediaUpload> {
  const stamp = new Date().toISOString().replaceAll(/[:.]/g, "-").replace("Z", "");
  const file = new File([frame.blob], `live-${stamp}.jpg`, { type: "image/jpeg" });
  return ingestFile(caseId, file, null);
}
