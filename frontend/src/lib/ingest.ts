/**
 * Every way an image gets into a case, and the one flow that follows.
 *
 * Four entry points — file picker, clipboard paste, drag-and-drop, screen
 * capture — reduce to two requests (`POST /api/media` multipart,
 * `POST /api/capture` JSON), report through one notice stack, and land the
 * operator on the media detail route so detected boxes appear exactly where
 * they are about to tag them.
 *
 * The selected case lives here rather than inside `MediaLibraryView` because a
 * paste is global: Cmd+V has to work on the person page too, and it still has
 * to know which case the bytes belong to.
 */

import { useEffect, useSyncExternalStore } from "react";

import { ApiError, errorMessage, postForm, postJson } from "../api/client";
import type {
  CaptureMode,
  CaptureRequest,
  CaptureStatus,
  FolderEnroll,
  Health,
  MediaImport,
  MediaUpload,
} from "../api/types";
import { truncateHash } from "./display";
import { navigate } from "./router";

/* ------------------------------------------------------------ selected case */

const CASE_STORAGE_KEY = "faceymatch.selected-case";

let selectedCase = "";
try {
  selectedCase = window.localStorage.getItem(CASE_STORAGE_KEY) ?? "";
} catch {
  // Storage can be blocked; an in-memory selection still works for the session.
}

const caseListeners = new Set<() => void>();

export function setSelectedCase(caseId: string): void {
  if (caseId === selectedCase) {
    return;
  }
  selectedCase = caseId;
  try {
    window.localStorage.setItem(CASE_STORAGE_KEY, caseId);
  } catch {
    // Losing persistence is not worth failing an upload over.
  }
  for (const listener of caseListeners) {
    listener();
  }
}

// useSyncExternalStore needs a stable subscribe identity, so these small
// subscribe helpers are the callback, not a rename of one.
function subscribeCase(onChange: () => void): () => void {
  caseListeners.add(onChange);
  return () => {
    caseListeners.delete(onChange);
  };
}

/**
 * Forget a stored selection the backend no longer lists.
 *
 * Fail closed on purpose: nothing here picks a case on the operator's behalf.
 * A write on an ingest path is not a UI preference, it is the creation of
 * evidence against a case whose `authorization_basis` is a claim about what
 * that data is, so the destination is always an explicit choice.
 */
export function reconcileSelectedCase(known: readonly { id: string }[]): void {
  if (selectedCase !== "" && !known.some((item) => item.id === selectedCase)) {
    setSelectedCase("");
  }
}

export function useSelectedCase(): string {
  return useSyncExternalStore(subscribeCase, () => selectedCase);
}

/* ------------------------------------------------------------------ notices */

export type IngestTone = "success" | "info" | "error";

export type IngestNotice = {
  id: number;
  tone: IngestTone;
  text: string;
  /** Set when the operator can jump straight to the media this notice is about. */
  mediaId: string | null;
};

/** Keep the stack short: a multi-file drop should not bury the app in toasts. */
const MAX_NOTICES = 5;
const AUTO_DISMISS_MS = 9_000;

let notices: readonly IngestNotice[] = [];
const noticeListeners = new Set<() => void>();
let nextNoticeId = 1;

function publishNotices(next: readonly IngestNotice[]): void {
  notices = next;
  for (const listener of noticeListeners) {
    listener();
  }
}

function subscribeNotices(onChange: () => void): () => void {
  noticeListeners.add(onChange);
  return () => {
    noticeListeners.delete(onChange);
  };
}

export function useIngestNotices(): readonly IngestNotice[] {
  return useSyncExternalStore(subscribeNotices, () => notices);
}

export function dismissIngestNotice(id: number): void {
  publishNotices(notices.filter((notice) => notice.id !== id));
}

function notify(tone: IngestTone, text: string, mediaId: string | null = null): void {
  const id = nextNoticeId;
  nextNoticeId += 1;
  publishNotices([...notices, { id, tone, text, mediaId }].slice(-MAX_NOTICES));
  if (tone !== "error") {
    // Failures stay until dismissed; a success the operator already saw does not.
    window.setTimeout(() => {
      dismissIngestNotice(id);
    }, AUTO_DISMISS_MS);
  }
}

/* -------------------------------------------------------------- file naming */

/**
 * The backend picks its decoder from the suffix
 * (`app/pipeline/decode.require_supported_image`), so anything posted needs a
 * name it recognises. A pasted bitmap arrives as `image.png` at best and
 * unnamed at worst.
 */
const SUPPORTED_SUFFIX: Readonly<Record<string, true>> = {
  ".jpg": true,
  ".jpeg": true,
  ".png": true,
  ".webp": true,
  ".heic": true,
  ".heif": true,
};

const SUFFIX_FOR_TYPE: Readonly<Record<string, string>> = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp",
  "image/heic": ".heic",
  "image/heif": ".heif",
};

function suffixOfName(name: string): string | null {
  const dot = name.lastIndexOf(".");
  if (dot <= 0) {
    return null;
  }
  const suffix = name.slice(dot).toLowerCase();
  return SUPPORTED_SUFFIX[suffix] === true ? suffix : null;
}

/** `20260912-142233`: sortable, filesystem-safe, local time like the rest of the UI. */
function timestampSlug(): string {
  const now = new Date();
  const pad = (value: number): string => String(value).padStart(2, "0");
  return (
    `${String(now.getFullYear())}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  );
}

export type ImagePayload = {
  files: readonly File[];
  /** Names of transferred items we will not post, so the operator is told why. */
  rejected: readonly string[];
};

function transferFiles(data: DataTransfer): readonly File[] {
  if (data.files.length > 0) {
    return Array.from(data.files);
  }
  // Some clipboard payloads only expose `items` rather than `files`.
  const files: File[] = [];
  for (const item of Array.from(data.items)) {
    if (item.kind !== "file") {
      continue;
    }
    const file = item.getAsFile();
    if (file !== null) {
      files.push(file);
    }
  }
  return files;
}

/**
 * Pull the image files out of a clipboard or drop payload.
 *
 * `renameStem` set (clipboard: no meaningful filename) renames every file;
 * null keeps the operator's own filenames when the suffix is usable.
 */
export function collectImages(data: DataTransfer, renameStem: string | null): ImagePayload {
  const files: File[] = [];
  const rejected: string[] = [];
  for (const file of transferFiles(data)) {
    const named = suffixOfName(file.name);
    const suffix = named ?? SUFFIX_FOR_TYPE[file.type.toLowerCase()] ?? null;
    if (suffix === null) {
      rejected.push(file.name !== "" ? file.name : file.type !== "" ? file.type : "unnamed item");
      continue;
    }
    if (renameStem === null && named !== null) {
      files.push(file);
      continue;
    }
    const ordinal = files.length === 0 ? "" : `-${String(files.length + 1)}`;
    // `new File` wraps the same blob: it copies no pixels.
    files.push(
      new File([file], `${renameStem ?? "image"}-${timestampSlug()}${ordinal}${suffix}`, {
        type: file.type,
      }),
    );
  }
  return { files, rejected };
}

/* ------------------------------------------------------------- ingest paths */

/**
 * One refusal, shaped the same way for every path, so the operator learns a
 * single rule instead of four phrasings: the destination is theirs to choose.
 * `action` names the write being refused ("saving a frame", "adding an image").
 */
export function noCaseRefusal(action: string): string {
  return `Select a case before ${action}: evidence has to belong to one.`;
}

/** Where the picker lives, for a refusal that can surface away from it. */
export const CASE_PICKER_HINT = "Choose one in the Media tab.";

/**
 * The single request builder for `POST /api/media`, and the authoritative
 * refusal when no case is chosen.
 *
 * Every ingest path funnels through here — picker, paste, drop, and the live
 * frame that `persistFrame` turns into evidence — so a missing case cannot
 * reach the wire from any of them. Callers still check first to report it in
 * their own idiom (a toast, or the live view's inline error); this throw is the
 * backstop, not the message the operator normally sees.
 */
export async function ingestFile(
  caseId: string,
  file: File,
  sourceUrl: string | null,
  acquisition: { mode: "screen_capture"; captureMode: "screen" } | null = null,
): Promise<MediaUpload> {
  if (caseId === "") {
    throw new Error(noCaseRefusal("adding an image"));
  }
  const form = new FormData();
  form.set("case_id", caseId);
  form.set("file", file, file.name);
  if (sourceUrl !== null && sourceUrl !== "") {
    form.set("source_url", sourceUrl);
  }
  // Left unset for an ordinary upload: the backend defaults to `upload` and refuses a
  // capture_mode without a screen_capture acquisition.
  if (acquisition !== null) {
    form.set("acquisition", acquisition.mode);
    form.set("capture_mode", acquisition.captureMode);
  }
  return postForm<MediaUpload>("/api/media", form);
}

/**
 * The two folder paths, kept here with every other ingest request.
 *
 * Neither touches the notice stack: both return a report worth reading, which
 * the folder panel renders itself rather than flattening to one line.
 */
export async function importFolder(caseId: string, folderPath: string): Promise<MediaImport> {
  return postJson<MediaImport>("/api/media/import", { case_id: caseId, folder_path: folderPath });
}

export async function enrollFolder(
  caseId: string,
  folderPath: string,
  reason: string,
): Promise<FolderEnroll> {
  return postJson<FolderEnroll>("/api/persons/enroll_folder", {
    case_id: caseId,
    folder_path: folderPath,
    reason,
  });
}

/**
 * Upload each file in turn, report each one, and land on the faces when the
 * batch produced exactly one media. A multi-file drop stays put: navigating
 * away would hide the other results.
 */
export async function ingestFiles(
  caseId: string,
  payload: ImagePayload,
  sourceUrl: string | null,
): Promise<void> {
  if (caseId === "") {
    notify("error", `${noCaseRefusal("adding an image")} ${CASE_PICKER_HINT}`);
    return;
  }
  for (const name of payload.rejected) {
    notify("error", `${name}: not a supported image (JPEG, PNG, WebP, HEIC).`);
  }
  const landed: string[] = [];
  for (const file of payload.files) {
    try {
      const result = await ingestFile(caseId, file, sourceUrl);
      landed.push(result.media_id);
      notify(
        result.reused ? "info" : "success",
        result.reused
          ? `${file.name}: identical bytes already ingested (${truncateHash(result.sha256)}).`
          : `${file.name}: ingested, processing queued.`,
        result.media_id,
      );
    } catch (error) {
      notify("error", `${file.name}: ${errorMessage(error)}`);
    }
  }
  const only = landed.length === 1 ? landed[0] : undefined;
  if (only !== undefined) {
    navigate({ view: "viewer", mediaId: only });
  }
}

/* ------------------------------------------------------------ screen capture */

export type CaptureCapability = {
  supported: boolean;
  /** Why capture is unavailable; null when it is available. */
  reason: string | null;
};

let captureCapability: CaptureCapability | null = null;
const capabilityListeners = new Set<() => void>();

function publishCapability(next: CaptureCapability): void {
  if (next.supported === captureCapability?.supported && next.reason === captureCapability.reason) {
    return;
  }
  captureCapability = next;
  for (const listener of capabilityListeners) {
    listener();
  }
}

function subscribeCapability(onChange: () => void): () => void {
  capabilityListeners.add(onChange);
  return () => {
    capabilityListeners.delete(onChange);
  };
}

/** Capture support, or null while it is still unknown. */
export function useCaptureCapability(): CaptureCapability | null {
  return useSyncExternalStore(subscribeCapability, () => captureCapability);
}

/**
 * Adopt the capture capability `GET /api/healthz` reports.
 *
 * A bundle can outlive the backend that serves it (and this field ships with
 * the capture endpoint itself), so an absent `capture` leaves capability
 * unknown: the button stays enabled and a real 503 supplies the reason. The
 * widened local is that tolerance, not a cast around the contract.
 */
export function syncCaptureCapability(health: Health): void {
  const capture: CaptureStatus | undefined = health.capture;
  if (capture === undefined) {
    return;
  }
  publishCapability({ supported: capture.available, reason: capture.reason });
}

/**
 * Ask the backend to run a screen grab and ingest it.
 *
 * Refuses before the request when no case is chosen: a grab is evidence too,
 * and `POST /api/capture` writes it the moment the operator releases the
 * crosshair.
 *
 * A cancelled grab (Esc over the crosshair) is the operator changing their
 * mind, not a failure, so it reports nothing at all.
 */
export async function captureFromScreen(caseId: string, mode: CaptureMode): Promise<void> {
  if (caseId === "") {
    notify("error", noCaseRefusal("capturing the screen"));
    return;
  }
  const body: CaptureRequest = { case_id: caseId, mode, source_url: null };
  try {
    const result = await postJson<MediaUpload>("/api/capture", body);
    publishCapability({ supported: true, reason: null });
    notify(
      result.reused ? "info" : "success",
      result.reused
        ? `Capture matches media already ingested (${truncateHash(result.sha256)}).`
        : "Screen capture ingested, processing queued.",
      result.media_id,
    );
    navigate({ view: "viewer", mediaId: result.media_id });
  } catch (error) {
    if (!(error instanceof ApiError)) {
      notify("error", `Screen capture failed: ${errorMessage(error)}`);
      return;
    }
    if (error.status === 400 && error.detail.toLowerCase().includes("cancel")) {
      return;
    }
    if (error.status === 503) {
      publishCapability({ supported: false, reason: error.detail });
      notify("error", `Screen capture unavailable: ${error.detail}`);
      return;
    }
    notify("error", `Screen capture failed: ${error.detail}`);
  }
}

/* ---------------------------------------------------------------- app hooks */

/**
 * App-wide clipboard ingest plus a drop guard.
 *
 * A text paste is left alone entirely: no notice, no error, and text fields
 * keep their own paste behaviour. The drop guard only stops the browser from
 * navigating to a file dropped outside the library dropzone — that zone calls
 * `preventDefault` first, so a handled drop is untouched here.
 */
export function useGlobalIngest(): void {
  const caseId = useSelectedCase();

  useEffect(() => {
    const onPaste = (event: ClipboardEvent): void => {
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return;
      }
      const data = event.clipboardData;
      if (data === null) {
        return;
      }
      const payload = collectImages(data, "clipboard");
      if (payload.files.length === 0) {
        // No image on the clipboard (plain text, a URL): do nothing at all.
        return;
      }
      event.preventDefault();
      void ingestFiles(caseId, payload, null);
    };

    const guard = (event: DragEvent): void => {
      if (!event.defaultPrevented) {
        event.preventDefault();
      }
    };

    window.addEventListener("paste", onPaste);
    window.addEventListener("dragover", guard);
    window.addEventListener("drop", guard);
    return () => {
      window.removeEventListener("paste", onPaste);
      window.removeEventListener("dragover", guard);
      window.removeEventListener("drop", guard);
    };
  }, [caseId]);
}
