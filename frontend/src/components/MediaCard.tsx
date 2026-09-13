import { useState } from "react";

import { ApiError, deleteJson, errorMessage } from "../api/client";
import type { Media, MediaPurgeResult } from "../api/types";
import { truncateHash } from "../lib/display";
import { hrefFor } from "../lib/router";

import type { GalleryNotice } from "./GalleryControls";

/**
 * One file in the library, as the picture it is.
 *
 * The preview is `GET /api/media/{id}/thumbnail`, a derived downscale of the
 * stored original — not evidence, and nothing can be enrolled or tagged from it
 * (invariant 13); the tile links to the media page, where the decision is made
 * against the stored detection. A file the worker has not finished, or a video,
 * has no preview yet and says which.
 *
 * The delete button sits outside the anchor rather than inside it, for the same
 * reason as the person tile's: a button nested in a link is invalid, and
 * clicking it must not also navigate.
 */
export function MediaCard({
  media,
  selected,
  onSelect,
  onOutcome,
  onDeleted,
}: {
  media: Media;
  selected: boolean;
  onSelect: (mediaId: string) => void;
  onOutcome: (notice: GalleryNotice) => void;
  onDeleted: () => void;
}) {
  const href = hrefFor({ view: "viewer", mediaId: media.id });
  const name = truncateHash(media.sha256);
  const faces = `${String(media.detection_count)} face${media.detection_count === 1 ? "" : "s"}`;
  return (
    <div className="media-tile">
      {/* Outside the anchor, like the delete button: selecting a file must not open it. */}
      <label className="review-select">
        <input
          type="checkbox"
          checked={selected}
          aria-label={`Select ${name}`}
          onChange={() => onSelect(media.id)}
        />
        Select
      </label>
      <a
        className={selected ? "media-card selected" : "media-card"}
        href={href}
        aria-label={`${name}: ${media.status}, ${faces}`}
      >
        {media.kind === "image" && media.status === "done" ? (
          <img
            src={`/api/media/${encodeURIComponent(media.id)}/thumbnail`}
            alt={`Preview of ${truncateHash(media.sha256)}`}
            loading="lazy"
          />
        ) : (
          <div className="crop-missing">
            {media.kind === "video" ? "Video" : "No preview yet"}
          </div>
        )}
        <div className="media-card-body">
          <strong className="mono">{name}</strong>
          <div className="media-card-facts">
            <span className={`status-chip status-${media.status}`}>{media.status}</span>
            <span className="muted">{faces}</span>
          </div>
        </div>
      </a>
      <div className="media-tile-actions">
        <DeleteMediaButton media={media} onOutcome={onOutcome} onDeleted={onDeleted} />
      </div>
    </div>
  );
}

/**
 * Delete one file and everything derived from it (spec 12). Irreversible.
 *
 * The same one-click confirm as the person delete, and for the same reason: a
 * form that has to be argued with gets clicked through rather than read. What
 * it says on the way back is what it actually removed, because this purge can
 * reach further than the file — a face enrolled from it leaves the gallery with
 * it, and the person is left `unenrolled` rather than deleted.
 */
export function DeleteMediaButton({
  media,
  onOutcome,
  onDeleted,
}: {
  media: Media;
  onOutcome: (notice: GalleryNotice) => void;
  onDeleted: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const name = truncateHash(media.sha256);

  async function remove(): Promise<void> {
    setBusy(true);
    try {
      const result = await deleteJson<MediaPurgeResult>(
        `/api/media/${encodeURIComponent(media.id)}`,
      );
      onOutcome({ tone: "success", text: purgeText(name, result) });
      onDeleted();
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 404) {
        // The intended state is the actual state; another window got there first.
        onDeleted();
      } else {
        onOutcome({
          tone: "error",
          text:
            failure instanceof ApiError
              ? `Could not delete ${name} (${String(failure.status)}): ${failure.detail}`
              : `Could not delete ${name}: ${errorMessage(failure)}`,
        });
      }
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  }

  if (!confirming) {
    return (
      <button
        type="button"
        className="icon-button danger"
        title={`Delete ${name}`}
        aria-label={`Delete ${name}`}
        onClick={() => setConfirming(true)}
      >
        <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
          <path
            d="M6 2h4v1h4v1.5H2V3h4V2zM3.5 6h9l-.7 8H4.2L3.5 6zM6.5 7.5v5M9.5 7.5v5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.2"
          />
        </svg>
      </button>
    );
  }

  return (
    <span className="delete-confirm">
      <button type="button" className="danger" disabled={busy} onClick={() => void remove()}>
        {busy ? "Deleting…" : "Delete"}
      </button>
      <button type="button" disabled={busy} onClick={() => setConfirming(false)}>
        Cancel
      </button>
    </span>
  );
}

/**
 * What actually went, in the order it matters: the faces, then the gallery, then
 * the bytes. The template and unenrolment clauses are conditional because they
 * are usually zero, and a sentence that always recites zeroes stops being read.
 */
function purgeText(name: string, result: MediaPurgeResult): string {
  const parts = [
    `${name} is deleted: ${String(result.detections)} detection(s) and ${String(result.tracks)} track(s) with it`,
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
    result.object_removed
      ? "the stored bytes are gone"
      : "the stored bytes stay: another case holds the same file",
  );
  return `${parts.join("; ")}.`;
}
