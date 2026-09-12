import { dismissIngestNotice, useIngestNotices } from "../lib/ingest";
import { hrefFor } from "../lib/router";

/**
 * Results of the global ingest paths.
 *
 * A paste or a capture can start anywhere and ends with a route change to the
 * new media, so the result cannot live inside the media library panel: it
 * would unmount before the operator read it. This stack is app-level and
 * survives the navigation.
 */
export function IngestNotices() {
  const notices = useIngestNotices();
  if (notices.length === 0) {
    return null;
  }
  return (
    <div className="ingest-notices" aria-live="polite" aria-label="Ingest results">
      {notices.map((notice) => (
        <div
          className={`notice ingest-notice ${notice.tone === "success" ? "success" : notice.tone === "error" ? "error" : ""}`}
          key={notice.id}
          role={notice.tone === "error" ? "alert" : "status"}
        >
          <p>{notice.text}</p>
          <div className="ingest-notice-actions">
            {notice.mediaId !== null && (
              <a className="button-link" href={hrefFor({ view: "viewer", mediaId: notice.mediaId })}>
                Open
              </a>
            )}
            <button
              type="button"
              aria-label={`Dismiss: ${notice.text}`}
              onClick={() => {
                dismissIngestNotice(notice.id);
              }}
            >
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
