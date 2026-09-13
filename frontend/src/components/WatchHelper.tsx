import { useState } from "react";

import { ApiError, errorMessage, postJson } from "../api/client";
import type { WatchLaunch, WatchStatus } from "../api/types";
import { useResource } from "../lib/useResource";

/**
 * Start the watch helper (spec 6.11) from the web UI.
 *
 * The helper follows one window, display, or region of this machine outside the
 * browser, which is the one thing this view cannot do: a hidden tab throttles
 * the sampling loop to nothing. It is the same tier 2 match either way.
 *
 * Three rules this control exists to keep:
 *
 *   - The press is the whole trigger. Nothing here launches on mount, retries a
 *     refusal, or polls a status that could start anything.
 *   - The request carries no body, because nothing a caller sends may reach the
 *     helper's command line.
 *   - A backend that says the helper is unavailable gets no button at all, only
 *     the sentence explaining why. `GET /api/watch` that is still loading, or
 *     that a bundle older than the endpoint cannot reach, is also not an offer.
 */
export function WatchHelper() {
  const status = useResource<WatchStatus>("/api/watch");
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<{ ok: boolean; text: string } | null>(null);

  async function launch(): Promise<void> {
    setBusy(true);
    setOutcome(null);
    try {
      const started = await postJson<WatchLaunch>("/api/watch/launch", null);
      setOutcome({
        ok: true,
        text:
          `Watch helper started (pid ${String(started.pid)}). Pick a window, display, or ` +
          `region in its own window and press Start there; it captures nothing until you do. ` +
          `Anything it reports goes to ${started.log_path}.`,
      });
    } catch (failure) {
      setOutcome({
        ok: false,
        text:
          failure instanceof ApiError
            ? `Could not start the watch helper (${String(failure.status)}): ${failure.detail}`
            : `Could not start the watch helper: ${errorMessage(failure)}`,
      });
    } finally {
      setBusy(false);
      // One refetch after the attempt, so a helper that started — or one that
      // has since been closed — is reflected without a loop that could start
      // anything on its own.
      status.reload();
    }
  }

  if (status.state.phase !== "ready") {
    return null;
  }
  const watch = status.state.data;

  return (
    <div className="notice watch-helper">
      {watch.available && (
        <button type="button" disabled={busy} onClick={() => void launch()}>
          {busy ? "Starting the helper…" : "Open the watch helper"}
        </button>
      )}
      <span>
        {watch.available
          ? watch.running
            ? `A watch helper started from here is running (pid ${String(watch.pid)}); it follows one window, display, or region outside the browser. Stop it from its own window.`
            : "The watch helper follows one window, display, or region of this machine outside the browser, through the same match path, and stores nothing of its own."
          : (watch.reason ?? "The watch helper is unavailable on this install.")}
      </span>
      {outcome !== null && (
        <p
          className={outcome.ok ? "compact" : "field-error"}
          role={outcome.ok ? "status" : "alert"}
        >
          {outcome.text}
        </p>
      )}
    </div>
  );
}
