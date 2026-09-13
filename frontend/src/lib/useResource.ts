/**
 * Loading state for a single GET endpoint, with optional polling.
 *
 * A poll tick never drops back to `loading`: the media library refreshes every
 * second while a file is processed, and blanking the table on each tick would
 * make it unusable. Errors during a poll replace the data so a backend that
 * goes away is visible, but a stale-then-error flicker is fine here.
 *
 * Two things a poll loop must not do, both of which this one used to:
 *
 *   - Poll a hidden tab. Nobody is reading it, and the requests compete with
 *     the live loop for the single uvicorn worker.
 *   - Poll a settled resource forever. A `done` media page fired 2 req/s until
 *     it was navigated away from, permanently, for a row that cannot change.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { errorMessage, getJson } from "../api/client";

export type Loadable<T> =
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "ready"; data: T };

export type Resource<T> = {
  state: Loadable<T>;
  /** Refetch now, without clearing the current data. */
  reload: () => void;
};

export type ResourceOptions<T> = {
  /**
   * Stop polling once this holds. The resource has reached a state it cannot
   * leave on its own — a finished job, a processed file — so another request
   * can only ever return the same bytes. `reload()` still works.
   */
  stopWhen?: (data: T) => boolean;
};

export function useResource<T>(
  path: string,
  pollMs = 0,
  options: ResourceOptions<T> = {},
): Resource<T> {
  const [state, setState] = useState<Loadable<T>>({ phase: "loading" });
  const [token, setToken] = useState(0);
  const pathRef = useRef(path);
  // Read through a ref: a caller writing `stopWhen: (m) => m.status === "done"`
  // inline gives a new function every render, and depending on it would restart
  // the poll loop on every tick it caused.
  const stopWhenRef = useRef(options.stopWhen);
  stopWhenRef.current = options.stopWhen;

  const reload = useCallback(() => {
    setToken((value) => value + 1);
  }, []);

  useEffect(() => {
    if (pathRef.current !== path) {
      pathRef.current = path;
      setState({ phase: "loading" });
    }
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let settled = false;

    const schedule = (): void => {
      if (pollMs <= 0 || settled || controller.signal.aborted || document.hidden) {
        return;
      }
      timer = setTimeout(() => void run(), pollMs);
    };

    const run = async (): Promise<void> => {
      try {
        const data = await getJson<T>(path, controller.signal);
        setState({ phase: "ready", data });
        settled = stopWhenRef.current?.(data) ?? false;
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setState({ phase: "error", message: errorMessage(error) });
      }
      schedule();
    };

    // A tab that becomes visible again is looking at data as old as the time it
    // spent hidden, so it refetches immediately rather than waiting a tick.
    const onVisibility = (): void => {
      if (document.hidden) {
        clearTimeout(timer);
        return;
      }
      if (!settled && !controller.signal.aborted) {
        clearTimeout(timer);
        void run();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    void run();

    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      controller.abort();
      clearTimeout(timer);
    };
  }, [path, pollMs, token]);

  return { state, reload };
}
