/**
 * Loading state for a single GET endpoint, with optional polling.
 *
 * A poll tick never drops back to `loading`: the media library refreshes every
 * second while a file is processed, and blanking the table on each tick would
 * make it unusable. Errors during a poll replace the data so a backend that
 * goes away is visible, but a stale-then-error flicker is fine here.
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

export function useResource<T>(path: string, pollMs = 0): Resource<T> {
  const [state, setState] = useState<Loadable<T>>({ phase: "loading" });
  const [token, setToken] = useState(0);
  const pathRef = useRef(path);

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

    const run = async (): Promise<void> => {
      try {
        const data = await getJson<T>(path, controller.signal);
        setState({ phase: "ready", data });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setState({ phase: "error", message: errorMessage(error) });
      }
      if (pollMs > 0 && !controller.signal.aborted) {
        timer = setTimeout(() => void run(), pollMs);
      }
    };

    void run();

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [path, pollMs, token]);

  return { state, reload };
}
