/**
 * How the operator wants a library laid out, per library.
 *
 * A module-level store behind `useSyncExternalStore`, backed by localStorage,
 * so the choice survives a reload and every mount of the view agrees on it —
 * the same shape the selected case uses. Tiles are the default in both
 * galleries because the fastest way to find the right face, or the right file,
 * is to look at it; the list is for the columns the tiles leave out.
 *
 * One factory rather than one module per view: the two stores differ by a
 * storage key and nothing else, and a second copy is a second place for the
 * default to drift.
 */

import { useSyncExternalStore } from "react";

export type Layout = "tiles" | "list";

type LayoutPreference = {
  use: () => Layout;
  set: (next: Layout) => void;
};

function layoutPreference(storageKey: string): LayoutPreference {
  let layout: Layout = "tiles";
  try {
    layout = window.localStorage.getItem(storageKey) === "list" ? "list" : "tiles";
  } catch {
    // Storage can be blocked; an in-memory choice still works for the session.
  }

  const listeners = new Set<() => void>();
  // useSyncExternalStore needs a stable subscribe identity.
  const subscribe = (onChange: () => void): (() => void) => {
    listeners.add(onChange);
    return () => {
      listeners.delete(onChange);
    };
  };

  return {
    use: () => useSyncExternalStore(subscribe, () => layout),
    set: (next: Layout) => {
      if (next === layout) {
        return;
      }
      layout = next;
      try {
        window.localStorage.setItem(storageKey, next);
      } catch {
        // Losing persistence is not worth failing a layout switch over.
      }
      for (const listener of listeners) {
        listener();
      }
    },
  };
}

const persons = layoutPreference("faceymatch.persons-layout");
const media = layoutPreference("faceymatch.media-layout");

export const usePersonsLayout = persons.use;
export const setPersonsLayout = persons.set;
export const useMediaLayout = media.use;
export const setMediaLayout = media.set;
