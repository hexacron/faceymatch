/**
 * How the operator wants the gallery laid out.
 *
 * Persisted the same way the selected case is: a module-level store behind
 * `useSyncExternalStore`, backed by localStorage, so the choice survives a
 * reload and every mount of the view agrees on it. Tiles are the default
 * because a face system's gallery is faces; the table is for the columns the
 * tiles leave out.
 */

import { useSyncExternalStore } from "react";

export type PersonsLayout = "tiles" | "list";

const LAYOUT_STORAGE_KEY = "faceymatch.persons-layout";

let layout: PersonsLayout = "tiles";
try {
  layout = window.localStorage.getItem(LAYOUT_STORAGE_KEY) === "list" ? "list" : "tiles";
} catch {
  // Storage can be blocked; an in-memory choice still works for the session.
}

const listeners = new Set<() => void>();

export function setPersonsLayout(next: PersonsLayout): void {
  if (next === layout) {
    return;
  }
  layout = next;
  try {
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, next);
  } catch {
    // Losing persistence is not worth failing a layout switch over.
  }
  for (const listener of listeners) {
    listener();
  }
}

// useSyncExternalStore needs a stable subscribe identity.
function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  return () => {
    listeners.delete(onChange);
  };
}

export function usePersonsLayout(): PersonsLayout {
  return useSyncExternalStore(subscribe, () => layout);
}
