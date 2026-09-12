/**
 * Hash routing.
 *
 * A hash router needs no server-side rewrite, which matters because the
 * backend serves the built bundle as static files with no SPA fallback. The
 * route is a discriminated union so every view gets its parameters typed.
 */

import { useSyncExternalStore } from "react";

import type { Rect } from "./geometry";

export type Route =
  | { view: "status" }
  | { view: "media" }
  /**
   * `focus` carries the box the operator clicked in live mode, in fractions of
   * the frame, so the detail view can open the tag panel on that same face
   * instead of defaulting to the first track. It rides in the hash rather than
   * in module state so a reload of the handoff URL still lands on the face.
   */
  | { view: "viewer"; mediaId: string; focus?: Rect }
  | { view: "persons" }
  | { view: "person"; personId: string }
  | { view: "review" }
  | { view: "live" };

function parseFocus(raw: string | undefined): Rect | null {
  if (raw === undefined) {
    return null;
  }
  // `hrefFor` writes bare commas, but a hash that went through an encoder on
  // the way here arrives as %2C; a focus box that fails to parse silently
  // sends the operator back to whichever track is first, so decode it.
  const parts = decodeURIComponent(raw)
    .split(",")
    .map((value) => Number.parseFloat(value));
  const [x, y, w, h] = parts;
  if (parts.length !== 4 || x === undefined || y === undefined || w === undefined || h === undefined) {
    return null;
  }
  if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) {
    return null;
  }
  return { x, y, w, h };
}

export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#\/?/, "");
  const parts = path.split("/").filter((part) => part.length > 0);
  const [head, second, third, fourth] = parts;
  switch (head) {
    case undefined:
    case "status":
      return { view: "status" };
    case "media": {
      if (second === undefined) {
        return { view: "media" };
      }
      const focus = third === "box" ? parseFocus(fourth) : null;
      return focus === null
        ? { view: "viewer", mediaId: second }
        : { view: "viewer", mediaId: second, focus };
    }
    case "persons":
      return second === undefined ? { view: "persons" } : { view: "person", personId: second };
    case "review":
      return { view: "review" };
    case "live":
      return { view: "live" };
    default:
      return { view: "status" };
  }
}

export function hrefFor(route: Route): string {
  switch (route.view) {
    case "status":
      return "#/status";
    case "media":
      return "#/media";
    case "viewer": {
      const base = `#/media/${encodeURIComponent(route.mediaId)}`;
      const focus = route.focus;
      if (focus === undefined) {
        return base;
      }
      // Digits, dots, and commas are all legal in a fragment: no encoding, so
      // the hash stays readable and round-trips through `parseFocus`.
      const box = [focus.x, focus.y, focus.w, focus.h].map((value) => value.toFixed(5)).join(",");
      return `${base}/box/${box}`;
    }
    case "persons":
      return "#/persons";
    case "person":
      return `#/persons/${encodeURIComponent(route.personId)}`;
    case "review":
      return "#/review";
    case "live":
      return "#/live";
  }
}

export function navigate(route: Route): void {
  window.location.hash = hrefFor(route);
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener("hashchange", onChange);
  return () => {
    window.removeEventListener("hashchange", onChange);
  };
}

export function useRoute(): Route {
  // `useSyncExternalStore` needs a stable snapshot value, and `parseRoute`
  // allocates, so subscribe to the raw hash string and parse in the caller.
  const hash = useSyncExternalStore(subscribe, () => window.location.hash);
  return parseRoute(hash);
}
