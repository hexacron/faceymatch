/**
 * Shared presentation constants and formatters.
 *
 * The band colours live here because both the DOM (pills, badges) and the
 * overlay canvas draw them, and a divergence between the two would make the
 * legend lie about the boxes.
 */

import type { Band } from "../api/types";

export const BAND_COLOR: Record<Band, string> = {
  strong: "#7fbf7f",
  possible: "#d9b96f",
  ambiguous: "#c98fd0",
  unknown: "#8b95a4",
};

/** Colour for a track with no match at all. */
export const NO_BAND_COLOR = "#5a626e";

export const SELECTED_COLOR = "#6fb3d2";

/**
 * Render a similarity as a raw number. Spec 6.4: the UI shows the band first
 * and the score second, and never labels the score a probability, so this is
 * deliberately a bare fixed-point number with no unit and no percentage.
 */
export function formatScore(score: number | null): string {
  return score === null ? "\u2014" : score.toFixed(3);
}

export function truncateHash(hash: string): string {
  return hash.length > 12 ? `${hash.slice(0, 12)}\u2026` : hash;
}

/** Trim microseconds to milliseconds; timestamps stay in UTC. */
export function formatTs(ts: string): string {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?/.exec(ts);
  if (match === null) {
    return ts;
  }
  const [, date, time, fraction] = match;
  const millis = (fraction ?? "").padEnd(3, "0").slice(0, 3);
  return `${date ?? ""} ${time ?? ""}.${millis}Z`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${String(bytes)} B`;
  }
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit] ?? "GB"}`;
}
