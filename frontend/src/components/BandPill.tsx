import type { Band, IdentitySource } from "../api/types";
import { BAND_COLOR, formatScore, NO_BAND_COLOR } from "../lib/display";

/**
 * Band first, raw score second (spec 6.4). The score is printed as a bare
 * number and is never called a probability or shown as a percentage.
 */
export function BandPill({ band, score }: { band: Band | null; score: number | null }) {
  const color = band === null ? NO_BAND_COLOR : BAND_COLOR[band];
  return (
    <span className="band-pill" style={{ borderColor: color, color }}>
      <span className="band-name">{band ?? "no match"}</span>
      <span className="band-score mono" title="cosine similarity, not a probability">
        {formatScore(score)}
      </span>
    </span>
  );
}

/** Operator decisions get a distinct badge so they never read as auto (C5, 6.5). */
export function SourceBadge({ source }: { source: IdentitySource | null }) {
  if (source === null) {
    return <span className="pill">untagged</span>;
  }
  return (
    <span className={source === "operator" ? "badge-operator" : "badge-auto"}>
      {source === "operator" ? "\u2713 operator" : "auto"}
    </span>
  );
}
