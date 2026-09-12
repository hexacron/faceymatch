import { useEffect, useMemo, useState } from "react";

import { errorMessage, postJson } from "../api/client";
import type {
  Band,
  ReviewBulkRequest,
  ReviewBulkResult,
  ReviewItem,
  ReviewList,
} from "../api/types";
import { BandPill } from "../components/BandPill";
import { Loaded } from "../components/Loading";
import { truncateHash } from "../lib/display";
import { hrefFor } from "../lib/router";
import { useResource } from "../lib/useResource";

const REVIEW_BANDS: readonly Band[] = ["possible", "ambiguous"];

export default function ReviewView() {
  const [band, setBand] = useState<Band>("possible");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<ReviewBulkResult | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const path = useMemo(() => `/api/review?band=${encodeURIComponent(band)}&limit=100`, [band]);
  const review = useResource<ReviewList>(path, 2_000);

  useEffect(() => {
    setSelected(new Set());
    setResult(null);
    setFailure(null);
  }, [band]);

  function toggle(trackId: string): void {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(trackId)) {
        next.delete(trackId);
      } else {
        next.add(trackId);
      }
      return next;
    });
  }

  async function apply(decision: "confirm" | "reject", items: readonly ReviewItem[]): Promise<void> {
    const chosen = items.filter((item) => selected.has(item.track_id));
    if (chosen.length === 0) {
      return;
    }
    const body: ReviewBulkRequest = {
      decisions: chosen.map((item) => ({
        track_id: item.track_id,
        decision,
        ...(decision === "confirm" ? { person_id: item.person_id } : {}),
      })),
    };
    setPending(true);
    setFailure(null);
    setResult(null);
    try {
      const response = await postJson<ReviewBulkResult>("/api/review/bulk", body);
      setResult(response);
      setSelected(new Set());
      review.reload();
    } catch (error) {
      setFailure(errorMessage(error));
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <div className="view-heading">
        <div>
          <h1>Review queue</h1>
          <p className="tagline">Optional operator review for non-strong gallery candidates.</p>
        </div>
        <button type="button" onClick={review.reload}>Refresh</button>
      </div>
      <div className="toolbar panel">
        <label>
          Match band
          <select value={band} onChange={(event) => setBand(event.currentTarget.value as Band)}>
            {REVIEW_BANDS.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
      </div>
      <Loaded state={review.state} label="review queue">
        {(list) => (
          <>
            {list.items.length === 0 ? (
              <p className="notice">No {band} matches are waiting for review.</p>
            ) : (
              <>
                <div className="bulk-bar panel">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={selected.size === list.items.length}
                      onChange={(event) => setSelected(
                        event.currentTarget.checked
                          ? new Set(list.items.map((item) => item.track_id))
                          : new Set(),
                      )}
                    />
                    Select all ({selected.size} selected)
                  </label>
                  <button type="button" disabled={pending || selected.size === 0} onClick={() => void apply("confirm", list.items)}>
                    Confirm selected
                  </button>
                  <button className="danger" type="button" disabled={pending || selected.size === 0} onClick={() => void apply("reject", list.items)}>
                    Reject selected
                  </button>
                </div>
                <div className="review-grid">
                  {list.items.map((item) => (
                    <article className={selected.has(item.track_id) ? "review-card selected" : "review-card"} key={item.track_id}>
                      <label className="review-select">
                        <input type="checkbox" checked={selected.has(item.track_id)} onChange={() => toggle(item.track_id)} />
                        Select {item.name}
                      </label>
                      {item.crop_sha256 === null ? (
                        <div className="crop-missing">No crop</div>
                      ) : (
                        <img src={`/api/crops/${encodeURIComponent(item.crop_sha256)}`} alt={`Face crop for ${item.name}`} />
                      )}
                      <div className="review-card-body">
                        <strong>{item.name}</strong>
                        <BandPill band={item.band} score={item.score} />
                        <a href={hrefFor({ view: "viewer", mediaId: item.media_id })}>Open in media</a>
                        <span className="muted mono" title={item.track_id}>track {truncateHash(item.track_id)}</span>
                      </div>
                    </article>
                  ))}
                </div>
              </>
            )}
            {result !== null && (
              <div className={result.errors.length === 0 ? "notice success" : "notice error"} role="status">
                Applied {result.applied} decisions.
                {result.errors.length > 0 && (
                  <ul>{result.errors.map((item) => <li key={item.track_id}>{truncateHash(item.track_id)}: {item.error}</li>)}</ul>
                )}
              </div>
            )}
          </>
        )}
      </Loaded>
      {failure !== null && <p className="notice error" role="alert">{failure}</p>}
    </>
  );
}
